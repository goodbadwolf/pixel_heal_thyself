"""Base trainer for PHT models."""

import math
import os
import random
import time
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import lpips
import numpy as np
import torch
from torch import optim
from torch.nn import Module
from torch.optim import lr_scheduler

from pht.config.base import Config
from pht.logger import logger
from pht.models.afgsa.dataset import Dataset
from pht.models.afgsa.discriminators import MultiScaleDiscriminator
from pht.models.afgsa.gen_hdf5 import Hdf5Constructor
from pht.models.afgsa.metric import calculate_psnr, calculate_rmse, calculate_ssim
from pht.models.afgsa.model import DiscriminatorVGG, print_model_structure
from pht.models.afgsa.prefetch_dataloader import DataLoaderX
from pht.models.afgsa.preprocessing import (
    postprocess_specular,
    preprocess_normal,
    preprocess_specular,
)
from pht.models.afgsa.util import (
    create_folder,
    save_img_group,
    tensor2img,
)
from pht.models.losses import (
    GANLoss,
    GradientPenaltyLoss,
    L1ReconstructionLoss,
    RaHingeGANLoss,
    SSIMLoss,
)
from pht.utils import run_once_multiprocessing

# Global constants
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
permutation = [0, 3, 1, 2]  # NHWC → NCHW


@run_once_multiprocessing
def set_determinism(
    seed: int,
    deterministic: bool = True,
    _prefix: str | None = None,
) -> None:
    """Set determinism for the training process."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(
    worker_id: int,
    base_seed: int,
    deterministic: bool,
) -> None:
    """Set determinism for the training process."""
    set_determinism(
        base_seed + worker_id,
        deterministic,
        _prefix=f"Trainer worker {worker_id}",
    )


class BaseTrainer(ABC):
    """Base class for all model trainers in the PHT project."""

    def __init__(self, cfg: Config) -> None:
        """
        Initialize the trainer with configuration.

        Args:
            cfg: Typed configuration object

        """
        self.cfg = cfg
        self.deterministic = cfg.trainer.deterministic
        # Extract model name from the trainer class name (e.g., AFGSATrainer -> AFGSA)
        self.model_name = self.__class__.__name__.replace("Trainer", "")

        # Set up deterministic training if requested
        set_determinism(self.cfg.seed, self.deterministic, self.model_name)

    @abstractmethod
    def create_generator(self) -> Module:
        """
        Create and return the generator model.

        Returns:
            A PyTorch model that serves as the generator

        """

    def create_discriminator(self) -> Module:
        """
        Create and return the discriminator model.

        Returns:
            A PyTorch model that serves as the discriminator

        """
        if self.cfg.model.discriminator.use_multiscale_discriminator:
            return MultiScaleDiscriminator(
                in_nc=self.cfg.model.input_channels,
                patch_size=self.cfg.data.patches.patch_size,
            ).to(device)
        return DiscriminatorVGG(3, 64, self.cfg.data.patches.patch_size).to(device)

    def create_losses(
        self,
    ) -> Tuple[Module, Module, Module, Optional[Module], Optional[Module]]:
        """
        Create and return the loss functions.

        Returns:
            A tuple containing (l1_loss, gan_loss, gp_loss, lpips_loss, ssim_loss)

        """
        l1_loss = L1ReconstructionLoss().to(device)
        gan_loss = (
            RaHingeGANLoss().to(device)
            if self.cfg.model.discriminator.use_multiscale_discriminator
            else GANLoss("wgan").to(device)
        )
        gp_loss = GradientPenaltyLoss(device).to(device)
        lpips_loss = (
            lpips.LPIPS(net="vgg").to(device)
            if self.cfg.model.losses.use_lpips_loss
            else None
        )
        ssim_loss = (
            SSIMLoss(window_size=11).to(device)
            if self.cfg.model.losses.use_ssim_loss
            else None
        )
        return l1_loss, gan_loss, gp_loss, lpips_loss, ssim_loss

    def create_optimizers(
        self,
        G: Module,  # noqa: N803
        D: Module,  # noqa: N803
    ) -> Tuple[
        optim.Optimizer,
        lr_scheduler.LRScheduler,
        optim.Optimizer,
        lr_scheduler.LRScheduler,
    ]:
        """
        Create and return the optimizers and schedulers.

        Args:
            G: Generator model
            D: Discriminator model

        Returns:
            A tuple containing (optimizer_G, scheduler_G, optimizer_D, scheduler_D)

        """
        milestones = [
            i * self.cfg.trainer.lr_milestone - 1
            for i in range(1, self.cfg.trainer.epochs // self.cfg.trainer.lr_milestone)
        ]

        optimizer_generator = optim.Adam(
            G.parameters(),
            lr=self.cfg.trainer.lr_g,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler_generator = lr_scheduler.MultiStepLR(
            optimizer_generator,
            milestones=milestones,
            gamma=0.5,
        )

        optimizer_discriminator = optim.Adam(
            D.parameters(),
            lr=self.cfg.trainer.lr_d,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler_discriminator = lr_scheduler.MultiStepLR(
            optimizer_discriminator,
            milestones=milestones,
            gamma=0.5,
        )

        return (
            optimizer_generator,
            scheduler_generator,
            optimizer_discriminator,
            scheduler_discriminator,
        )

    def print_training_config(self) -> None:
        """Print the training configuration."""
        logger.info(f"Creating {self.model_name}")
        logger.info(f"{self.model_name} padding mode: {self.padding_mode}")
        logger.info(f"{self.model_name} curve order: {self.cfg.model.curve_order}")
        logger.info(f"{self.model_name} L1 lossW: {self.cfg.model.losses.l1_loss_w}")
        logger.info(f"{self.model_name} GAN lossW: {self.cfg.model.losses.gan_loss_w}")
        logger.info(f"{self.model_name} GP lossW: {self.cfg.model.losses.gp_loss_w}")
        if self.cfg.model.losses.use_lpips_loss:
            logger.info(
                f"{self.model_name} LPIPS lossW: {self.cfg.model.losses.lpips_loss_w}",
            )
        if self.cfg.model.losses.use_ssim_loss:
            logger.info(
                f"{self.model_name} SSIM lossW: {self.cfg.model.losses.ssim_loss_w}",
            )
        if self.cfg.model.discriminator.use_multiscale_discriminator:
            logger.info(f"{self.model_name} multiscale discriminator")
        if self.cfg.model.discriminator.use_film:
            logger.info(f"{self.model_name} use FiLM")

    def setup_dataloaders(self) -> Tuple[DataLoaderX, DataLoaderX, int, int]:
        """
        Set up and return the training and validation dataloaders.

        Returns:
            A tuple containing (train_dataloader, val_dataloader, train_num_samples, val_num_samples)

        """
        train_save_path = os.path.join(self.cfg.data.patches.dir, "train.h5")
        val_save_path = os.path.join(self.cfg.data.patches.dir, "val.h5")

        # Create datasets if they don't exist
        exist = True
        for path in [train_save_path, val_save_path]:
            if not os.path.exists(path):
                exist = False
        if not exist:
            logger.info(f"Creating dataset: patches in {self.cfg.data.patches.dir}")
            os.makedirs(self.cfg.data.patches.dir, exist_ok=True)
            constructor = Hdf5Constructor(
                self.cfg.data.images.dir,
                self.cfg.data.patches.dir,
                self.cfg.data.patches.patch_size,
                self.cfg.data.patches.num_patches,
                self.cfg.seed,
                self.cfg.data_ratio,
                scale=self.cfg.data.images.scale,
            )
            constructor.construct_hdf5()

        # Create dataloaders
        train_dataset = Dataset(train_save_path)
        train_num_samples = len(train_dataset)
        if self.deterministic:
            g = torch.Generator()
            g.manual_seed(self.cfg.seed)
            train_dataloader = DataLoaderX(
                train_dataset,
                batch_size=self.cfg.trainer.batch_size,
                shuffle=True,
                generator=g,
                num_workers=7,
                pin_memory=True,
                worker_init_fn=lambda wid: worker_init_fn(
                    wid,
                    self.cfg.seed,
                    self.deterministic,
                ),
            )
        else:
            train_dataloader = DataLoaderX(
                train_dataset,
                batch_size=self.cfg.trainer.batch_size,
                shuffle=True,
                num_workers=7,
                pin_memory=True,
            )

        val_dataset = Dataset(val_save_path)
        val_num_samples = len(val_dataset)

        # For validation, we want deterministic behavior regardless of training mode
        if self.deterministic:
            g = torch.Generator()
            g.manual_seed(self.cfg.seed)
            val_dataloader = DataLoaderX(
                val_dataset,
                batch_size=1,
                shuffle=False,
                generator=g,
                num_workers=7,
                pin_memory=True,
                worker_init_fn=lambda wid: worker_init_fn(
                    wid,
                    self.cfg.seed,
                    self.deterministic,
                ),
            )
        else:
            val_dataloader = DataLoaderX(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=7,
                pin_memory=True,
            )

        return train_dataloader, val_dataloader, train_num_samples, val_num_samples

    def train(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Train the model."""
        logger.info(
            f"Starting training: model={self.model_name}, seed={self.cfg.seed}, "
            f"batch_size={self.cfg.trainer.batch_size}, epochs={self.cfg.trainer.epochs}",
        )
        logger.info(f"Loading dataset: patches from {self.cfg.data.patches.dir}")
        train_dataloader, val_dataloader, train_num_samples, val_num_samples = (
            self.setup_dataloaders()
        )

        self.padding_mode = "replicate" if self.deterministic else "reflect"
        # Create models and training components
        self.print_training_config()
        G = self.create_generator()  # noqa: N806
        D = self.create_discriminator()  # noqa: N806

        # Load model if specified
        if self.cfg.trainer.load_model:
            G.load_state_dict(
                torch.load(os.path.join(self.cfg.trainer.model_path, "G.pt")),
            )
            D.load_state_dict(
                torch.load(os.path.join(self.cfg.trainer.model_path, "D.pt")),
            )

        print_model_structure(G)
        print_model_structure(D)

        l1_loss, gan_loss, gp_loss, lpips_loss, ssim_loss = self.create_losses()
        (
            optimizer_generator,
            scheduler_generator,
            optimizer_discriminator,
            scheduler_discriminator,
        ) = self.create_optimizers(G, D)

        # Training loop
        accumulated_generator_loss = 0
        accumulated_discriminator_loss = 0
        total_iterations = math.ceil(train_num_samples / self.cfg.trainer.batch_size)
        save_img_interval = val_num_samples // self.cfg.trainer.num_saved_imgs
        root_save_path = self.cfg.paths.output_dir

        logger.info("Start training")
        end = None
        for epoch in range(self.cfg.trainer.epochs):
            start = time.time()
            for i_batch, batch_sample in enumerate(train_dataloader):
                # Process input data
                aux_features = batch_sample["aux"]
                aux_features[:, :, :, :3] = torch.FloatTensor(
                    preprocess_normal(aux_features[:, :, :, :3]),
                )
                aux_features = aux_features.permute(permutation).to(device)
                noisy = batch_sample["noisy"]
                noisy = preprocess_specular(noisy)
                noisy = noisy.permute(permutation).to(device)
                gt = batch_sample["gt"]
                gt = preprocess_specular(gt)
                gt = gt.permute(permutation).to(device)

                end_io = time.time()
                io_took = end_io - end if i_batch != 0 else end_io - start

                output = G(noisy, aux_features)

                # Train discriminator
                optimizer_discriminator.zero_grad()
                pred_d_fake = D(output.detach())
                pred_d_real = D(gt)
                if self.cfg.model.discriminator.use_multiscale_discriminator:
                    discriminator_loss = gan_loss(pred_d_real, pred_d_fake)
                else:
                    try:
                        loss_d_real = gan_loss(pred_d_real, True)
                        loss_d_fake = gan_loss(pred_d_fake, False)
                        loss_gp = gp_loss(D, gt, output.detach())
                    except Exception as e:
                        logger.error(f"Error in discriminator loss calculation: {e}")
                        logger.error(f"Stack trace: {traceback.format_exc()}")
                    discriminator_loss = (
                        loss_d_fake + loss_d_real
                    ) / 2 + self.cfg.model.losses.gp_loss_w * loss_gp
                discriminator_loss.backward()
                optimizer_discriminator.step()
                accumulated_discriminator_loss += (
                    discriminator_loss.item() / self.cfg.trainer.batch_size
                )

                # Train generator
                optimizer_generator.zero_grad()
                pred_g_fake = D(output)
                try:
                    if self.cfg.model.discriminator.use_multiscale_discriminator:
                        with torch.no_grad():
                            pred_d_real_ng = D(gt)
                        loss_g_fake = gan_loss(pred_g_fake, pred_d_real_ng)
                    else:
                        loss_g_fake = gan_loss(pred_g_fake, True)
                    loss_l1 = l1_loss(output, gt)
                except Exception as e:
                    logger.error(f"Error in generator loss calculation: {e}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    break
                generator_loss = (
                    self.cfg.model.losses.gan_loss_w * loss_g_fake
                    + self.cfg.model.losses.l1_loss_w * loss_l1
                )

                def assert_nchw(x, name) -> None:  # noqa: ANN001
                    assert x.ndim == 4 and x.shape[1] == 3, f"{name} not NCHW/3‑ch"  # noqa: PLR2004, PT018, RUF001

                assert_nchw(output, "output")
                assert_nchw(gt, "gt")

                if self.cfg.model.losses.use_lpips_loss:

                    def to_lpips_range(x_log):  # noqa: ANN001, ANN202
                        x_lin = torch.exp(x_log) - 1.0
                        x_rgb = (x_lin / (x_lin.max() + 1e-6)).clamp(0, 1)
                        return x_rgb * 2 - 1

                    lpips_output = to_lpips_range(output)
                    lpips_gt = to_lpips_range(gt)
                    loss_lpips = lpips_loss(lpips_output, lpips_gt).mean()
                    generator_loss += self.cfg.model.losses.lpips_loss_w * loss_lpips
                if self.cfg.model.losses.use_ssim_loss:
                    loss_ssim = ssim_loss(output, gt)
                    generator_loss += self.cfg.model.losses.ssim_loss_w * loss_ssim
                generator_loss.backward()
                optimizer_generator.step()
                accumulated_generator_loss += (
                    generator_loss.item() / self.cfg.trainer.batch_size
                )

                iter_took = time.time() - start if i_batch == 0 else time.time() - end
                end = time.time()
                if i_batch % 10 == 0 or i_batch == total_iterations - 1:
                    logger.debug(
                        f"[Train] epoch={epoch + 1} iter={i_batch + 1}/{total_iterations} "
                        f"g_loss={accumulated_generator_loss / (i_batch + 1):.4f} "
                        f"d_loss={accumulated_discriminator_loss / (i_batch + 1):.4f} "
                        f"iter_time={iter_took:.2f}s io_time={io_took:.2f}s",
                    )

            logger.info(
                f"[Train] epoch={epoch + 1} summary: g_loss={accumulated_generator_loss / (i_batch + 1):.4f} "
                f"d_loss={accumulated_discriminator_loss / (i_batch + 1):.4f} time={int(end - start)}s",
            )

            # Save loss values
            with open(os.path.join(root_save_path, "train_loss.txt"), "a") as f:
                f.write(
                    f"Epoch: {epoch + 1} \tG loss: {accumulated_generator_loss / (i_batch + 1):.4f} "
                    f"\tD Loss: {accumulated_discriminator_loss / (i_batch + 1):.4f}\n",
                )

            scheduler_discriminator.step()
            scheduler_generator.step()
            accumulated_generator_loss = 0
            accumulated_discriminator_loss = 0

            # Validate and save model
            if epoch % self.cfg.trainer.save_interval == 0:
                self._validate_and_save(
                    epoch,
                    G,
                    D,
                    val_dataloader,
                    val_num_samples,
                    root_save_path,
                    save_img_interval,
                )

    def _validate_and_save(  # noqa: PLR0913
        self,
        epoch: int,
        G: Module,  # noqa: N803
        D: Module,  # noqa: N803
        val_dataloader: DataLoaderX,
        val_num_samples: int,
        root_save_path: str,
        save_img_interval: int,
    ) -> None:
        """
        Validate the model and save checkpoints.

        Args:
            epoch: Current training epoch
            G: Generator model
            D: Discriminator model
            val_dataloader: Validation data loader
            val_num_samples: Number of validation samples
            root_save_path: Base directory to save results
            save_img_interval: Interval at which to save validation images

        """
        current_save_path = create_folder(
            os.path.join(root_save_path, f"model_epoch{epoch + 1}"),
        )
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_mrse = 0.0
        start = time.time()

        with torch.no_grad():
            G.eval()
            # Save model
            torch.save(G.state_dict(), os.path.join(current_save_path, "G.pt"))
            torch.save(D.state_dict(), os.path.join(current_save_path, "D.pt"))

            for i_batch, batch_sample in enumerate(val_dataloader):
                aux_features = batch_sample["aux"]
                aux_features[:, :, :, :3] = torch.FloatTensor(
                    preprocess_normal(aux_features[:, :, :, :3]),
                )
                aux_features = aux_features.permute(permutation).to(device)
                noisy = batch_sample["noisy"]
                noisy = preprocess_specular(noisy)
                noisy = noisy.permute(permutation).to(device)
                gt = batch_sample["gt"]
                gt = gt.permute(permutation)

                output = G(noisy, aux_features)

                # Transfer to image
                output_c_n = postprocess_specular(output.cpu().numpy())
                gt_c_n = gt.numpy()
                noisy_c_n_255 = tensor2img(noisy.cpu().numpy(), post_spec=True)
                output_c_n_255 = tensor2img(output.cpu().numpy(), post_spec=True)
                gt_c_n_255 = tensor2img(gt.cpu().numpy())

                # Save image
                if i_batch % save_img_interval == 0:
                    save_img_group(
                        current_save_path,
                        i_batch,
                        noisy_c_n_255.copy(),
                        output_c_n_255.copy(),
                        gt_c_n_255.copy(),
                    )

                # Calculate metrics
                avg_mrse += calculate_rmse(output_c_n.copy(), gt_c_n.copy())
                avg_psnr += calculate_psnr(output_c_n_255.copy(), gt_c_n_255.copy())
                avg_ssim += calculate_ssim(output_c_n_255.copy(), gt_c_n_255.copy())

                end = time.time()
                if i_batch % 10 == 0 or i_batch == val_num_samples - 1:
                    logger.debug(
                        f"[Val] epoch={epoch + 1} iter={i_batch + 1}/{val_num_samples} "
                        f"mrse={avg_mrse / (i_batch + 1):.4f} psnr={avg_psnr / (i_batch + 1):.4f} "
                        f"ssim={avg_ssim / (i_batch + 1):.4f} "
                        f"val_time={end - start:.2f}s",
                    )
            G.train()

            avg_mrse /= val_num_samples
            avg_psnr /= val_num_samples
            avg_ssim /= val_num_samples
            logger.info(
                f"[Val] epoch={epoch + 1} summary: avg_mrse={avg_mrse:.4f} "
                f"avg_psnr={avg_psnr:.4f} avg_1-ssim={1 - avg_ssim:.4f} "
                f"time={int(end - start)}s",
            )

            # Save evaluation results
            with open(os.path.join(root_save_path, "evaluation.txt"), "a") as f:
                f.write(
                    f"Validation: {epoch + 1} \tAvg MRSE: {avg_mrse:.4f} "
                    f"\tAvg PSNR: {avg_psnr:.4f} \tAvg 1-SSIM: {1 - avg_ssim:.4f}\n",
                )
