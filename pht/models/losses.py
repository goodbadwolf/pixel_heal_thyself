"""PHT losses."""

from typing import Literal

import kornia
import torch
import torch.nn.functional as F  # noqa: N812
from torch import autograd, nn
from torchvision import models


class GradientPenaltyLoss(nn.Module):
    """Gradient penalty loss for GAN training."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.register_buffer("grad_outputs", torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input_data: torch.Tensor) -> torch.Tensor:
        """Get gradient outputs for the discriminator."""
        if self.grad_outputs.size() != input_data.size():
            self.grad_outputs.resize_(input_data.size()).fill_(1.0)
        return self.grad_outputs

    def forward(
        self,
        D: nn.Module,  # noqa: N803
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the gradient penalty loss."""
        # random weight term for interpolation between real and fake samples
        alpha = torch.rand(
            (real_data.shape[0], 1, 1, 1),
            dtype=torch.float32,
            device=self.device,
        )
        # get random interpolation between real and fake samples
        interpolates = alpha * fake_data.detach() + (1 - alpha) * real_data
        interpolates.requires_grad = True
        pred_d_interpolates = D(interpolates)

        grad_outputs = self.get_grad_outputs(pred_d_interpolates)
        grad_interp = torch.autograd.grad(
            outputs=pred_d_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_interp = grad_interp.reshape((grad_interp.size(0), -1))
        grad_interp_norm = grad_interp.norm(2, dim=1)

        return ((grad_interp_norm - 1) ** 2).mean()


class WDivGradientPenaltyLoss(nn.Module):
    """Wasserstein Divergence gradient penalty loss."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        discriminator: nn.Module,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        p: int = 6,
    ) -> torch.Tensor:
        """Forward pass for the Wasserstein Divergence gradient penalty loss."""
        # random weight term for interpolation between real and fake samples
        alpha = torch.rand(
            (real_data.shape[0], 1, 1, 1),
            dtype=torch.float32,
            device=real_data.device,
        )
        # get random interpolation between real and fake samples
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(
            True,
        )
        pred_d_interpolates = discriminator(interpolates)
        fake_grad_outputs = torch.ones(
            (real_data.shape[0], 1),
            dtype=real_data.dtype,
            device=real_data.device,
            requires_grad=False,
        )
        gradients = autograd.grad(
            outputs=pred_d_interpolates,
            inputs=interpolates,
            grad_outputs=fake_grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.contiguous().view(gradients.shape[0], -1)
        return (gradients.pow(2).sum(1) ** (p / 2)).mean()


class GANLoss(nn.Module):
    """GAN loss with multiple loss types support."""

    def __init__(
        self,
        loss_type: Literal["nsgan", "wgan", "lsgan", "hinge"] = "nsgan",
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.type = loss_type
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))

        if loss_type == "nsgan":
            self.criterion = nn.BCELoss()
        elif loss_type == "wgan":
            self.criterion = self.wgan_loss
        elif loss_type == "lsgan":
            self.criterion = nn.MSELoss()
        elif loss_type == "hinge":
            self.criterion = self.hinge_loss
        else:
            raise NotImplementedError(f"GAN type {loss_type} is not found!")

    def wgan_loss(self, input_data: torch.Tensor, target: bool) -> torch.Tensor:
        """WGAN loss."""
        return -1 * input_data.mean() if target else input_data.mean()

    def hinge_loss(
        self,
        input_data: torch.Tensor,
        target: bool,
        is_discriminator: bool,
    ) -> torch.Tensor:
        """Hinge loss."""
        criterion = nn.ReLU()
        if is_discriminator:
            return (
                criterion(1 - input_data).mean()
                if target
                else criterion(1 + input_data).mean()
            )
        return (-input_data).mean()

    def get_target_tensor(
        self,
        input_data: torch.Tensor,
        target_is_real: bool,
    ) -> torch.Tensor:
        """Get target tensor for the loss."""
        if self.type == "wgan":
            return target_is_real

        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(input_data)

    def forward(
        self,
        input_data: torch.Tensor,
        target_is_real: bool,
        is_discriminator: bool | None = None,
    ) -> torch.Tensor:
        """Forward pass for the GAN loss."""
        if self.type == "hinge":
            loss = self.criterion(input_data, target_is_real, is_discriminator)
        else:
            target_tensor = self.get_target_tensor(input_data, target_is_real)
            loss = self.criterion(input_data, target_tensor)
        return loss


class L1ReconstructionLoss(nn.Module):
    """L1 reconstruction loss."""

    def __init__(self) -> None:
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, input_data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass for the L1 reconstruction loss."""
        return self.l1_loss(input_data, target)


class BCELoss(nn.Module):
    """Binary Cross Entropy loss."""

    def __init__(self) -> None:
        super().__init__()
        self.bce_loss = torch.nn.BCELoss()

    def forward(self, input_data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass for the BCE loss."""
        return self.bce_loss(input_data, target)


class BCELossLogits(nn.Module):
    """Binary Cross Entropy loss with logits."""

    def __init__(self) -> None:
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass for the BCE loss with logits."""
        return self.bce_loss(input_data, target)


class ToneMappingLoss(nn.Module):
    """Tone mapping loss."""

    def __init__(self) -> None:
        super().__init__()
        self.l1_loss = L1ReconstructionLoss()

    def forward(self, input_data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass for the tone mapping loss."""
        return self.l1_loss(input_data / (input_data + 1), target / (target + 1))


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features."""

    def __init__(self) -> None:
        super().__init__()
        vgg16 = models.vgg16(pretrained=True).cuda()
        self.vgg_layers = vgg16.features
        self.l1_loss = L1ReconstructionLoss()
        self.layer_name_mapping = {
            "4": "pool_1",
            "9": "pool_2",
            "16": "pool_3",
        }

    def forward(self, input_data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass for the perceptual loss."""
        loss = 0
        for layer, module in self.vgg_layers._modules.items():
            input_data = module(input_data)
            target = module(target)
            if layer in self.layer_name_mapping:
                loss += self.l1_loss(input_data, target)
        return loss


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss."""

    def __init__(self, _window_size: int = 11) -> None:
        super().__init__()
        self.ms_ssim = kornia.losses.MS_SSIMLoss(reduction="mean")

    def forward(self, input_data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass for the SSIM loss."""
        # Kornia expects images in [0,1]; your data are log-mapped radiance.
        # Normalise to [0,1] by the empirical max on-the-fly.
        scale = torch.maximum(
            target.max(dim=1, keepdim=True)[0],
            torch.tensor(1.0, device=target.device),
        )
        return self.ms_ssim(input_data / scale, target / scale)


class RaHingeGANLoss(nn.Module):
    """Relativistic average hinge loss (RaGAN-H)."""

    def forward(
        self,
        real_preds: torch.Tensor,
        fake_preds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the RaHingeGAN loss."""
        loss = 0
        for pr, pf in zip(real_preds, fake_preds, strict=False):
            real_mean = pr.mean([0, 2, 3], keepdim=True)
            fake_mean = pf.mean([0, 2, 3], keepdim=True)
            loss += (F.relu(1.0 - (pr - fake_mean))).mean()
            loss += (F.relu(1.0 + (pf - real_mean))).mean()
        return loss * 0.5
