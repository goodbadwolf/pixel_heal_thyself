import torch.optim as optim
from torch.optim import lr_scheduler
from pht.models.afgsa.prefetch_dataloader import *
from pht.models.afgsa.model import *
from pht.models.afgsa.loss import *
from pht.models.afgsa.gen_hdf5 import *
from pht.models.afgsa.dataset import *
from pht.models.afgsa.util import *
from pht.models.afgsa.metric import *
from pht.models.afgsa.discriminators import MultiScaleDiscriminator
import time
import math
import numpy as np
import lpips
from omegaconf import DictConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
permutation = [0, 3, 1, 2]


def train_SANet(cfg, train_dataloader, train_num_samples, val_dataloader, val_num_samples, root_save_path):
    deterministic = cfg.trainer.get("deterministic", False)
    print("\t-Creating AFGSANet")
    padding_mode = 'replicate' if deterministic else 'reflect'
    print("\t\t-AFGSANet padding mode: %s" % padding_mode)
    print("\t\t-AFGSANet curve order: %s" % cfg.trainer.curve_order)
    print("\t\t-AFGSANet L1 lossW: %s" % cfg.trainer.l1_loss_w)
    print("\t\t-AFGSANet GAN lossW: %s" % cfg.trainer.gan_loss_w)
    print("\t\t-AFGSANet GP lossW: %s" % cfg.trainer.gp_loss_w)
    if cfg.trainer.use_lpips_loss:
        print("\t\t-AFGSANet LPIPS lossW: %s" % cfg.trainer.lpips_loss_w)
    if cfg.trainer.use_ssim_loss:
        print("\t\t-AFGSANet SSIM lossW: %s" % cfg.trainer.ssim_loss_w)
    if cfg.trainer.use_multiscale_discriminator:
        print("\t\t-AFGSANet multiscale discriminator")
    if cfg.trainer.use_film:
        print("\t\t-AFGSANet use FiLM")
    G = AFGSANet(cfg.model.in_ch, cfg.model.aux_in_ch, cfg.model.base_ch,
                 num_sa=cfg.model.num_sa,
                 block_size=cfg.model.block_size,
                 halo_size=cfg.model.halo_size,
                 num_heads=cfg.model.num_heads,
                 num_gcp=cfg.trainer.num_gradient_checkpoint,
                 padding_mode=padding_mode,
                 curve_order=cfg.trainer.curve_order,
                 use_film=cfg.trainer.use_film).to(device)
    if cfg.trainer.use_multiscale_discriminator:
        D = MultiScaleDiscriminator(in_nc=cfg.model.in_ch, patch_size=cfg.data.patches.patch_size).to(device)
    else:
        D = DiscriminatorVGG(3, 64, cfg.data.patches.patch_size).to(device)
    if cfg.trainer.get("load_model", False):
        G.load_state_dict(torch.load(os.path.join(cfg.trainer.get("model_path", None), 'G.pt')))
        D.load_state_dict(torch.load(os.path.join(cfg.trainer.get("model_path", None), 'D.pt')))
    print_model_structure(G)
    print_model_structure(D)

    l1_loss = L1ReconstructionLoss().to(device)
    if cfg.trainer.use_multiscale_discriminator:
        gan_loss = RaHingeGANLoss().to(device)
    else:
        gan_loss = GANLoss('wgan').to(device)
    gp_loss = GradientPenaltyLoss(device).to(device)
    lpips_loss = lpips.LPIPS(net='vgg').to(device) if cfg.trainer.use_lpips_loss else None
    ssim_loss = SSIMLoss(window_size=11).to(device) if cfg.trainer.use_ssim_loss else None

    milestones = [i * cfg.trainer.lr_milestone - 1 for i in range(1, cfg.trainer.epochs // cfg.trainer.lr_milestone)]
    optimizer_generator = optim.Adam(G.parameters(), lr=cfg.trainer.lrG, betas=(0.9, 0.999), eps=1e-8)
    scheduler_generator = lr_scheduler.MultiStepLR(optimizer_generator, milestones=milestones, gamma=0.5)
    optimizer_discriminator = optim.Adam(D.parameters(), lr=cfg.trainer.lrD, betas=(0.9, 0.999), eps=1e-8)
    scheduler_discriminator = lr_scheduler.MultiStepLR(optimizer_discriminator, milestones=milestones, gamma=0.5)

    accumulated_generator_loss = 0
    accumulated_discriminator_loss = 0
    total_iteraions = math.ceil(train_num_samples / cfg.trainer.batch_size)
    save_img_interval = val_num_samples // cfg.trainer.num_saved_imgs

    print("\t-Start training")
    for epoch in range(cfg.trainer.epochs):
        start = time.time()
        for i_batch, batch_sample in enumerate(train_dataloader):
            aux_features = batch_sample['aux']
            aux_features[:, :, :, :3] = torch.FloatTensor(preprocess_normal(aux_features[:, :, :, :3]))  # normal is not yet preprocessed
            aux_features = aux_features.permute(permutation).to(device)
            noisy = batch_sample['noisy']
            noisy = preprocess_specular(noisy)
            noisy = noisy.permute(permutation).to(device)
            gt = batch_sample['gt']
            gt = preprocess_specular(gt)
            gt = gt.permute(permutation).to(device)

            end_io = time.time()
            if i_batch != 0:
                io_took = end_io - end
            else:
                io_took = end_io - start

            output = G(noisy, aux_features)

            # train discriminator
            optimizer_discriminator.zero_grad()
            pred_d_fake = D(output.detach())
            pred_d_real = D(gt)
            if cfg.trainer.use_multiscale_discriminator:
                discriminator_loss = gan_loss(pred_d_real, pred_d_fake)
            else:
                try:
                    loss_d_real = gan_loss(pred_d_real, True)
                    loss_d_fake = gan_loss(pred_d_fake, False)
                    loss_gp = gp_loss(D, gt, output.detach())
                except:
                    break
                discriminator_loss = (loss_d_fake + loss_d_real) / 2 + cfg.trainer.gp_loss_w * loss_gp
            discriminator_loss.backward()
            optimizer_discriminator.step()
            accumulated_discriminator_loss += discriminator_loss.item() / cfg.trainer.batch_size

            # train generator
            optimizer_generator.zero_grad()
            pred_g_fake = D(output)
            try:
                if cfg.trainer.use_multiscale_discriminator:
                    with torch.no_grad():
                        pred_d_real_ng = D(gt)
                    loss_g_fake = gan_loss(pred_g_fake, pred_d_real_ng)
                else:
                    loss_g_fake = gan_loss(pred_g_fake, True)
                loss_l1 = l1_loss(output, gt)
            except:
                break
            generator_loss = cfg.trainer.gan_loss_w * loss_g_fake + cfg.trainer.l1_loss_w * loss_l1

            def assert_nchw(x, name):
                assert x.ndim == 4 and x.shape[1] == 3, f"{name} not NCHW/3‑ch"

            assert_nchw(output, 'output')
            assert_nchw(gt, 'gt')

            if cfg.trainer.use_lpips_loss:
                def to_lpips_range(x_log):
                    x_lin = torch.exp(x_log) - 1.0
                    x_rgb = (x_lin / (x_lin.max() + 1e-6)).clamp(0, 1)
                    return x_rgb * 2 - 1
                lpips_output = to_lpips_range(output)
                lpips_gt = to_lpips_range(gt)
                loss_lpips = lpips_loss(lpips_output, lpips_gt).mean()
                generator_loss += cfg.trainer.lpips_loss_w * loss_lpips
            if cfg.trainer.use_ssim_loss:
                loss_ssim = ssim_loss(output, gt)
                generator_loss += cfg.trainer.ssim_loss_w * loss_ssim
            generator_loss.backward()
            optimizer_generator.step()
            accumulated_generator_loss += generator_loss.item() / cfg.trainer.batch_size

            if i_batch == 0:
                iter_took = time.time() - start
            else:
                iter_took = time.time() - end
            end = time.time()
            print("\r\t-Epoch: %d \tTook: %f sec \tIteration: %d/%d \tIter Took: %f sec \tI/O Took: %f sec \tG Loss: %f \tD Loss: %f" %
                  (epoch + 1, end - start, i_batch + 1, total_iteraions, iter_took, io_took,
                   accumulated_generator_loss/(i_batch+1), accumulated_discriminator_loss/(i_batch+1)), end='')

        end = time.time()
        print("\r\t-Epoch: %d \tG loss: %f \tD Loss: %f \tTook: %d seconds" %
              (epoch + 1, accumulated_generator_loss/(i_batch+1), accumulated_discriminator_loss/(i_batch+1),
               end - start))
        # save loss values
        with open(os.path.join(root_save_path, "train_loss.txt"), 'a') as f:
            f.write("Epoch: %d \tG loss: %f \tD Loss: %f\n" % (epoch + 1, accumulated_generator_loss/(i_batch+1),
                                                               accumulated_discriminator_loss/(i_batch+1)))

        scheduler_discriminator.step()
        scheduler_generator.step()
        accumulated_generator_loss = 0
        accumulated_discriminator_loss = 0

        # validate and save model, example images
        if epoch % cfg.trainer.save_interval == 0:
            current_save_path = create_folder(os.path.join(root_save_path, 'model_epoch%d' % (epoch + 1)))
            avg_psnr = 0.0
            avg_ssim = 0.0
            avg_mrse = 0.0
            start = time.time()
            with torch.no_grad():
                G.eval()
                # save model
                torch.save(G.state_dict(), os.path.join(current_save_path, "G.pt"))
                torch.save(D.state_dict(), os.path.join(current_save_path, "D.pt"))

                for i_batch, batch_sample in enumerate(val_dataloader):
                    aux_features = batch_sample['aux']
                    aux_features[:, :, :, :3] = torch.FloatTensor(preprocess_normal(aux_features[:, :, :, :3]))  # normal is not yet preprocessed
                    aux_features = aux_features.permute(permutation).to(device)
                    noisy = batch_sample['noisy']
                    noisy = preprocess_specular(noisy)
                    noisy = noisy.permute(permutation).to(device)
                    gt = batch_sample['gt']
                    gt = gt.permute(permutation)

                    output = G(noisy, aux_features)

                    # transfer to image
                    output_c_n = postprocess_specular(output.cpu().numpy())
                    gt_c_n = gt.numpy()
                    noisy_c_n_255 = tensor2img(noisy.cpu().numpy(), post_spec=True)
                    output_c_n_255 = tensor2img(output.cpu().numpy(), post_spec=True)
                    gt_c_n_255 = tensor2img(gt.cpu().numpy())

                    # save image
                    if i_batch % save_img_interval == 0:
                        save_img_group(current_save_path, i_batch, noisy_c_n_255.copy(), output_c_n_255.copy(),
                                       gt_c_n_255.copy())

                    # calculate mrse
                    avg_mrse += calculate_rmse(output_c_n.copy(), gt_c_n.copy())
                    # calculate psnr
                    avg_psnr += calculate_psnr(output_c_n_255.copy(), gt_c_n_255.copy())
                    # calculate ssim
                    avg_ssim += calculate_ssim(output_c_n_255.copy(), gt_c_n_255.copy())

                    end = time.time()
                    print("\r\t-Validation: %d \tTook: %f seconds \tIteration: %d/%d" %
                          (epoch + 1, end - start, i_batch + 1, val_num_samples), end='')
                G.train()

                avg_mrse /= val_num_samples
                avg_psnr /= val_num_samples
                avg_ssim /= val_num_samples
                print("\r\t-Validation: %d \tTook: %d seconds \tAvg MRSE: %f \tAvg PSNR: %f \tAvg 1-SSIM: %f" %
                      (epoch + 1, end - start, avg_mrse, avg_psnr, 1-avg_ssim))
                # save evaluation results
                with open(os.path.join(root_save_path, "evaluation.txt"), 'a') as f:
                    f.write("Validation: %d \tAvg MRSE: %f \tAvg PSNR: %f \tAvg 1-SSIM: %f\n" %
                            (epoch + 1, avg_mrse, avg_psnr, 1-avg_ssim))


def train(cfg: DictConfig):
    train_save_path = os.path.join(cfg.data.patches.root, "train.h5")
    val_save_path = os.path.join(cfg.data.patches.root, "val.h5")
    print(f"Loading dataset: patches from {cfg.data.patches.root}")
    exist = True
    for path in [train_save_path, val_save_path]:
        if not os.path.exists(path):
            exist = False
    if not exist:
        constructor = Hdf5Constructor(cfg.data.in_dir, cfg.data.patches.root,
                                      cfg.data.patches.patch_size,
                                      cfg.data.patches.num_patches,
                                      cfg.seed,
                                      cfg.data_ratio, resize=cfg.data.resize)
        constructor.construct_hdf5()

    if cfg.trainer.get("deterministic", False):
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = False

    train_dataset = Dataset(train_save_path)
    train_num_samples = len(train_dataset)
    if cfg.trainer.get("deterministic", False):
        g = torch.Generator()
        g.manual_seed(cfg.seed)
        train_dataloader = DataLoaderX(train_dataset,
                                       batch_size=cfg.trainer.batch_size,
                                       shuffle=True,
                                       generator=g,
                                       num_workers=7,
                                       pin_memory=True,
                                       worker_init_fn=lambda wid: set_global_seed(cfg.seed + wid))
    else:
        train_dataloader = DataLoaderX(train_dataset,
                                       batch_size=cfg.trainer.batch_size,
                                       shuffle=True,
                                       num_workers=7,
                                       pin_memory=True)

    val_dataset = Dataset(val_save_path)
    val_num_samples = len(val_dataset)
    if cfg.trainer.get("deterministic", False):
        g = torch.Generator()
        g.manual_seed(cfg.seed)
        val_dataloader = DataLoaderX(val_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     generator=g,
                                     num_workers=7,
                                     pin_memory=True)
    else:
        val_dataloader = DataLoaderX(val_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=7,
                                     pin_memory=True)

    root_save_path = create_folder(cfg.paths.out_dir, still_create=False)
    train_SANet(cfg, train_dataloader, train_num_samples,
                val_dataloader, val_num_samples, root_save_path)
    print("Finish training!")


def run(cfg: DictConfig):
    cfg.trainer.curve_order = CurveOrder(cfg.trainer.curve_order)

    if cfg.trainer.get("deterministic", False):
        set_global_seed(cfg.seed)

    create_folder(cfg.paths.out_dir)
    create_folder(cfg.data.patches.root)
    train(cfg)

# expose for Hydra entrypoint
__all__ = ["run"]
