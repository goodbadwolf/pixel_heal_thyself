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
data_ratio = (0.95, 0.05)


def train_SANet(args, train_dataloader, train_num_samples, val_dataloader, val_num_samples, root_save_path):
    print("\t-Creating AFGSANet")
    padding_mode = 'replicate' if args.deterministic else 'reflect'
    print("\t\t-AFGSANet padding mode: %s" % padding_mode)
    print("\t\t-AFGSANet curve order: %s" % args.curveOrder)
    print("\t\t-AFGSANet L1 lossW: %s" % args.l1LossW)
    print("\t\t-AFGSANet GAN lossW: %s" % args.ganLossW)
    print("\t\t-AFGSANet GP lossW: %s" % args.gpLossW)
    if args.useLPIPSLoss:
        print("\t\t-AFGSANet LPIPS lossW: %s" % args.lpipsLossW)
    if args.useSSIMLoss:
        print("\t\t-AFGSANet SSIM lossW: %s" % args.ssimLossW)
    if args.useMultiscaleDiscriminator:
        print("\t\t-AFGSANet multiscale discriminator")
    if args.useFilm:
        print("\t\t-AFGSANet use FiLM")
    G = AFGSANet(args.inCh, args.auxInCh, args.baseCh, num_sa=args.numSA, block_size=args.blockSize,
                 halo_size=args.haloSize, num_heads=args.numHeads, num_gcp=args.numGradientCheckpoint,
                 padding_mode=padding_mode, curve_order=args.curveOrder, use_film=args.useFilm).to(device)
    if args.useMultiscaleDiscriminator:
        D = MultiScaleDiscriminator(in_nc=args.inCh, patch_size=args.patchSize).to(device)
    else:
        D = DiscriminatorVGG(3, 64, args.patchSize).to(device)
    if args.loadModel:
        G.load_state_dict(torch.load(os.path.join(args.modelPath, 'G.pt')))
        D.load_state_dict(torch.load(os.path.join(args.modelPath, 'D.pt')))
    print_model_structure(G)
    print_model_structure(D)

    l1_loss = L1ReconstructionLoss().to(device)
    if args.useMultiscaleDiscriminator:
        gan_loss = RaHingeGANLoss().to(device)
    else:
        gan_loss = GANLoss('wgan').to(device)
    gp_loss = GradientPenaltyLoss(device).to(device)
    lpips_loss = lpips.LPIPS(net='vgg').to(device) if args.useLPIPSLoss else None
    ssim_loss = SSIMLoss(window_size=11).to(device) if args.useSSIMLoss else None

    milestones = [i * args.lrMilestone - 1 for i in range(1, args.epochs//args.lrMilestone)]
    optimizer_generator = optim.Adam(G.parameters(), lr=args.lrG, betas=(0.9, 0.999), eps=1e-8)
    scheduler_generator = lr_scheduler.MultiStepLR(optimizer_generator, milestones=milestones, gamma=0.5)
    optimizer_discriminator = optim.Adam(D.parameters(), lr=args.lrD, betas=(0.9, 0.999), eps=1e-8)
    scheduler_discriminator = lr_scheduler.MultiStepLR(optimizer_discriminator, milestones=milestones, gamma=0.5)

    accumulated_generator_loss = 0
    accumulated_discriminator_loss = 0
    total_iteraions = math.ceil(train_num_samples / args.batchSize)
    save_img_interval = val_num_samples // args.numSavedImgs

    print("\t-Start training")
    for epoch in range(args.epochs):
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
            if args.useMultiscaleDiscriminator:
                discriminator_loss = gan_loss(pred_d_real, pred_d_fake)
            else:
                try:
                    loss_d_real = gan_loss(pred_d_real, True)
                    loss_d_fake = gan_loss(pred_d_fake, False)
                    loss_gp = gp_loss(D, gt, output.detach())
                except:
                    break
                discriminator_loss = (loss_d_fake + loss_d_real) / 2 + args.gpLossW * loss_gp
            discriminator_loss.backward()
            optimizer_discriminator.step()
            accumulated_discriminator_loss += discriminator_loss.item() / args.batchSize

            # train generator
            optimizer_generator.zero_grad()
            pred_g_fake = D(output)
            try:
                if args.useMultiscaleDiscriminator:
                    with torch.no_grad():
                        pred_d_real_ng = D(gt)
                    loss_g_fake = gan_loss(pred_g_fake, pred_d_real_ng)
                else:
                    loss_g_fake = gan_loss(pred_g_fake, True)
                loss_l1 = l1_loss(output, gt)
            except:
                break
            generator_loss = args.ganLossW * loss_g_fake + args.l1LossW * loss_l1

            def assert_nchw(x, name):
                assert x.ndim == 4 and x.shape[1] == 3, f"{name} not NCHW/3‑ch"

            # before loss calls
            assert_nchw(output, 'output')
            assert_nchw(gt, 'gt')

            if args.useLPIPSLoss:
                def to_lpips_range(x_log):
                    x_lin = torch.exp(x_log) - 1.0
                    x_rgb = (x_lin / (x_lin.max() + 1e-6)).clamp(0, 1)
                    return x_rgb * 2 - 1
                lpips_output = to_lpips_range(output)
                lpips_gt = to_lpips_range(gt)
                loss_lpips = lpips_loss(lpips_output, lpips_gt).mean()
                generator_loss += args.lpipsLossW * loss_lpips
            if args.useSSIMLoss:
                loss_ssim = ssim_loss(output, gt)
                generator_loss += args.ssimLossW * loss_ssim
            generator_loss.backward()
            optimizer_generator.step()
            accumulated_generator_loss += generator_loss.item() / args.batchSize

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
        if epoch % args.saveInterval == 0:
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


def train(args, data_ratio):
    train_save_path = os.path.join(args.datasetDir, "train.h5")
    val_save_path = os.path.join(args.datasetDir, "val.h5")
    print(f"Loading dataset: patches from {args.datasetDir}")
    exist = True
    for path in [train_save_path, val_save_path]:
        if not os.path.exists(path):
            exist = False
    if not exist:
        constructor = Hdf5Constructor(args.inDir, args.datasetDir, args.patchSize, args.numPatches, args.seed,
                                      data_ratio)
        constructor.construct_hdf5()

    if not args.deterministic:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = False

    train_dataset = Dataset(train_save_path)
    train_num_samples = len(train_dataset)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=7,
    #                               pin_memory=True)  # original
    if args.deterministic:
        g = torch.Generator()
        g.manual_seed(args.seed)
        train_dataloader = DataLoaderX(train_dataset, batch_size=args.batchSize, shuffle=True, generator=g, num_workers=7,
                                       pin_memory=True, worker_init_fn=lambda wid: set_global_seed(args.seed + wid))  # prefetch
    else:
        train_dataloader = DataLoaderX(train_dataset, batch_size=args.batchSize, shuffle=True, num_workers=7,
                                       pin_memory=True)  # prefetch

    val_dataset = Dataset(val_save_path)
    val_num_samples = len(val_dataset)
    if args.deterministic:
        g = torch.Generator()
        g.manual_seed(args.seed)
        val_dataloader = DataLoaderX(val_dataset, batch_size=1, shuffle=False, generator=g, num_workers=7, pin_memory=True)
    else:
        val_dataloader = DataLoaderX(val_dataset, batch_size=1, shuffle=False, num_workers=7, pin_memory=True)

    # root_save_path = create_folder(os.path.join(args.outDir, 'AFGSA'), still_create=True)  # path to save model, imgs
    root_save_path = create_folder(args.outDir, still_create=False)  # path to save model, imgs

    train_SANet(args, train_dataloader, train_num_samples, val_dataloader, val_num_samples, root_save_path)
    print("Finish training!")


def run(cfg: DictConfig):
    # 1) optional deterministic seeding
    if cfg.trainer.get("deterministic", False):
        set_global_seed(cfg.seed)

    class A: pass
    args = A()
    # I/O & seed
    args.inDir      = cfg.data.in_dir
    args.datasetDir = cfg.data.patches.root
    args.outDir     = cfg.paths.out_dir
    args.seed       = cfg.seed
    # dataset parameters
    args.patchSize  = cfg.data.patches.patch_size
    args.numPatches = cfg.data.patches.num_patches
    # training schedule
    args.epochs         = cfg.trainer.epochs
    args.batchSize      = cfg.trainer.batch_size
    args.saveInterval   = cfg.trainer.save_interval
    args.numSavedImgs   = cfg.trainer.num_saved_imgs
    # optimizers / schedulers
    args.lrG        = cfg.trainer.lrG
    args.lrD        = cfg.trainer.lrD
    args.lrGamma    = cfg.trainer.lr_gamma
    args.lrMilestone= cfg.trainer.lr_milestone
    # loss weights
    args.l1LossW    = cfg.trainer.l1_loss_w
    args.ganLossW   = cfg.trainer.gan_loss_w
    args.gpLossW    = cfg.trainer.gp_loss_w
    # miscellaneous
    args.deterministic             = cfg.trainer.deterministic
    args.numGradientCheckpoint     = cfg.trainer.num_gradient_checkpoint
    args.useLPIPSLoss              = cfg.trainer.use_lpips_loss
    args.lpipsLossW                = cfg.trainer.lpips_loss_w
    args.useSSIMLoss               = cfg.trainer.use_ssim_loss
    args.ssimLossW                 = cfg.trainer.ssim_loss_w
    args.useMultiscaleDiscriminator= cfg.trainer.use_multiscale_discriminator
    args.useFilm                   = cfg.trainer.use_film
    # model hyperparameters
    args.blockSize  = cfg.model.block_size
    args.haloSize   = cfg.model.halo_size
    args.numHeads   = cfg.model.num_heads
    args.numSA      = cfg.model.num_sa
    args.inCh       = cfg.model.in_ch
    args.auxInCh    = cfg.model.aux_in_ch
    args.baseCh     = cfg.model.base_ch
    args.curveOrder = CurveOrder(cfg.trainer.curve_order)
    # optional reload
    args.loadModel  = cfg.trainer.get("load_model", False)
    args.modelPath  = cfg.trainer.get("model_path", None)

    # 3) ensure output dirs exist
    create_folder(args.outDir)
    create_folder(args.datasetDir)

    # 4) launch the original training
    train(args, data_ratio)

# expose for Hydra entrypoint
__all__ = ["run"]