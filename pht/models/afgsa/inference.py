from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pyexr

# internal modules
from pht.models.afgsa.model import AFGSANet           # network
from pht.models.afgsa.util import create_folder, save_img, tensor2img
from pht.models.afgsa.metric import (
    calculate_rmse,
    calculate_psnr,
    calculate_ssim,
)
from pht.models.afgsa.preprocessing import (          # stays unchanged
    preprocess_specular,
    postprocess_specular,
    preprocess_normal,
    preprocess_depth,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_PERM = [0, 3, 1, 2]                                 # NHWC → NCHW


# --------------------------------------------------------------------------- #
def _infer_single(cfg) -> None:
    """Runs a single-frame inference using the configuration tree."""
    # ------------------------------------------------------------------ model
    m_cfg = cfg.model
    model_path = os.path.join(cfg.inference.model_dir, cfg.inference.model_file)
    print(f"-AFGSA modelPath = {model_path}")
    print(f"-AFGSA inDir     = {cfg.inference.images.dir}")
    print(f"-AFGSA file      = {cfg.inference.file_name}")

    net = AFGSANet(
        m_cfg.input_channels,
        m_cfg.aux_input_channels,
        m_cfg.base_ch,
        num_sa=m_cfg.num_sa,
        block_size=m_cfg.block_size,
        halo_size=m_cfg.halo_size,
        num_heads=m_cfg.num_heads,
        num_gcp=cfg.trainer.num_gradient_checkpoint  # in case we share this flag
    ).to(device)

    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # ------------------------------------------------------------- I/O - noisy
    noisy_exr = pyexr.open(
        os.path.join(cfg.inference.images.dir, f"{cfg.inference.file_name}.exr")
    )
    width_in, height_in = noisy_exr.width / 1000, noisy_exr.height / 1000
    exr_all = noisy_exr.get_all()

    normal = preprocess_normal(np.nan_to_num(exr_all["normal"]))
    depth  = preprocess_depth(np.nan_to_num(exr_all["depth"]))
    albedo = exr_all["albedo"]

    aux = np.concatenate((normal, depth, albedo), axis=2)[None]      # NHWC, batch=1
    aux_t = torch.as_tensor(aux).permute(_PERM).to(device)
    aux_t = F.pad(
        aux_t,
        (
            0,
            aux_t.shape[3] % m_cfg.block_size,
            0,
            aux_t.shape[2] % m_cfg.block_size,
        ),
        "constant",
        0,
    )

    noisy = np.clip(np.nan_to_num(exr_all["default"]), 0, np.max(exr_all["default"]))[None]
    noisy_t = torch.as_tensor(preprocess_specular(noisy)).permute(_PERM).to(device)
    noisy_t = F.pad(
        noisy_t,
        (
            0,
            noisy_t.shape[3] % m_cfg.block_size,
            0,
            noisy_t.shape[2] % m_cfg.block_size,
        ),
        "constant",
        0,
    )

    # ---------------------------------------------------- split (OOM-safe)
    center = (aux_t.shape[3] + aux_t.shape[3] % m_cfg.block_size) // 2
    split1 = center + cfg.inference.overlap * m_cfg.block_size
    split2 = center - cfg.inference.overlap * m_cfg.block_size

    noisy_chunks = (noisy_t[..., :split1], noisy_t[..., split2:])
    aux_chunks   = (aux_t  [..., :split1], aux_t  [..., split2:])
    outputs = []

    print("\tStart denoising")
    with torch.no_grad():
        for noisy_c, aux_c in zip(noisy_chunks, aux_chunks):
            outputs.append(net(noisy_c, aux_c))

    out = np.concatenate(
        (
            outputs[0].cpu().numpy()[..., :center],
            outputs[1].cpu().numpy()[..., cfg.inference.overlap * m_cfg.block_size :],
        ),
        axis=3,
    )
    out_post = np.transpose(postprocess_specular(out), (0, 2, 3, 1))[0]

    save_path = os.path.join(cfg.paths.output_dir, "inferences")
    create_folder(save_path)
    save_filename = Path(cfg.inference.file_name).stem + "_clean.exr"
    print(f"-AFGSA outDir    = {save_path}")
    print(f"-AFGSA outFile   = {save_filename}")
    pyexr.write(os.path.join(save_path, save_filename), out_post)
    # noisy_img_255  = tensor2img(np.transpose(noisy, _PERM))[0]
    output_img_255 = tensor2img(out, post_spec=True)[0]

    #save_img(os.path.join(save_path, "noisy.png"),  noisy_img_255,
    #         figsize=(width_in, height_in), dpi=1000)
    #save_img(os.path.join(save_path, "output_ours.png"), output_img_255,
    #         figsize=(width_in, height_in), dpi=1000)

    print("\tDone!")

    # -------------------------------------------------------------- metrics
    if cfg.inference.load_gt:
        # cfg.inference.file_name remove the _32 at the end of the stem and then replace it with _1024
        gt_filename = f"{cfg.inference.file_name[:-3]}_1024{cfg.inference.gt_suffix}.exr"
        gt_exr = pyexr.open(
            os.path.join(cfg.inference.images.dir, gt_filename)).get_all()
        gt = np.clip(np.nan_to_num(gt_exr["default"]), 0, np.max(gt_exr["default"]))
        # pyexr.write(os.path.join(save_path, "gt.exr"), gt)

        rmse = calculate_rmse(out_post.copy(), gt.copy())
        psnr = calculate_psnr(output_img_255.copy(), tensor2img(gt.transpose(2, 0, 1)))
        ssim = calculate_ssim(output_img_255.copy(), tensor2img(gt.transpose(2, 0, 1)))

        eval_filename = f"{cfg.inference.file_name}_evaluation.txt"
        print(f"-AFGSA evalFile  = {eval_filename}")
        with open(os.path.join(save_path, eval_filename), "w") as f:
            f.write(f"RMSE: {rmse:.6f}\tPSNR: {psnr:.6f}\t1-SSIM: {1-ssim:.6f}\n")
        print(f"\tRMSE: {rmse:.6f}\tPSNR: {psnr:.6f}\t1-SSIM: {1-ssim:.6f}")


# --------------------------------------------------------------------------- #
def run(cfg) -> None:
    """
    Hydra entry-point (mirrors train.run).
    Creates output dirs, delegates to _infer_single.
    """
    create_folder(cfg.paths.output_dir)
    if cfg.inference.file_name is None:
        exr_paths = sorted(Path(cfg.inference.images.dir).glob(cfg.inference.pattern))
        for p in exr_paths:
            if p.stem.startswith("._"):
                continue
            cfg.inference.file_name = p.stem
            _infer_single(cfg)
    else:
        _infer_single(cfg)



# expose for Hydra to import
__all__ = ["run"]
