from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pyexr

# internal modules
from pht.models.mamba.model import MambaDenoiserNet, PositionalEncoding2D
from pht.models.afgsa.util import create_folder, save_img, tensor2img
from pht.models.afgsa.metric import (
    calculate_rmse,
    calculate_psnr,
    calculate_ssim,
)
from pht.models.afgsa.preprocessing import (
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
    print(f"-Mamba modelPath = {model_path}")
    print(f"-Mamba inDir     = {cfg.inference.in_dir}")
    print(f"-Mamba file      = {cfg.inference.file_name}")

    # Create positional encoding for the model
    pos_encoder = PositionalEncoding2D(
        m_cfg.base_ch,
        cfg.data.patches.patch_size,
        cfg.data.patches.patch_size,
    ).to(device)

    net = MambaDenoiserNet(
        m_cfg.in_ch,
        m_cfg.aux_in_ch,
        m_cfg.base_ch,
        pos_encoder,
        num_blocks=m_cfg.num_blocks,
        d_state=m_cfg.d_state,
        d_conv=m_cfg.d_conv,
        expansion=m_cfg.expansion,
        num_gcp=m_cfg.num_gcp,
    ).to(device)

    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # ------------------------------------------------------------- I/O - noisy
    noisy_exr = pyexr.open(
        os.path.join(cfg.inference.in_dir, f"{cfg.inference.file_name}.exr")
    )
    width_in, height_in = noisy_exr.width / 1000, noisy_exr.height / 1000
    exr_all = noisy_exr.get_all()

    normal = preprocess_normal(np.nan_to_num(exr_all["normal"]))
    depth  = preprocess_depth(np.nan_to_num(exr_all["depth"]))
    albedo = exr_all["albedo"]

    aux = np.concatenate((normal, depth, albedo), axis=2)[None]      # NHWC, batch=1
    aux_t = torch.as_tensor(aux).permute(_PERM).to(device)
    
    # For Mamba models, we don't need block size padding like in AFGSA
    # but we'll keep similar structure for compatibility

    noisy = np.clip(np.nan_to_num(exr_all["default"]), 0, np.max(exr_all["default"]))[None]
    noisy_t = torch.as_tensor(preprocess_specular(noisy)).permute(_PERM).to(device)

    # ---------------------------------------------------- split (OOM-safe)
    center = aux_t.shape[3] // 2
    split1 = center + cfg.inference.overlap * 8  # Using block_size=8 as default
    split2 = center - cfg.inference.overlap * 8

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
            outputs[1].cpu().numpy()[..., cfg.inference.overlap * 8:],
        ),
        axis=3,
    )
    out_post = np.transpose(postprocess_specular(out), (0, 2, 3, 1))[0]

    save_path = os.path.join(cfg.paths.out_dir, "inferences")
    create_folder(save_path)
    save_filename = Path(cfg.inference.file_name).stem + "_clean.exr"
    print(f"-Mamba outDir    = {save_path}")
    print(f"-Mamba outFile   = {save_filename}")
    pyexr.write(os.path.join(save_path, save_filename), out_post)
    output_img_255 = tensor2img(out, post_spec=True)[0]

    print("\tDone!")

    # -------------------------------------------------------------- metrics
    if cfg.inference.load_gt:
        # cfg.inference.file_name remove the _32 at the end of the stem and then replace it with _1024
        gt_filename = f"{cfg.inference.file_name[:-3]}_1024{cfg.inference.gt_suffix}.exr"
        gt_exr = pyexr.open(
            os.path.join(cfg.inference.in_dir, gt_filename)).get_all()
        gt = np.clip(np.nan_to_num(gt_exr["default"]), 0, np.max(gt_exr["default"]))

        rmse = calculate_rmse(out_post.copy(), gt.copy())
        psnr = calculate_psnr(output_img_255.copy(), tensor2img(gt.transpose(2, 0, 1)))
        ssim = calculate_ssim(output_img_255.copy(), tensor2img(gt.transpose(2, 0, 1)))

        eval_filename = f"{cfg.inference.file_name}_evaluation.txt"
        print(f"-Mamba evalFile  = {eval_filename}")
        with open(os.path.join(save_path, eval_filename), "w") as f:
            f.write(f"RMSE: {rmse:.6f}\tPSNR: {psnr:.6f}\t1-SSIM: {1-ssim:.6f}\n")
        print(f"\tRMSE: {rmse:.6f}\tPSNR: {psnr:.6f}\t1-SSIM: {1-ssim:.6f}")


# --------------------------------------------------------------------------- #
def run(cfg) -> None:
    """
    Hydra entry-point (mirrors train.run).
    Creates output dirs, delegates to _infer_single.
    """
    create_folder(cfg.paths.out_dir)
    if cfg.inference.file_name is None:
        exr_paths = sorted(Path(cfg.inference.in_dir).glob(cfg.inference.pattern))
        for p in exr_paths:
            if p.stem.startswith("._"):
                continue
            cfg.inference.file_name = p.stem
            _infer_single(cfg)
    else:
        _infer_single(cfg)


# expose for Hydra to import
__all__ = ["run"]