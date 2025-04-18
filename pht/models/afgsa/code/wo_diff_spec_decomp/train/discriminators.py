import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

class PatchDiscriminator(nn.Module):
    """
    N‑layer 70×70‑style PatchGAN whose depth is chosen so the
    last feature map is ≥ min_feat×min_feat (default 4×4),
    **based on input_size passed from the caller**.
    """
    def __init__(self, in_nc: int, base_nf: int = 64,
                 input_size: int = 128, min_feat: int = 4):
        super().__init__()

        kw = 4; pad = 1
        layers = []
        nf_in, nf_out = in_nc, base_nf
        cur_size = input_size

        # keep striding until the next stride would go below min_feat
        while cur_size // 2 >= min_feat:
            layers += [SN(nn.Conv2d(nf_in, nf_out, kw, 2, pad)),
                       nn.LeakyReLU(0.2, True)]
            nf_in, nf_out = nf_out, min(nf_out * 2, base_nf * 8)
            cur_size //= 2                    # track feature‑map side

        # one last 1‑stride conv → 1‑channel patch logits
        layers += [SN(nn.Conv2d(nf_in, 1, kw, 1, pad))]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_nc=3, patch_size=128):
        super().__init__()
        self.D1 = PatchDiscriminator(in_nc, input_size=patch_size)          # 128×128
        self.D2 = PatchDiscriminator(in_nc, input_size=patch_size // 2)     # 64×64
        self.D3 = PatchDiscriminator(in_nc, input_size=patch_size // 4)     # 32×32

    def forward(self, x):
        return [
            self.D1(x),
            self.D2(nn.functional.avg_pool2d(x, 2)),
            self.D3(nn.functional.avg_pool2d(x, 4))
        ]
