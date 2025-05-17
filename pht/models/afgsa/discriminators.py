"""AFGSA discriminators."""

import torch
from torch import nn
from torch.nn.utils import spectral_norm as SN  # noqa: N812


class PatchDiscriminator(nn.Module):
    """
    N-layer 70x70-style PatchGAN whose depth is chosen so the last feature map is ≥ min_featxmin_feat (default 4x4).

    **based on input_size passed from the caller**.
    """

    def __init__(
        self,
        in_nc: int,
        base_nf: int = 64,
        input_size: int = 128,
        min_feat: int = 4,
    ) -> None:
        super().__init__()

        kw = 4
        pad = 1
        layers = []
        nf_in, nf_out = in_nc, base_nf
        cur_size = input_size

        # keep striding until the next stride would go below min_feat
        while cur_size // 2 >= min_feat:
            layers += [
                SN(nn.Conv2d(nf_in, nf_out, kw, 2, pad)),
                nn.LeakyReLU(0.2, True),
            ]
            nf_in, nf_out = nf_out, min(nf_out * 2, base_nf * 8)
            cur_size //= 2  # track feature-map side

        # one last 1-stride conv → 1-channel patch logits
        layers += [SN(nn.Conv2d(nf_in, 1, kw, 1, pad))]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator."""

    def __init__(self, in_nc: int = 3, patch_size: int = 128) -> None:
        super().__init__()
        self.D1 = PatchDiscriminator(in_nc, input_size=patch_size)  # 128x128
        self.D2 = PatchDiscriminator(in_nc, input_size=patch_size // 2)  # 64x64
        self.D3 = PatchDiscriminator(in_nc, input_size=patch_size // 4)  # 32x32

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass."""
        return [
            self.D1(x),
            self.D2(nn.functional.avg_pool2d(x, 2)),
            self.D3(nn.functional.avg_pool2d(x, 4)),
        ]
