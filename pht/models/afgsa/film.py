"""AFGSA FiLM."""

import torch
from torch import nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.  Takes a conditioning tensor `cond and predicts per-channel γ, β to modulate `x`.

        x' = γ * x + β
    ── args ──────────────────────────────────────────────────────────
    input_channels          channels of x   (C)
    cond_ch        channels of cond
    use_spatial    if True, γ/β have HxW resolution (SPADE-like).
    """  # noqa: RUF002

    def __init__(
        self,
        input_channels: int,
        cond_ch: int,
        hidden: int = 128,
        use_spatial: bool = False,
    ) -> None:
        super().__init__()
        out_ch = input_channels * 2  # γ and β  # noqa: RUF003
        self.use_spatial = use_spatial
        self.affine = nn.Sequential(
            nn.Conv2d(cond_ch, hidden, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden, out_ch, 1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # cond → γβ tensors
        gamma_beta = self.affine(cond)
        if self.use_spatial:  # (N,2C,H,W)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        else:  # global γβ (N,2C,1,1)
            gamma_beta = gamma_beta.mean([2, 3], keepdim=True)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma * x + beta
