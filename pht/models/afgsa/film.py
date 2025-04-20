# film.py
import torch, torch.nn as nn

class FiLM(nn.Module):
    """
    Feature‑wise Linear Modulation.  Takes a conditioning tensor `cond`
    and predicts per‑channel γ, β to modulate `x`.
        x' = γ * x + β
    ── args ──────────────────────────────────────────────────────────
    in_ch          channels of x   (C)
    cond_ch        channels of cond
    use_spatial    if True, γ/β have H×W resolution (SPADE‑like)
    """
    def __init__(self, in_ch: int, cond_ch: int,
                 hidden: int = 128, use_spatial: bool = False):
        super().__init__()
        out_ch = in_ch * (2 if use_spatial else 2)   # γ and β
        self.use_spatial = use_spatial
        self.affine = nn.Sequential(
            nn.Conv2d(cond_ch, hidden, 1), nn.ReLU(True),
            nn.Conv2d(hidden, out_ch, 1)
        )

    def forward(self, x, cond):
        # cond → γβ tensors
        gamma_beta = self.affine(cond)
        if self.use_spatial:                       # (N,2C,H,W)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        else:                                     # global γβ (N,2C,1,1)
            gamma_beta = gamma_beta.mean([2,3], keepdim=True)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma * x + beta
