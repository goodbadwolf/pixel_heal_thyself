# discriminator_ms.py
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

class PatchDiscriminator(nn.Module):
    """N‑layer PatchGAN with Spectral Norm (64×64 receptive field)."""
    def __init__(self, in_nc=3, base_nf=64, n_layers=4):
        super().__init__()
        kw = 4; pad = 1
        layers = [SN(nn.Conv2d(in_nc, base_nf, kw, 2, pad)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
            layers += [
                SN(nn.Conv2d(base_nf*nf_mult_prev, base_nf*nf_mult, kw, 2, pad)),
                nn.LeakyReLU(0.2, True)
            ]
        layers += [SN(nn.Conv2d(base_nf*nf_mult, 1, kw, 1, pad))]
        self.model = nn.Sequential(*layers)

    def forward(self, x):              # output: N×1×h×w patch map
        return self.model(x)

class MultiScaleDiscriminator(nn.Module):
    """Three PatchGANs operating at 1×, ½×, ¼× input scales."""
    def __init__(self, in_nc=3):
        super().__init__()
        self.D1 = PatchDiscriminator(in_nc)
        self.D2 = PatchDiscriminator(in_nc)
        self.D3 = PatchDiscriminator(in_nc)

    def forward(self, x):
        out1 = self.D1(x)
        out2 = self.D2(F.avg_pool2d(x, 2))
        out3 = self.D3(F.avg_pool2d(x, 4))
        return [out1, out2, out3]      # list for RaGAN loss
