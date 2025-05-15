import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from mamba_ssm import Mamba2


def mm_conv_block(
    input_channels,
    out_ch,
    kernel_size,
    stride=1,
    dilation=1,
    padding=0,
    padding_mode="zeros",
    act_type="relu",
    groups=1,
    inplace=True,
):
    c = nn.Conv2d(
        input_channels,
        out_ch,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
        padding_mode=padding_mode,
        groups=groups,
    )
    a = nn.ReLU(inplace) if act_type == "relu" else nn.LeakyReLU(0.2, inplace)
    return nn.Sequential(c, a)


class MambaBlock(nn.Module):
    def __init__(
        self,
        ch,
        d_state=64,
        d_conv=5,
        expansion=2,
        padding_mode="reflect",
        checkpoint=True,
    ):
        super(MambaBlock, self).__init__()
        self.checkpoint = checkpoint
        self.norm1 = nn.LayerNorm(ch)
        self.mamba = Mamba2(
            d_model=ch, d_state=d_state, d_conv=d_conv, expand=expansion
        )
        self.feed_forward = nn.Sequential(
            mm_conv_block(
                ch,
                ch,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
                act_type="relu",
            ),
            mm_conv_block(
                ch,
                ch,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
                act_type="relu",
            ),
        )

    def forward(self, x):
        noisy, aux = x
        b, c, h, w = noisy.shape
        noisy_flat = noisy.permute(0, 2, 3, 1).reshape(b, h * w, c)
        noisy_norm = self.norm1(noisy_flat)

        if self.checkpoint:
            mamba_out = checkpoint(self.mamba, noisy_norm)
        else:
            mamba_out = self.mamba(noisy_norm)

        mamba_out = mamba_out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        noisy = noisy + mamba_out
        noisy = noisy + self.feed_forward(noisy)

        return (noisy, aux)


class MambaDenoiserNet(nn.Module):
    def __init__(
        self,
        input_channels,
        aux_input_channels,
        base_ch,
        pos_encoder,
        num_blocks=5,
        d_state=64,
        d_conv=5,
        expansion=2,
        num_gcp=2,
        padding_mode="reflect",
    ):
        super(MambaDenoiserNet, self).__init__()
        assert num_gcp <= num_blocks

        self.conv1 = mm_conv_block(input_channels, 256, kernel_size=1, act_type="relu")
        self.conv3 = mm_conv_block(
            input_channels,
            256,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            act_type="relu",
        )
        self.conv5 = mm_conv_block(
            input_channels,
            256,
            kernel_size=5,
            padding=2,
            padding_mode=padding_mode,
            act_type="relu",
        )
        self.conv_map = mm_conv_block(256 * 3, base_ch, kernel_size=1, act_type="relu")

        self.conv_a1 = mm_conv_block(
            aux_input_channels, 256, kernel_size=1, act_type="relu"
        )
        self.conv_a3 = mm_conv_block(
            aux_input_channels,
            256,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            act_type="leakyrelu",
        )
        self.conv_a5 = mm_conv_block(
            aux_input_channels,
            256,
            kernel_size=5,
            padding=2,
            padding_mode=padding_mode,
            act_type="leakyrelu",
        )
        self.conv_aenc1 = mm_conv_block(
            256 * 3, base_ch, kernel_size=1, act_type="leakyrelu"
        )
        self.conv_aenc2 = mm_conv_block(
            base_ch, base_ch, kernel_size=1, act_type="leakyrelu"
        )

        mamba_blocks = []
        for i in range(1, num_blocks + 1):
            use_checkpoint = i > (num_blocks - num_gcp)
            mamba_blocks.append(
                MambaBlock(
                    base_ch,
                    d_state=d_state,
                    d_conv=d_conv,
                    expansion=expansion,
                    checkpoint=use_checkpoint,
                    padding_mode=padding_mode,
                )
            )
        self.mamba_blocks = nn.Sequential(*mamba_blocks)

        self.decoder = nn.Sequential(
            mm_conv_block(
                base_ch,
                base_ch,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
                act_type="relu",
            ),
            mm_conv_block(
                base_ch,
                base_ch,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
                act_type="relu",
            ),
            mm_conv_block(
                base_ch,
                3,
                kernel_size=3,
                padding=1,
                padding_mode="zeros",
                act_type=None,
            ),
        )

        self.pos_encoder = pos_encoder

    def forward(self, noisy, aux):
        n1 = self.conv1(noisy)
        n3 = self.conv3(noisy)
        n5 = self.conv5(noisy)
        out = self.conv_map(torch.cat([n1, n3, n5], dim=1))

        out = self.pos_encoder(out)

        a1 = self.conv_a1(aux)
        a3 = self.conv_a3(aux)
        a5 = self.conv_a5(aux)
        a = self.conv_aenc1(torch.cat([a1, a3, a5], dim=1))
        a = self.conv_aenc2(a)

        out, _ = self.mamba_blocks((out, a))
        out = self.decoder(out) + noisy
        return out


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channelsannels=3, base_channels=64):
        super(PatchGANDiscriminator, self).__init__()

        self.net = nn.Sequential(
            # Input: [B, 3, H, W]
            nn.Conv2d(
                input_channelsannels, base_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, height, width):
        super(PositionalEncoding2D, self).__init__()
        pe = torch.zeros(channels, height, width)
        y_pos = torch.arange(0, height).unsqueeze(1).repeat(1, width)
        x_pos = torch.arange(0, width).unsqueeze(0).repeat(height, 1)

        div_term = torch.exp(
            torch.arange(0, channels, 2) * -(math.log(10000.0) / channels)
        )

        pe[0::2, :, :] = torch.sin(
            y_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2)
        )
        pe[1::2, :, :] = torch.cos(
            x_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2)
        )
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe.to(x.device)
