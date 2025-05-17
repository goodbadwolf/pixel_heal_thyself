"""AFGSA model."""

import math
from enum import StrEnum

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from hilbertcurve.hilbertcurve import HilbertCurve
from torch import nn
from torch.nn import init
from torch.utils.checkpoint import checkpoint

from pht.logger import logger
from pht.models.afgsa.film import FiLM


def print_model_structure(model: nn.Module) -> None:
    """Print model structure."""
    blank = " "
    logger.debug(f"\t {'-' * 95}")
    logger.debug(
        f"\t |{' ' * 13 + 'weight name' + ' ' * 13}|{' ' * 15 + 'weight shape' + ' ' * 15}|{' ' * 3 + 'number' + ' ' * 3}|",  # noqa: E501
    )
    logger.debug(f"\t {'-' * 95}")
    num_para = 0

    for _index, (key, w_variable) in enumerate(model.named_parameters()):
        key = key + (35 - len(key)) * blank if len(key) <= 35 else key[:32] + "..."  # noqa: PLR2004, PLW2901
        shape = str(w_variable.shape)
        shape = (
            shape + (40 - len(shape)) * blank
            if len(shape) <= 40  # noqa: PLR2004
            else shape[:37] + "..."
        )
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:  # noqa: PLR2004
            str_num = str_num + (10 - len(str_num)) * blank

        logger.debug(f"\t | {key} | {shape} | {str_num} |")
    logger.debug(f"\t {'-' * 95}")
    logger.debug(f"\t Total number of parameters: {num_para}")
    logger.debug(f"\t CUDA: {next(model.parameters()).is_cuda}")
    logger.debug(f"\t {'-' * 95}\n")


def norm(norm_type: str, out_ch: int) -> nn.Module:
    """Make normalization layer."""
    norm_type = norm_type.lower()
    if norm_type == "batch":
        layer = nn.BatchNorm2d(out_ch, affine=True)
    elif norm_type == "instance":
        layer = nn.InstanceNorm2d(out_ch, affine=False)
    else:
        raise NotImplementedError(f"Normalization layer [{norm_type:s}] is not found")
    return layer


def act(
    act_type: str,
    inplace: bool = True,
    neg_slope: float = 0.2,
    n_prelu: int = 1,
) -> nn.Module:
    """Make activation layer."""
    # helper selecting activation
    # neg_slope: for lrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(f"Activation layer [{act_type:s}] is not found")
    return layer


def sequential(*args: nn.Module) -> nn.Sequential:
    """Make sequential layer."""
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# conv norm activation
def conv_block(  # noqa: PLR0913
    input_channels: int,
    out_ch: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0,
    padding_mode: str = "zeros",
    norm_type: str | None = None,
    act_type: str = "relu",
    groups: int = 1,
    inplace: bool = True,
) -> nn.Sequential:
    """Conv block."""
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
    n = norm(norm_type, out_ch) if norm_type else None
    a = act(act_type, inplace) if act_type else None
    return sequential(c, n, a)


class DiscriminatorVGG128(nn.Module):
    """Discriminator VGG 128."""

    def __init__(
        self,
        in_nc: int,
        base_nf: int,
        norm_type: str = "batch",
        act_type: str = "leakyrelu",
    ) -> None:
        """Initialize Discriminator VGG 128."""
        super().__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = conv_block(
            in_nc,
            base_nf,
            kernel_size=3,
            norm_type=None,
            act_type=act_type,
            padding=1,
        )
        conv1 = conv_block(
            base_nf,
            base_nf,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            padding=1,
        )
        # 64, 64
        conv2 = conv_block(
            base_nf,
            base_nf * 2,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            padding=1,
        )
        conv3 = conv_block(
            base_nf * 2,
            base_nf * 2,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            padding=1,
        )
        # 32, 128
        conv4 = conv_block(
            base_nf * 2,
            base_nf * 4,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            padding=1,
        )
        conv5 = conv_block(
            base_nf * 4,
            base_nf * 4,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            padding=1,
        )
        # 16, 256
        conv6 = conv_block(
            base_nf * 4,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            padding=1,
        )
        conv7 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            padding=1,
        )
        # 8, 512
        conv8 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=3,
            stride=1,
            norm_type=norm_type,
            act_type=act_type,
            padding=1,
        )
        conv9 = conv_block(
            base_nf * 8,
            base_nf * 8,
            kernel_size=4,
            stride=2,
            norm_type=norm_type,
            act_type=act_type,
            padding=1,
        )
        # 4, 512
        self.features = nn.Sequential(
            conv0,
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            conv6,
            conv7,
            conv8,
            conv9,
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = x.reshape((x.size(0), -1))
        return self.classifier(x)


class DiscriminatorVGG(nn.Module):
    """Discriminator VGG."""

    def __init__(
        self,
        in_nc: int,
        base_nf: int,
        input_size: int,
        norm_type: str = "batch",
        act_type: str = "leakyrelu",
    ) -> None:
        """Initialize Discriminator VGG."""
        super().__init__()
        num_downsample = int(np.log2(input_size / 4))

        features = []
        curr_nc = in_nc

        # First conv without downsampling
        features.append(
            conv_block(
                curr_nc,
                base_nf,
                kernel_size=3,
                norm_type=None,
                act_type=act_type,
                padding=1,
            ),
        )
        curr_nc = base_nf

        # Downsampling layers
        for i in range(num_downsample):
            # Calculate channel multiplier (double channels up to 8x base_nf)
            next_nc = min(base_nf * (2 ** (i + 1)), base_nf * 8)

            # Add regular conv
            features.append(
                conv_block(
                    curr_nc,
                    next_nc,
                    kernel_size=3,
                    stride=1,
                    norm_type=norm_type,
                    act_type=act_type,
                    padding=1,
                ),
            )

            # Add downsampling conv
            features.append(
                conv_block(
                    next_nc,
                    next_nc,
                    kernel_size=4,
                    stride=2,
                    norm_type=norm_type,
                    act_type=act_type,
                    padding=1,
                ),
            )

            curr_nc = next_nc

        self.features = nn.Sequential(*features)

        # Calculate the final feature dimension for the classifier
        final_spatial_size = input_size // (2**num_downsample)
        final_flat_features = curr_nc * final_spatial_size * final_spatial_size

        self.classifier = nn.Sequential(
            nn.Linear(final_flat_features, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = x.reshape((x.size(0), -1))
        return self.classifier(x)


class CurveOrder(StrEnum):
    """Curve order."""

    RASTER = "raster"
    HILBERT = "hilbert"
    ZORDER = "zorder"


def make_curve_indices(block_size: int, mode: CurveOrder) -> torch.Tensor:
    """
    Make curve indices.

    Return a 1-D LongTensor that, when used as
        q = q[:, order, :]
    rearranges a raster-flattened block (row-major) into the
    chosen curve order.
    """
    if mode is CurveOrder.RASTER:
        return torch.arange(block_size * block_size)

    # helper: (row-major index) ➜ (x,y)
    def xy(i: int) -> tuple[int, int]:
        """XY."""
        return (i % block_size, i // block_size)

    if mode is CurveOrder.HILBERT:
        p = int(math.log2(block_size))
        assert block_size == 1 << p, "Hilbert: block_size must be power of two"
        hc = HilbertCurve(p, 2)

        # distance along the curve for every raster cell
        dist = [hc.distance_from_point(xy(i)) for i in range(block_size * block_size)]
        order = torch.tensor(dist).argsort()

    elif mode is CurveOrder.ZORDER:
        # Morton code = bit-interleave of y and x
        def morton(x: int, y: int) -> int:
            """Morton code."""

            def part1(v: int) -> int:
                """Part 1."""
                v = (v | (v << 8)) & 0x00FF00FF
                v = (v | (v << 4)) & 0x0F0F0F0F
                v = (v | (v << 2)) & 0x33333333
                return (v | (v << 1)) & 0x55555555

            return (part1(y) << 1) | part1(x)

        mort = [morton(*xy(i)) for i in range(block_size * block_size)]
        order = torch.tensor(mort).argsort()

    return order.to(torch.long)


class AFGSA(nn.Module):
    """AFGSA."""

    def __init__(  # noqa: PLR0913
        self,
        ch: int,
        block_size: int = 8,
        halo_size: int = 3,
        num_heads: int = 4,
        bias: bool = False,
        curve_order: CurveOrder = CurveOrder.RASTER,
        use_film: bool = False,
    ) -> None:
        """Initialize AFGSA."""
        super().__init__()
        self.block_size = block_size
        self.halo_size = halo_size
        self.num_heads = num_heads
        self.head_ch = ch // num_heads
        assert ch % num_heads == 0, "ch should be divided by # heads"

        self.curve_order = curve_order
        self.register_buffer(
            "curve_indices",
            make_curve_indices(block_size, curve_order),
        )
        self.register_buffer("inv_curve_indices", torch.argsort(self.curve_indices))

        # relative positional embedding: row and column embedding each with dimension 1/2 head_ch
        self.rel_h = nn.Parameter(
            torch.randn(1, block_size + 2 * halo_size, 1, self.head_ch // 2),
            requires_grad=True,
        )
        self.rel_w = nn.Parameter(
            torch.randn(1, 1, block_size + 2 * halo_size, self.head_ch // 2),
            requires_grad=True,
        )

        self.use_film = use_film
        if use_film:
            self.alpha = nn.Parameter(torch.zeros(1))
            self.film = FiLM(
                input_channels=ch,
                cond_ch=ch,
                hidden=128,
                use_spatial=True,
            )
        else:
            self.conv_map = conv_block(ch * 2, ch, kernel_size=1, act_type="relu")
        self.q_conv = nn.Conv2d(ch, ch, kernel_size=1, bias=bias)
        self.k_conv = nn.Conv2d(ch, ch, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(ch, ch, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, noisy: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.use_film:  # noqa: SIM108
            # n_aux = noisy + self.alpha * (self.film(noisy, aux) - noisy)
            n_aux = self.film(noisy, aux)
        else:
            n_aux = self.conv_map(torch.cat([noisy, aux], dim=1))
        b, c, h, w, block, halo, heads = (
            *noisy.shape,
            self.block_size,
            self.halo_size,
            self.num_heads,
        )
        assert h % block == 0 and w % block == 0, (  # noqa: PT018
            "feature map dimensions must be divisible by the block size"
        )

        q = self.q_conv(n_aux)
        q = rearrange(q, "b c (h k1) (w k2) -> (b h w) (k1 k2) c", k1=block, k2=block)
        q *= self.head_ch**-0.5  # b*#blocks, flattened_query, c

        q = q[:, self.curve_indices, :]

        k = self.k_conv(n_aux)
        k = F.unfold(k, kernel_size=block + halo * 2, stride=block, padding=halo)
        k = rearrange(k, "b (c a) l -> (b l) a c", c=c)

        v = self.v_conv(noisy)
        v = F.unfold(v, kernel_size=block + halo * 2, stride=block, padding=halo)
        v = rearrange(v, "b (c a) l -> (b l) a c", c=c)

        # b*#blocks*#heads, flattened_vector, head_ch
        q, v = (rearrange(i, "b a (h d) -> (b h) a d", h=heads) for i in (q, v))
        # positional embedding
        k = rearrange(
            k,
            "b (k1 k2) (h d) -> (b h) k1 k2 d",
            k1=block + 2 * halo,
            h=heads,
        )
        k_h, k_w = k.split(self.head_ch // 2, dim=-1)
        k = torch.cat([k_h + self.rel_h, k_w + self.rel_w], dim=-1)
        k = rearrange(k, "b k1 k2 d -> b (k1 k2) d")

        # b*#blocks*#heads, flattened_query, flattened_neighborhood
        sim = torch.einsum("b i d, b j d -> b i j", q, k)
        attn = F.softmax(sim, dim=-1)
        # b*#blocks*#heads, flattened_query, head_ch
        out = torch.einsum("b i j, b j d -> b i d", attn, v)

        out = out[:, self.inv_curve_indices, :]

        return rearrange(
            out,
            "(b h w n) (k1 k2) d -> b (n d) (h k1) (w k2)",
            b=b,
            h=(h // block),
            w=(w // block),
            k1=block,
            k2=block,
        )

    def reset_parameters(self) -> None:
        """Reset parameters."""
        init.kaiming_normal_(self.q_conv.weight, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.k_conv.weight, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.v_conv.weight, mode="fan_out", nonlinearity="relu")
        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class TransformerBlock(nn.Module):
    """Transformer block."""

    def __init__(  # noqa: PLR0913
        self,
        ch: int,
        block_size: int = 8,
        halo_size: int = 3,
        num_heads: int = 4,
        checkpoint: bool = True,
        padding_mode: str = "reflect",
        curve_order: CurveOrder = CurveOrder.RASTER,
        use_film: bool = False,
    ) -> None:
        """Initialize Transformer block."""
        super().__init__()
        self.checkpoint = checkpoint
        self.attention = AFGSA(
            ch,
            block_size=block_size,
            halo_size=halo_size,
            num_heads=num_heads,
            curve_order=curve_order,
            use_film=use_film,
        )
        self.feed_forward = nn.Sequential(
            conv_block(
                ch,
                ch,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
                act_type="relu",
            ),
            conv_block(
                ch,
                ch,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
                act_type="relu",
            ),
        )

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        noisy = (
            x[0] + checkpoint(self.attention, x[0], x[1])
            if self.checkpoint
            else x[0] + self.attention(x[0], x[1])
        )
        noisy = noisy + self.feed_forward(noisy)
        return (noisy, x[1])


class AFGSANet(nn.Module):
    """AFGSANet."""

    def __init__(  # noqa: PLR0913
        self,
        input_channels: int,
        aux_input_channels: int,
        base_ch: int,
        num_sa: int = 5,
        block_size: int = 8,
        halo_size: int = 3,
        num_heads: int = 4,
        num_gcp: int = 2,
        padding_mode: str = "reflect",
        curve_order: CurveOrder = CurveOrder.RASTER,
        use_film: bool = False,
    ) -> None:
        """Initialize AFGSANet."""
        super().__init__()
        assert num_gcp <= num_sa

        self.conv1 = conv_block(input_channels, 256, kernel_size=1, act_type="relu")
        self.conv3 = conv_block(
            input_channels,
            256,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            act_type="relu",
        )
        self.conv5 = conv_block(
            input_channels,
            256,
            kernel_size=5,
            padding=2,
            padding_mode=padding_mode,
            act_type="relu",
        )
        self.conv_map = conv_block(256 * 3, base_ch, kernel_size=1, act_type="relu")

        self.conv_a1 = conv_block(
            aux_input_channels,
            256,
            kernel_size=1,
            act_type="relu",
        )
        self.conv_a3 = conv_block(
            aux_input_channels,
            256,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            act_type="leakyrelu",
        )
        self.conv_a5 = conv_block(
            aux_input_channels,
            256,
            kernel_size=5,
            padding=2,
            padding_mode=padding_mode,
            act_type="leakyrelu",
        )
        self.conv_aenc1 = conv_block(
            256 * 3,
            base_ch,
            kernel_size=1,
            act_type="leakyrelu",
        )
        self.conv_aenc2 = conv_block(
            base_ch,
            base_ch,
            kernel_size=1,
            act_type="leakyrelu",
        )

        transformer_blocks = []
        # to train on a RTX 3090, use gradient checkpoint for 3 Transformer blocks here (5 in total)
        for i in range(1, num_sa + 1):
            if i <= (num_sa - num_gcp):
                transformer_blocks.append(
                    TransformerBlock(
                        base_ch,
                        block_size=block_size,
                        halo_size=halo_size,
                        num_heads=num_heads,
                        checkpoint=False,
                        padding_mode=padding_mode,
                        curve_order=curve_order,
                        use_film=use_film,
                    ),
                )
            else:
                transformer_blocks.append(
                    TransformerBlock(
                        base_ch,
                        block_size=block_size,
                        halo_size=halo_size,
                        num_heads=num_heads,
                        padding_mode=padding_mode,
                        curve_order=curve_order,
                        use_film=use_film,
                    ),
                )
        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        self.decoder = nn.Sequential(
            conv_block(
                base_ch,
                base_ch,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
                act_type="relu",
            ),
            conv_block(
                base_ch,
                base_ch,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
                act_type="relu",
            ),
            conv_block(
                base_ch,
                3,
                kernel_size=3,
                padding=1,
                padding_mode="zeros",
                act_type=None,
            ),
        )

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        n1 = self.conv1(x)
        n3 = self.conv3(x)
        n5 = self.conv5(x)
        out = self.conv_map(torch.cat([n1, n3, n5], dim=1))

        a1 = self.conv_a1(aux)
        a3 = self.conv_a3(aux)
        a5 = self.conv_a5(aux)
        a = self.conv_aenc1(torch.cat([a1, a3, a5], dim=1))
        a = self.conv_aenc2(a)

        out = self.transformer_blocks([out, a])
        out = self.decoder(out[0])
        out += x
        return out
