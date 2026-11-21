"""U-Net as a one-step transport map F_theta(z, c): the image-space U-Net of De Bortoli et al., "Distributional
Diffusion Models with Scoring Rules" (ICML 2025, arXiv 2502.02483), App. I.2.

Own implementation, written from the paper text (no code was released with the paper; theirs is JAX). Where the
paper is silent, the choice is ours; each is listed in docs/experiments_unet.md ("Design decisions"). The layout of
the down/up path follows the common DDPM U-Net (Ho et al. 2020), from memory; no third-party code is imported.

What App. I.2 fixes:
- base channels 256, channel multipliers (1, 2, 2, 2), 2 residual blocks per level;
- attention at the second level (16x16 for CIFAR-10) and in the bottleneck, 8 heads, applied after the residual block;
- residual block: norm -> 3x3 conv -> GELU -> dropout -> 3x3 conv, plus the input through a 1x1 conv;
- RMSNorm, made adaptive: scale and bias from an MLP of the conditioning vector;
- conditioning = MLP(sinusoidal(t)) + class embedding (summed).

As for the DiT map (tmap.model.TransportMap): t is the constant 999, the input is z ~ N(0, I) with the image's shape,
the output is used directly, and the last conv is zero-initialised so that F == 0 at step 0 (an assumption, not in
App. I.2). Dropout defaults to 0 (the paper gives no rate; DiT uses none).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding [cos, sin] of a (B,) tensor of timesteps (the DDPM / DiT convention)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class AdaRMSNorm(nn.Module):
    """RMSNorm over channels at every spatial position, with scale and bias from the conditioning vector:
    y = x / rms_C(x) * (1 + gamma(e)) + beta(e), [gamma, beta] = Linear(GELU(e)). Normalisation in float32."""

    def __init__(self, channels: int, emb_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.mod = nn.Sequential(nn.GELU(), nn.Linear(emb_dim, 2 * channels))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=1, keepdim=True) + self.eps)
        gamma, beta = self.mod(emb).float().chunk(2, dim=1)
        return (xf * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]).to(dt)


class ResBlock(nn.Module):
    """App. I.2: norm -> 3x3 conv -> GELU -> dropout -> 3x3 conv, plus 1x1 conv(input)."""

    def __init__(self, cin: int, cout: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = AdaRMSNorm(cin, emb_dim)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1)

    def forward(self, x, emb):
        h = self.conv1(self.norm(x, emb))
        h = self.conv2(self.drop(F.gelu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Pre-norm multi-head self-attention over the H*W positions, residual: x + proj(MHSA(AdaRMSNorm(x)))."""

    def __init__(self, channels: int, emb_dim: int, num_heads: int = 8):
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"channels {channels} not divisible by heads {num_heads}")
        self.heads = num_heads
        self.norm = AdaRMSNorm(channels, emb_dim)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x, emb):
        B, C, H, W = x.shape
        h = self.norm(x, emb).flatten(2).transpose(1, 2)                    # (B, HW, C)
        q, k, v = self.qkv(h).view(B, H * W, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        h = F.scaled_dot_product_attention(q, k, v)                          # (B, heads, HW, C/heads)
        h = self.proj(h.transpose(1, 2).reshape(B, H * W, C))
        return x + h.transpose(1, 2).reshape(B, C, H, W)


class Level(nn.Module):
    """One residual block, optionally followed by attention (App. I.2: attention after the residual block)."""

    def __init__(self, cin, cout, emb_dim, dropout, attn, heads):
        super().__init__()
        self.res = ResBlock(cin, cout, emb_dim, dropout)
        self.attn = AttnBlock(cout, emb_dim, heads) if attn else None

    def forward(self, x, emb):
        x = self.res(x, emb)
        return self.attn(x, emb) if self.attn is not None else x


class UNet(nn.Module):
    """Down path: per level num_res_blocks blocks (each output kept as a skip), then a stride-2 3x3 conv (not after
    the last level). Bottleneck: ResBlock -> attention -> ResBlock. Up path: per level num_res_blocks blocks, each
    taking concat(h, skip) of the matching down block, then nearest x2 + 3x3 conv (not after the top level).
    Head: AdaRMSNorm -> GELU -> 3x3 conv (zero-initialised)."""

    def __init__(self, input_size=32, in_channels=3, out_channels=3, base_channels=256, channel_mult=(1, 2, 2, 2),
                 num_res_blocks=2, attention_resolutions=(16,), num_heads=8, dropout=0.0, emb_dim=None,
                 num_classes=10, mid_attention=True):
        super().__init__()
        emb_dim = emb_dim or 4 * base_channels
        self.base_channels, self.emb_dim = base_channels, emb_dim
        self.time_mlp = nn.Sequential(nn.Linear(base_channels, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(num_classes, emb_dim)
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        skips, ch, res = [], base_channels, input_size
        for i, mult in enumerate(channel_mult):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(Level(ch, base_channels * mult, emb_dim, dropout, res in attention_resolutions,
                                    num_heads))
                ch = base_channels * mult
                skips.append(ch)
            self.down.append(blocks)
            last = i == len(channel_mult) - 1
            self.downsample.append(nn.Identity() if last else nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            res = res if last else res // 2

        self.mid1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = AttnBlock(ch, emb_dim, num_heads) if mid_attention else None
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout)

        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mult))):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(Level(ch + skips.pop(), base_channels * mult, emb_dim, dropout,
                                    res in attention_resolutions, num_heads))
                ch = base_channels * mult
            self.up.append(blocks)
            self.upsample.append(nn.Identity() if i == 0 else nn.Conv2d(ch, ch, 3, padding=1))
            res = res if i == 0 else res * 2
        assert not skips and res == input_size

        self.norm_out = AdaRMSNorm(ch, emb_dim)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.base_channels).to(x.dtype)) + self.class_emb(c)
        h = self.conv_in(x)
        hs = []
        for blocks, ds in zip(self.down, self.downsample):
            for b in blocks:
                h = b(h, emb)
                hs.append(h)
            h = ds(h)
        h = self.mid1(h, emb)
        if self.mid_attn is not None:
            h = self.mid_attn(h, emb)
        h = self.mid2(h, emb)
        for blocks, us in zip(self.up, self.upsample):
            for b in blocks:
                h = b(torch.cat([h, hs.pop()], dim=1), emb)
            if not isinstance(us, nn.Identity):
                h = us(F.interpolate(h, scale_factor=2.0, mode="nearest"))
        return self.conv_out(F.gelu(self.norm_out(h, emb)))


class UNetTransportMap(nn.Module):
    """F_theta(z, c) = UNet(z, t_fixed, c); the same interface as tmap.model.TransportMap."""

    def __init__(self, input_size=32, in_channels=3, out_channels=None, num_classes=10, timestep_value=999.0,
                 **unet_kw):
        super().__init__()
        out_channels = out_channels or in_channels
        self.net = UNet(input_size=input_size, in_channels=in_channels, out_channels=out_channels,
                        num_classes=num_classes, **unet_kw)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.input_size, self.num_classes = input_size, num_classes
        self.register_buffer("t_value", torch.tensor(float(timestep_value)), persistent=False)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t = self.t_value.expand(z.shape[0])
        return self.net(z, t, c)

    def sample(self, c: torch.Tensor, generator: torch.Generator | None = None, z: torch.Tensor | None = None):
        if z is None:
            z = torch.randn(c.shape[0], self.in_channels, self.input_size, self.input_size,
                            generator=generator, device=c.device)
        return self(z, c)


def build_unet(cfg: dict) -> UNetTransportMap:
    """From a full config dict (model.backbone == "unet"); see configs/expA_cifar10_unet_pixel.yaml."""
    m = cfg["model"]
    if m.get("timestep_mode", "constant") != "constant" or m.get("output", "direct") != "direct":
        raise ValueError("only timestep_mode=constant, output=direct are implemented")
    return UNetTransportMap(input_size=m["input_size"], in_channels=m["in_channels"],
                            out_channels=m.get("out_channels"), num_classes=cfg["data"]["num_classes"],
                            timestep_value=m.get("timestep_value", 999),
                            base_channels=m.get("base_channels", 256),
                            channel_mult=tuple(m.get("channel_mult", (1, 2, 2, 2))),
                            num_res_blocks=m.get("num_res_blocks", 2),
                            attention_resolutions=tuple(m.get("attention_resolutions", (16,))),
                            num_heads=m.get("num_heads", 8), dropout=m.get("dropout", 0.0),
                            emb_dim=m.get("emb_dim"), mid_attention=m.get("mid_attention", True))
