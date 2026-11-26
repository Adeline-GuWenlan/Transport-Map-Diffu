"""Frozen feature encoder for the cifar_C_dino feature-space energy term.

`DinoV2` and `RandomConv` are copied from transport_map `src/tmap/features.py` at commit 4b9e8f9 (read-only checkout
/scratch/wg2381/transport_map): DINOv2 ViT-S/14 (Oquab et al. 2023, arXiv 2304.07193) via timm `vit_small_patch14_dinov2.lvd142m`
(weights in $HF_HOME; set HF_HUB_OFFLINE=1 in jobs), 384-d CLS after the final LayerNorm. Input [-1, 1] -> [0, 1] -> bicubic
resize to 224 -> ImageNet mean/std; not clamped; frozen, eval mode, differentiable w.r.t. the input.

`feature_scale` adapts transport_map's `scale: auto` rule (tmap.train.feature_scale): s = (mean same-class pixel distance) /
(mean same-class feature distance) on the first n_per_class train images per class, and lam_f = weight * s. transport_map used
beta = 1 on both sides because its pixel term had beta = 1. Here the pixel term of cifar_C has beta = 0.1 and the feature term
beta_f = 1, so the numerator is the mean of ||y - y'||^beta_pixel and the denominator the mean of ||phi(y) - phi(y')||^beta_f:
this keeps the rule's intent (feature term on the pixel term's scale) [G: our adaptation].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

DINOV2_TIMM = {"dinov2_vits14": "vit_small_patch14_dinov2.lvd142m"}


class DinoV2(nn.Module):
    def __init__(self, name: str = "dinov2_vits14", img_size: int = 224):
        super().__init__()
        import timm
        self.enc = timm.create_model(DINOV2_TIMM[name], pretrained=True, img_size=img_size, num_classes=0)
        self.enc.eval().requires_grad_(False)
        cfg = self.enc.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)
        self.img_size, self.dim, self.name = img_size, self.enc.num_features, name

    def train(self, mode: bool = True):          # always frozen, always eval mode
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = (x.float() + 1) / 2
        if u.shape[-1] != self.img_size:
            u = F.interpolate(u, size=(self.img_size, self.img_size), mode="bicubic", align_corners=False)
        return self.enc((u - self.mean) / self.std)


class RandomConv(nn.Module):
    """Fixed random conv features (tests and CPU smoke only; no download)."""

    def __init__(self, channels: int = 3, dim: int = 16, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w1 = nn.Parameter(torch.randn(8, channels, 3, 3, generator=g) * 0.3, requires_grad=False)
        self.w2 = nn.Parameter(torch.randn(dim, 8, 3, 3, generator=g) * 0.3, requires_grad=False)
        self.dim, self.name = dim, "random_conv"

    def forward(self, x):
        h = torch.tanh(F.conv2d(x.float(), self.w1, stride=2, padding=1))
        return F.conv2d(h, self.w2, stride=2, padding=1).mean((2, 3))


def build_encoder(fc: dict) -> nn.Module:
    name = fc.get("encoder", "dinov2_vits14")
    if name in DINOV2_TIMM:
        return DinoV2(name, int(fc.get("img_size", 224)))
    if name == "random_conv":
        return RandomConv(3, int(fc.get("dim", 16)))
    raise ValueError(f"unknown feature encoder {name}")


def _mean_offdiag_pow(x: torch.Tensor, beta: float) -> float:
    x = x.double().flatten(1)
    d = torch.cdist(x, x)
    n = len(x)
    off = ~torch.eye(n, dtype=torch.bool)
    return float((d[off] ** beta).mean())


@torch.no_grad()
def feature_scale(enc, x: torch.Tensor, y: torch.Tensor, beta_pixel: float, beta_feat: float, n_per_class: int,
                  device, bf16: bool) -> dict:
    px, ft = [], []
    for c in sorted(torch.unique(y).tolist()):
        xc = x[y == c][:n_per_class]
        fs = []
        for i in range(0, len(xc), 100):
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=bf16):
                fs.append(enc(xc[i:i + 100].to(device)).float().cpu())
        px.append(_mean_offdiag_pow(xc, beta_pixel))
        ft.append(_mean_offdiag_pow(torch.cat(fs), beta_feat))
    p, f = sum(px) / len(px), sum(ft) / len(ft)
    return {"scale": p / f, "pixel_mean_dist_pow": p, "feature_mean_dist_pow": f, "n_per_class": n_per_class,
            "beta_pixel": beta_pixel, "beta_feat": beta_feat}
