"""Frozen feature encoders: for the feature-space energy-score loss term (Experiment B) and for feature-space
energy distances at evaluation.

  dinov2_vits14 : DINOv2 ViT-S/14 (Oquab et al. 2023, https://arxiv.org/abs/2304.07193), self-supervised on LVD-142M,
                  weights Apache-2.0, loaded with timm as `vit_small_patch14_dinov2.lvd142m`
                  (https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m). Output: the 384-d CLS token after
                  the final LayerNorm, the global embedding FD-DINOv2 also uses (Stein et al. 2023,
                  https://arxiv.org/abs/2306.04675). Input: [-1, 1] images -> [0, 1] -> bicubic resize to img_size
                  (224 = 16x16 patches of 14 px) -> ImageNet mean/std. Not clamped, so the map gets a gradient
                  for pixels outside [-1, 1] too. Differentiable w.r.t. the input; parameters frozen, eval mode.
  inception     : clean-fid's legacy_tensorflow Inception-v3 pool3 (2048-d), the FID feature space. uint8 input,
                  not differentiable here; evaluation only.
  random_conv   : a small fixed random conv net, for tests (no download).

The loss itself is in train.py: lambda * per-class energy score of phi(F(z, c)) against phi(y), the same unbiased
class-grouped estimator as the pixel term (energy_score.energy_score_loss). A proper scoring rule of phi(X) plus
the strictly proper pixel term is still strictly proper, so the minimiser (P_theta = Q) is unchanged.
"""
from __future__ import annotations

import os
import shutil

import torch
import torch.nn.functional as F
from torch import nn

DINOV2_TIMM = {"dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
               "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m"}


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
        if u.shape[1] == 1:
            u = u.expand(-1, 3, -1, -1)
        if u.shape[-1] != self.img_size:
            u = F.interpolate(u, size=(self.img_size, self.img_size), mode="bicubic", align_corners=False)
        return self.enc((u - self.mean) / self.std)


class RandomConv(nn.Module):
    """Fixed random conv features (tests only)."""

    def __init__(self, channels: int = 3, dim: int = 16, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w1 = nn.Parameter(torch.randn(8, channels, 3, 3, generator=g) * 0.3, requires_grad=False)
        self.w2 = nn.Parameter(torch.randn(dim, 8, 3, 3, generator=g) * 0.3, requires_grad=False)
        self.dim, self.name = dim, "random_conv"

    def forward(self, x):
        h = torch.tanh(F.conv2d(x.float(), self.w1, stride=2, padding=1))
        return F.conv2d(h, self.w2, stride=2, padding=1).mean((2, 3))


def build_feature_encoder(fc: dict, channels: int = 3) -> nn.Module:
    name = fc.get("encoder", "dinov2_vits14")
    if name in DINOV2_TIMM:
        return DinoV2(name, int(fc.get("img_size", 224)))
    if name == "random_conv":
        return RandomConv(channels, int(fc.get("dim", 16)))
    raise ValueError(f"unknown feature encoder {name}")


@torch.no_grad()
def encode(enc: nn.Module, x: torch.Tensor, batch: int = 256, bf16: bool = False) -> torch.Tensor:
    """Features of x (any device) in chunks, float32 on CPU."""
    dev = next(enc.buffers(), next(enc.parameters(), None)).device
    out = []
    for i in range(0, len(x), batch):
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=bf16 and dev.type == "cuda"):
            out.append(enc(x[i:i + batch].to(dev)).float().cpu())
    return torch.cat(out)


def inception_extractor(device):
    """clean-fid legacy_tensorflow Inception pool3 (the FID network); maps uint8-valued float images to 2048-d."""
    from cleanfid.features import build_feature_extractor
    from .data import resolve
    src = resolve("data/fid/inception/inception-2015-12-05.pt")
    if not os.path.exists("/tmp/inception-2015-12-05.pt") and os.path.exists(src):
        shutil.copy(src, "/tmp/inception-2015-12-05.pt")
    return build_feature_extractor("legacy_tensorflow", torch.device(device), use_dataparallel=False)


@torch.no_grad()
def inception_features(extractor, x: torch.Tensor, device, batch: int = 250) -> torch.Tensor:
    from .sample import to_uint8
    out = []
    for i in range(0, len(x), batch):
        out.append(extractor(to_uint8(x[i:i + batch]).float().to(device)).float().cpu())
    return torch.cat(out)
