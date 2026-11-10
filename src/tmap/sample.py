"""Sampling: per-class grids with fixed noise.

    python -m tmap.sample --ckpt runs/<run>/ckpt/latest.pt --out grid.png [--weights ema|raw] [--n-per-class 10]
"""
from __future__ import annotations

import argparse
import os

import torch


def fixed_noise(model, classes, n_per_class: int, seed: int = 1234):
    """Deterministic (z, c): rows = classes, n_per_class columns; z on CPU then moved to the model's device."""
    g = torch.Generator().manual_seed(seed)
    c = torch.tensor([cl for cl in classes for _ in range(n_per_class)], dtype=torch.long)
    z = torch.randn(len(c), model.in_channels, model.input_size, model.input_size, generator=g)
    return z, c


@torch.no_grad()
def generate(model, c: torch.Tensor, z: torch.Tensor | None = None, seed: int = 0, batch: int = 500,
             autocast_bf16: bool = False) -> torch.Tensor:
    """F(z, c) for all c, in chunks; returns float32 on CPU. Model is put in eval mode and restored."""
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    if z is None:
        g = torch.Generator().manual_seed(seed)
        z = torch.randn(len(c), model.in_channels, model.input_size, model.input_size, generator=g)
    out = []
    for i in range(0, len(c), batch):
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=autocast_bf16 and dev.type == "cuda"):
            out.append(model(z[i:i + batch].to(dev), c[i:i + batch].to(dev)).float().cpu())
    model.train(was_training)
    return torch.cat(out, 0)


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float -> uint8 [0, 255] (clamped)."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def save_grid(imgs: torch.Tensor, path: str, nrow: int, pad: int = 2):
    """imgs in [-1, 1], (B, C, H, W); written as PNG with nrow images per row (1-channel -> gray)."""
    from torchvision.utils import save_image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_image((imgs.clamp(-1, 1) + 1) / 2, path, nrow=nrow, padding=pad, pad_value=1.0)


def load_checkpoint_model(ckpt_path: str, weights: str = "ema", device: str = "cpu"):
    from .model import build_model
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg)
    model.load_state_dict(ck["ema" if weights == "ema" else "model"])
    return model.to(device).eval(), cfg, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="ema", choices=["ema", "raw"])
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    model, cfg, ck = load_checkpoint_model(a.ckpt, a.weights, a.device)
    classes = list(range(cfg["data"]["num_classes"]))
    z, c = fixed_noise(model, classes, a.n_per_class, a.seed)
    x = generate(model, c, z)
    save_grid(x, a.out, nrow=a.n_per_class)
    print(f"wrote {a.out}: {len(classes)} classes (rows) x {a.n_per_class}, step {ck['step']}, {a.weights} weights")


if __name__ == "__main__":
    main()
