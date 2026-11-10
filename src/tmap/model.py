"""DiT as a one-step transport map F_theta(z, c) (docs/stage2_choices.md, "Map F_theta(z, c)").

Backbone: class DiT from third_party/DiT/models.py (facebookresearch/DiT @ ed81ce2, CC-BY-NC 4.0), imported
unmodified. F(z, c) = DiT(z, t_fixed, c): z ~ N(0, I) with the image's shape, t-embedder kept with a constant
t = 999 (reserved for the later per-noise-level variant), learn_sigma = False, class_dropout_prob = 0, output used
directly. DiT's adaLN-Zero init zeroes the final layer, so F == 0 at step 0.
"""
from __future__ import annotations

import os
import sys

import torch
from torch import nn

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIT_DIR = os.environ.get("TMAP_DIT_DIR", os.path.join(_ROOT, "third_party", "DiT"))


def import_dit():
    """Import third_party/DiT/models.py without writing __pycache__ into the clone."""
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        if DIT_DIR not in sys.path:
            sys.path.insert(0, DIT_DIR)
        import models as dit_models  # noqa: E402  (third_party/DiT/models.py)
    finally:
        sys.dont_write_bytecode = old
    path = os.path.realpath(dit_models.__file__)
    if path != os.path.realpath(os.path.join(DIT_DIR, "models.py")):
        raise ImportError(f"imported the wrong `models` module: {path}")
    return dit_models


class TransportMap(nn.Module):
    """F_theta(z, c) = DiT(z, t_fixed, c)."""

    def __init__(self, input_size=32, patch_size=4, in_channels=1, out_channels=None, hidden_size=256, depth=6,
                 num_heads=4, mlp_ratio=4.0, num_classes=10, class_dropout_prob=0.0, timestep_value=999.0,
                 learn_sigma=False):
        super().__init__()
        if learn_sigma:
            raise ValueError("learn_sigma must be False for a transport map")
        dit = import_dit()
        self.net = dit.DiT(input_size=input_size, patch_size=patch_size, in_channels=in_channels,
                           hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                           class_dropout_prob=class_dropout_prob, num_classes=num_classes, learn_sigma=False)
        out_channels = out_channels or in_channels
        if out_channels != in_channels:
            # For the later per-noise-level variant (input [x_t, xi]): new zero-initialised head (as DiT's).
            fl = dit.FinalLayer(hidden_size, patch_size, out_channels)
            for mod in (fl.adaLN_modulation[-1], fl.linear):
                nn.init.zeros_(mod.weight)
                nn.init.zeros_(mod.bias)
            self.net.final_layer = fl
            self.net.out_channels = out_channels
        self.in_channels, self.out_channels = in_channels, out_channels
        self.input_size, self.num_classes = input_size, num_classes
        self.register_buffer("t_value", torch.tensor(float(timestep_value)), persistent=False)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t = self.t_value.expand(z.shape[0]).to(z.dtype)
        return self.net(z, t, c)

    def sample(self, c: torch.Tensor, generator: torch.Generator | None = None, z: torch.Tensor | None = None):
        if z is None:
            z = torch.randn(c.shape[0], self.in_channels, self.input_size, self.input_size,
                            generator=generator, device=c.device)
        return self(z, c)


def build_model(cfg: dict) -> TransportMap:
    """Build from a full config dict (keys model.*, data.num_classes), as configs/*.yaml."""
    m = cfg["model"]
    if m.get("backbone", "dit") != "dit":
        raise ValueError(m["backbone"])
    if m.get("timestep_mode", "constant") != "constant" or m.get("output", "direct") != "direct":
        raise ValueError("only timestep_mode=constant, output=direct are implemented")
    return TransportMap(input_size=m["input_size"], patch_size=m["patch_size"], in_channels=m["in_channels"],
                        out_channels=m.get("out_channels"), hidden_size=m["hidden_size"], depth=m["depth"],
                        num_heads=m["num_heads"], mlp_ratio=m.get("mlp_ratio", 4.0),
                        num_classes=cfg["data"]["num_classes"], class_dropout_prob=m.get("class_dropout_prob", 0.0),
                        timestep_value=m.get("timestep_value", 999), learn_sigma=m.get("learn_sigma", False))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
