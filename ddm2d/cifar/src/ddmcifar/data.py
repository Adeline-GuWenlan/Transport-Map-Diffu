"""CIFAR-10 from transport_map's .npy files (uint8 N x 3 x 32 x 32, read-only), mapped to [-1, 1].

Data order: a seeded torch.Generator draws one permutation per epoch and one hflip bit per drawn image. The sampler's state
(generator state, permutation, position, epoch) goes into the checkpoint, so a resumed run sees exactly the same batches.
All arms with the same seed see the same images, labels and flips in the same order.
"""
from __future__ import annotations

import os

import numpy as np
import torch

CIFAR_ROOT = os.environ.get("DDMCIFAR_DATA", "/scratch/wg2381/transport_map/data/cifar10")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_cifar(split: str = "train", root: str = CIFAR_ROOT, n: int | None = None):
    """(images float32 [-1, 1] N x 3 x 32 x 32, labels int64 N) as CPU tensors. n: first n images (tests, smoke)."""
    x = np.load(os.path.join(root, f"{split}_images.npy"), mmap_mode="r")
    y = np.load(os.path.join(root, f"{split}_labels.npy"))
    if n is not None:
        x, y = x[:n], y[:n]
    x = torch.from_numpy(np.ascontiguousarray(x)).float().div_(127.5).sub_(1.0)
    return x, torch.from_numpy(np.asarray(y, dtype=np.int64))


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float -> uint8 (clamped, rounded), the same map as transport_map's tmap.sample.to_uint8."""
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


class EpochSampler:
    """Batches of `batch` images without replacement within an epoch (a partial last batch is skipped), random hflip."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch: int, seed: int, hflip: bool = True):
        self.x, self.y, self.batch, self.hflip = x, y, batch, hflip
        self.gen = torch.Generator().manual_seed(seed)
        self.perm = torch.randperm(len(x), generator=self.gen)
        self.pos, self.epoch = 0, 0

    def next(self):
        if self.pos + self.batch > len(self.perm):
            self.perm = torch.randperm(len(self.x), generator=self.gen)
            self.pos, self.epoch = 0, self.epoch + 1
        idx = self.perm[self.pos:self.pos + self.batch]
        self.pos += self.batch
        xb, yb = self.x[idx], self.y[idx]
        if self.hflip:
            f = torch.rand(len(idx), generator=self.gen) < 0.5
            xb = torch.where(f[:, None, None, None], xb.flip(-1), xb)
        return xb, yb

    def state_dict(self):
        return {"gen": self.gen.get_state(), "perm": self.perm.clone(), "pos": self.pos, "epoch": self.epoch}

    def load_state_dict(self, sd):
        self.gen.set_state(sd["gen"])
        self.perm, self.pos, self.epoch = sd["perm"].clone(), sd["pos"], sd["epoch"]
