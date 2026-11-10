"""Data: in-memory uint8 arrays (data/<name>/{split}_{images,labels}.npy, see data/README.md) and a
class-stratified batch sampler (docs/stage2_choices.md, "Batch n x m").

Preprocessing (data/README.md): pad 28 -> 32 with uint8 0 (2 px per side) for the MNIST-type sets, then
x / 127.5 - 1 to [-1, 1]; optional random horizontal flip (CIFAR-10 training only).
"""
from __future__ import annotations

import os

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


class ImageArray:
    """uint8 images (N, C, H, W) as a torch tensor in RAM plus int64 labels; optional subset."""

    def __init__(self, root: str, split: str, image_size: int = 32, classes=None, n_per_class=None,
                 subset_seed: int = 0):
        d = resolve(root)
        x = np.load(os.path.join(d, f"{split}_images.npy"), mmap_mode="r")
        y = np.load(os.path.join(d, f"{split}_labels.npy"))
        idx = np.arange(len(y))
        if classes is not None:
            idx = idx[np.isin(y, np.asarray(classes))]
        if n_per_class is not None:
            rng = np.random.default_rng(subset_seed)
            idx = np.sort(np.concatenate([rng.choice(idx[y[idx] == c], n_per_class, replace=False)
                                          for c in np.unique(y[idx])]))
        self.images = torch.from_numpy(np.ascontiguousarray(x[idx]))
        self.labels = torch.from_numpy(y[idx].astype(np.int64))
        self.image_size = image_size
        self.classes = sorted(torch.unique(self.labels).tolist())
        self.by_class = {c: torch.nonzero(self.labels == c).squeeze(1).numpy() for c in self.classes}

    def __len__(self):
        return len(self.labels)

    def to_float(self, imgs_u8: torch.Tensor) -> torch.Tensor:
        """uint8 (B, C, h, w) -> float32 (B, C, S, S) in [-1, 1]; zero-pad (uint8 0 -> -1) to image_size."""
        x = imgs_u8.float()
        pad = (self.image_size - x.shape[-1]) // 2
        if pad:
            x = torch.nn.functional.pad(x, (pad, pad, pad, pad), value=0.0)
        return x / 127.5 - 1.0

    def get(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.as_tensor(idx, dtype=torch.long)
        return self.to_float(self.images[idx]), self.labels[idx]

    def class_tensor(self, c: int, n: int | None = None) -> torch.Tensor:
        """The first n images of class c (all if n is None), float in [-1, 1]."""
        ids = self.by_class[c] if n is None else self.by_class[c][:n]
        return self.get(ids)[0]


class ClassStratifiedSampler:
    """Each batch: classes_per_batch distinct classes (uniform, without replacement) x n_per_class distinct images
    of each class (without replacement within the batch). Batches are i.i.d. across steps. State is a numpy
    Generator, saved in checkpoints for exact resume.

    Weighting: with equal n_c in every batch, the estimator's n_c/N weights are equal; averaged over the random
    class choice this targets sum_c pi_c(...) with pi_c = 1/K for balanced data (Fashion-MNIST, CIFAR-10).
    """

    def __init__(self, data: ImageArray, classes_per_batch: int, n_per_class: int, seed: int = 0,
                 hflip: bool = False):
        if classes_per_batch > len(data.classes):
            raise ValueError(f"classes_per_batch {classes_per_batch} > number of classes {len(data.classes)}")
        if n_per_class < 2:
            raise ValueError("n_per_class must be >= 2 (the interaction term needs m_c >= 2)")
        self.data, self.k, self.n, self.hflip = data, classes_per_batch, n_per_class, hflip
        self.rng = np.random.default_rng(seed)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        cls = self.rng.choice(np.asarray(self.data.classes), self.k, replace=False)
        idx = np.concatenate([self.rng.choice(self.data.by_class[int(c)], self.n, replace=False) for c in cls])
        x, y = self.data.get(idx)
        if self.hflip:
            flip = torch.from_numpy(self.rng.random(len(idx)) < 0.5)
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        return x, y

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, s):
        self.rng.bit_generator.state = s["rng"]


def load_split(cfg: dict, split_key: str = "train_split") -> ImageArray:
    dc = cfg["data"]
    sub = dc.get("subset") or {}
    split = dc[split_key]
    use_subset = split_key == "train_split" or sub.get("apply_to_heldout", False)
    return ImageArray(dc["root"], split, dc["image_size"],
                      classes=sub.get("classes") if (sub and (use_subset or "classes" in sub)) else None,
                      n_per_class=sub.get("n_per_class") if use_subset else None,
                      subset_seed=sub.get("seed", 0))
