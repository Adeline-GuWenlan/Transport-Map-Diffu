"""2D datasets: the paper's two-Gaussian mixture and the FFJORD checkerboard.

GMM: arXiv 2502.02483 v3, Sec. 6.1 and App. I.1 (0.5 N((3,3), 0.5^2 I) + 0.5 N((-3,3), 0.5^2 I), 102,400 training points).
Checkerboard: re-implementation of FFJORD lib/toy_data.py inf_train_gen("checkerboard") (https://github.com/rtqichen/ffjord, commit 994864a,
MIT license, Chen & Grathwohl 2018), with an explicit numpy Generator instead of the global np.random state. Paper Table 2 cites
Galashov et al. (2025) for this dataset, which does not specify a generator (docs/audit.md section 4), so this is a documented approximation.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

GMM_MEANS = np.array([[3.0, 3.0], [-3.0, 3.0]])
GMM_STD = 0.5
GMM_WEIGHTS = np.array([0.5, 0.5])
TRAIN_SIZE = 102_400
EVAL_SEED_BASE = 10_000          # eval sets use seeds >= 10_000; training sets use the data seed (0) -> disjoint streams

ROOT = Path(__file__).resolve().parents[2]          # .../ddm2d
CACHE = Path(os.environ.get("DDM2D_DATA_CACHE", ROOT / "data" / "cache"))


def sample_gmm(n: int, rng: np.random.Generator) -> np.ndarray:
    comp = rng.integers(0, 2, size=n)
    return GMM_MEANS[comp] + GMM_STD * rng.standard_normal((n, 2))


def sample_checkerboard(n: int, rng: np.random.Generator) -> np.ndarray:
    """FFJORD checkerboard: x1 ~ U[-2,2), x2 ~ U[0,1) - 2*{0,1} + (floor(x1) mod 2), then *2. Support: 8 squares of side 2 in [-4,4]^2."""
    x1 = rng.random(n) * 4 - 2
    x2_ = rng.random(n) - rng.integers(0, 2, size=n) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    return np.concatenate([x1[:, None], x2[:, None]], 1) * 2


def in_checkerboard(x: np.ndarray) -> np.ndarray:
    """Membership of the FFJORD checkerboard support (scaled coordinates): inside [-4,4]^2 and floor(x1/2)+floor(x2/2) even."""
    x = np.asarray(x)
    inside = np.all((x >= -4) & (x < 4), axis=1)
    parity = (np.floor(x[:, 0] / 2) + np.floor(x[:, 1] / 2)) % 2 == 0
    return inside & parity


SAMPLERS = {"gmm": sample_gmm, "checkerboard": sample_checkerboard}


# ---------------------------------------------------------------- ddm2dx extension (E2): GMMs with unequal weights / per-component stds
# Spec: claude_jobs/ddm2dx/prompts/_task.md, E2. The paper GMM path above is unchanged (sample_gmm draws components with rng.integers).
GMM2W_MEANS = GMM_MEANS.copy()
GMM2W_STDS = np.array([0.5, 0.5])
GMM2W_WEIGHTS = np.array([0.8, 0.2])

GMM9W_MEANS = np.array([[-4.0 + 4.0 * (k % 3), 4.0 - 4.0 * (k // 3)] for k in range(9)])   # 3x3 grid, row-major from (-4, 4)
GMM9W_STDS = np.array([0.5, 0.3, 0.4, 0.3, 0.6, 0.3, 0.4, 0.5, 0.3])
GMM9W_WEIGHTS = np.array([0.30, 0.02, 0.15, 0.05, 0.20, 0.03, 0.10, 0.07, 0.08])

# name -> (means [K, 2], stds [K], weights [K]); "gmm" is the paper GMM expressed in the same form (for the general posterior)
GMM_SPECS = {
    "gmm": (GMM_MEANS, np.full(2, GMM_STD), GMM_WEIGHTS),
    "gmm2w": (GMM2W_MEANS, GMM2W_STDS, GMM2W_WEIGHTS),
    "gmm9w": (GMM9W_MEANS, GMM9W_STDS, GMM9W_WEIGHTS),
}


def sample_weighted_gmm(n: int, rng: np.random.Generator, means, stds, weights, return_comp: bool = False):
    """Isotropic GMM with per-component weights and stds. Component by inverse CDF of one uniform per point, then Gaussian noise."""
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    comp = np.searchsorted(cdf, rng.random(n), side="right").clip(max=len(weights) - 1)
    x = means[comp] + stds[comp][:, None] * rng.standard_normal((n, means.shape[1]))
    return (x, comp) if return_comp else x


def _spec_sampler(name: str):
    means, stds, weights = GMM_SPECS[name]
    return lambda n, rng: sample_weighted_gmm(n, rng, means, stds, weights)


SAMPLERS["gmm2w"] = _spec_sampler("gmm2w")
SAMPLERS["gmm9w"] = _spec_sampler("gmm9w")


def generate(name: str, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return SAMPLERS[name](n, rng).astype(np.float32)


def train_set(name: str, n: int = TRAIN_SIZE, seed: int = 0, cache: bool = True) -> np.ndarray:
    """Fixed training set (float32), cached in data/cache/ (gitignored). Regenerated bit-identically from (name, n, seed)."""
    path = CACHE / f"{name}_train_n{n}_seed{seed}.npy"
    if cache and path.exists():
        return np.load(path)
    x = generate(name, n, seed)
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, x)
        os.replace(tmp, path)
    return x


def eval_set(name: str, n: int = 4096, k: int = 0) -> np.ndarray:
    """Independent evaluation reference set number k: fresh draws from p_0 with seed EVAL_SEED_BASE + k (never the training points)."""
    return generate(name, n, EVAL_SEED_BASE + k)


def fingerprint(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
