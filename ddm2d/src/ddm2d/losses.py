"""Training losses. Eq. (6) (standard diffusion, x0-prediction) and Eq. (14) (empirical generalized energy diffusion loss).

arXiv 2502.02483 v3: Eq. (14) p. 3 and the pseudo-code of App. F p. 30. Per data point i:
  conf_i  = (1/m) sum_j ||x0_i - xhat_ij||^beta
  inter_i = 1/(m(m-1)) sum_{j != j'} ||xhat_ij - xhat_ij'||^beta      (j = j' terms excluded)
  loss    = mean_i w_i (conf_i - lambda/2 * inter_i)
which equals (1/(nm)) sum_{ij} w_i [ ||.||^beta - lambda/(2(m-1)) sum_{j' != j} ||.||^beta ] of Eq. (14).
"""
from __future__ import annotations

import torch

DELTA = 1e-12   # smoothing inside ||d||^beta = (||d||^2 + DELTA)^(beta/2) for beta != 2 (not specified by the paper)


def norm_pow(sq, beta: float, delta: float = DELTA):
    """||d||^beta from the squared norm sq."""
    if beta == 2.0:
        return sq
    return (sq + delta) ** (beta / 2.0)


def energy_terms(x0, xhat, beta: float, delta: float = DELTA):
    """x0: [n, d]; xhat: [n, m, d]. Returns per-data-point (conf [n], inter [n])."""
    n, m, d = xhat.shape
    if m < 2:
        raise ValueError("Eq. (14) needs m >= 2")
    conf = norm_pow(((x0[:, None, :] - xhat) ** 2).sum(-1), beta, delta).mean(1)
    diff = xhat[:, :, None, :] - xhat[:, None, :, :]                 # [n, m, m, d]
    pd = norm_pow((diff ** 2).sum(-1), beta, delta)                  # [n, m, m]
    offdiag = ~torch.eye(m, dtype=torch.bool, device=xhat.device)
    inter = pd[:, offdiag].sum(-1) / (m * (m - 1))
    return conf, inter


def energy_loss(x0, xhat, w, beta: float, lam: float, delta: float = DELTA):
    """Eq. (14). w: per-data-point weights [n]. Returns (loss, dict of detached diagnostics)."""
    conf, inter = energy_terms(x0, xhat, beta, delta)
    per = conf - 0.5 * lam * inter
    loss = (w * per).mean()
    return loss, {"conf": conf.mean().detach(), "inter": inter.mean().detach(), "wconf": (w * conf).mean().detach()}


def diffusion_loss(x0, xhat, w):
    """Eq. (6): mean_i w_i ||x0_i - xhat_i||^2. x0, xhat: [n, d]."""
    sq = ((x0 - xhat) ** 2).sum(-1)
    loss = (w * sq).mean()
    return loss, {"conf": sq.mean().detach(), "inter": torch.zeros(()), "wconf": loss.detach()}


# ---------------------------------------------------------------- ddm2dx extension (E4): other distances rho in Eq. (14)
# Generalized kernel score of arXiv 2502.02483 v3, Sec. 2 (main.tex lines 267-287): S_{lam,rho}(p, y) = lam/2 E rho(X, X') - E rho(X, y),
# with rho a negative definite kernel or rho = -k for a positive definite k. The loss per data point i is
#   conf_i - lam/2 inter_i,  conf_i = (1/m) sum_j rho(x0_i, xhat_ij),  inter_i = 1/(m(m-1)) sum_{j != j'} rho(xhat_ij, xhat_ij')
# (j = j' excluded, so inter_i is an unbiased estimate of E rho(X, X')). Options ("rho" in the config; "rho_c" its parameter):
#   l2            ||d||_2^beta                                    (default; delegates to energy_loss above, bit-for-bit)
#   l1            ||d||_1^beta, |d_k| smoothed as sqrt(d_k^2 + DELTA) so the gradient at d = 0 is finite for every beta
#   pseudo_huber  sqrt(||d||^2 + c^2) - c                         (iCT, Song & Dhariwal 2023, arXiv 2310.14189 Sec. 3.3); c = rho_c
#   imq           -(||d||^2 + c)^(-1/2)                           (main.tex Eq. imq_k); c = rho_c
#   rbf           -exp(-||d||^2 / (2 sigma^2))                    (main.tex Eq. rbf_k); rho_c = sigma^2
#   laplace       -exp(-||d|| / sigma), ||d|| = sqrt(||d||^2 + DELTA)   (main.tex Eq. exp_k); rho_c = sigma
# beta is used by l2 and l1 only. For rbf / laplace, sigma comes from the median heuristic of main.tex line 1747: median_sq_dist below.
RHOS = ("l2", "l1", "pseudo_huber", "imq", "rbf", "laplace")
POWER_RHOS = ("l2", "l1")


def rho_of_diff(diff, rho: str, beta: float, c: float, delta: float = DELTA):
    """rho evaluated on difference vectors diff [..., d] -> [...]."""
    if rho == "l2":
        return norm_pow((diff ** 2).sum(-1), beta, delta)
    if rho == "l1":
        l1 = torch.sqrt(diff ** 2 + delta).sum(-1)
        return l1 if beta == 1.0 else l1 ** beta
    sq = (diff ** 2).sum(-1)
    if rho == "pseudo_huber":
        return torch.sqrt(sq + c * c) - c
    if rho == "imq":
        return -((sq + c) ** -0.5)
    if rho == "rbf":
        return -torch.exp(-sq / (2.0 * c))
    if rho == "laplace":
        return -torch.exp(-torch.sqrt(sq + delta) / c)
    raise ValueError(f"unknown rho {rho!r}")


def rho_terms(x0, xhat, rho: str, beta: float, c: float, delta: float = DELTA):
    """Generic version of energy_terms for any rho. x0: [n, d]; xhat: [n, m, d]. Returns (conf [n], inter [n])."""
    n, m, d = xhat.shape
    if m < 2:
        raise ValueError("Eq. (14) needs m >= 2")
    conf = rho_of_diff(x0[:, None, :] - xhat, rho, beta, c, delta).mean(1)
    pd = rho_of_diff(xhat[:, :, None, :] - xhat[:, None, :, :], rho, beta, c, delta)    # [n, m, m]
    offdiag = ~torch.eye(m, dtype=torch.bool, device=xhat.device)
    inter = pd[:, offdiag].sum(-1) / (m * (m - 1))
    return conf, inter


def rho_loss(x0, xhat, w, rho: str, beta: float, lam: float, c: float = 0.0, delta: float = DELTA, generic: bool = False):
    """Eq. (14) with a general rho. rho = "l2" calls energy_loss (bit-for-bit the existing path) unless generic=True (tests only)."""
    if rho == "l2" and not generic:
        return energy_loss(x0, xhat, w, beta, lam, delta)
    conf, inter = rho_terms(x0, xhat, rho, beta, c, delta)
    per = conf - 0.5 * lam * inter
    loss = (w * per).mean()
    return loss, {"conf": conf.mean().detach(), "inter": inter.mean().detach(), "wconf": (w * conf).mean().detach()}


def median_sq_dist(x, n_batches: int = 100, batch: int = 512, seed: int = 0) -> float:
    """Median of ||x_i - x_j||^2 over all pairs i < j inside each of n_batches random batches of the data (main.tex line 1747:
    "median ... on a subset of 100 batches of size 512 from the dataset"). numpy RNG with its own seed; no global RNG is touched.
    RBF: sigma^2 = gamma * M^2. For the Laplace (exp) kernel, sigma = gamma * M with M from median_dist."""
    return float(_pair_stat(x, n_batches, batch, seed, sq=True))


def median_dist(x, n_batches: int = 100, batch: int = 512, seed: int = 0) -> float:
    """Median of ||x_i - x_j|| (same pairs as median_sq_dist)."""
    return float(_pair_stat(x, n_batches, batch, seed, sq=False))


def _pair_stat(x, n_batches, batch, seed, sq):
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(batch, 1)
    vals = []
    for _ in range(n_batches):
        b = x[rng.choice(len(x), batch, replace=False)]
        d2 = ((b[:, None, :] - b[None, :, :]) ** 2).sum(-1)[iu]
        vals.append(d2 if sq else np.sqrt(d2))
    return np.median(np.concatenate(vals))
