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
