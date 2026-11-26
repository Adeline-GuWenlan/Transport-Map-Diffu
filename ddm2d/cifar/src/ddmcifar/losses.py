"""Training losses for the CIFAR-10 trial (ddm2dx E3).

Arm A (standard diffusion, velocity target as the paper's diffusion baseline): x_t = (1 - t) x_0 + t x_1, v = x_0 - x_1,
  loss = mean_i w_i ||v_hat_i - v_i||^2   (squared norm over the 3072 pixels, as Eq. (6) uses ||.||^2).
Arms B / C: Eq. (14) of arXiv 2502.02483 v3 over flattened images, generalised to a distance rho (prompts/_task.md, E4):
  conf_i  = (1/m) sum_j rho(x0_i, xhat_ij)
  inter_i = 1/(m(m-1)) sum_{j != j'} rho(xhat_ij, xhat_ij')
  loss    = mean_i w_i (conf_i - lam/2 inter_i)
rho = "l2pow":        ||d||^beta, computed as (||d||^2 + 1e-12)^(beta/2) for beta != 2 (ddm2d's losses.norm_pow, same delta);
rho = "pseudo_huber": sqrt(||d||^2 + c^2) - c (iCT, Song & Dhariwal 2023, arXiv 2310.14189 Sec. 3.3; c = 0.00054 sqrt(3072)).
Optional feature term (cifar_C_dino): lam_f * the same estimator on frozen features phi (rho = ||.||^beta_f, lambda 1).
Everything is computed in float32. Differences are formed explicitly (no ||a||^2 + ||b||^2 - 2ab expansion), so rho near d = 0 is exact.
"""
from __future__ import annotations

import torch

DELTA = 1e-12


def rho_from_sq(sq: torch.Tensor, rho: str = "l2pow", beta: float = 2.0, c: float = 0.0) -> torch.Tensor:
    if rho == "l2pow":
        return sq if beta == 2.0 else (sq + DELTA) ** (beta / 2.0)
    if rho == "pseudo_huber":
        return torch.sqrt(sq + c * c) - c
    raise ValueError(f"unknown rho {rho}")


def energy_terms(x0: torch.Tensor, xhat: torch.Tensor, rho: str = "l2pow", beta: float = 2.0, c: float = 0.0):
    """x0: [n, ...]; xhat: [n, m, ...] (same trailing shape). Returns per-data-point (conf [n], inter [n]), float32."""
    n, m = xhat.shape[:2]
    if m < 2:
        raise ValueError("Eq. (14) needs m >= 2")
    a = x0.float().reshape(n, 1, -1)
    b = xhat.float().reshape(n, m, -1)
    conf = rho_from_sq(((a - b) ** 2).sum(-1), rho, beta, c).mean(1)
    iu, ju = torch.triu_indices(m, m, offset=1, device=b.device)       # unordered pairs j < j' (rho is symmetric)
    sq = ((b[:, iu] - b[:, ju]) ** 2).sum(-1)                           # [n, m(m-1)/2]
    inter = rho_from_sq(sq, rho, beta, c).sum(1) * 2.0 / (m * (m - 1))
    return conf, inter


def energy_loss(x0, xhat, w, lam: float, rho: str = "l2pow", beta: float = 2.0, c: float = 0.0):
    """Eq. (14) with distance rho. w: per-data-point weights [n]. Returns (loss, dict of detached diagnostics)."""
    conf, inter = energy_terms(x0, xhat, rho, beta, c)
    loss = (w * (conf - 0.5 * lam * inter)).mean()
    return loss, {"conf": conf.mean().detach(), "inter": inter.mean().detach()}


def velocity_loss(v, vhat, w):
    """Arm A: mean_i w_i ||vhat_i - v_i||^2."""
    sq = ((vhat.float() - v.float()) ** 2).flatten(1).sum(1)
    loss = (w * sq).mean()
    return loss, {"conf": sq.mean().detach(), "inter": torch.zeros((), device=sq.device)}
