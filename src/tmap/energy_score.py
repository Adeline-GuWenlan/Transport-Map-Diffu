"""Energy-score loss and energy-distance estimators (docs/energy_score_loss.tex, boxed eq. and Sec. 7, eq. (9)).

Loss (stage 1, boxed; negatively oriented, smaller is better):

    L(theta) = E_{(Y,C)~Q} E_{Z,Z' iid N(0,I)} [ ||F(Z,C) - Y||^beta - 1/2 ||F(Z,C) - F(Z',C)||^beta ],  beta = 1,

with (Z, Z') independent of (Y, C).  L = 1/2 sum_c pi_c D_beta(P_theta^c, Q_c) + const.

Minibatch estimator (stage 1 eq. (9), Prop. 15; unbiased for L):
    sum_c (n_c / N) [ mean_{i in c, j in c} ||x_j - y_i||^beta
                      - w / (m_c (m_c - 1)) sum_{j != k in c} ||x_j - x_k||^beta ],      w = 1/2
  * the interaction term uses only same-class generated pairs (never pooled across classes);
  * the j = k diagonal is excluded by index (U-statistic), divide by m_c (m_c - 1);
  * ||u||^beta uses the "double-where" trick: exact value, gradient 0 at u = 0 (DiT's zero-init head makes
    every generated pair identical at step 0; a plain sqrt gives NaN there).

Reference implementation this follows: docs/stage1_checks/es_core.py (torch_pow_dist, torch_es_loss_grouped).
"""
from __future__ import annotations

import torch

__all__ = ["pow_dist", "energy_score_loss", "energy_distance_u", "energy_distance_v", "pairwise_dist_eval",
           "per_class_energy_distance"]


def pow_dist(a: torch.Tensor, b: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """||a_i - b_j||^beta for a (n, d), b (m, d); exact (difference form), gradient 0 where a_i == b_j.

    Memory is O(n m d); call it per class (n, m = batch sizes per class), not on the full batch.
    """
    diff = a[:, None, :] - b[None, :, :]
    sq = (diff * diff).sum(-1)
    pos = sq > 0
    sq_safe = torch.where(pos, sq, torch.ones_like(sq))
    return torch.where(pos, sq_safe ** (beta / 2), torch.zeros_like(sq))


def energy_score_loss(x: torch.Tensor, y: torch.Tensor, y_labels: torch.Tensor, x_labels: torch.Tensor | None = None,
                      beta: float = 1.0, interaction_weight: float = 0.5, statistic: str = "u",
                      return_parts: bool = False):
    """Class-grouped unbiased minibatch estimator of the expected conditional energy score (stage 1 eq. (9)).

    x: (M, ...) generated samples F(z, c), z fresh N(0, I) independent of y.
    y: (N, ...) data points.  y_labels: (N,) classes.  x_labels: (M,) classes of x (default: y_labels, i.e. r = 1).
    Every class present in y needs m_c >= 2 generated samples of that class.
    Distances are computed in float32 (or the input dtype if float64), whatever the network's precision.
    statistic: "u" (the estimator; default) or "v" (keeps the zero diagonal; biased, for tests only).
    Returns the scalar loss, or (loss, dict(confinement=..., interaction=...)) with n_c/N-weighted parts.
    """
    if x_labels is None:
        x_labels = y_labels
    dt = torch.float64 if x.dtype == torch.float64 else torch.float32
    x = x.reshape(x.shape[0], -1).to(dt)
    y = y.reshape(y.shape[0], -1).to(dt)
    N = y.shape[0]
    conf_tot = x.new_zeros(())
    inter_tot = x.new_zeros(())
    for c in torch.unique(y_labels):
        xc = x[x_labels == c]
        yc = y[y_labels == c]
        m, n = xc.shape[0], yc.shape[0]
        if m < 2:
            raise ValueError(f"class {int(c)}: needs m_c >= 2 generated samples, got {m}")
        conf = pow_dist(xc, yc, beta).mean()
        dxx = pow_dist(xc, xc, beta)
        if statistic == "u":
            off = ~torch.eye(m, dtype=torch.bool, device=x.device)
            inter = dxx[off].sum() / (m * (m - 1))
        elif statistic == "v":
            inter = dxx.mean()
        else:
            raise ValueError(statistic)
        conf_tot = conf_tot + (n / N) * conf
        inter_tot = inter_tot + (n / N) * inter
    loss = conf_tot - interaction_weight * inter_tot
    if return_parts:
        return loss, {"confinement": conf_tot.detach(), "interaction": inter_tot.detach()}
    return loss


# ----------------------------------------------------------------------------- evaluation (no grad)
@torch.no_grad()
def pairwise_dist_eval(a: torch.Tensor, b: torch.Tensor, beta: float = 1.0, chunk: int = 2048) -> torch.Tensor:
    """||a_i - b_j||^beta in float64 via the Gram form (|a|^2 + |b|^2 - 2 a.b, clamped at 0), chunked.

    For evaluation only (no gradient). float64 keeps the cancellation error ~1e-13 relative, which is
    negligible next to typical image distances (~10). Returns float64 (n, m) on a's device.
    """
    a = a.reshape(a.shape[0], -1).double()
    b = b.reshape(b.shape[0], -1).double()
    bb = (b * b).sum(1)
    out = []
    for i in range(0, a.shape[0], chunk):
        ai = a[i:i + chunk]
        sq = (ai * ai).sum(1)[:, None] + bb[None, :] - 2.0 * ai @ b.T
        out.append(sq.clamp_min_(0.0) ** (beta / 2))
    return torch.cat(out, 0)


def _offdiag_mean(K: torch.Tensor) -> torch.Tensor:
    m = K.shape[0]
    return (K.sum() - torch.diagonal(K).sum()) / (m * (m - 1))


@torch.no_grad()
def energy_distance_u(x: torch.Tensor, y: torch.Tensor, beta: float = 1.0, yy_term: float | None = None) -> float:
    """Unbiased U-statistic for D_beta(P, Q) = 2E||X-Y|| - E||X-X'|| - E||Y-Y'||; x ~ P (m >= 2), y ~ Q (n >= 2).

    Can be < 0 on a finite sample; its expectation is >= 0 and is 0 iff P = Q.
    yy_term: precomputed U-mean of ||y - y'||^beta (to reuse across calls with the same y).
    """
    cross = pairwise_dist_eval(x, y, beta).mean()
    xx = _offdiag_mean(pairwise_dist_eval(x, x, beta))
    yy = _offdiag_mean(pairwise_dist_eval(y, y, beta)) if yy_term is None else yy_term
    return float(2 * cross - xx - yy)


@torch.no_grad()
def energy_distance_v(x: torch.Tensor, y: torch.Tensor, beta: float = 1.0) -> float:
    """Biased V-statistic (keeps the zero diagonals); for comparison only."""
    return float(2 * pairwise_dist_eval(x, y, beta).mean() - pairwise_dist_eval(x, x, beta).mean()
                 - pairwise_dist_eval(y, y, beta).mean())


@torch.no_grad()
def per_class_energy_distance(x: torch.Tensor, x_labels: torch.Tensor, y: torch.Tensor, y_labels: torch.Tensor,
                              beta: float = 1.0, yy_cache: dict | None = None) -> dict:
    """Per-class held-out unbiased D_beta between generated x (labels x_labels) and reference y (labels y_labels).

    Returns {"per_class": {c: D}, "mean": mean over classes, "within_x": {c: U-mean||x-x'||},
    "within_y": {c: U-mean||y-y'||}}.  yy_cache (dict) memoises the y-y term per class.
    """
    per, wx, wy = {}, {}, {}
    for c in torch.unique(y_labels).tolist():
        xc, yc = x[x_labels == c], y[y_labels == c]
        if yy_cache is not None and c in yy_cache:
            yy = yy_cache[c]
        else:
            yy = float(_offdiag_mean(pairwise_dist_eval(yc, yc, beta)))
            if yy_cache is not None:
                yy_cache[c] = yy
        xx = float(_offdiag_mean(pairwise_dist_eval(xc, xc, beta)))
        cross = float(pairwise_dist_eval(xc, yc, beta).mean())
        per[c] = 2 * cross - xx - yy
        wx[c], wy[c] = xx, yy
    return {"per_class": per, "mean": sum(per.values()) / len(per), "within_x": wx, "within_y": wy}
