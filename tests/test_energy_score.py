"""Estimator tests: closed-form 1-D Gaussian energy distance, unbiasedness, grouping, zero-distance gradient."""
import math
import os
import sys

import numpy as np
import pytest
import torch
from scipy.stats import norm

from tmap.energy_score import (energy_distance_u, energy_distance_v, energy_score_loss, pairwise_dist_eval,
                               per_class_energy_distance, pow_dist)


def folded_normal_mean(delta, s):
    """E|W| for W ~ N(delta, s^2)."""
    return s * math.sqrt(2 / math.pi) * math.exp(-delta ** 2 / (2 * s ** 2)) + delta * (1 - 2 * norm.cdf(-delta / s))


def ed_gauss1d(mu1, s1, mu2, s2):
    """Closed-form D_1(N(mu1, s1^2), N(mu2, s2^2)) = 2E|X-Y| - E|X-X'| - E|Y-Y'|."""
    return 2 * folded_normal_mean(mu1 - mu2, math.hypot(s1, s2)) - 2 * s1 / math.sqrt(math.pi) - 2 * s2 / math.sqrt(math.pi)


@pytest.mark.parametrize("mu1,s1,mu2,s2", [(0.0, 1.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0), (0.0, 2.0, 0.0, 1.0),
                                           (0.5, 1.5, -0.3, 0.7)])
def test_energy_distance_matches_1d_gaussian_closed_form(mu1, s1, mu2, s2):
    g = torch.Generator().manual_seed(0)
    reps, m, n = 300, 200, 200
    vals = []
    for _ in range(reps):
        x = mu1 + s1 * torch.randn(m, 1, generator=g, dtype=torch.float64)
        y = mu2 + s2 * torch.randn(n, 1, generator=g, dtype=torch.float64)
        vals.append(energy_distance_u(x, y))
    vals = np.array(vals)
    se = vals.std(ddof=1) / math.sqrt(reps)
    target = ed_gauss1d(mu1, s1, mu2, s2)
    assert abs(vals.mean() - target) < 4 * se + 1e-4, (vals.mean(), target, se)


def test_loss_expectation_matches_closed_form_1d():
    """E[loss] = E|X - Y| - 1/2 E|X - X'| exactly (unbiased), here for X ~ N(1, 0.5^2), Y ~ N(0, 1), m = n = 4."""
    mu1, s1, mu2, s2 = 1.0, 0.5, 0.0, 1.0
    target = folded_normal_mean(mu1 - mu2, math.hypot(s1, s2)) - 0.5 * 2 * s1 / math.sqrt(math.pi)
    g = torch.Generator().manual_seed(1)
    reps, B = 4000, 4
    x = mu1 + s1 * torch.randn(reps, B, 1, generator=g, dtype=torch.float64)
    y = mu2 + s2 * torch.randn(reps, B, 1, generator=g, dtype=torch.float64)
    lab = torch.zeros(B, dtype=torch.long)
    vals = np.array([float(energy_score_loss(x[i], y[i], lab)) for i in range(reps)])
    se = vals.std(ddof=1) / math.sqrt(reps)
    assert abs(vals.mean() - target) < 4 * se, (vals.mean(), target, se)


def test_estimators_unbiased_at_p_equals_q():
    """P = Q: U-statistic D has mean 0; the V-statistic (diagonal kept) is biased upward. Small m makes bias visible."""
    g = torch.Generator().manual_seed(2)
    reps, m = 3000, 4
    u, v, loss = [], [], []
    lab = torch.zeros(m, dtype=torch.long)
    for _ in range(reps):
        x = torch.randn(m, 2, generator=g, dtype=torch.float64)
        y = torch.randn(m, 2, generator=g, dtype=torch.float64)
        u.append(energy_distance_u(x, y))
        v.append(energy_distance_v(x, y))
        loss.append(float(energy_score_loss(x, y, lab)) - float(energy_score_loss(x, y, lab, statistic="v")))
    u, v = np.array(u), np.array(v)
    se_u = u.std(ddof=1) / math.sqrt(reps)
    assert abs(u.mean()) < 4 * se_u, (u.mean(), se_u)
    assert v.mean() > 10 * v.std(ddof=1) / math.sqrt(reps), "V-statistic should be visibly biased at m = 4"
    # U-loss minus V-loss = -1/2 (U-mean - V-mean) of ||x - x'|| < 0 always (U-mean excludes zeros): sign check
    assert np.all(np.array(loss) < 0)


def test_loss_expectation_unbiased_2d_p_equals_q():
    """E[loss] at P = Q = N(0, I_2) equals E||X - Y|| - 1/2 E||X - X'|| = 1/2 E||X - X'|| = 1/2 * sqrt(pi) (d = 2)."""
    target = 0.5 * math.sqrt(math.pi)          # E||X - X'|| for X, X' ~ N(0, I_2) is sqrt(2) * sqrt(pi/2) = sqrt(pi)
    g = torch.Generator().manual_seed(3)
    reps, m = 4000, 3
    lab = torch.zeros(m, dtype=torch.long)
    vals = np.array([float(energy_score_loss(torch.randn(m, 2, generator=g, dtype=torch.float64),
                                             torch.randn(m, 2, generator=g, dtype=torch.float64), lab))
                     for _ in range(reps)])
    se = vals.std(ddof=1) / math.sqrt(reps)
    assert abs(vals.mean() - target) < 4 * se, (vals.mean(), target, se)


def test_grouping_is_per_class_and_weighted():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(12, 3, 4, 4, generator=g, dtype=torch.float64)
    y = torch.randn(12, 3, 4, 4, generator=g, dtype=torch.float64)
    lab = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    perm = torch.randperm(12, generator=g)
    a = energy_score_loss(x, y, lab)
    b = energy_score_loss(x[perm], y[perm], lab[perm])
    assert torch.allclose(a, b), "loss must not depend on the order of the batch"
    manual = 0.0
    for c, w in [(0, 6 / 12), (1, 3 / 12), (2, 3 / 12)]:
        xc, yc = x[lab == c].flatten(1), y[lab == c].flatten(1)
        m = len(xc)
        dxy = torch.cdist(xc, yc).mean()
        dxx = torch.cdist(xc, xc)
        manual += w * (dxy - 0.5 * (dxx.sum() - dxx.diagonal().sum()) / (m * (m - 1)))
    assert torch.allclose(a, torch.as_tensor(manual), atol=1e-10)
    pooled = (torch.cdist(x.flatten(1), x.flatten(1)).sum()) / (12 * 11)
    conf = sum(w * torch.cdist(x[lab == c].flatten(1), y[lab == c].flatten(1)).mean()
               for c, w in [(0, .5), (1, .25), (2, .25)])
    assert not torch.allclose(a, conf - 0.5 * pooled), "interaction must not be pooled across classes"


def test_r_samples_per_datapoint():
    """m_c = r * n_c generated samples with their own labels."""
    g = torch.Generator().manual_seed(5)
    y = torch.randn(6, 5, generator=g, dtype=torch.float64)
    yl = torch.tensor([0, 0, 0, 1, 1, 1])
    x = torch.randn(12, 5, generator=g, dtype=torch.float64)
    xl = yl.repeat_interleave(2)
    loss = energy_score_loss(x, y, yl, xl)
    assert torch.isfinite(loss)
    with pytest.raises(ValueError):
        energy_score_loss(x[:1], y[:3], yl[:3], xl[:1])


def test_zero_distance_gradient_is_finite():
    """DiT's zero-init head: all generated samples identical (= 0) at step 0. No NaN in the gradient."""
    x = torch.zeros(16, 1, 8, 8, requires_grad=True)
    y = torch.randn(16, 1, 8, 8)
    lab = torch.arange(2).repeat_interleave(8)
    loss = energy_score_loss(x, y, lab)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    # only the confinement term moves x at step 0: grad = -(1/N) * sum_i (y_i - x)/||y_i - x|| / n_c ... non-zero
    assert x.grad.abs().sum() > 0
    d = pow_dist(torch.zeros(3, 4, requires_grad=True), torch.zeros(3, 4))
    assert torch.equal(d, torch.zeros(3, 3))


def test_eval_distances_match_exact():
    g = torch.Generator().manual_seed(6)
    a = torch.rand(50, 1024, generator=g) * 2 - 1
    b = torch.rand(40, 1024, generator=g) * 2 - 1
    assert torch.allclose(pairwise_dist_eval(a, b), pow_dist(a.double(), b.double()), atol=1e-9)
    assert torch.allclose(pairwise_dist_eval(a, a).diagonal(), torch.zeros(50, dtype=torch.float64), atol=1e-6)


def test_matches_stage1_reference(root):
    sys.path.insert(0, os.path.join(root, "docs", "stage1_checks"))
    from es_core import torch_es_loss_grouped
    g = torch.Generator().manual_seed(7)
    x = torch.randn(40, 16, generator=g, dtype=torch.float64)
    y = torch.randn(40, 16, generator=g, dtype=torch.float64)
    lab = torch.arange(5).repeat_interleave(8)
    assert torch.allclose(energy_score_loss(x, y, lab), torch_es_loss_grouped(x, lab, y, lab), atol=1e-12)


def test_per_class_energy_distance_detects_wrong_class():
    g = torch.Generator().manual_seed(8)
    y = torch.cat([torch.randn(200, 3, generator=g) - 2, torch.randn(200, 3, generator=g) + 2])
    yl = torch.cat([torch.zeros(200), torch.ones(200)]).long()
    x_ok = torch.cat([torch.randn(200, 3, generator=g) - 2, torch.randn(200, 3, generator=g) + 2])
    r_ok = per_class_energy_distance(x_ok, yl, y, yl)
    r_bad = per_class_energy_distance(x_ok, 1 - yl, y, yl)
    assert abs(r_ok["mean"]) < 0.05 and r_bad["mean"] > 3
