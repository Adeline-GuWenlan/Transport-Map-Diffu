"""Check (b): beta = 2 cannot separate distributions with the same mean; beta in (0, 2) can.

b1  P = N(0,1), Q = N(0,4): Monte Carlo of the unbiased estimators of D_1 and D_2, vs closed forms
    D_1 = (2 sqrt(10) - 6)/sqrt(pi) and D_2 = 2 (mu_P - mu_Q)^2 = 0.
b2  Expected score S_beta(N(0, sigma^2), Q) - S_beta(Q, Q) as a function of sigma, closed form,
    for several beta: unique minimum at sigma = 2 for beta < 2, flat for beta = 2.
b3  Weighted interaction term  E|X-Y| - (lam/2) E|X-X'|  over the Gaussian family N(0, s^2), Q = N(0,1):
    numerical minimiser vs the formula s^2 = lam^2 / (2 - lam^2).
Outputs: results/check_b.json, ../figures/check_b_beta2.pdf
"""
import json
import os
import sys
import time

import numpy as np
from scipy import optimize
from scipy.special import gamma

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from es_core import BLUE, ORANGE, AQUA, YELLOW, VIOLET, INK2, MUTED, energy_distance_u

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "results")
rng = np.random.default_rng(1)
t0 = time.time()
out = {}

# ---------------------------------------------------------------- b1
R, n = 2000, 200
D1, D2 = np.empty(R), np.empty(R)
for r in range(R):
    x = rng.standard_normal(n)
    y = 2.0 * rng.standard_normal(n)
    D1[r] = energy_distance_u(x, y, 1.0)
    D2[r] = energy_distance_u(x, y, 2.0)
D1_true = (2 * np.sqrt(10) - 6) / np.sqrt(np.pi)
out["b1"] = dict(n=n, reps=R, D1_closed=D1_true, D1_mean=D1.mean(), D1_se=D1.std(ddof=1) / np.sqrt(R),
                 D1_frac_positive=float((D1 > 0).mean()), D2_closed=0.0, D2_mean=D2.mean(),
                 D2_se=D2.std(ddof=1) / np.sqrt(R), D2_frac_positive=float((D2 > 0).mean()))

# power of a permutation-free sign test is not the point; also report a 2-sample comparison
# where the means differ, so beta = 2 does see a mean shift:
D2s = np.array([energy_distance_u(rng.standard_normal(n) + 0.5, rng.standard_normal(n), 2.0) for _ in range(500)])
out["b1_mean_shift_beta2"] = dict(shift=0.5, D2_closed=2 * 0.25, D2_mean=D2s.mean(),
                                  D2_se=D2s.std(ddof=1) / np.sqrt(500))

# ---------------------------------------------------------------- b2
def abs_moment(s, beta):     # E|N(0, s^2)|^beta
    return s ** beta * 2 ** (beta / 2) * gamma((beta + 1) / 2) / np.sqrt(np.pi)

def score(sigma, beta, tau=2.0):     # S_beta(N(0, sigma^2), N(0, tau^2))
    return abs_moment(np.sqrt(sigma ** 2 + tau ** 2), beta) - 0.5 * abs_moment(np.sqrt(2) * sigma, beta)

sig = np.linspace(0.0, 5.0, 501)
betas = [0.5, 1.0, 1.5, 1.9, 2.0]
b2 = {}
for b in betas:
    vals = score(sig, b) - score(2.0, b)
    b2[str(b)] = dict(argmin_sigma=float(sig[np.argmin(vals)]), min=float(vals.min()),
                      excess_at_sigma0=float(vals[0]), excess_at_sigma4=float(vals[400]))
out["b2"] = b2

# ---------------------------------------------------------------- b3
def wscore(s, lam):           # E|X - Y| - lam/2 E|X - X'|, X ~ N(0, s^2), Y ~ N(0, 1)
    return np.sqrt(2 / np.pi) * np.sqrt(1 + s * s) - lam * s / np.sqrt(np.pi)

b3 = []
for lam in [0.25, 0.5, 0.75, 1.0, 1.25]:
    res = optimize.minimize_scalar(lambda s: wscore(s, lam), bounds=(0, 50), method="bounded",
                                   options=dict(xatol=1e-10))
    b3.append(dict(lam=lam, s_numeric=float(res.x), s_formula=float(np.sqrt(lam ** 2 / (2 - lam ** 2)))))
for lam in [1.5, 2.0]:
    b3.append(dict(lam=lam, score_at_s=[float(wscore(s, lam)) for s in (1, 10, 100, 1000)],
                   note="decreasing without bound in s (lam >= sqrt 2)"))
out["b3_weighted_interaction"] = b3

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
ax = axes[0]
bins = np.linspace(min(D1.min(), D2.min()), max(D1.max(), np.quantile(D2, 0.995)), 70)
ax.hist(D2, bins=bins, color=ORANGE, alpha=0.75, label=r"$\hat D_2$ ($\beta=2$)")
ax.hist(D1, bins=bins, color=BLUE, alpha=0.75, label=r"$\hat D_1$ ($\beta=1$)")
ax.axvline(0, color=ORANGE, lw=1.5, ls="--")
ax.axvline(D1_true, color=BLUE, lw=1.5, ls="--")
ax.set_xlabel(r"estimate of $D_\beta(N(0,1),N(0,4))$, $n=m=200$")
ax.set_ylabel("count (2000 reps)")
ax.set_title("same mean, different scale", fontsize=9, color=INK2)
ax.legend(frameon=False, fontsize=7.5)
ax = axes[1]
cols = [VIOLET, BLUE, AQUA, YELLOW, ORANGE]
for b, c in zip(betas, cols):
    vals = score(sig, b) - score(2.0, b)
    ax.plot(sig, vals / abs_moment(2.0, b), color=c, lw=2, label=rf"$\beta={b}$")
ax.axvline(2.0, color=MUTED, lw=1, ls=":")
ax.set_xlabel(r"model scale $\sigma$ in $P=N(0,\sigma^2)$; data $Q=N(0,2^2)$")
ax.set_ylabel(r"$[S_\beta(P,Q)-S_\beta(Q,Q)]\,/\,E|Y|^\beta$")
ax.set_title("expected score minus its minimum", fontsize=9, color=INK2)
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "check_b_beta2.pdf"))

out["runtime_sec"] = time.time() - t0
with open(os.path.join(RES, "check_b.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
