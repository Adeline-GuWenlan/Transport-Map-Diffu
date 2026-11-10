"""Check (a): the energy-distance / energy-score estimators.

a1  U-statistic of D_beta has mean ~0 when P = Q (individual draws can be < 0); the V-statistic is biased up.
a2  Estimator mean matches the closed form of D_1 for 1-D Gaussians (via E|X - Y| = folded-normal mean),
    and the folded-normal formula itself matches numerical integration.
a3  Fourier constant C(d, beta) in  int (1 - cos<t,x>)/||t||^{d+beta} dt = C ||x||^beta, by quadrature,
    and the characteristic-function form of D_1 for two 1-D Gaussians.
a4  Variance of the loss and of its gradient as the number of generated samples m changes (n fixed),
    for per-data-point groups (engression-style) and for one pooled group.
Outputs: results/check_a.json, ../figures/check_a_variance.pdf
"""
import json
import os
import sys
import time

import numpy as np
from scipy import integrate, special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from es_core import (BLUE, ORANGE, AQUA, INK2, MUTED, energy_distance_u, energy_distance_v,
                     energy_distance_gauss1d, folded_normal_mean, fourier_constant, eight_gaussians)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
rng = np.random.default_rng(0)
out = {}
t0 = time.time()

# ---------------------------------------------------------------- a1: P = Q and P != Q, d = 2
R, n, m = 4000, 32, 32
for name, shift in [("P_eq_Q", 0.0), ("P_shift_0.5", 0.5)]:
    U, V = np.empty(R), np.empty(R)
    for r in range(R):
        x = rng.standard_normal((m, 2)); x[:, 0] += shift
        y = rng.standard_normal((n, 2))
        U[r] = energy_distance_u(x, y)
        V[r] = energy_distance_v(x, y)
    # closed form: X - Y and X - X' are 2-D Gaussians with covariance 2I, so their norms are Rice
    # distributed; D_1 = 2 E||N((shift,0), 2I)|| - 2 E||N(0, 2I)||
    from scipy.stats import rice
    ref = 2 * rice.mean(shift / np.sqrt(2), scale=np.sqrt(2)) - 2 * rice.mean(0.0, scale=np.sqrt(2))
    out[f"a1_{name}"] = dict(n=n, m=m, reps=R, U_mean=U.mean(), U_se=U.std(ddof=1) / np.sqrt(R),
                             U_frac_negative=float((U < 0).mean()), V_mean=V.mean(),
                             V_se=V.std(ddof=1) / np.sqrt(R), closed_form=ref)

# ---------------------------------------------------------------- a2: 1-D Gaussians, closed form
quad_checks = []
for delta, s in [(0.0, 1.0), (1.3, 0.7), (-2.0, 2.5)]:
    num, _ = integrate.quad(lambda w: abs(w) * np.exp(-(w - delta) ** 2 / (2 * s * s)) / np.sqrt(2 * np.pi * s * s),
                            -np.inf, np.inf)
    quad_checks.append(dict(delta=delta, s=s, closed=float(folded_normal_mean(delta, s)), quad=num))
out["a2_folded_normal_vs_quad"] = quad_checks

cases = [(0, 1, 0, 1), (0, 1, 0.5, 1), (0, 1, 0, 2), (1, 0.5, -1, 1.5), (0, 1, 3, 1)]
R2, n2 = 3000, 64
a2 = []
for mu1, s1, mu2, s2 in cases:
    est = np.array([energy_distance_u(mu1 + s1 * rng.standard_normal(n2), mu2 + s2 * rng.standard_normal(n2))
                    for _ in range(R2)])
    cf = float(energy_distance_gauss1d(mu1, s1, mu2, s2))
    se = est.std(ddof=1) / np.sqrt(R2)
    a2.append(dict(P=f"N({mu1},{s1}^2)", Q=f"N({mu2},{s2}^2)", closed_form=cf, est_mean=est.mean(), est_se=se,
                   z=(est.mean() - cf) / se))
out["a2_gauss1d"] = a2

# ---------------------------------------------------------------- a3: Fourier constant
def fourier_integral(d, beta):
    """Numerically integrate (1 - cos<t, e1>)/||t||^{d+beta} over R^d, radial form."""
    if d == 1:
        ang = lambda r: 1 - np.cos(r)
        area = 2.0
    elif d == 2:
        ang = lambda r: 1 - special.j0(r)          # mean of cos(r cos phi) over the circle is J0(r)
        area = 2 * np.pi
    elif d == 3:
        ang = lambda r: 1 - np.sinc(r / np.pi)      # mean over the sphere is sin(r)/r
        area = 4 * np.pi
    f = lambda r: ang(r) * r ** (-1 - beta)
    Rmax = 4000.0
    edges = np.concatenate([[0, 1], np.arange(2 * np.pi, Rmax + 1e-9, 2 * np.pi)])
    total = sum(integrate.quad(f, a, b, limit=200)[0] for a, b in zip(edges[:-1], edges[1:]))
    total += edges[-1] ** (-beta) / beta           # tail of the "1" part; oscillating part is O(R^{-1-beta})
    return area * total

a3 = []
for d in (1, 2, 3):
    for beta in (0.5, 1.0, 1.5):
        num = fourier_integral(d, beta)
        a3.append(dict(d=d, beta=beta, formula=float(fourier_constant(d, beta)), quadrature=num,
                       rel_err=abs(num / fourier_constant(d, beta) - 1)))
out["a3_fourier_constant"] = a3

# characteristic-function form of D_1 for two 1-D Gaussians
mu1, s1, mu2, s2 = 0.0, 1.0, 1.0, 2.0
phi_diff2 = lambda t: (np.exp(-s1 ** 2 * t ** 2) + np.exp(-s2 ** 2 * t ** 2)
                       - 2 * np.exp(-(s1 ** 2 + s2 ** 2) * t ** 2 / 2) * np.cos((mu1 - mu2) * t)) / t ** 2
cf_int = 2 * integrate.quad(phi_diff2, 0, np.inf, limit=400)[0] / fourier_constant(1, 1.0)
out["a3_charfn_form"] = dict(P="N(0,1)", Q="N(1,2^2)", via_charfn=cf_int,
                             closed_form=float(energy_distance_gauss1d(mu1, s1, mu2, s2)))

# ---------------------------------------------------------------- a4: variance vs m
# Target Q = 8 Gaussians on a circle; model x = s * z, z ~ N(0, I_2), at s = 1. Loss = S_1(P_s, Q) estimate.
# Gradient d/ds computed analytically.
def loss_grad_pooled(y, z, s):
    x = s * z
    dif = x[:, None, :] - y[None, :, :]                 # (M, n, 2)
    r = np.linalg.norm(dif, axis=-1)
    conf = r.mean()
    dconf = (np.einsum("mnk,mk->mn", dif, z) / r).mean()
    M = z.shape[0]
    dz = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    inter = dz.sum() / (M * (M - 1))                    # E||z - z'||, so E||x-x'|| = s * inter
    return conf - 0.5 * s * inter, dconf - 0.5 * inter

def loss_grad_perpoint(y, z, s):                        # z: (n, m, 2), group i = data point i
    x = s * z
    dif = x - y[:, None, :]
    r = np.linalg.norm(dif, axis=-1)                    # (n, m)
    conf = r.mean()
    dconf = ((dif * z).sum(-1) / r).mean()
    m = z.shape[1]
    dz = np.linalg.norm(z[:, :, None, :] - z[:, None, :, :], axis=-1)   # (n, m, m)
    inter = (dz.sum((1, 2)) / (m * (m - 1))).mean()
    return conf - 0.5 * s * inter, dconf - 0.5 * inter

n4, s = 64, 1.0
ms = [1, 2, 4, 8, 16, 32]
a4 = dict(n=n4, s=s, m=ms, pooled_loss_var=[], pooled_grad_var=[], perpoint_loss_var=[], perpoint_grad_var=[])
for mm in ms:
    R4 = 1500 if mm <= 8 else 600
    L1, G1, L2, G2 = [], [], [], []
    for _ in range(R4):
        y = eight_gaussians(n4, rng)
        l, g = loss_grad_pooled(y, rng.standard_normal((n4 * mm, 2)), s); L1.append(l); G1.append(g)
        if mm >= 2:
            l, g = loss_grad_perpoint(y, rng.standard_normal((n4, mm, 2)), s); L2.append(l); G2.append(g)
    a4["pooled_loss_var"].append(np.var(L1, ddof=1)); a4["pooled_grad_var"].append(np.var(G1, ddof=1))
    a4["perpoint_loss_var"].append(np.var(L2, ddof=1) if L2 else None)
    a4["perpoint_grad_var"].append(np.var(G2, ddof=1) if G2 else None)
# data-only floor: replace the model expectation by a very large sample (M = 8192), n = 64 data
Lf, Gf = [], []
zbig = None
for _ in range(300):
    y = eight_gaussians(n4, rng)
    l, g = loss_grad_pooled(y, rng.standard_normal((4096, 2)), s); Lf.append(l); Gf.append(g)
a4["floor_loss_var_M4096"] = np.var(Lf, ddof=1)
a4["floor_grad_var_M4096"] = np.var(Gf, ddof=1)
out["a4_variance_vs_m"] = a4

# figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
M = np.array(ms) * n4
for ax, key, floor, title in [(axes[0], "loss", a4["floor_loss_var_M4096"], "variance of loss estimate"),
                              (axes[1], "grad", a4["floor_grad_var_M4096"], "variance of d(loss)/ds")]:
    ax.loglog(M, a4[f"pooled_{key}_var"], "o-", color=BLUE, lw=2, ms=5, label="one pooled group")
    pp = [(Mi, v) for Mi, v in zip(M, a4[f"perpoint_{key}_var"]) if v is not None]
    ax.loglog([p[0] for p in pp], [p[1] for p in pp], "s-", color=ORANGE, lw=2, ms=5,
              label="per-data-point groups")
    ax.axhline(floor, color=MUTED, ls="--", lw=1.2, label="floor (n=64 data, M=4096)")
    ax.set_ylim(bottom=floor / 2)
    ax.set_xlabel("generated samples per batch  M = n m   (n = 64)")
    ax.set_title(title, fontsize=9, color=INK2)
    ax.grid(True, which="major", color="#e6e5e0", lw=0.6)
axes[0].legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "check_a_variance.pdf"))

out["runtime_sec"] = time.time() - t0
with open(os.path.join(RES, "check_a.json"), "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
