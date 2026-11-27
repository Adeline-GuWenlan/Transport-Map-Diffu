"""X3 theory check: pseudo-Huber rho(d) = sqrt(|d|^2 + c^2) - c = f(|d|^2) with the Bernstein representation
f(s) = int_0^inf (1 - exp(-s u)) mu(du),  mu(du) = exp(-c^2 u) u^(-3/2) / (2 sqrt(pi)) du   (from sqrt(s) = int (1-e^{-su}) u^{-3/2}/(2 Gamma(1/2)) du,
shifted by c^2). Hence the energy-type divergence D_rho(P, Q) = int MMD^2_{k_u}(P, Q) mu(du) with Gaussian kernels k_u = exp(-u |d|^2),
which is > 0 for P != Q (each k_u is characteristic, mu has full support). Numerical check of the representation; output in runs/ext/eval/."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
rows = []
for c in [0.1, 1.0]:
    for s in [0.01, 1.0, 25.0, 400.0]:
        f = lambda u: (1 - np.exp(-s * u)) * np.exp(-c * c * u) * u ** -1.5 / (2 * np.sqrt(np.pi))
        v = quad(f, 0, 1, limit=200)[0] + quad(f, 1, np.inf, limit=200)[0]
        rows.append({"c": c, "s": s, "integral": v, "closed_form": float(np.sqrt(s + c * c) - c), "abs_err": abs(v - (np.sqrt(s + c * c) - c))})
out = Path(__file__).resolve().parents[1] / "runs/ext/eval/pseudohuber_bernstein_check.json"
out.write_text(json.dumps(rows, indent=1) + "\n")
print(max(r["abs_err"] for r in rows))
