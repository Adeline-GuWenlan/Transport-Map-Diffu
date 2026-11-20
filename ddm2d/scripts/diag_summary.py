"""S4 DIAGNOSTIC: table of paper checkpoint vs + 10k steps at lr 1e-4 (scripts/diag_finetune.py), from runs/diag/diag_eval*.json.
Writes runs/diag/diag_summary.md. Diagnostic only; never part of Table 2 / Fig. 3.

Usage: python scripts/diag_summary.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "runs" / "diag"


def main():
    ev = {}
    for p in sorted(D.glob("diag_eval*.json")):
        ev.update(json.loads(p.read_text()))
    L = []
    for ds, nfes, tl in [("cb", ["5", "10", "50", "100", "1000"], 0.01), ("gmm", ["5", "10", "20", "30", "50", "70", "90", "100"], 0.01)]:
        L += [f"### {'checkerboard' if ds == 'cb' else 'GMM'} (MMD²_u, mean of 10 repeats, Table 2 / Fig. 3 protocol)", "",
              "| run | weights | " + " | ".join(f"NFE {n}" for n in nfes) + f" | t={tl}: RMS err of E[x̂] | t={tl}: draw std |",
              "|---|---|" + "---|" * (len(nfes) + 2)]
        per = {}
        for arm in "ABC":
            for s in range(3):
                run = f"{ds}_{arm}_s{s}"
                o, f = ev.get(f"runs/paper/{run}"), ev.get(f"runs/diag/{run}_lr0.0001_10000")
                if o is None or f is None:
                    continue
                for tag, e in [("paper config, 100k", o), ("+10k at lr 1e-4", f)]:
                    dn = [q for q in e["denoiser"] if q["t"] == tl][0]
                    L.append(f"| {run} | {tag} | " + " | ".join(f"{e['summary'][n]:.2e}" for n in nfes) +
                             f" | {dn['rms_err_mean']:.3f} | {dn['draw_std']:.3f} |")
                    per.setdefault((arm, tag), []).append([e["summary"][n] for n in nfes])
        for (arm, tag), v in per.items():
            if len(v) > 1:
                m = np.mean(v, 0)
                L.append(f"| **{ds} {arm} mean of {len(v)} seeds** | {tag} | " + " | ".join(f"{x:.2e}" for x in m) + " | | |")
        L.append("")
    (D / "diag_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
