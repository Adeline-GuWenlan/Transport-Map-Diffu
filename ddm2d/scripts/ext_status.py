"""X3 step 1: completeness of every run in runs/ext/runs.tsv (+ the reference runs runs/paper/gmm_{A,B,C}_s{0,1}).
Checks DONE.json at 100k steps, final.pt sha256 = DONE.json, final.pt loads, EMA params finite, no FAILED.txt. Writes runs/ext/STATUS.md."""
import csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
torch.set_num_threads(1)
from ddm2d.train import load_ema_model, sha256_file

rows = list(csv.DictReader(open(ROOT / "runs/ext/runs.tsv"), delimiter="\t"))
x3 = list(csv.DictReader(open(ROOT / "runs/ext/runs_x3.tsv"), delimiter="\t")) if (ROOT / "runs/ext/runs_x3.tsv").exists() else []
rows = rows + x3                     # X3 extras (job 24786603)
runs = [(r["experiment"], r["out_dir"]) for r in rows] + [("ref", f"runs/paper/gmm_{a}_s{s}") for a in "ABC" for s in (0, 1, 2)]
out, bad = [], 0
for exp, d in runs:
    p = ROOT / d
    done = json.loads((p / "DONE.json").read_text()) if (p / "DONE.json").exists() else None
    ok_steps = bool(done) and done["steps"] == 100_000
    sha_ok = bool(done) and sha256_file(p / "ckpt/final.pt") == done.get("final_ckpt_sha256")
    try:
        m = load_ema_model(p)
        finite = all(torch.isfinite(v).all().item() for v in m.state_dict().values())
        loads = True
    except Exception as e:  # noqa
        loads, finite = False, False
    failed = (p / "FAILED.txt").exists()
    good = ok_steps and sha_ok and loads and finite and not failed
    bad += not good
    out.append(f"| {exp} | `{d}` | {done['steps'] if done else '-'} | {done.get('n_segments', '-') if done else '-'} | "
               f"{'yes' if sha_ok else 'NO'} | {'yes' if loads else 'NO'} | {'yes' if finite else 'NO'} | {'yes' if failed else 'no'} | "
               f"{'OK' if good else 'INCOMPLETE'} |")
hdr = ["# Completeness of the 2D runs (ddm2dx X3, step 1)", "",
       f"`python scripts/ext_status.py`. {len(runs) - bad} / {len(runs)} runs complete "
       f"({len(rows)} ext runs from `runs.tsv` and `runs_x3.tsv` + 9 reference runs `runs/paper/gmm_{{A,B,C}}_s{{0,1,2}}`).", "",
       "Checks: `DONE.json` steps = 100000; sha256 of `ckpt/final.pt` equals `DONE.json`; the EMA weights of `final.pt` load "
       "(`train.load_ema_model`) and are all finite; no `FAILED.txt`.", "",
       "| exp | run | steps | segments | sha256 ok | loads | finite | FAILED.txt | status |", "|---|---|---|---|---|---|---|---|---|"]
(ROOT / "runs/ext/STATUS.md").write_text("\n".join(hdr + out) + "\n")
print(len(runs) - bad, "/", len(runs), "complete")
