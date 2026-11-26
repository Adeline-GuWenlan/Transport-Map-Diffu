"""ddm2dx X1: every configs/ext config resolves, and differs from its paper base only in the allowed keys."""
import json
from pathlib import Path

import pytest

from ddm2d import train as T

ROOT = Path(__file__).resolve().parents[1]
EXT = sorted((ROOT / "configs/ext").glob("*.json"))
ALLOWED = {"name", "_comment", "dataset", "beta", "lam", "rho", "rho_c"}


def test_count():
    assert len(EXT) == 37
    runs = (ROOT / "runs/ext/runs.tsv").read_text().strip().splitlines()[1:]
    assert len(runs) == 64 and all((ROOT / r.split("\t")[1]).exists() for r in runs)


@pytest.mark.parametrize("path", EXT, ids=lambda p: p.stem)
def test_ext_config(path):
    c = json.loads(path.read_text())
    base = json.loads((ROOT / f"configs/paper/gmm_{c['arm']}.json").read_text())
    rc = T.resolve_config(c)
    assert c["name"] == path.stem
    for k in set(c) | set(base):
        if k not in ALLOWED:
            assert c[k] == base[k], k
    if c["arm"] == "C":
        assert rc["lam"] == 1.0
    if c["arm"] == "B":
        assert rc["lam"] == 0.0
    assert c["dataset"] in ("gmm", "gmm2w", "gmm9w") and c["dataset"] in path.stem.split("_")[0]
