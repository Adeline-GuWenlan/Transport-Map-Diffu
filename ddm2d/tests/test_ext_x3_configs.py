"""ddm2dx X3: the extra configs in configs/ext/x3 resolve and differ from their paper base only in the allowed keys."""
import json
from pathlib import Path

import pytest

from ddm2d import train as T

ROOT = Path(__file__).resolve().parents[1]
X3 = sorted((ROOT / "configs/ext/x3").glob("*.json"))
ALLOWED = {"name", "_comment", "dataset", "beta", "lam", "rho", "rho_c"}


def test_count():
    assert len(X3) == 1
    for line in (ROOT / "runs/ext/runs_x3.tsv").read_text().strip().splitlines()[1:]:
        assert (ROOT / line.split("\t")[1]).exists()


@pytest.mark.parametrize("path", X3, ids=lambda p: p.stem)
def test_x3_config(path):
    c = json.loads(path.read_text())
    base = json.loads((ROOT / f"configs/paper/gmm_{c['arm']}.json").read_text())
    T.resolve_config(c)
    assert c["name"] == path.stem
    for k in set(c) | set(base):
        if k not in ALLOWED:
            assert c[k] == base[k], k
    assert (c["dataset"], c["arm"], c["beta"], c["lam"]) == ("gmm9w", "C", 1.5, 1.0)
