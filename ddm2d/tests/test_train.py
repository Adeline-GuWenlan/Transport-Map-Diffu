import json

import pytest
import torch

from ddm2d import data as D
from ddm2d import train as T


@pytest.fixture(autouse=True)
def small_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "CACHE", tmp_path / "cache")


def cfg(arm):
    c = {"dataset": "gmm", "arm": arm, "steps": 8, "batch": 16, "m": 4, "train_size": 40, "ckpt_every": 4, "log_every": 2, "warmup": 3}
    if arm == "B":
        c.update(beta=2.0, lam=0.0)
    return c


def final_state(out):
    sd = torch.load(out / "ckpt" / "final.pt", weights_only=False)
    return sd


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_resume_equals_uninterrupted(tmp_path, arm):
    c = cfg(arm)                         # train_size 40 / batch 16 -> epoch boundary (reshuffle) inside the run
    a, b = tmp_path / "a", tmp_path / "b"
    assert T.run(c, a, seed=1, threads=1, deadline=None, max_steps=None) == 0
    # interrupted: stop at step 3 via max_steps (checkpoint), then by the deadline right after the next step, then finish
    assert T.run(c, b, seed=1, threads=1, deadline=None, max_steps=3) == 0
    assert not (b / "DONE.json").exists()
    assert T.run(c, b, seed=1, threads=1, deadline=0.0, max_steps=None) == 3
    assert T.run(c, b, seed=1, threads=1, deadline=None, max_steps=None) == 0
    sa, sb = final_state(a), final_state(b)
    assert sa["step"] == sb["step"] == 8
    for key in ("model", "ema"):
        for k in sa[key]:
            assert torch.equal(sa[key][k], sb[key][k]), (key, k)
    for k, v in sa["opt"]["state"].items():
        for kk in v:
            assert torch.equal(v[kk], sb["opt"]["state"][k][kk])
    assert torch.equal(sa["gen"], sb["gen"])
    la = [json.loads(l) for l in open(a / "log.jsonl")]
    lb = [json.loads(l) for l in open(b / "log.jsonl")]
    assert [r["step"] for r in la] == [r["step"] for r in lb] == [2, 4, 6, 8]
    assert [r["loss"] for r in la] == [r["loss"] for r in lb]
    done = json.loads((b / "DONE.json").read_text())
    assert done["steps"] == 8 and done["n_segments"] == 3 and done["peak_rss_mb"] > 0
    assert (b / "ckpt_manifest.txt").read_text().count("final.pt") == 1
    # a finished run is not retrained
    assert T.run(c, b, seed=1, threads=1, deadline=None, max_steps=None) == 0


def test_warmup_and_ema_and_finite(tmp_path):
    assert [round(T.lr_factor(k, 4), 6) for k in range(6)] == [round(x, 6) for x in [0.146447, 0.5, 0.853553, 1.0, 1.0, 1.0]]
    tr = T.Trainer(cfg("C"), seed=0)
    p0 = [p.detach().clone() for p in tr.model.parameters()]
    tr.train_step()
    for pe, p, q in zip(tr.ema.parameters(), tr.model.parameters(), p0):
        assert torch.allclose(pe, 0.99 * q + 0.01 * p.detach(), atol=1e-7)
    assert tr.acc["finite"] and tr.acc["gnorm"] > 0


def test_different_seeds_differ_same_seed_same():
    l1 = [T.Trainer(cfg("C"), seed=0).train_step() for _ in range(1)]
    l2 = [T.Trainer(cfg("C"), seed=0).train_step() for _ in range(1)]
    l3 = [T.Trainer(cfg("C"), seed=1).train_step() for _ in range(1)]
    assert l1 == l2 and l1 != l3


def test_load_ema_model(tmp_path):
    out = tmp_path / "r"
    T.run(cfg("C"), out, seed=0, threads=1, deadline=None, max_steps=None)
    m = T.load_ema_model(out)
    sd = final_state(out)
    assert all(torch.equal(m.state_dict()[k], sd["ema"][k]) for k in sd["ema"])
