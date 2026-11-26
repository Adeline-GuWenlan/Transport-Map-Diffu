"""ddm2dx X1: train.py golden test (existing configs: same resolved config, same parameters after 5 steps) and the rho option."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from ddm2d import data as D
from ddm2d import train as T

ROOT = Path(__file__).resolve().parents[1]

# computed at commit bff64b1 (before any ddm2dx change) with 4 torch threads: sha256 of the resolved config JSON (sort_keys) and of
# the model state_dict bytes after 5 training steps of configs/paper/gmm_{A,B,C}.json, seed 0
GOLDEN = {
    "A": ("1fc23d8522db5b24f72274cb9ef9d37a48a839bcee26f2c92f52a9d910ba9882",
          "662c4b4418579357316a0b5f2c6d30aad210862a2624e9720b807a49d03cbe75"),
    "B": ("e084c65f189bd2246463bc7a37d2ed8e7516220937cd2399247c88de1dd52ba1",
          "d23e671786c9e4f7a8451d80f9456a5e398c5bbc7f89cf2acad087e49df5f7ed"),
    "C": ("286d9a2f445ea282a9c591baa842b4adc48237742783968c09d6ff9f9c7c96ce",
          "f6906cc7339acd67e70cecd61ace81dfe610993d4b870b0a5ac7b184d9f0f74d"),
}


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_golden_paper_configs(arm):
    cfg = json.loads((ROOT / f"configs/paper/gmm_{arm}.json").read_text())
    rc = T.resolve_config(cfg)
    assert hashlib.sha256(json.dumps(rc, sort_keys=True).encode()).hexdigest() == GOLDEN[arm][0]
    nt = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        tr = T.Trainer(cfg, 0)
        for _ in range(5):
            tr.train_step()
    finally:
        torch.set_num_threads(nt)
    h = hashlib.sha256()
    for v in tr.model.state_dict().values():
        h.update(v.numpy().tobytes())
    assert h.hexdigest() == GOLDEN[arm][1]


def test_explicit_l2_equals_default():
    base = json.loads((ROOT / "configs/paper/gmm_C.json").read_text())
    base.update(steps=3, batch=8, m=4)
    a = T.Trainer(base, 0)
    b = T.Trainer(dict(base, rho="l2"), 0)
    for _ in range(3):
        assert a.train_step() == b.train_step()


def test_rho_config_validation():
    ok = {"dataset": "gmm", "arm": "C", "beta": 1.0, "lam": 1.0}
    T.resolve_config(dict(ok, rho="imq", rho_c=1.0))
    with pytest.raises(ValueError):
        T.resolve_config(dict(ok, rho="cosine"))
    with pytest.raises(ValueError):
        T.resolve_config(dict(ok, rho="imq"))                       # missing rho_c
    with pytest.raises(ValueError):
        T.resolve_config(dict(ok, rho="rbf", rho_c=1.0, beta=0.1))  # beta unused -> must be 1.0
    with pytest.raises(ValueError):
        T.resolve_config(dict(ok, arm="A", rho="l1"))
    T.resolve_config(dict(ok, rho="l1", beta=0.5))


@pytest.fixture
def small_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "CACHE", tmp_path / "cache")


@pytest.mark.parametrize("rho,beta,c", [("l1", 0.5, 0.0), ("pseudo_huber", 1.0, 0.1), ("laplace", 1.0, 3.5)])
def test_rho_resume_equals_uninterrupted(tmp_path, small_cache, rho, beta, c):
    cfg = {"dataset": "gmm9w", "arm": "C", "steps": 8, "batch": 16, "m": 4, "train_size": 40, "ckpt_every": 4, "log_every": 2,
           "warmup": 3, "beta": beta, "lam": 1.0, "rho": rho, "rho_c": c}
    a, b = tmp_path / "a", tmp_path / "b"
    T.run(cfg, a, 3, 1, None, None, argv=["x"])
    T.run(cfg, b, 3, 1, None, 4, argv=["x"])
    T.run(cfg, b, 3, 1, None, None, argv=["x"])
    sa = torch.load(a / "ckpt" / "final.pt", weights_only=False)
    sb = torch.load(b / "ckpt" / "final.pt", weights_only=False)
    for k in sa["model"]:
        assert torch.equal(sa["model"][k], sb["model"][k])
    la = [json.loads(l) for l in open(a / "log.jsonl")]
    lb = [json.loads(l) for l in open(b / "log.jsonl")]
    assert [r["loss"] for r in la] == [r["loss"] for r in lb] and all(r["finite"] for r in la)
    assert sa["cfg"]["rho"] == rho
