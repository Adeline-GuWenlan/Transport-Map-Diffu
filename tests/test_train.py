"""End-to-end: a few training steps with a tiny DiT; resume from a checkpoint reproduces an uninterrupted run."""
import json
import os

import torch

from tmap.train import load_config, train

OVR = ["model.depth=1", "model.hidden_size=32", "model.num_heads=2", "model.patch_size=8",
       "batch.classes_per_batch=2", "batch.n_per_class=4", "train.log_every=2", "train.ckpt_every=3",
       "train.sample_every=1000", "eval.every=1000", "eval.energy_distance.n_per_class=8",
       "data.subset={classes: [0, 1], n_per_class: 32}", "train.time_limit_min=0", "num_threads=1"]


def test_train_and_exact_resume(tmp_path):
    a = load_config("configs/smoke_cpu.yaml", OVR + ["train.max_steps=6"])
    train(a, str(tmp_path / "straight"))
    b = load_config("configs/smoke_cpu.yaml", OVR + ["train.max_steps=3"])
    train(b, str(tmp_path / "resumed"))
    train(a, str(tmp_path / "resumed"))            # same out dir: resumes at step 3, runs to 6
    ka = torch.load(tmp_path / "straight" / "ckpt" / "latest.pt", weights_only=False)
    kb = torch.load(tmp_path / "resumed" / "ckpt" / "latest.pt", weights_only=False)
    assert ka["step"] == kb["step"] == 6
    for k in ka["model"]:
        assert torch.allclose(ka["model"][k], kb["model"][k], atol=1e-6), k
        assert torch.allclose(ka["ema"][k], kb["ema"][k], atol=1e-6), k
    recs = [json.loads(l) for l in open(tmp_path / "straight" / "metrics.jsonl")]
    assert [r["step"] for r in recs] == [2, 4, 6]
    for key in ("loss", "grad_norm", "lr", "steps_per_s", "imgs_per_s", "confinement", "interaction"):
        assert all(key in r and r[key] == r[key] for r in recs), key
    assert os.path.exists(tmp_path / "straight" / "samples" / "step0000006_ema.png")
    ev = [json.loads(l) for l in open(tmp_path / "straight" / "eval.jsonl")]
    assert ev[0]["step"] == 0 and ev[-1]["step"] == 6 and "ema_ed_mean" in ev[-1]
