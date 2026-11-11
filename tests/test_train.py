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


def _tiny_map():
    from tmap.model import TransportMap
    torch.manual_seed(0)
    m = TransportMap(input_size=8, patch_size=2, in_channels=3, hidden_size=32, depth=2, num_heads=2, num_classes=3)
    with torch.no_grad():                       # leave DiT's zero init so the map is not constant
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m


def test_chunked_two_pass_gradient_equals_one_pass():
    """grad_chunk > 0 (exact two-pass gradient) gives the one-pass loss and parameter gradients, with and without
    the feature-space term."""
    from tmap.features import RandomConv
    from tmap.train import forward_backward
    m = _tiny_map()
    g = torch.Generator().manual_seed(1)
    xl = torch.tensor([0] * 5 + [1] * 5 + [2] * 5)
    z = torch.randn(15, 3, 8, 8, generator=g)
    y = torch.randn(15, 3, 8, 8, generator=g)
    lc = {"beta": 1.0, "interaction_weight": 0.5}
    for enc in (None, RandomConv(3, 8)):
        out = []
        for chunk in (0, 4):
            m.zero_grad(set_to_none=True)
            loss, parts = forward_backward(m, enc, z, xl, y, xl, lc, 0.7, False, chunk)
            out.append((loss, parts, [p.grad.clone() for p in m.parameters() if p.grad is not None]))
        (l0, p0, g0), (l1, p1, g1) = out
        assert torch.allclose(l0, l1, rtol=1e-5), (l0, l1)
        assert ("feat_loss" in p0) == (enc is not None)
        assert len(g0) == len(g1) > 0
        for a, b in zip(g0, g1):
            assert torch.allclose(a, b, rtol=1e-4, atol=1e-7), (a - b).abs().max()


def test_feature_loss_chunked_snapshots_and_resume(tmp_path):
    """Feature term + two-pass gradient + snapshots train end to end; lambda is fixed once and survives resume."""
    ovr = OVR + ["loss.feature={encoder: random_conv, weight: 1.0, scale: auto, scale_n_per_class: 8}",
                 "train.grad_chunk=3", "train.snapshot_steps=[2, 4]"]
    a = load_config("configs/smoke_cpu.yaml", ovr + ["train.max_steps=4"])
    train(a, str(tmp_path / "straight"))
    b = load_config("configs/smoke_cpu.yaml", ovr + ["train.max_steps=2"])
    train(b, str(tmp_path / "resumed"))
    train(a, str(tmp_path / "resumed"))
    ka = torch.load(tmp_path / "straight" / "ckpt" / "latest.pt", weights_only=False)
    kb = torch.load(tmp_path / "resumed" / "ckpt" / "latest.pt", weights_only=False)
    assert ka["feat"]["lambda"] > 0 and ka["feat"] == kb["feat"]
    for k in ka["model"]:
        assert torch.allclose(ka["model"][k], kb["model"][k], atol=1e-6), k
    for s in (2, 4):
        snap = torch.load(tmp_path / "straight" / "ckpt" / f"step{s:07d}.pt", weights_only=False)
        assert snap["step"] == s and "opt" not in snap and set(snap) >= {"model", "ema", "config", "feat"}
    recs = [json.loads(l) for l in open(tmp_path / "straight" / "metrics.jsonl")]
    assert all("feat_loss" in r and "pixel_loss" in r for r in recs)
    lines = open(tmp_path / "resumed" / "feature_scale.jsonl").read().splitlines()
    assert len(lines) == 1                       # computed once, not again on resume
