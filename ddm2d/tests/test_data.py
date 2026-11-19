import numpy as np

from ddm2d import data as D


def test_gmm_determinism_and_moments():
    a = D.generate("gmm", 102_400, 0)
    b = D.generate("gmm", 102_400, 0)
    assert a.dtype == np.float32 and a.shape == (102_400, 2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a[:4096], D.eval_set("gmm", 4096, 0))
    left = a[:, 0] < 0
    assert abs(left.mean() - 0.5) < 0.01
    for sel, mu in [(~left, (3, 3)), (left, (-3, 3))]:
        assert np.allclose(a[sel].mean(0), mu, atol=0.01)
        assert np.allclose(a[sel].std(0), 0.5, atol=0.01)


def test_checkerboard_support_and_determinism():
    x = D.generate("checkerboard", 200_000, 0)
    assert np.array_equal(x, D.generate("checkerboard", 200_000, 0))
    assert D.in_checkerboard(x).all()
    assert x.min() >= -4 and x.max() < 4
    # 8 occupied squares of side 2, equal mass
    cells = (np.floor(x[:, 0] / 2) + 2) * 4 + (np.floor(x[:, 1] / 2) + 2)
    counts = np.bincount(cells.astype(int), minlength=16)
    occ = counts[counts > 0]
    assert len(occ) == 8 and np.allclose(occ / len(x), 1 / 8, atol=0.005)
    # complement is empty: uniform points off the board are rejected
    u = np.random.default_rng(1).uniform(-4, 4, (100_000, 2))
    assert abs(D.in_checkerboard(u).mean() - 0.5) < 0.01


def test_checkerboard_matches_ffjord_formula():
    """Same arithmetic as FFJORD inf_train_gen('checkerboard') given the same uniforms."""
    rng = np.random.default_rng(5)
    n = 1000
    u1, u2, b = rng.random(n), rng.random(n), rng.integers(0, 2, n)
    x1 = u1 * 4 - 2
    x2 = u2 - b * 2 + (np.floor(x1) % 2)
    ref = np.stack([x1, x2], 1) * 2
    ours = D.sample_checkerboard(n, np.random.default_rng(5))
    assert np.allclose(ours, ref)


def test_train_set_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "CACHE", tmp_path)
    a = D.train_set("gmm", 1000, 0)
    assert (tmp_path / "gmm_train_n1000_seed0.npy").exists()
    assert np.array_equal(a, D.train_set("gmm", 1000, 0))
    assert D.fingerprint(a) == D.fingerprint(D.generate("gmm", 1000, 0))
