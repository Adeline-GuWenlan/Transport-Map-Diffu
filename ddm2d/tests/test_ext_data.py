"""ddm2dx X1: golden test (existing datasets unchanged) and the new weighted GMMs gmm2w / gmm9w."""
import numpy as np
import pytest

from ddm2d import data as D

# sha256 fingerprints computed with the code at commit bff64b1, before any ddm2dx change
GOLDEN = {
    ("gmm", "train"): "8f7aef9fe6be89bba71f22d1a26d8438aa7a684cae2fac81a3567899c32a5af1",
    ("gmm", 0): "51b178347a857d5c2a417847579c713ebc92566c479572c4d1c0bab970ad2b00",
    ("gmm", 1): "fa23e2fd91e8f8996b4e8628ead7ecd38f572554335e74dcde3b926dbf66081c",
    ("gmm", 9): "c4088c88ba7e7e5db58708901f55a44576fd9369d00938ca18898525fb7b7ad2",
    ("checkerboard", "train"): "b145bb416f19073dc211e23b9720aa488a1774b68f863533b5438f8644fa67cb",
    ("checkerboard", 0): "3a3d6dea6f8064e508273fab0176f06d10c23954f536d4b00766edd66f68ec2b",
    ("checkerboard", 1): "fe2b2d237f2ff5a5c08e38d61fb29d135b2aa5d6acbc8ffe2bc9f8f1172157fb",
    ("checkerboard", 9): "5cd2209409a7dbaa28c5d10c87f019140f1572e7ea79a09b95cd56651b117593",
}


@pytest.mark.parametrize("key", list(GOLDEN))
def test_golden_existing_datasets(key):
    name, which = key
    x = D.generate(name, D.TRAIN_SIZE, 0) if which == "train" else D.eval_set(name, 4096, which)
    assert D.fingerprint(x) == GOLDEN[key]


def test_golden_cached_train_sets():
    for name in ["gmm", "checkerboard"]:
        assert D.fingerprint(D.train_set(name, cache=False)) == GOLDEN[(name, "train")]


def test_gmm9w_spec():
    means, stds, w = D.GMM_SPECS["gmm9w"]
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert means[0].tolist() == [-4, 4] and means[2].tolist() == [4, 4] and means[8].tolist() == [4, -4] and means[4].tolist() == [0, 0]
    assert stds.tolist() == [0.5, 0.3, 0.4, 0.3, 0.6, 0.3, 0.4, 0.5, 0.3]


@pytest.mark.parametrize("name", ["gmm2w", "gmm9w"])
def test_weighted_gmm_moments_and_determinism(name):
    means, stds, w = D.GMM_SPECS[name]
    n = D.TRAIN_SIZE
    a = D.generate(name, n, 0)
    assert a.dtype == np.float32 and a.shape == (n, 2)
    assert np.array_equal(a, D.generate(name, n, 0))
    ev = D.eval_set(name, 4096, 0)
    assert not np.array_equal(a[:4096], ev) and not np.array_equal(ev, D.eval_set(name, 4096, 1))
    x, comp = D.sample_weighted_gmm(n, np.random.default_rng(0), means, stds, w, return_comp=True)
    assert np.array_equal(x.astype(np.float32), a)
    for k in range(len(w)):
        sel = comp == k
        se = np.sqrt(w[k] * (1 - w[k]) / n)
        assert abs(sel.mean() - w[k]) < 5 * se
        assert np.allclose(x[sel].mean(0), means[k], atol=5 * stds[k] / np.sqrt(sel.sum()) + 1e-3)
        assert np.allclose(x[sel].std(0), stds[k], rtol=0.05)


def test_gmm2w_only_changes_weights():
    means, stds, w = D.GMM_SPECS["gmm2w"]
    assert np.array_equal(means, D.GMM_MEANS) and np.all(stds == D.GMM_STD) and w.tolist() == [0.8, 0.2]
    x = D.generate("gmm2w", D.TRAIN_SIZE, 0)
    assert abs((x[:, 0] > 0).mean() - 0.8) < 0.006
