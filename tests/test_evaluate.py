"""Evaluation helpers: baselines order correctly, class-confusion and 1-NN checks recognise real data."""
import torch

from tmap.data import ImageArray
from tmap.evaluate import baselines, class_confusion, nn_classifier_accuracy, reference_set


def test_baselines_and_class_checks_on_real_data():
    train = ImageArray("data/fashion_mnist", "train", 32, classes=[1, 5, 8], n_per_class=300)
    test = ImageArray("data/fashion_mnist", "test", 32, classes=[1, 5, 8])
    ref_x, ref_l = reference_set(test, 100)
    b = baselines(train, ref_x, ref_l, 100)
    assert b["train"]["mean"] < 0.5 < b["class_mean"]["mean"] < b["other_class"]["mean"]
    assert b["class_mean"]["mean"] < b["noise"]["mean"] < b["zero"]["mean"]
    # real train images used as "samples": right class on the diagonal, 1-NN agrees
    x, lx = reference_set(train, 100)
    M, acc = class_confusion(x, lx, ref_x, ref_l, n=100)
    assert acc == 1.0 and len(M) == 3
    assert nn_classifier_accuracy(x, lx, test, n_train_per_class=100, n_query_per_class=50) > 0.9


def test_blockiness_detects_patch_grid():
    from tmap.evaluate import blockiness, radial_power
    test = ImageArray("data/fashion_mnist", "test", 32)
    x = test.get(range(200))[0]
    assert 0.8 < blockiness(x, 4) < 1.3
    g = torch.Generator().manual_seed(0)
    blocks = torch.rand(50, 1, 8, 8, generator=g).repeat_interleave(4, -1).repeat_interleave(4, -2)
    blocks = blocks + 0.01 * torch.randn(blocks.shape, generator=g)
    assert blockiness(blocks, 4) > 5
    p = radial_power(x)
    assert set(p) == {"1-4", "4-8", "8-12", "12-17"} and p["1-4"] > p["12-17"]
