"""Data loading and the class-stratified sampler (Fashion-MNIST, data/README.md)."""
import torch

from tmap.data import ClassStratifiedSampler, ImageArray


def test_fashion_mnist_shapes_range_and_padding():
    d = ImageArray("data/fashion_mnist", "test", image_size=32)
    assert len(d) == 10000 and d.images.dtype == torch.uint8 and d.images.shape[1:] == (1, 28, 28)
    x, y = d.get([0, 1, 2])
    assert x.shape == (3, 1, 32, 32) and x.dtype == torch.float32 and y.dtype == torch.long
    assert x.min() >= -1 and x.max() <= 1
    assert torch.all(x[..., :2, :] == -1) and torch.all(x[..., :, -2:] == -1), "2-px zero padding -> -1"


def test_class_stratified_sampler():
    d = ImageArray("data/fashion_mnist", "test", image_size=32)
    s = ClassStratifiedSampler(d, classes_per_batch=8, n_per_class=8, seed=0)
    x, y = s.next()
    assert x.shape == (64, 1, 32, 32)
    counts = torch.bincount(y, minlength=10)
    assert (counts > 0).sum() == 8 and set(counts[counts > 0].tolist()) == {8}
    st = s.state_dict()
    x1, y1 = s.next()
    s.load_state_dict(st)
    x2, y2 = s.next()
    assert torch.equal(x1, x2) and torch.equal(y1, y2), "sampler state must give exact resume"


def test_subset_and_hflip():
    d = ImageArray("data/cifar10", "test", image_size=32, classes=[3], n_per_class=20)
    assert len(d) == 20 and d.classes == [3]
    s = ClassStratifiedSampler(d, 1, 20, seed=0, hflip=True)
    x, y = s.next()
    assert x.shape == (20, 3, 32, 32) and torch.all(y == 3)
