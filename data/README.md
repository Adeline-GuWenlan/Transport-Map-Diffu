# data/

Downloaded and checked by stage 3 on 2026-11-06 (cs627, Slurm job 22047089). Download commands are stage 2's
(docs/stage2_choices.md §3.3). Every raw sha256 matched stage 2's value. Decode: `python -I scripts/prepare_data.py`
(about 20 s). It re-checks the raw hashes, then writes the `.npy` arrays and `meta.json`. Nothing is unpickled.

```
data/
  fashion_mnist/raw/*.gz                     4 IDX files (30 MB)       smoke dataset (configs/smoke_cpu.yaml)
  fashion_mnist/{train,test}_{images,labels}.npy, meta.json
  mnist/raw/*.gz                             4 IDX files (12 MB)       fallback only
  mnist/{train,test}_{images,labels}.npy, meta.json
  cifar10/raw/*.parquet                      2 files (137 MB)          real dataset (configs/base_gpu.yaml, smoke_cpu_cifar10.yaml)
  cifar10/{train,test}_{images,labels}.npy, meta.json, official_crosscheck.json
  fid/stats/cifar10_{legacy_tensorflow,clean}_train_32.npz             clean-fid reference stats (mu 2048, sigma 2048x2048)
  fid/inception/inception-2015-12-05.pt                                Inception-v3 TorchScript used by clean-fid
  fid/edm/cifar10-32x32.npz                                            NVIDIA EDM FID reference (cross-check only)
  fid/check_fid_refs.json                                              output of scripts/check_fid_refs.py
```
Total 645 MB, 31 files.

## Array format (all three datasets)
- `*_images.npy`: **uint8, N × C × H × W**, native resolution, in source-file order. Fashion-MNIST and MNIST
  are 1 × 28 × 28, CIFAR-10 is 3 × 32 × 32 (RGB). Use `np.load(..., mmap_mode="r")`.
- `*_labels.npy`: int64, N.
- No padding, scaling or augmentation is stored. The loader does them (configs): pad 28 → 32 with uint8 0
  (2 px per side), then `x / 127.5 − 1` to [−1, 1], so padding becomes −1. CIFAR-10 gets hflip in training only.
  `scripts/check_env.py::real_batch` is a reference implementation of this, together with the class-stratified draw.
- `meta.json` records the source, raw sha256, class names, per-split counts, pixel mean/std, the number of
  unique images, and sha256 of each `.npy`.

## Datasets

| | Fashion-MNIST | MNIST (fallback) | CIFAR-10 |
|---|---|---|---|
| source | http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/ (Zalando Research; github.com/zalandoresearch/fashion-mnist) | https://ossci-datasets.s3.amazonaws.com/mnist/ (mirror used by torchvision) | HF `uoft-cs/cifar10` @ `0b2714987fa478483af9968de7c934580d0bb9a2`, `plain_text/*.parquet` (mirror of https://www.cs.toronto.edu/~kriz/cifar.html) |
| licence | MIT (repo LICENSE) | CC BY-SA 3.0 (assumption: as stated in torchvision docs; not checked at the source) | none stated; cite Krizhevsky 2009 |
| train / test | 60000 / 10000 | 60000 / 10000 | 50000 / 10000 (official split) |
| per class (train / test) | 6000 / 1000, exactly balanced | unbalanced: train 5923 6742 5958 6131 5842 5421 5918 6265 5851 5949, test 980 1135 1032 1010 982 892 958 1028 974 1009 | 5000 / 1000, exactly balanced |
| classes (label order) | T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot | 0–9 | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck (parquet metadata matches the official order) |
| pixel mean / std in [0, 1] (train) | 0.2860 / 0.3530 | 0.1307 / 0.3081 | R 0.4914 G 0.4822 B 0.4465 / 0.2470 0.2435 0.2616 |
| unique images (train, test) | 60000, 10000 | 60000, 10000 | 50000, 10000 |
| test images byte-identical to a train image | 0 | 0 | 0 |

The CIFAR-10 channel means match the widely used values (0.4914, 0.4822, 0.4465). The MNIST counts are its
standard, unbalanced ones. The class-stratified sampler handles that.

### Raw files, sha256
```
fashion_mnist/raw  3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84  train-images-idx3-ubyte.gz  26421880 B
                   a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845  train-labels-idx1-ubyte.gz     29515 B
                   346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073  t10k-images-idx3-ubyte.gz    4422102 B
                   67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5  t10k-labels-idx1-ubyte.gz       5148 B
mnist/raw          440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609  train-images-idx3-ubyte.gz   9912422 B
                   3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c  train-labels-idx1-ubyte.gz     28881 B
                   8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6  t10k-images-idx3-ubyte.gz    1648877 B
                   f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6  t10k-labels-idx1-ubyte.gz       4542 B
cifar10/raw        8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883  train-00000-of-00001.parquet 119705255 B
                   841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de  test-00000-of-00001.parquet   23940850 B
```

### Decoded arrays, sha256 (also in each meta.json)
```
fashion_mnist  4d6d54d7b7f1d1146412ac0d5ebfad200214b8dbc262a70b79e84877afef3279  train_images.npy
               eb6efccc70db136ce327076edeadbbcba39b38ab16fd8f89b0625f62fc605f24  train_labels.npy
               4cdcdb7b44ef667e143916344990c85c236bee299a09d137df3050c35ba6eb91  test_images.npy
               33e7c41e1d27e7ea46528a86fd6d54e0eb0d60f705e95ae5e9c956635f269cbc  test_labels.npy
mnist          ce5e2952ff69d663f3b4d67d193df149bc38c265afa4be151bc306df10232c26  train_images.npy
               10226b938104d9231ead4e9a0627f17e66cd18baf31a28784f1b72147057decf  train_labels.npy
               b5eacb271cad91d8c5bf59205ea56ee965234df25fec7b7a62a5092130040326  test_images.npy
               996074233ccae69ee65b8551233f608adc1bce84b6b440e3531478a847958149  test_labels.npy
cifar10        304a769ab0682c43bbdc4f766303066aef2073d37441d820ecc2666ccb3432b3  train_images.npy
               9e2ee9261f6a7c6509b35aaa7161bf15d6d99673bbcac04130702d12a6da88d3  train_labels.npy
               efbfa3b24c4f91c6febe8a57d87b6448f2c1226d8b2aa24d14db1eea85ad46c6  test_images.npy
               fc48d9ecfdbeacce2dacf004498170f2df12e75e3485475017d2663b587a92f3  test_labels.npy
```

### CIFAR-10: HF mirror vs official release
CROSSCHECK_PLACEHOLDER

## FID reference files (CIFAR-10 only; configs/base_gpu.yaml, eval.fid)
| file | source | sha256 |
|---|---|---|
| `fid/stats/cifar10_legacy_tensorflow_train_32.npz` | https://www.cs.cmu.edu/~clean-fid/stats/cifar10_legacy_tensorflow_train_32.npz | `8e2ddeabe3515e74058f52218e93f0c87adb935cd1352f4cff883bac0c4eaa9a` |
| `fid/stats/cifar10_clean_train_32.npz` | https://www.cs.cmu.edu/~clean-fid/stats/cifar10_clean_train_32.npz | `a7b9763a8316c2b03997d92ee9c3900588b563326521fba05373f16e179b0fb9` |
| `fid/inception/inception-2015-12-05.pt` | https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt | `f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4` |
| `fid/edm/cifar10-32x32.npz` | https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz | `4ada108cdfd8e43409ed427f5f76631beac9b9617cdcaf416d41b1e161d85969` |

Where clean-fid looks for them (read from the installed `cleanfid/features.py` and `fid.py`, v0.1.35):
- **Stats.** It looks in `<site-packages>/cleanfid/stats/` and downloads them if absent. Stage 3 copied both
  `.npz` files there: `/scratch/wg2381/.conda/envs/tmap/lib/python3.11/site-packages/cleanfid/stats/`.
  `env/setup_env.sh` repeats the copy on a rebuild, and `check_env.py` verifies the hashes.
- **Inception weights.** The path is hard-coded to **`/tmp/inception-2015-12-05.pt`**
  (`features.py::feature_extractor`), which is node-local, and is downloaded if missing. On a new node, before
  computing FID, run `cp data/fid/inception/inception-2015-12-05.pt /tmp/`, or let it download from the
  NVIDIA CDN, which is reachable from compute nodes. `scripts/check_fid_refs.py` does the copy.
- The Inception forward runs on CPU (`check_env.py`). Use `compute_fid(..., device=torch.device("cpu"),
  use_dataparallel=False)` there, because clean-fid's default device is `cuda`.

FID_PLACEHOLDER
