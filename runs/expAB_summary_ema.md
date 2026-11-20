# Experiments A / B summary (EMA weights)

| run | seed | step | FID-10k | pixel D1 | Inception D1 | DINOv2 D1 | blockiness | 1-NN class acc | D1-confusion acc | diversity |
|---|---|---|---|---|---|---|---|---|---|---|
| expA_s0 | 0 | 5000 | 348.4 | 12.706 | 15.546 | 60.400 | 6.71 | 0.100 | 0.10 | 0.13 |
| expA_s0 | 0 | 10000 | 367.6 | 3.902 | 15.609 | 57.488 | 7.78 | 0.163 | 0.20 | 0.52 |
| expA_s0 | 0 | 20000 | 351.6 | 1.896 | 15.267 | 52.485 | 7.48 | 0.202 | 0.70 | 0.66 |
| expA_s0 | 0 | 30000 | 294.2 | 0.571 | 11.905 | 43.123 | 11.41 | 0.196 | 0.90 | 0.88 |
| expA_s1 | 1 | 5000 | 353.8 | 12.489 | 15.700 | 62.865 | 10.46 | 0.100 | 0.10 | 0.14 |
| expA_s1 | 1 | 10000 | 385.8 | 4.680 | 16.268 | 58.869 | 7.47 | 0.175 | 0.20 | 0.47 |
| expA_s1 | 1 | 20000 | 307.5 | 1.975 | 12.678 | 47.617 | 3.59 | 0.209 | 0.70 | 0.64 |
| expA_s1 | 1 | 30000 | 349.0 | 1.012 | 14.009 | 43.659 | 12.38 | 0.210 | 0.60 | 0.98 |
| expB_s0 | 0 | 5000 | 407.8 | 17.167 | 21.130 | 60.552 | 1.00 | 0.100 | 0.10 | 0.01 |
| expB_s0 | 0 | 10000 | 361.8 | 15.490 | 16.690 | 47.281 | 1.19 | 0.105 | 0.10 | 0.07 |
| expB_s0 | 0 | 20000 | 75.5 | 0.869 | 1.874 | 1.547 | 1.38 | 0.181 | 0.80 | 0.79 |
| expB_s0 | 0 | 30000 | 51.9 | 0.156 | 1.161 | 0.432 | 1.51 | 0.202 | 1.00 | 0.93 |
| expB_s1 | 1 | 5000 | 399.0 | 17.139 | 20.608 | 56.934 | 1.00 | 0.100 | 0.10 | 0.02 |
| expB_s1 | 1 | 10000 | 346.9 | 14.803 | 14.833 | 42.382 | 1.17 | 0.063 | 0.20 | 0.13 |
| expB_s1 | 1 | 20000 | 96.4 | 0.760 | 2.605 | 1.632 | 1.51 | 0.176 | 0.90 | 0.82 |
| expB_s1 | 1 | 30000 | 69.0 | 0.263 | 1.745 | 0.422 | 1.54 | 0.200 | 1.00 | 0.92 |

References (held-out test set, 1000/class): pixel D1 right-class real 0.011, wrong-class real 2.059; stage-4 block-shuffle thresholds (CIFAR-10): 16x16 0.214, 4x4 0.397, pixels 0.431. Inception D1 right/wrong-class real 0.002 / 3.260; DINOv2 D1 0.005 / 15.672. Real test images: 1-NN class acc 0.288, blockiness 1.01. DINOv2 D1 is the training space of B (circular for B); FID and Inception D1 are not.

## Experiment C rule (numbers only; look at the sample montage too)

- expA_s0 (seed 0), 20k -> 30k relative improvement: FID 0.163, Inception D1 0.220, pixel D1 0.699; FID monotone 10k->20k->30k: True. **quality still improving -> candidate for the 200k run (if the grids show structure)**
- expA_s1 (seed 1), 20k -> 30k relative improvement: FID -0.135, Inception D1 -0.105, pixel D1 0.488; FID monotone 10k->20k->30k: False. **in between -> decide from the grids and from B**
- seed 0 at 30k, B vs A: FID 51.9 vs 294.2; Inception D1 1.161 vs 11.905; pixel D1 0.156 vs 0.571
- seed 1 at 30k, B vs A: FID 69.0 vs 349.0; Inception D1 1.745 vs 14.009; pixel D1 0.263 vs 1.012

Figures: `runs/expAB_metrics_ema.png`, `runs/expAB_samples_ema.png`.
