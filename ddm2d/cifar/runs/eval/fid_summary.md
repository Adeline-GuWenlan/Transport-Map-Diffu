Final (30k) EMA, FID-10k, sampler seed 0 (the protocol). Lower is better.

| run | NFE 2 | NFE 4 | NFE 8 | NFE 10 | NFE 15 | NFE 30 | NFE 100 |
|---|---|---|---|---|---|---|---|
| A (diffusion, v-target) | 102.86 | 56.01 | 33.61 | 29.03 | 22.58 | 15.43 | 9.20 |
| B (β=2, λ=0) | 117.98 | 63.29 | 39.51 | 34.30 | 27.31 | 19.41 | 13.99 |
| B (β=1, λ=0) | 112.68 | 63.06 | 38.86 | 33.69 | 26.62 | 18.29 | 11.83 |
| B (pseudo-Huber, λ=0) | 113.07 | 62.44 | 38.55 | 33.24 | 26.11 | 18.10 | 11.73 |
| C (β=0.1, λ=1) | 201.27 | 72.08 | 23.01 | 15.91 | 9.29 | 7.75 | 10.49 |
| C + DINOv2 term | 49.35 | 42.57 | 44.37 | 46.75 | 49.71 | 54.91 | 62.43 |

EMA snapshots (checkpoint selection), FID-10k, sampler seed 0.

| run | 10k NFE 10 | 20k NFE 10 | 30k NFE 10 | 10k NFE 100 | 20k NFE 100 | 30k NFE 100 |
|---|---|---|---|---|---|---|
| A (diffusion, v-target) | 34.24 | 30.77 | 29.03 | 14.40 | 10.30 | 9.20 |
| B (β=2, λ=0) | 41.20 | 36.58 | 34.30 | 20.16 | 15.83 | 13.99 |
| B (β=1, λ=0) | 38.87 | 36.15 | 33.69 | 16.92 | 13.17 | 11.83 |
| B (pseudo-Huber, λ=0) | 39.59 | 36.11 | 33.24 | 17.50 | 13.33 | 11.73 |
| C (β=0.1, λ=1) | 43.68 | 23.92 | 15.91 | 14.20 | 10.39 | 10.49 |
| C + DINOv2 term | 71.36 | 63.68 | 46.75 | 108.36 | 107.58 | 62.43 |

Sampler-seed check on CPU (fp32; the protocol's seed 0 ran on the GPU in bf16), final EMA. Separate from the protocol.

| run | NFE | seed | device / precision | FID-10k |
|---|---|---|---|---|
| B (β=1, λ=0) | 4 | 0 | GPU / bf16 (protocol) | 63.06 |
| B (β=1, λ=0) | 4 | 1 | CPU / fp32 | 64.20 |
| B (β=1, λ=0) | 4 | 2 | CPU / fp32 | 63.46 |

Real-data FID-10k floor (real CIFAR-10 images vs the same train statistics).

| set | FID-10k |
|---|---|
| test[0:10000] | 3.15 |
| train randperm(seed 0)[0:10000] | 2.10 |
| train randperm(seed 0)[10000:20000] | 2.11 |
| train randperm(seed 0)[20000:30000] | 2.11 |
