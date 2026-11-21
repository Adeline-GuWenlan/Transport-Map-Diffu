Status: DONE

S2 完成：4 个 UNet run 已经用 `scripts/exp_gpu.slurm`（未修改）按 DiT 的方式提交，现在都在 L40S 上正常训练。
Budget gate 通过：S1 bench 预计每个 run 不到 30 GPU-h，4 个 run 合计约 56 GPU-h，在 120 GPU-h 预算之内。
4 个 run 统一用 `grad_chunk=256`：B 在 512 会 OOM，而 prompt 要求 4 个 run 用同一个值。gradient 是 exact 的，所以只影响速度；
A 因此和 DiT A 一样是 two-pass，wall time 更可比，代价是每个 A run 多约 2.2 GPU-h。
`--time` A 14 h、B 20 h（= ⌈1.2 × training + eval⌉）。
14:14 EST 的 health check：loss 和 grad norm 都 finite，且在下降（A 1k 步 18.48 → 13.4k 步 18.35；B 37.80 → 37.07），
held-out pixel D1 raw 在 5k/10k 时为 0.10–0.29，速度与 bench 相差不到 1.3%（A 1.30–1.31 s/step，B 1.93），显存与 bench 一致（17.1 / 23.9 GiB），没有 requeue。
预计 A 约 11/21 20:30 EST 跑完（含 evaluation），B 约 11/22 01:45–01:55 EST。
下一步：S3 等这 4 个 job 跑完，检查 evaluation 是否齐全，再写 `summarize_backbones.py` 和最终报告。

## Runs
| run | job | config | seed | TRAIN_SET | --time |
|---|---|---|---|---|---|
| runs/unetA_s0 | 23968737 | configs/expA_cifar10_unet_pixel.yaml | 0 | `seed=0 train.grad_chunk=256` | 14:00:00 |
| runs/unetB_s0 | 23968738 | configs/expB_cifar10_unet_pixel-dinov2.yaml | 0 | `seed=0` | 20:00:00 |
| runs/unetA_s1 | 23968739 | configs/expA_cifar10_unet_pixel.yaml | 1 | `seed=1 train.grad_chunk=256` | 14:00:00 |
| runs/unetB_s1 | 23968740 | configs/expB_cifar10_unet_pixel-dinov2.yaml | 1 | `seed=1` | 20:00:00 |

Ids are in `claude_jobs/unet/jobids/S2.txt`. The health table, with step, s/step, memory, loss, grad norm, D1 and projected
finish per run, is in `docs/experiments_unet.md`, "Health check (S2, 2026-11-21 14:14 EST)" [E].

## Decisions
- **grad_chunk 256 for all 4 runs** (S1 had proposed 512 for A). Why: the S2 prompt says "keep it the same for all 4", and B does
  not fit at 512 (bench OOM). The A config still says 512 and was not edited, so the A runs override it in `TRAIN_SET`. Each run's
  `config.yaml` shows 256. The gradient is exact, so this changes speed only.
- **Health check in the same allocation.** All 4 jobs started within 8 min of submission, so I did not wait for the 2 h requeue.
  My first allocation (23968568) ended while I waited for more metric lines, before I wrote `.continue`. The wrapper requeued me
  as 23974056, and no work was lost.
- unetB_s1 waited briefly in `QOSMaxGRESPerUser`, because the user's other GPU jobs (val_subset_P*) share the GPU QOS. A requeue
  could hit the same wait. That costs time only, not work.

## Observation (not a result)
At matched steps the UNet's grad norm is about 2–3× DiT-B/2's (A s0 at 13.4k: UNet 8.3, DiT 3.2 [E: metrics.jsonl]). It falls
steadily. Same clip (2), unchanged. For S3 to look at when it compares the runs.

## Commits (pushed to origin main)
871b7b6 submit and run table · 12e0d3a health check, metrics/eval/sample/log snapshots · (this standup).

## Compute
- GPU-h used by the chain at 14:14 EST: about 19.5 (bench 0.13 + 4 × about 4.85 h). Projected total for the 4 runs: about 55 GPU-h [G].
  Budget 120.
- No other jobs were submitted. No scancel.

## Open questions for S3
- Commit the final `metrics.jsonl`, `eval.jsonl`, `eval/*.json`, PNGs, `feature_scale.jsonl` and slurm `.out` files.
- When writing the comparison, note that UNet runs at chunk 256 and DiT at 128. Both are exact, and only speed differs.
