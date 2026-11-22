Status: DONE

S3 完成：4 个 UNet run（`runs/unetA_s{0,1}`、`runs/unetB_s{0,1}`）全部 COMPLETED：各跑满 30k steps，没有 requeue，也没有 divergence，
5k/10k/20k/30k 四个 snapshot 的 evaluation（EMA 和 raw）都齐全，所以不需要 relaunch。
**30k EMA FID-10k（2 个 seed 的 mean ± half-range）：DiT-B/2 A 321.6 ± 27.4，UNet A 158.1 ± 17.9，DiT-B/2 B 60.4 ± 8.6，UNet B 26.9 ± 1.9**
（raw weights 下 UNet B 是 17.9，DiT B 是 62.9）。
结论：在 loss、数据、seed、optimizer 和预算都相同的情况下，backbone 影响很大。
- Pixel-only A：UNet 完全消除了 DiT A 的 2×2 mosaic（blockiness 1.02，DiT A 是 11.89），FID 也减半；但 30k 时仍然生成不出物体，只是平滑的色块。所以"只用 pixel ES、30k 步、生成不出物体"这一点和 backbone 无关。
- B（+DINOv2 ES）：UNet 明显更好。10k、20k、30k 时两个 seed 的 FID 和 Inception D1 都优于 DiT 的两个 seed。UNet 10k 步时的 FID 已经是 39.5，好于 DiT B 30k 的 60.4。样本是干净、大多类别正确的物体。成本也不更高：16.3 GPU-h/run，DiT 是 16.9。UNet B 只在 DINOv2 D1（它自己的训练 feature，circular）、pixel D1 和 diversity 上比 DiT B 差。
- 整个 chain 用了 55.1 GPU-h（预算 120）。
- 下一步建议（没有提交任何 job）：把 UNet B 从 30k checkpoint 续训到 100k（约 37 GPU-h/seed）；补 FID-50k 和第三个 seed；不要在 pixel-only A 上花长 run。

## Where things are
- Doc: `docs/experiments_unet.md`, new sections: "Results (S3)", "Does the backbone matter, given the loss?", "Measured cost", "Caveats" and
  "What to run next".
- README: section "Backbone comparison: UNet vs DiT-B/2".
- Tables: `runs/backbone_summary_ema.md` and `runs/backbone_summary_raw.md`. Each has 7 metrics × 4 steps × 8 runs, seed mean ± half-range and a cost table.
  Figures: `runs/backbone_metrics_{ema,raw}.png`, `runs/backbone_samples_{ema,raw}.png`.
- Script: `scripts/summarize_backbones.py`. It imports the readers of `scripts/summarize_ab.py`, which is unchanged. DiT numbers are read from
  the committed `runs/exp{A,B}_s*/eval/*.json`.

## 30k EMA, seed mean ± half-range [E: runs/backbone_summary_ema.md]
| arm | FID-10k | Inception D1 | pixel D1 | DINOv2 D1 | blockiness | 1-NN class | diversity |
|---|---|---|---|---|---|---|---|
| DiT A | 321.6 ± 27.4 | 12.957 | 0.792 | 43.391 | 11.89 | 0.203 | 0.93 |
| UNet A | 158.1 ± 17.9 | 5.044 | 0.878 | 27.431 | 1.02 | 0.235 | 0.86 |
| DiT B | 60.4 ± 8.6 | 1.453 | 0.209 | 0.427 (circular) | 1.52 | 0.201 | 0.92 |
| UNet B | 26.9 ± 1.9 | 0.540 | 0.464 | 1.398 (circular) | 0.96 | 0.243 | 0.85 |

Grid reading at 30k EMA (mine, [G] as a judgement):
- DiT A: 2×2 mosaics.
- UNet A: smooth blobs and "camouflage" shapes, no objects. s0 shows little variety.
- DiT B: blurry objects; animal classes mixed.
- UNet B: clean, varied objects, nearly all class-correct. About 5 of 100 s1 cells are degenerate.

## Completeness check [E]
| run | job | steps | ckpt 5k/10k/20k/30k | eval json ×4 | EMA+raw PNGs | wall |
|---|---|---|---|---|---|---|
| unetA_s0 | 23968737 | 30000 | yes | yes | yes | 11:09:07 |
| unetA_s1 | 23968739 | 30000 | yes | yes | yes | 11:09:41 |
| unetB_s0 | 23968738 | 30000 | yes | yes | yes | 16:19:22 |
| unetB_s1 | 23968740 | 30000 | yes | yes | yes | 16:18:42 |

## Commits (pushed to origin main)
- 3273495 run outputs (144 files, same file set as DiT commit 8a15563, no checkpoints)
- f34d0e0 summarize_backbones.py and its outputs
- 8d8048c doc and README
- (this standup)

## Compute
- S3 submitted no GPU job and cancelled nothing.
- Chain total: **55.06 GPU-h** of 120 (S1 bench 0.13, plus the 4 UNet runs at 11.15 + 11.16 + 16.32 + 16.31).
  These are wall times of the allocation segments in `runs/slurm-tmap_u*.out`.

## Deviations and notes
- The S2 health-check note "UNet grad norm 2–3× DiT" holds for B only. DiT A's grad norm grew late, to 32.7 / 119.3 at 30k, against UNet A's 7.2 / 5.2.
- The GPU-h in the cost table come from the slurm `.out` timestamps, not from `sacct`, so they can be traced to a committed file. Each DiT run's
  one requeue gap is excluded; its restart overhead is included.
- The UNet ran at `grad_chunk` 256 and DiT at 128. Both give exact gradients; only speed differs. Wall time is therefore not a clean backbone cost.
- Uncommitted files I did not create (none were left in `runs/` after my commits) were not touched.

## Open questions for the user
- Continue UNet B to 100k–200k? This is the strongest one-step map so far, and it was still improving (−13% FID from 20k to 30k).
- Is a size- or locality-ablation (DiT patch 1, or UNet without attention) worth running to explain the gap?
