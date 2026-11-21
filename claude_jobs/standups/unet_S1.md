Status: DONE

S1 完成：按 De Bortoli et al. 2025 App. I.2（arXiv 2502.02483 v3, p. 33）自己实现了 UNet backbone（`src/tmap/unet.py`），
`build_model` 通过 `model.backbone: unet` 分发。UNet 140.0 M params / 20.0 GMACs per sample，DiT-B/2 用同一 counter 是 129.7 M / 21.8，
参数比 1.08×，在 0.6–1.6× 范围内。全部 35 个 pytest 通过，其中包括改代码前记录的 DiT golden test。所以 DiT 行为没变。
determinism check 证明同一 seed 下 UNet 和 DiT 看到完全相同的 batch、flip 和 z（seed 0、1 都是）。
两个 config 和 DiT 版只差 name、comments、model block 和 `grad_chunk`（A 512，B 256）。
GPU bench（job 23951657，L40S）：UNet A 1.012 s/step（30.6 GiB），UNet B 1.906 s/step（23.9 GiB），B 在 chunk 512 时 OOM。
预计每个 30k-step run：A 8.97 GPU-h，B 16.64 GPU-h，4 个 run 共 51.2 GPU-h，低于 30 GPU-h/run 的停止线和 120 GPU-h 的总预算。
下一步：S2 用 `scripts/exp_gpu.slurm` 提交 4 个 run，建议 `--time` A 10:00:00、B 18:00:00。

## Details
Main write-up: `docs/experiments_unet.md`. It covers the source text with page numbers, the design-decision table,
fairness checks, size and cost, the smoke run, the bench and the projection.

| check | result | evidence |
|---|---|---|
| DiT unchanged | golden loss, grad-norm and parameter checksums after 3 real `tmap.train` steps (pixel; random-conv feature) recorded on 8a15563 before any edit; still pass | `tests/test_dit_unchanged.py` [E] |
| UNet tests | shapes, F ≡ 0 at init, class conditioning after 1 step, gradients reach every parameter, chunked = one-pass gradient (with and without a feature term), checkpoint round trip via `tmap.sample.load_checkpoint_model`, full-size configs at 140.01 M, config fairness | `tests/test_unet.py` [E] |
| full suite | 35 passed | `runs/pytest_unet_S1.log` [E] |
| config diff | only name, comments, model block, `grad_chunk` | `docs/unet_config_diff.txt` [E] |
| same draws | sampler classes, indices, flips, y, z (sha256) and grid noise identical across backbones, seeds 0 and 1 | `runs/determinism_unet_S1*.json` [E] |
| cost | UNet 53.4 MiB/sample saved activations (bf16) vs DiT 110.2 (method reproduces the 111 MiB in experiments_AB) | `runs/cost_unet_S1.json` [E] |
| CPU smoke | 3.98 M-param UNet, 1737 steps, loss 26.70 → 18.66, held-out pixel D1 17.80 → 0.621, finite | `runs/smoke_unet_cpu/` [E], smoke only |
| GPU bench | A c512 1.012 s/step 30.6 GiB; A c256 1.297; B c256 1.906 s/step 23.9 GiB; B c512 OOM; eval path OK (57 s) | `runs/bench_gpu_unet_23951657/`, `runs/slurm-tmap_bench_unet-23951657.out` [E] |
| projection | A 8.97, B 16.64 GPU-h per run; 51.2 for 4 runs | [G], from the bench × 1.03 + 0.28 h eval; the DiT bench predicted its runs within 2% |

## Commits (all pushed to origin main)
99460c8 golden DiT test · fc04510 UNet, dispatch, tests, configs · 27b7a1e cost, grad_chunk, config diff ·
d38c154 determinism check · dc12830 bench script, draft doc · e4af1db bench result, projection, smoke · (this standup).

## Compute
- GPU: 1 bench job, 23951657, 7.9 min ≈ **0.13 GPU-h** (budget 120).
- CPU: in the S1 agent allocations only.

## Deviations and decisions to know
- **grad_chunk.** UNet A uses 512 (= N, one pass); DiT A used 128 (two-pass). The gradient is exact either way, but UNet A
  skips the extra forward, so wall time is not like-for-like. At chunk 256, UNet A runs 1.297 s/step against DiT A's
  1.494 (both from benches). UNet B uses 256.
- **FLOP count.** DiT-B/2's config quotes 23.0 GMACs (the DiT paper's count). `torch.utils.flop_counter` gives 21.79, so the
  UNet's 20.04 is compared with 21.79.
- **Paper-silent choices** (full table in the doc): one adaptive RMSNorm per residual block (literal reading of
  App. I.2); stride-2 conv down and nearest + conv up; 2 blocks per level on both paths with concat skips; attention on
  both paths at 16 px plus the bottleneck; emb 1024; zero-initialised last conv; dropout 0. The UNet config carries
  `patch_size: 2` only for `tmap.evaluate`'s blockiness grid.
- **No third-party U-Net code** was read or cloned, so `third_party/` is unchanged.
- **pypdf** was installed into `/scratch/wg2381/.cache-home/unet_site` (`pip --target`). It was used only to read the arXiv PDF;
  the project never imports it.
- **Smoke run in two segments.** Segment 1 was SIGTERM'd by the agent tool limit at step 897; it checkpointed and
  resumed exactly. Both commands are in `command.txt`.
- **Bench checkpoints deleted.** The bench's 9.4 GB of checkpoints (my own, gitignored) were removed after reading them.
- The S1 agent was interrupted twice by usage limits (requeued as 23928807 and 23952845). No work was lost.

## Open questions for S2/S3
- `--time` for the runs: A 10:00:00, B 18:00:00 suggested. `l40s_public` has no wall limit.
- If UNet B's real speed drifts above about 1.8 GPU-h per 1k steps, recheck the budget. It is far from the 30 GPU-h line.
