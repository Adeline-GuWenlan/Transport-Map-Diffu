# third_party/

Upstream code, unmodified. Do not edit files here; wrap them or copy into `src/` and say so.

| Dir | Repo | Commit | License | Cloned by | What we use |
|---|---|---|---|---|---|
| `DiT/` | https://github.com/facebookresearch/DiT | `ed81ce2229091fd4ecc9a223645f95cf379d582b` (2023-06-02) | CC-BY-NC 4.0 (`DiT/LICENSE.txt`) | stage 2, `git clone --depth 1` on 2026-11-06 | `models.py`: class `DiT`, `FinalLayer`, size table. Imported via `sys.path`; needs `timm` (`PatchEmbed`, `Attention`, `Mlp`). Not used: `diffusion/`, `train.py`, `sample*.py` (read for the recipe only) |

Verify: `git -C third_party/DiT rev-parse HEAD` and `git -C third_party/DiT status --porcelain` (must be empty).
`scripts/check_env.py` checks both, before and after importing the model.

## Stage 3 notes (2026-11-06)
- Stage 3 cloned nothing else. Stage 2 chose only DiT (docs/stage2_choices.md). The clone is shallow
  (`--depth 1`), and its `origin` is the SSH form `git@github.com:facebookresearch/DiT.git`. A fresh copy needs no
  key: `git clone https://github.com/facebookresearch/DiT && git -C DiT checkout ed81ce2`.
- **Wrapped, not copied.** Code puts `third_party/DiT` on `sys.path` (env var `TMAP_DIT_DIR`, set by
  `env/activate.sh`) and runs `from models import DiT`. DiT is not put on `PYTHONPATH`, because its module name
  `models` is generic. `env/activate.sh` sets `PYTHONDONTWRITEBYTECODE=1`, so the import leaves no
  `__pycache__/` in the clone.
- How we build it: `DiT(input_size=32, patch_size=p, in_channels=C, hidden_size, depth, num_heads, mlp_ratio,
  class_dropout_prob=0.0, num_classes=10, learn_sigma=False)`, called as `model(z, t, y)` with a constant
  `t = 999` (`configs/*.yaml`; `scripts/check_env.py::build_model`).
- Facts about this commit that the check confirmed, and stage 4 relies on:
  - `pos_embed` is a frozen `nn.Parameter(..., requires_grad=False)` (`models.py:174`). Its `.grad` is
    `None`. Leave it out of grad checks, and out of the optimizer if you like (AdamW skips it either way).
  - adaLN-Zero init zeroes every block's modulation and the whole final layer (`models.py:209–216`). So
    F(z, c) ≡ 0 at step 0, and at step 0 only `final_layer.linear.{weight,bias}` get non-zero gradient. All
    trainable tensors get gradient after 2–3 AdamW steps (measured).
- clean-fid 0.1.35 (MIT, https://github.com/GaParmar/clean-fid) is installed from PyPI (`--no-deps`), not
  cloned. See data/README.md for its stats and Inception files.
