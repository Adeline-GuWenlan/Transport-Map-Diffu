# Stage 2 of 4: choose the backbone, optimizer and dataset

Goal: decide, with evidence, what stage 3 installs and stage 4 trains: the DiT backbone (code base and
size), the optimizer and its settings, and the dataset. Write `docs/stage2_choices.md`. Also write two
configs stage 4 can use as they are: `configs/smoke_cpu.yaml` and `configs/base_gpu.yaml`.

Read first: CLAUDE.md, `claude_jobs/standups/stage1.md`, `docs/energy_score_loss.tex`. The loss and its
estimator are settled there; design around them. For example, the estimator needs m ≥ 2 generated samples
per data point or per class in every batch.

## Backbone
- Shallow-clone the reference DiT code (github.com/facebookresearch/DiT) into `third_party/` and record
  the commit. Read it: `models.py`, the size table (S/B/L/XL × patch 2/4/8), conditioning (timestep and
  class embeddings, adaLN-Zero), `learn_sigma`, and the latent setup (SD VAE, 256/8 = 32 latent).
- Look at the alternatives and say whether any suits us better: e.g. SiT, fast-DiT, U-ViT, LightningDiT.
  Also look for public code that trains DiT-style one-step generators with energy distance, scoring rules
  or MMD. Check every license (DiT's is CC-BY-NC).
- Decide how DiT becomes a transport map F_θ(z, c):
  - the input z, the same shape as an image (pixel or latent);
  - what happens to the timestep input (fix it, drop it, or keep it as a noise-level conditioner for later);
  - the output head (no learn_sigma);
  - whether extra noise channels are needed.
  Also choose pixel space or VAE latent space for the first test, and say why.
- Pick two sizes: a CPU smoke model (one training step in well under a second at batch 64 on 8 cores)
  and a GPU model for the real test. Give parameter counts and FLOPs that you measured or computed.

## Optimizer
- What DiT used. Verify it in the paper and code (AdamW, lr 1e-4, wd 0, batch 256, EMA 0.9999?).
- What papers that train with energy distance, scoring rules or MMD used.
- Choose: the optimizer, lr and schedule, betas, weight decay, gradient clipping, EMA, batch size, and the
  n × m batch structure.
- Note anything specific to the energy score: gradient variance as m changes, and the scale of the loss
  in pixel space versus latent space.

## Dataset
- Requirements: open, well labelled, licensed for research, and downloadable here without an account.
- Compare at least MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, and a downsampled ImageNet (32 or 64) or
  similar. For each, give: size, resolution, classes, license, download source (torchvision, Hugging Face
  id or URL), and whether standard FID reference stats exist.
- Recommend one dataset for the CPU smoke test and one for the real test, ideally from the same family so
  results carry over.
- Evaluation plan: FID (which implementation, which reference stats), held-out energy distance,
  per-class sample grids, and a nearest-neighbour check against the train set for memorization.

## Rules
Investigate only: do not build the env (stage 3) or the training code (stage 4). Small CPU timing
experiments are fine; use the helper env from stage 1 or a throwaway env. Cite a URL for every external fact.

## Standup: `claude_jobs/standups/stage2.md`
- A decision table: choice, alternatives considered, why.
- The exact package list stage 3 must install, with versions where they matter.
- Repos to clone, with commit hashes.
- Dataset ids and download commands.
- Open questions.
