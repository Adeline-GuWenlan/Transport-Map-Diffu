# transport_map

One-step generative transport maps trained with the energy score, on a DiT backbone.

## Idea
Learn a map F_θ that pushes Gaussian noise z ~ N(0, I) to the data distribution in one forward pass
(class-conditional: F_θ(z, c)). Train it with a strictly proper scoring rule, the energy score
(equivalently, the energy distance), instead of a denoising loss. First goal: test whether this loss
works at all on a small, well-labelled, open image dataset. The diffusion connection (per-noise-level
conditional energy scores) comes after that test.

## Loss as proposed by the user (UNVERIFIED)
> Loss(F_θ) = E_y[ E_x ‖F_θ(x) − y‖ ] − ½ · E_{x,x'}[ F_θ(x) − F_θ(x') ],
> y from the original dataset, x starts from a Gaussian, and "x and x' both from the natural distribution".

The user expects this may be wrong. Stage 1 derives the loss from the strict definition of the energy
score (Gneiting & Raftery 2007) in `docs/energy_score_loss.tex`. Once that exists, its boxed loss is the
spec, and code implements it, not the line above.

## Stages
Unattended Claude agents on CPU nodes, scheduled 5 h apart from `claude_jobs/`.

| # | job | main outputs |
|---|-----|--------------|
| 1 | derive + verify the loss from the energy score | `docs/energy_score_loss.{tex,pdf}`, `docs/stage1_checks/` |
| 2 | choose backbone, optimizer, dataset | `docs/stage2_choices.md`, `configs/` |
| 3 | environment, cloned code, data | `env/`, `third_party/`, `data/`, `scripts/check_env.py` |
| 4 | implement, test, run until it works | `src/tmap/`, `tests/`, `runs/`, `README.md` |

## Rules for agents
- First read every file in `claude_jobs/standups/`. If an earlier stage is INCOMPLETE, finish the part
  of it your stage depends on, and say so in your standup.
- Your Slurm job is your only compute: a `cpu_short` node, 8 CPUs, 16 GB, no GPU. Do not sbatch or srun
  other jobs. Put timeouts on long commands, and keep to the deadline in your prompt header.
- Python env: `/scratch/wg2381/.conda/envs/tmap` (stage 3 builds it). Activate with
  `eval "$(conda shell.bash hook)" && conda activate /scratch/wg2381/.conda/envs/tmap`.
  Conda uses conda-forge only (`~/.condarc`). HOME has a 30k-file quota: never install there.
  `~/.cache` already points into /scratch.
- Code you clone goes in `third_party/` unmodified, with repo, commit and license recorded.
- Check facts against primary sources (paper, code) and cite URLs. Label assumptions as assumptions.
- Before your deadline, even if unfinished, write `claude_jobs/standups/stage<N>.md`.
  Its first line is `Status: DONE` or `Status: INCOMPLETE`.
- Do not edit the pipeline machinery (`claude_jobs/*.slurm`, `*.sh`, `prompts/`) or other stages'
  standups. Do not write outside this project, apart from the env and caches.
