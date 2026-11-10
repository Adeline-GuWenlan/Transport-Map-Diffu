# Stage 4 of 4: make the pipeline run end to end

Goal: a pipeline that trains a DiT transport map F_θ: N(0, I) → data with the stage 1 energy-score loss.
It must run end to end on this CPU node and produce evidence that the loss works, or clear evidence that it
does not. Keep going until it actually runs; fix what breaks.

Read first: CLAUDE.md, all of `claude_jobs/standups/`, `docs/energy_score_loss.tex` (the loss and estimator
are fixed there), `docs/stage2_choices.md`, `configs/`, `env/activate.sh`.

## Build: `src/tmap/`, installable with `pip install -e .`
- `energy_score.py`: the loss exactly as stage 1 boxed it, its unbiased minibatch estimator, the β
  parameter, and the class-conditional grouping.
- `model.py`: DiT as a transport map, following stage 2.
- `data.py`.
- `train.py`: YAML config, seed, checkpoint and resume, and loss / grad-norm / throughput logged to CSV or JSONL.
- `sample.py`: sample grids per class.
- `evaluate.py`: held-out energy distance. Add FID if stages 2–3 set it up and it fits on CPU at small
  sample counts; label CPU FID as indicative only.
- `tests/` (pytest):
  - the estimator against the closed-form 1-D Gaussian energy distance;
  - the estimator is unbiased;
  - shape checks;
  - gradients reach every parameter;
  - the 2-D toy from stage 1 trains to a low energy distance in under a minute.

## Run
1. `pytest` passes.
2. Overfit check on a tiny subset (e.g. one class, 256 images): the loss should drop clearly and the
   samples should approach the data.
3. Smoke run with `configs/smoke_cpu.yaml` on stage 2's dataset, for as long as the budget allows (aim for
   at least 60 min of training), checkpointing as you go. Put the outputs in `runs/<date>_<name>/`: the
   config, metrics, a loss-curve PNG, sample grids at several steps, and eval numbers against baselines
   (e.g. Gaussian noise as a ceiling and the train set itself as a floor).
4. Look at the figures yourself. Report honestly: does training lower the held-out energy distance, do the
   samples look like their classes, and is there collapse, blur or mode dropping?
5. Write `scripts/train_gpu.slurm` for the real run with `configs/base_gpu.yaml`: account
   `torch_pr_527_general`, a GPU partition such as `l40s_public`, and safe to requeue through resume.
   Do not submit it.
6. Write `README.md` at the project root: what this is, and how to set up, test, train, sample and
   evaluate, with the exact commands you ran.

## Standup: `claude_jobs/standups/stage4.md`
- Does it run? Yes or no, plus the command.
- Test results.
- Smoke-run numbers and figure paths.
- What the results say about the loss.
- Known issues.
- A suggested next experiment on GPU.
