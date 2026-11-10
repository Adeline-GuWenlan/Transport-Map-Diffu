# Stage 3 of 4: environment, code and data

Goal: everything stage 4 needs is installed, downloaded and checked. That means one conda env, the cloned
code at pinned commits, the dataset(s) on disk, and a check script that passes.

Read first: CLAUDE.md, `claude_jobs/standups/stage1.md` and `stage2.md`, and `docs/stage2_choices.md`.

## Do
1. **Env** at `/scratch/wg2381/.conda/envs/tmap`: conda-forge base, the Python version stage 2 named (else
   3.11), and pip for PyTorch. Use the CUDA build of PyTorch so the same env can run the later GPU job; it
   must also run on these CPU nodes. Write:
   - `env/setup_env.sh`, which rebuilds the env from scratch;
   - `env/requirements.txt`, pinned from `pip freeze` once everything works;
   - `env/activate.sh`, which activates the env and points the HF, torch and pip caches at /scratch.
2. **Code.** Clone the repos stage 2 chose into `third_party/` at the recorded commits. Write
   `third_party/README.md` with each repo, commit, license, and what we use from it. Do not modify upstream
   files: wrap them, or copy into `src/`, and note which.
3. **Data** under `data/`, raw plus any preprocessed form, using stage 2's download commands. In
   `data/README.md` record the source, size, sha256 or file counts, class balance, and train/test split.
   If FID reference stats need downloading, or computing within CPU time, do it and record where they are.
4. **`scripts/check_env.py`.** It must:
   - print versions (python, torch and its CUDA build, timm, …);
   - import the DiT model class from `third_party/`;
   - build the CPU smoke model from `configs/smoke_cpu.yaml`;
   - run one forward and backward pass on a real data batch, and time it;
   - exit non-zero on any failure.
   Run it on this node and paste its output into the standup.

## Rules
Write no training code beyond what `check_env.py` needs; that is stage 4. If stage 2's choice turns out
uninstallable or the dataset unavailable, pick the closest alternative and record why in the standup.

## Standup: `claude_jobs/standups/stage3.md`
- How to activate the env.
- What is installed (key versions).
- Clone commits.
- Data paths and checks.
- The `check_env.py` output.
- Anything stage 4 must know.
