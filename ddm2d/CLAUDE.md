# ddm2d: reproduce the 2D experiments of Distributional Diffusion Models (De Bortoli et al., ICML 2025)

This directory, on branch `ddm2d-repro`, is a separate sub-project of transport_map. It reproduces the paper's 2D GMM (§6.1, Fig. 3–4, App. I.1)
and checkerboard (App. K.1, Table 2) experiments: standard diffusion vs. distributional diffusion (DDM) trained with the generalized energy score.

- Spec: `docs/task_spec.md` (the user's words, binding).
- It is run by the agent chain in `/scratch/wg2381/claude_jobs/ddm2d/` (rules: `prompts/_rules.md` there).
- **Precedence:** for work inside `ddm2d/`, those rules replace the "Stages" and "Rules for agents" sections of the transport_map
  `CLAUDE.md` one level up. In particular, this chain may submit its own `cpu_short` training jobs within the budget in its rules, and its
  standups live in `ddm2d/standups/` and `/scratch/wg2381/claude_jobs/logs/ddm2d_*_done.md`, not in `claude_jobs/standups/`.
- Nothing outside `ddm2d/` is changed on this branch except the one-line pointer in the root `CLAUDE.md`.
