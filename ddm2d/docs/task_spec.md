# Task spec, as given by the user on 2026-11-17 (verbatim; binding)

The user pasted this in the orchestrating session and asked for it to be run as a `claude_jobs` agent chain. Later the user added
(2026-11-18): "有一点进展就push一点": push a little every time there is a little progress.

---

# Reproduce Distributional Diffusion Models — 2D Experiments

## Objective

Reproduce the 2D experiments from:

**Distributional Diffusion Models with Scoring Rules** (De Bortoli et al., ICML 2025)

Paper: [https://arxiv.org/pdf/2502.02483](https://arxiv.org/pdf/2502.02483)

Focus exclusively on:

1. **2D Gaussian Mixture** — Section 6.1, Figures 3–4, Appendix I.1.
2. **2D Checkerboard** — Appendix K.1, Table 2.

Implement, train, evaluate, and compare standard diffusion against distributional diffusion models (DDM). Match the paper's architecture, training hyperparameters, noise schedule, sampling algorithm, metrics, and visualization criteria as closely as possible.

**The experiment must run end-to-end and produce real checkpoints, generated samples, evaluation metrics, and plots.** Code implementation alone is insufficient.

Do not start CIFAR-10 or other image experiments.

## 1. Verified sources

### Paper and reference implementation

- Original paper: [https://arxiv.org/pdf/2502.02483](https://arxiv.org/pdf/2502.02483)
- Third-party PyTorch implementation: [https://github.com/edluyuan/ddm](https://github.com/edluyuan/ddm)
- Toy training entry point: [https://github.com/edluyuan/ddm/blob/main/run_example.py](https://github.com/edluyuan/ddm/blob/main/run_example.py)
- Model backbone: [https://github.com/edluyuan/ddm/blob/main/dddm/model.py](https://github.com/edluyuan/ddm/blob/main/dddm/model.py)
- Training implementation: [https://github.com/edluyuan/ddm/blob/main/dddm/training.py](https://github.com/edluyuan/ddm/blob/main/dddm/training.py)
- Scoring-rule loss: [https://github.com/edluyuan/ddm/blob/main/dddm/losses.py](https://github.com/edluyuan/ddm/blob/main/dddm/losses.py)
- Sampling: [https://github.com/edluyuan/ddm/blob/main/dddm/sampling.py](https://github.com/edluyuan/ddm/blob/main/dddm/sampling.py)
- Diffusion schedule: [https://github.com/edluyuan/ddm/blob/main/dddm/schedules.py](https://github.com/edluyuan/ddm/blob/main/dddm/schedules.py)

This is a **third-party implementation**. Its default toy MLP, training budget, and particle count differ from the paper. Review it before adapting it.

The paper's Appendix I.1 takes precedence wherever there is a discrepancy.

### Dataset sources

**Gaussian Mixture**

- Generator: [https://github.com/edluyuan/ddm/blob/main/dddm/data.py](https://github.com/edluyuan/ddm/blob/main/dddm/data.py)

Generate 102,400 2D points from the paper's exact mixture:

- Equal mixture weights: 0.5 / 0.5
- Mean 1: (3, 3)
- Mean 2: (-3, 3)
- Isotropic standard deviation: 0.5

**Checkerboard**

- Reference generator: [https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py](https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py)
- Function: `inf_train_gen("checkerboard", ...)`

The paper references Galashov et al. (2025) for the Checkerboard experiment. Its appendix does not fully specify the generator's scaling, dataset size, evaluation bandwidth, or all Checkerboard-specific hyperparameters.

Inspect the cited source and seek the exact original configuration. If it remains unavailable, use the verified FFJORD generator as a documented approximation.

Generate and cache reproducible datasets with fixed seeds. Preserve an independent evaluation set.

## 2. Define the baseline models

Implement the following three experimental arms.

### A. Standard Diffusion — primary baseline

- Train using the conventional diffusion regression objective, Equation (6).
- Predict the conditional mean of clean data.
- Use the paper's 2D backbone and training configuration.
- Implement the corresponding standard diffusion sampling procedure.
- Use this model as the primary reference for Figure 3 and Checkerboard Table 2.

### B. Diffusion-like DDM — architecture-matched control

- Same stochastic-denoiser architecture as the distributional model.
- Generalized energy loss, Equation (14).
- beta = 2, lambda = 0.
- The stochastic input should have no necessary role in the optimal prediction.
- This is the additional control included in the paper's 2D experiment.

### C. Distributional Diffusion — proposed method

- Train with Equation (14), the conditional generalized energy score.
- beta = 0.1, lambda = 1.
- Number of prediction particles: m = 32.
- Model inputs: noisy sample, diffusion time, and independently sampled latent noise.
- Implement sampling with Algorithm 2 and the Gaussian bridge in Equation (4).

Use the same training data, optimization budget, evaluation points, and reproducibility controls across all three models.

The final comparisons must clearly distinguish these three arms.

## 3. Match the original 2D backbone

Use Appendix I.1 as the architectural specification:

- Nine-layer MLP backbone.
- Hidden dimension = 64.
- GELU activation.
- Sinusoidal timestep embedding of dimension 2048.
- A two-layer timestep embedding MLP with hidden/output dimension 2048.
- Four preprocessing MLP layers for the time and noisy-data branches, followed by four joint MLP layers and an output layer, as described in the appendix.
- For the distributional model, concatenate independent latent Gaussian noise with the noisy input.
- Follow the paper's four-dimensional intermediate output and two-dimensional prediction extraction for the stochastic model.

Inspect the original architectural description carefully and document any ambiguity regarding layer counting or the exact location of concatenation.

Add unit tests for input/output dimensions, independent particle sampling, and compatibility with both training losses.

Do not silently retain the third-party repository's smaller SiLU architecture.

## 4. Training configuration

Match the paper's 2D setup:

| Parameter                            | Paper configuration                 |
| ------------------------------------ | ----------------------------------- |
| Training dataset                     | 102,400 synthetic GMM points        |
| Optimizer steps                      | 100,000                             |
| Batch size                           | 128                                 |
| Optimizer                            | Adam                                |
| Learning rate                        | 1e-3                                |
| Adam beta1 / beta2                   | 0.9 / 0.999                         |
| Adam epsilon                         | 1e-8                                |
| Learning-rate warmup                 | First 100 iterations, cosine warmup |
| Global gradient/update norm clipping | 1                                   |
| EMA decay                            | 0.99                                |
| Forward noise schedule               | alpha(t)=1-t, sigma(t)=t            |
| Safety epsilon                       | 1e-2                                |
| Loss weighting                       | Sigmoid weighting                   |
| DDM population m                     | 32                                  |
| DDM beta / lambda                    | 0.1 / 1                             |
| Diffusion-like beta / lambda         | 2 / 0                               |

The paper specifies these values for Gaussian Mixture. For Checkerboard, reuse this configuration as the initial controlled setup when its experiment-specific values cannot be verified. Explicitly distinguish borrowed settings from confirmed Checkerboard settings.

Use the same time sampling and weighting convention across methods where applicable.

Do not silently substitute the repository defaults.

Log the exact config, resolved command, Git commit, random seed, training steps, GPU, peak memory, and runtime.

## 5. First make the complete pipeline runnable

Before starting full training:

1. Set up the Python/PyTorch environment and confirm the available GPU.
2. Generate both datasets and plot their ground-truth distributions.
3. Instantiate all three model variants.
4. Validate the forward diffusion distribution and Gaussian bridge equations.
5. Validate the generalized energy loss against a small, explicit reference calculation.
6. Run several forward/backward optimization steps and verify finite losses and gradients.
7. Train a small smoke-test model for each dataset and each arm.
8. Save and reload checkpoints, including optimizer, EMA, and random state.
9. Generate samples using the actual sampling algorithms.
10. Compute MMD and save at least one end-to-end evaluation figure.

Only proceed to the 100k-step experiments once these tests pass.

Use separate `smoke` and `paper` configurations. No smoke-test measurements should appear as paper-scale reproduction results.

If execution fails, diagnose and fix the actual problem. Do not report an experiment as completed unless its outputs exist and can be inspected.

## 6. Gaussian Mixture evaluation

Reproduce the paper's Figure 3 and Figure 4.

### Figure 3 — left: MMD² versus sampling steps

- Compare Standard Diffusion, diffusion-like DDM, and Distributional Diffusion.
- Use RBF-kernel squared MMD, Equations (9b) and (10).
- RBF bandwidth sigma = 1.
- Use 4,096 samples, following Appendix I.1.
- Evaluate multiple sampling-step counts, including 5, 10, 20, 50, and 100.
- Plot sampling steps on the x-axis and squared MMD on the y-axis.
- Preserve the paper's distinction between the MMD used for evaluation and the generalized energy score used for training.

Implement a memory-efficient, numerically checked MMD estimator. Explicitly document biased versus unbiased MMD² estimation and match the paper where possible.

### Figure 3 — right: posterior standard deviation

For multiple diffusion times between 0 and 1:

- Sample noisy observations from the forward diffusion.
- For each observation, generate eight independent conditional predictions.
- Compute the standard deviation across those predictions.
- Average over 4,096 noisy observations.
- Plot the result as a function of diffusion time.
- Include the exact analytical Gaussian-mixture posterior as ground truth.
- Compare against the distributional model and the beta=2, lambda=0 control.

Implement the analytical posterior using the mixture formula given in Appendix I.1.

### Figure 4 — conditional posterior visualization

Reproduce the two-row visualization:

- First row: set clean x0 to (3, 3).
- Second row: set clean x0 to (-3, 3).
- Generate a noisy observation x_t from each.
- Use the paper's five displayed noise levels, approximately t = 0.99, 0.88, 0.45, 0.23, and 0.01.
- Draw 4,096 conditional samples from the exact posterior.
- Draw 4,096 conditional predictions from the trained DDM.
- Compare with predictions from the beta=2, lambda=0 control.

Use consistent axes and distinguish the true posterior, DDM predictions, deterministic-control predictions, clean point, and noisy point.

Attempt to reproduce the geometry and layout of Figure 4 closely.

## 7. Checkerboard evaluation

Reproduce Appendix K.1, Table 2.

Use the paper's sampling budgets:

| NFE  | Standard Diffusion | Distributional Diffusion |
| ---- | ------------------ | ------------------------ |
| 5    | 7.00e-3            | 1.57e-4                  |
| 10   | 2.10e-3            | 1.96e-4                  |
| 50   | 2.70e-4            | 1.55e-4                  |
| 100  | 1.96e-4            | 2.00e-4                  |
| 1000 | 1.71e-4            | 1.83e-4                  |

These are the **published reference numbers**, never predetermined outputs of our implementation.

The table labels its metric MMD without fully specifying all numerical estimation details.

Before comparing numeric values, investigate the original metric convention, including whether the reported statistic is MMD or MMD², its kernel, bandwidth, sample size, and estimator.

If exact details remain unavailable:

- Reproduce the qualitative comparison using a documented metric implementation.
- Report the metric definition explicitly.
- Mark direct numeric agreement with Table 2 as unverified.
- Keep the paper's reference values separate from our measured values.

Produce:

- Ground-truth Checkerboard scatter/density plot.
- Generated-sample scatter plots for each method at 5, 10, 50, and 100 sampling steps.
- A metric-versus-NFE curve, including 1000 steps.
- A comparison table with our values, the paper's values, and deviations when numerically comparable.

Use identical evaluation samples/seeds and the same metric implementation for both methods.

## 8. Evaluation integrity

- Fix random seeds and record all deviations from the paper.
- Use EMA weights for evaluation consistently.
- Evaluate both models with the same NFE budget and sample count.
- Do not select a different evaluation protocol for the proposed method.
- Document the stochastic bridge churn parameter separately from the training safety epsilon.
- Make sure the interpretation of an NFE corresponds to the actual number of denoiser evaluations.
- Check numerical stability near t=0 and t=1.
- Save raw metric values as CSV/JSON.
- If feasible, repeat experiments with three seeds and report variability as an additional reproducibility analysis. Keep published single-run values separately identifiable.

Avoid claiming an exact reproduction when dataset conventions, kernel parameters, architecture, or optimization settings remain different.

## 9. Required deliverables

Create a concise, reproducible repository containing:

- Data-generation modules for GMM and Checkerboard.
- Paper-matched 2D MLP backbone.
- Standard diffusion and distributional diffusion training implementations.
- Shared optimization/evaluation infrastructure.
- Configuration files for smoke tests and paper-scale runs.
- Tests for diffusion schedule, loss, sampler, dataset generation, and metric correctness.
- Completed checkpoints for all three GMM arms and both Checkerboard arms.
- Training-loss curves.
- Generated-sample figures.
- Reproductions of Figure 3, Figure 4, and Table 2.
- Raw evaluation metrics and a reproducibility report.
- README with the exact commands needed to regenerate the results.

Include a `paper_vs_implementation.md` audit showing each important paper setting, our implemented value, and whether it is verified, matched, approximated, or unknown.

## 10. Execution sequence and completion criteria

**Phase 1 — Audit:** Inspect references, identify mismatches, and write the implementation/configuration plan.

**Phase 2 — Baselines:** Implement and successfully run Standard Diffusion and the beta=2, lambda=0 matched-architecture control.

**Phase 3 — DDM:** Implement and successfully run beta=0.1, lambda=1 distributional diffusion with m=32.

**Phase 4 — GMM reproduction:** Train the complete three-arm comparison and regenerate Figure 3 and Figure 4.

**Phase 5 — Checkerboard reproduction:** Train the relevant models and regenerate the Table 2 comparison and additional sample plots.

**Phase 6 — Validation:** Compare our measured metrics with the published results and investigate discrepancies.

Completion requires actual executed training, readable saved checkpoints, successfully generated samples, computed metrics, and figures corresponding to the paper's evaluation criteria.

Proceed through the phases autonomously, using the available compute resources. If access to a GPU or another necessary resource is blocked, report the exact blocker and the completed verified work. Never fabricate training runs, metrics, or reproduced figures.

**Primary success criterion:** Determine whether the distributional method reproduces the paper's substantial low-NFE improvement while maintaining an explicitly matched baseline and evaluation protocol.
