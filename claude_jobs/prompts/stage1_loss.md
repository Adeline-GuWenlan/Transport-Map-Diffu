# Stage 1 of 4: derive and verify the energy-score loss

Goal: a self-contained LaTeX note, `docs/energy_score_loss.tex`, compiled to `docs/energy_score_loss.pdf`.
It derives the training loss for a one-step transport map from the strict definition of the energy score.
It also settles whether the loss the user typed (CLAUDE.md, "Loss as proposed") is correct. Where it is not,
say exactly what is wrong and give the corrected form. Back every claim with a proof, a verified citation,
or a numerical check you ran. The user expects their formula may be wrong, so check it; don't defend it.

## The note, in this order
1. **Setup.** Data distribution Q on R^d (images as vectors), labels c where used. Source μ = N(0, I).
   Map F_θ, model P_θ = (F_θ)_# μ.
2. **Scoring rules from the start.** Definition, orientation, expected score S(P, Q), proper, strictly proper.
   Gneiting & Raftery state scores positively oriented (larger is better). We minimize, so fix one sign
   convention and keep it throughout.
3. **The energy score, as strictly defined** in Gneiting & Raftery (2007, JASA 102(477), 359–378):
   ES_β(P, y) with β ∈ (0, 2) and its moment condition. Prove strict propriety for β ∈ (0, 2) through the
   energy distance (Székely & Rizzo), with the key steps written out (e.g. the Fourier-integral
   representation of ‖x‖^β), not only cited. Show by counterexample that β = 2 is proper but not strictly proper.
4. **Expected score minus its minimum = ½ × energy distance.** So minimizing the expected energy score
   over θ minimizes the energy distance to the data, up to a constant that does not depend on θ.
5. **The loss for the pushforward model**, with expectations over z, z' ~ μ and y ~ Q, and the
   class-conditional version F_θ(z, c) with the expectation over c. Prove that the conditional version is
   still strictly proper.
6. **Line-by-line check of the user's formula.** Settle each point explicitly:
   - nested expectation versus joint expectation;
   - the second term: is it a norm (to what power) or a raw difference? What is E[F_θ(x) − F_θ(x')]?
   - which distribution x and x' are drawn from (the data, or the source Gaussian), and whether they are
     independent of y and of each other;
   - "Euclidean distance": norm or squared norm, i.e. β = 1 or β = 2;
   - the ½ factor and the sign;
   - what the minimizer becomes when a term is dropped or miswritten.
   End the section with the final loss in a box and a one-line verdict on the original.
7. **Minibatch estimator.** The unbiased U-statistic for n data points and m generated samples (per class
   if conditional). Variance as m changes. The gradient of ‖·‖ at 0 when β = 1 and how to handle it.
   Compute cost.
8. **Related work, brief and accurate, with citations you verified.** Energy distance as an MMD with the
   distance kernel (Sejdinovic et al. 2013). Engression (Shen & Meinshausen). Scoring-rule / distributional
   diffusion models, where the energy score is conditioned on (x_t, t). Say how our one-step test relates,
   and what the natural next step toward diffusion is.
9. **Numerical checks.** Small and CPU-only, under 30 min in total. Include the figures in the PDF.
   a. The energy-distance estimator has expectation ≥ 0, is ≈ 0 for P = Q, and matches the closed form of
      E|X − Y| for 1-D Gaussians.
   b. β = 2 cannot separate two distributions with the same mean (e.g. N(0,1) vs N(0,4)); β = 1 can.
   c. Train a small MLP map R^2 → R^2 from N(0, I) to a 2-D mixture (e.g. 8 Gaussians on a circle), with
      (i) the corrected loss and (ii) the user's formula read literally, plus any other plausible reading
      of it. Show the samples, and report the energy distance to the target for each.
   Code goes in `docs/stage1_checks/`, figures in `docs/figures/`.

## Tools
There is no project env yet (stage 3 builds it), and there is no system LaTeX. Make a small helper env with
conda (conda-forge) at `/scratch/wg2381/.conda/envs/tmap-tex` holding `tectonic`, numpy, scipy and
matplotlib, plus CPU-only PyTorch for check (c). Compile with tectonic and fix every error and warning
that matters. Then check that the PDF rendered: read it back, including the equations and figures.
Check definitions and citations against the primary sources with web search or fetch. Do not cite from memory.

## Standup: `claude_jobs/standups/stage1.md`
- The verdict on the user's formula, in one plain paragraph.
- The final loss, as one line of LaTeX.
- The estimator stage 4 must implement: β, the n × m batch structure, the conditional grouping.
- Results of the numerical checks.
- File paths.
- Open questions.
