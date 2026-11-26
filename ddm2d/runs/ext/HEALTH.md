# Health of the 2D training runs (ddm2dx X1)

Check at 2026-11-26 13:42–13:45 EST, 20 min after the first jobs started (`python scripts/ext_health.py --md runs/ext/HEALTH_table.md`;
table in `HEALTH_table.md`). All 11 jobs (24751248–24751261) were RUNNING. tr01/tr02 started 13:21, tr03–tr11 at 13:30, once the
per-user CPU cap allowed. Nodes: cs647, cs672, cs662, cs686, cs693 (tr05, tr06, tr07), cs703 (tr08, tr09), cs719 (tr10, tr11).

| check | result |
|---|---|
| every run of every job started and logs | yes, 64 / 64 |
| finite losses and gradient norms in every logged window | yes, 64 / 64; no `FAILED.txt`, no traceback in `runs/ext/*.out` |
| speed | 0.118–0.206 s/step. Expected (smoke): 0.11–0.13 alone, 1.0–1.6× in packed jobs |
| fits in one 5-h segment (deadline 4 h 48 min)? | runs at ≤ 0.17 s/step: yes (3.1–4.4 h). The 18 runs of tr05–tr07 share node cs693 at 0.19–0.21 s/step (5.1–5.5 h). They will checkpoint at the deadline and finish in an automatic continuation segment |

## Things X3 should look at
- **β = 3 (λ = 0), seed 0: isolated loss / gradient spikes.** The 100-step window at step 6800 had max gnorm 4.7e3, and the window
  at 8100 had max gnorm 1.1e5 and mean loss 7.4. The next window was back to 0.48–0.52. Clipping at 1 bounds each update. ‖d‖³
  weighs outliers very heavily, so this is plausibly the loss itself, not a bug. It was not changed: changing it for one run would break
  the matched protocol. Seed 1 had no spike by step 8800. The ddm2d report already notes isolated spikes for the β = 2 arms
  (`docs/reproducibility_report.md` §3, loss curves). Smaller isolated spikes so far: `gmm_C_b1.99_s0` (83), `gmm_B_b1.99_s1` (110),
  `gmm_C_pH0.1_s0` (206), `gmm2w_A_s1` (60).
- **RBF runs have small gradient norms** (mean 0.02–0.05, vs 0.2–3 for the other ρ). With σ² = M² (γ = 1) the kernel is wide compared
  with the posterior scale, so its gradients are small. Adam is invariant to the scale of the gradient, so this does not by itself
  slow training. Whether the RBF runs learn the posterior spread is for the evaluation to show.
- Kernel losses (IMQ, RBF, Laplace) are negative by construction (ρ = −k). Compare them only through the evaluation metrics, never
  through their loss values.
