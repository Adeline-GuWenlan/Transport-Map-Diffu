"""Turn results/*.json into LaTeX macros and tables used by docs/energy_score_loss.tex."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
J = {k: json.load(open(os.path.join(RES, f"check_{k}.json"))) for k in "abcdef"
     if os.path.exists(os.path.join(RES, f"check_{k}.json"))}


def g(x, p=3):
    """Format a number for LaTeX with p significant digits; scientific notation when small/large."""
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        m, e = f"{x:.{p - 1}e}".split("e")
        return rf"{m}\times10^{{{int(e)}}}"
    return f"{x:.{p}g}"


mac = {}
a, b = J["a"], J["b"]
a1, a1s = a["a1_P_eq_Q"], a["a1_P_shift_0.5"]
mac.update(AoneReps=str(a1["reps"]), AoneEqUmean=g(a1["U_mean"], 2), AoneEqUse=g(a1["U_se"], 2),
           AoneEqFracNeg=f"{100 * a1['U_frac_negative']:.0f}", AoneEqVmean=g(a1["V_mean"]),
           AoneShUmean=f"{a1s['U_mean']:.4f}", AoneShUse=f"{a1s['U_se']:.4f}",
           AoneShRef=f"{a1s.get('closed_form', a1s.get('large_sample_ref')):.4f}")
mac["AtwoQuadErr"] = g(max(abs(q["closed"] - q["quad"]) for q in a["a2_folded_normal_vs_quad"]), 1)
cf = a["a3_charfn_form"]
mac.update(AthreeCF=f"{cf['via_charfn']:.10f}", AthreeClosed=f"{cf['closed_form']:.10f}")
a4 = a["a4_variance_vs_m"]
ratios = [pp / po for pp, po in zip(a4["perpoint_loss_var"], a4["pooled_loss_var"]) if pp is not None]
mac["AfourRatioLoss"] = f"{max(ratios):.0f}"
b1 = b["b1"]
mac.update(BoneDoneClosed=f"{b1['D1_closed']:.4f}", BoneDoneMean=f"{b1['D1_mean']:.4f}",
           BoneDoneSe=f"{b1['D1_se']:.4f}", BoneDoneFracPos=f"{100 * b1['D1_frac_positive']:.0f}",
           BoneDtwoMean=g(b1["D2_mean"], 2), BoneDtwoSe=g(b1["D2_se"], 2),
           BoneShiftMean=f"{b['b1_mean_shift_beta2']['D2_mean']:.3f}")
b3 = [r for r in b["b3_weighted_interaction"] if "s_numeric" in r]
mac["BthreeMaxErr"] = g(max(abs(r["s_numeric"] - r["s_formula"]) for r in b3), 1)
lam2 = [r for r in b["b3_weighted_interaction"] if r["lam"] == 2.0][0]
mac["BthreeLamTwo"] = g(lam2["score_at_s"][-1])
if "e" in J:
    mac["EcdistDiag"] = f"{J['e']['cdist_mm_path_65x3072']['max_abs_diag']:.2f}"
    mac["TorchVersion"] = J["e"]["torch_version"]
import numpy, scipy  # noqa: E401
mac["NumpyVersion"], mac["ScipyVersion"] = numpy.__version__, scipy.__version__
# total wall time of the final runs, from the `time` lines in logs/check_*.log
import re
tot = 0.0
for fn in sorted(os.listdir(os.path.join(HERE, "logs"))):
    if fn.startswith("check_") and fn.endswith(".log"):
        m = re.findall(r"^real\s+(\d+)m([\d.]+)s", open(os.path.join(HERE, "logs", fn)).read(), re.M)
        if m:
            tot += 60 * int(m[-1][0]) + float(m[-1][1])
mac["TotalRuntime"] = f"{tot / 60:.0f}"
if "c" in J:
    base = J["c"]["baselines"]
    mac.update(CbaseTT=g(base["target_vs_target"], 2), CbaseGauss=g(base["gaussian_source_vs_target"]),
               CbasePoint=g(base["point_mass_origin_vs_target"]))

if "c" in J:
    R = J["c"]["results"]
    tag = {"es_beta1": "CesOne", "es_beta1.5": "CesOneHalf", "literal_rawdiff": "CLiteral",
           "data_pairs": "CDataPairs", "beta2_squared": "CBetaTwo", "coef1_no_half": "CCoefOne",
           "mixed_norm_sq": "CMixed"}
    for k, t in tag.items():
        sm = R[k]["summary"]
        mac[t + "Done"] = g(sm["D1_mean"])
        seeds = [v for kk, v in R[k].items() if kk != "summary"]
        mac[t + "MaxStd"] = g(max(max(v["sample_std"]) for v in seeds), 2)
        mac[t + "MinStd"] = g(min(min(v["sample_std"]) for v in seeds), 2)
        mac[t + "LossMin"] = g(min(v["loss_min"] for v in seeds), 2)
    mac["CTargetStd"] = g(np.mean(J["c"]["target_std"]), 3)
if "d" in J:
    d = J["d"]
    stds = lambda mode: [d[mode][sd][f"class{c}"]["std"] for sd in d[mode] for c in (0, 1)]
    d1s = lambda mode: [d[mode][sd][f"class{c}"]["D1"] for sd in d[mode] for c in (0, 1)]
    mac.update(DgrStdMin=f"{min(stds('grouped')):.2f}", DgrStdMax=f"{max(stds('grouped')):.2f}",
               DpoStdMin=f"{min(stds('pooled')):.2f}", DpoStdMax=f"{max(stds('pooled')):.2f}",
               DgrDoneMax=g(max(d1s('grouped')), 2), DpoDoneMin=g(min(d1s('pooled')), 2),
               DpoDoneMax=g(max(d1s('pooled')), 2), DsStar=f"{d['gaussian_family_pooled_minimiser_s']:.3f}",
               DderivFormula=f"{d['pooled_deriv_at_s1_formula']:.5f}", DderivNumeric=f"{d['pooled_deriv_at_s1_numeric']:.5f}",
               DlamHalf=f"{d['lambda_half_limit']:.3f}",
               DpoMeanMin=f"{min(abs(d['pooled'][sd][f'class{c}']['mean']) for sd in d['pooled'] for c in (0, 1)):.2f}",
               DpoMeanMax=f"{max(abs(d['pooled'][sd][f'class{c}']['mean']) for sd in d['pooled'] for c in (0, 1)):.2f}")
if os.path.exists(os.path.join(RES, "check_d2.json")):
    d2 = json.load(open(os.path.join(RES, "check_d2.json")))
    for mode, t in (("grouped", "Gr"), ("pooled", "Po")):
        v = [d2[mode][sd] for sd in d2[mode]]
        mac[f"DtwoD{t}Done"] = ", ".join(f"{x[f'D1_class{c}']:.3f}" for x in v for c in (0, 1))
        mac[f"DtwoD{t}Std"] = ", ".join(f"{x[f'within_mode_std_class{c}']:.3f}" for x in v for c in (0, 1))
    mac["DtwoDTargetStd"] = f"{d2['target_within_mode_std']:.3f}"
if "f" in J:
    f_ = J["f"]
    mac.update(FsafeDone=g(f_["safe_dist"]["final_D1"]), FsafeNaN=str(f_["safe_dist"]["steps_with_nan_grad"]),
               FnaiveFirstNaN="yes" if f_["naive_sqrt"]["first_step_grad_has_nan"] else "no",
               FnaiveSteps=str(f_["naive_sqrt"]["steps_with_nan_grad"]),
               FnaiveDone=("NaN" if not np.isfinite(f_["naive_sqrt"]["final_D1"]) else g(f_["naive_sqrt"]["final_D1"])))

if os.path.exists(os.path.join(RES, "check_c2.json")):
    c2 = json.load(open(os.path.join(RES, "check_c2.json")))
    mac.update(CtwoShortDone=g(c2["3000"]["D1"]), CtwoLongDone=g(c2["20000"]["D1"], 2),
               CtwoShortNear=f"{100 * c2['3000']['frac_near_mode']:.0f}",
               CtwoLongNear=f"{100 * c2['20000']['frac_near_mode']:.0f}",
               CtwoTargetNear=f"{100 * c2['target_frac_near_mode']:.0f}")

with open(os.path.join(RES, "results_macros.tex"), "w") as f:
    f.write("% generated by make_tex_results.py from results/*.json -- do not edit\n")
    for k, v in mac.items():
        f.write(rf"\newcommand{{\res{k}}}{{{v}}}" + "\n")

# ---- table a2
rows = [r"\begin{tabular}{llrrr}", r"\toprule", r"$P$ & $Q$ & closed-form $D_1$ & estimator mean $\pm$ s.e. & $z$\\",
        r"\midrule"]
for r in a["a2_gauss1d"]:
    P = r["P"].replace("N(", r"$\N(").replace("^2)", "^2)$")
    Q = r["Q"].replace("N(", r"$\N(").replace("^2)", "^2)$")
    rows.append(rf"{P} & {Q} & {r['closed_form']:.4f} & {r['est_mean']:.4f} $\pm$ {r['est_se']:.4f} & {r['z']:+.2f}\\")
rows += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(RES, "table_a2.tex"), "w").write("\n".join(rows) + "\n")

# ---- table a3
rows = [r"\begin{tabular}{rrrrr}", r"\toprule", r"$d$ & $\beta$ & $C(d,\beta)$, eq.~\eqref{eq:fourier} & quadrature & rel.\ error\\",
        r"\midrule"]
for r in a["a3_fourier_constant"]:
    rows.append(rf"{r['d']} & {r['beta']} & {r['formula']:.8f} & {r['quadrature']:.8f} & ${g(r['rel_err'], 1)}$\\")
rows += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(RES, "table_a3.tex"), "w").write("\n".join(rows) + "\n")

# ---- table c
if "c" in J:
    names = {"es_beta1": r"corrected loss, $\beta=1$", "es_beta1.5": r"corrected loss, $\beta=1.5$",
             "literal_rawdiff": r"literal: $-\tfrac12\,\texttt{(F(x)-F(x')).mean()}$",
             "data_pairs": r"$x,x'$ from the data: $-\tfrac12\E\norm{y-y'}$",
             "beta2_squared": r"squared norms in both terms ($\beta=2$)",
             "coef1_no_half": r"coefficient $1$ instead of $\tfrac12$",
             "mixed_norm_sq": r"$\norm{\cdot}$, then $-\tfrac12\norm{\cdot}^2$"}
    rows = [r"\begin{tabular}{lrrrr}", r"\toprule",
            r"loss & held-out $\widehat D_1$ & sample std (x, y) & mean radius & outside $[-4,4]^2$\\",
            r"\midrule"]
    R = J["c"]["results"]
    for k, lab in names.items():
        s = R[k]["summary"]
        s0 = R[k]["0"] if "0" in R[k] else R[k][0]
        d1 = "diverged (NaN)" if not np.isfinite(s["D1_mean"]) else rf"${g(s['D1_mean'])}\pm{g(s['D1_sd'], 2)}$"
        rows.append(rf"{lab} & {d1} & {s0['sample_std'][0]:.3g}, {s0['sample_std'][1]:.3g} & "
                    rf"{s0['mean_radius']:.3g} & {100 * s0['frac_outside_4']:.0f}\%\\")
    rows += [r"\midrule", rf"target (8 Gaussians) & ${g(J['c']['baselines']['target_vs_target'], 2)}$ & "
             rf"{J['c']['target_std'][0]:.3g}, {J['c']['target_std'][1]:.3g} & 2.00 & 0\%\\", r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(RES, "table_c.tex"), "w").write("\n".join(rows) + "\n")
print(open(os.path.join(RES, "results_macros.tex")).read())
