# Supplementary Materials

**Manuscript**: A Linearly Implicit BDF2–Taylor Finite Element Method for
Nonlocal Power-Law Curvature Flow: Conditional Convergence and Computational
Validation  
**Authors**: Chengde Huoye  
**Date**: 26 October 2026

This document, the reviewer-facing PDF `A1_Supplementary_Materials.pdf`, and
the accompanying `MCA_Supplementary_Materials.zip`
contain the supplementary material that accompanies the main paper: the exact
finite-\(N\) reduction, the full E0–E9 tables, the solver and experiment
scripts, the raw JSON/CSV data, and reproduction commands. The four original
E1/E2/E3/E5 figures and the E1/E2 summary tables are in the main text. All
paths in this guide are relative to the archive root
`MCA_Supplementary_Materials/`.

---

## S1. Exact finite-\(N\) regular-polygon reduction

On a regular \(N\)-gon, let \(c_N=\cos(\pi/N)\). The discrete curvature is
\(\kappa=1/(Rc_N)\), and the frozen curvature equation gives
\(\kappa^{m+1}=R^{m+1}/(\widetilde R^2c_N)\). The exact recurrence used in
E0 and E3 is
\[
R^{m+1}
=\frac{(4R^m-R^{m-1})c_N/(2\tau)-(\nu-1)s_*^{\nu}}
{3c_N/(2\tau)+(1-\nu s_*^{\nu-1})/(\widetilde R^2c_N)},
\qquad
s_*=\frac{2}{R^mc_N}-\frac{1}{R^{m-1}c_N}.
\]
The semi-discrete limit is the modified ODE
\[
\frac{dR}{dt}
=\frac{(Rc_N)^{-\nu}-(Rc_N)^{-1}}{c_N},
\]
not the continuum ODE \(dR/dt=R^{-\nu}-R^{-1}\). The \(O(h^2)\) difference
is the finite-\(N\) \(\cos(\pi/N)\) effect; using the continuum ODE as a
reference on a finite \(N\)-gon leaves an \(O(h^2)\) floor.

The 2D solver reproduces this recurrence to \(4.9\times10^{-14}\) in E0.
The exact circle defect is
\[
D_h=2N\tan(\pi/N)-2\pi,
\]
reproduced to \(8.9\times10^{-16}\).

---

## S2. Full E0–E9 tables

The main paper reports selected values. The complete tables are in
`A1_Numerics/data/tables/`:

| Main-paper item | Full table(s) |
|---|---|
| E0 implementation gate | `table1_e0_gate.csv`, `data/e0_verification.json` |
| E1 closure exponent | `table2_e1_closure.csv`, `table2b_e1_Dh.csv` |
| E2 spatial orders and A5 ratio (legacy label A7) | `table3_e2_spatial.csv`, `table3b_e2_summary.csv`, `table3c_e2_A7_time.csv` |
| E3 time/startup/joint | `table4_e3_circle.csv`, `table4b_e3_startup.csv`, `table5_e3_ellipse_joint.csv`, `table5b_e3_fixedN.csv`, `table6_e3_coupling.csv` |
| E3b one-step curvature | `table14_e3b_curvature.csv`, `data/e3b_startup_curvature.json` |
| E4 long-time | `table7_e4_safe.csv`, `table8_e4_coarse.csv` |
| E5 cost/conditioning | `table9_e5_cost.csv`, `table10_e5_condition.csv` |
| E6 prior-work comparison | `table11_e6_circle.csv`, `table12_e6_stability.csv`, `table13_e6_n_sensitivity.csv` |
| E7 extended A5 ratio | `table15_e7_a5_summary.csv` (flower quantitative range uses N≤256; N=512 is at the reference floor), `table15b_e7_a5_detail.csv`, `data/e7_a5_extension.json` |
| E8 two-parameter stability map | `table16_e8_stability.csv`, `data/e8_stability_map.json` |
| E9 work–precision | `table17_e9_work_precision.csv`, `data/e9_work_precision.json` |

The raw JSON files are `A1_Numerics/data/e0_verification.json` through
`A1_Numerics/data/e9_work_precision.json`; the manifest
`A1_Numerics/data/manifest.json` records the environment, commands, timings,
and SHA-256 hashes.

The main paper reports only the first-order BGN baseline in E6. The
supplementary `table11`/`table12` also contain the legacy CNLF–EX rows;
those rows are prior-work context and are **not** used in the paper's
claims.

### S2b. Supplementary tables S1–S4

The main text refers to the following Supplementary tables. The E1/E2/E3/E5
figures and the E1/E2 summary tables are in the main text; the SM retains the
full per-\(N\) data and the auxiliary tables.

| Item | Content | File(s) |
|---|---|---|
| Table S1 | E2 full per-\(N\) errors and orders | `table3_e2_spatial.csv` |
| Table S2 | E1 closure defect \(D_h\) and mesh quality | `table2b_e1_Dh.csv` |
| Table S3 | Constants used in the proof | `tableS3_constants.md` |
| Table S4 | Experiment configurations | `tableS4_experiment_configurations.md` |

The reviewer-facing PDF `A1_Supplementary_Materials.pdf` typesets Tables S1–S4;
the file names above are the archive sources.

---

## S3. Code and data map

```
A1_Numerics/
  code/
    a1n_config.py                 # paths, ladders, tolerances
    solver/geometry.py            # BGN geometry, stiffness, curvature
    solver/bdf2_taylor.py         # BDF2-Taylor step and time loop
    solver/scalar_circle.py       # exact finite-N recurrence / modified ODE
    solver/legacy_adapter.py      # read-only BGN1 baseline adapter
    metrics/shapes.py             # circle/ellipse/flower/superellipse
    metrics/alignment.py          # arc-length alignment, spline reference
    metrics/norms.py              # L2/H1/curvature/A5 metrics (legacy label A7)
    experiments/e0_verify.py ... e6_compare.py
    experiments/e3b_startup_curvature.py, e7_a5_extension.py,
      e8_stability_map.py, e9_work_precision.py
    make_tables.py, make_figures.py, checks.py, run_all.py, run_phase4.py
  data/
    e0_verification.json ... e9_work_precision.json
    tables/*.csv, *.md
    manifest.json
  figures/
    fig1_closure.pdf ... fig6_n_sensitivity.pdf,
    fig6_e8_stability.pdf, fig7_e9_work_precision.pdf
paper_figures/
  make_paper_figures.py           # paper-specific Fig 4/5/6/7 (no CNLF-EX)
  fig4_stability_paper.pdf
  fig5_cost_paper.pdf
  fig6_e8_stability.pdf
  fig7_e9_work_precision.pdf
A1_Legacy_BGN1_Baseline/code/solver/
  bgn2_powerlaw.py                # legacy read-only BGN1/CNLF solver
  metrics.py, shapes.py           # legacy dependencies
```

The legacy files are included only so that the E6 BGN1 baseline can be
regenerated; the paper does not describe the legacy CNLF–EX route. The
new BDF2–Taylor solver is self-contained in `A1_Numerics/code`.

**Nomenclature note.** The manuscript calls the \(L^{2}\)
normal-superconvergence hypothesis A5. Some legacy code comments, table
filenames, and diagnostic labels still use A7 for the same ratio; this is a
naming legacy, not a different assumption. The archive folder name
`A1_Legacy_BGN1_Baseline` is retained for provenance and identifies the
read-only BGN1 baseline code used in E6; it is not a separate manuscript or
submission.

---

## S4. Reproduction

Environment: Python 3.13.12, NumPy 2.4.4, SciPy 1.17.1, Matplotlib
3.10.8, Windows 11. The E0–E6 chain takes about 20–25 minutes on the
reference machine; E0 is the hard gate. The Phase-4 additions
(E3b/E7/E8/E9) take about 20–25 minutes more, dominated by the E7
reference runs.

```bash
cd A1_Numerics/code
python -I experiments/e0_verify.py
python -I experiments/e1_closure.py
python -I experiments/e2_spatial.py
python -I experiments/e3_temporal.py
python -I experiments/e4_stability.py
python -I experiments/e5_cost.py
python -I experiments/e6_compare.py
python -I experiments/e3b_startup_curvature.py
python -I experiments/e7_a5_extension.py
python -I experiments/e8_stability_map.py
python -I experiments/e9_work_precision.py
python -I make_tables.py
python -I make_figures.py
python -I checks.py --write-manifest
```

The Phase-4 additions can be run together with
`python -I run_phase4.py` (E3b, E7, E8, E9 only; it does not rerun
E0–E6). The paper-specific figures `fig4_stability_paper.pdf`,
`fig5_cost_paper.pdf`, `fig6_e8_stability.pdf` and
`fig7_e9_work_precision.pdf` (without the legacy CNLF–EX curve) are
generated by `paper_figures/make_paper_figures.py`.

---

## S5. Summary of negative results

1. Tangential mesh mode: the predicted geometry drives node clustering at
   coarse \(\tau\) or on strongly non-circular shapes; the mesh ratio
   reaches \(10^3\)–\(10^4\) and the run aborts. The failure is
   non-monotone in \(\tau/h\); it is not a simple CFL condition.
2. Flower at \(t=0\): the closure exponent is pre-asymptotic at the
   affordable \(N\) (finest pair \(1.967\), not \(2.000\)).
3. Fixed-\(N\) ellipse time order: the time error lies below the
   \(O(h^2)\) spatial floor, so a clean fixed-\(N\) order cannot be
   extracted.
4. Full \(H^1\) order one: this is interpolation-dominated and is the
   theorem's rate; the nodal error is second order but is not the theorem
   norm.
5. No exact area preservation: the continuum flow is area preserving only
   at \(\nu=1\); the discrete \(\nu=1\) area gap is \(O(10^{-7})\).
6. Error constants grow with \(\nu\); no uniform-in-\(\nu\) claim.
7. BDF2–Taylor has a tighter tangential mesh window than the BGN1
   baseline; it is more accurate in time but less robust with respect to
   mesh quality.
8. The two-parameter E8 map is non-monotone: completed runs can have
   \(\Psi_{\max}\) of \(10^2\)–\(10^4\) while other, apparently similar
   \((\tau,h)\) pairs abort. \(\Psi_{\max}\) is a posteriori, not a
   predictor; the minimum edge and tangential drift are more informative in
   the tested runs.
9. E9 shows the efficiency trade-off: on the circle BDF2–Taylor is more
   accurate at comparable CPU time, but on the ellipse at \(N=512\) it
   aborts for \(\tau\le0.0125\), while BGN1 remains stable at first order.
10. E7 flower \(N=512\) at \(t=0.25,0.5\) has a full \(L^2\) error
    comparable to the reference-floor difference; the quantitative flower
    range and trend therefore use \(N\le256\), and the \(N=512\) row is
    retained only in the detailed table.
