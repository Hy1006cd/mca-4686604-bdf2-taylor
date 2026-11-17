# Supplementary Materials

**Manuscript**: A Linearly Implicit BDF2–Taylor Finite Element Method for
Nonlocal Power-Law Curvature Flow: Conditional Convergence and Computational
Validation  
**Author**: Chengde Huoye  
**Date**: 26 October 2026

This archive contains the code, data, figures, and reproduction instructions
for the E0–E9 numerical experiments in the manuscript. All paths are relative
to this folder. The reviewer-facing PDF `A1_Supplementary_Materials.pdf`
typesets the S1–S4 material; this README and
`A1_Supplementary_Materials.md` document the archive.

## Layout

- `A1_Supplementary_Materials.pdf` — reviewer-facing supplement (finite-\(N\)
  reduction, table map, Tables S1–S4, code/data map,
  reproduction, and negative results).
- `A1_Numerics/code/` — self-contained BDF2–Taylor solver, metrics, and
  experiments E0–E9.
- `A1_Numerics/data/` — raw JSON results, CSV/Markdown tables, and
  `manifest.json`.
- `A1_Numerics/figures/` — figures from the numerical study.
- `paper_figures/` — paper-specific Fig 4–7 (without CNLF–EX) and the
  script that regenerates them.
- `A1_Legacy_BGN1_Baseline/code/solver/` — read-only legacy BGN1 solver
  used only for the E6 first-order baseline. The folder name is retained
  for provenance; it is not a separate manuscript or submission. The paper
  does not describe the legacy CNLF–EX route.
- `A1_Supplementary_Materials.md` — archive guide, including the exact
  finite-\(N\) reduction and the full table map.

**E7 reference-floor note.** The flower reference is \(N_{\rm ref}=1024\)
at \(T=0.5\). The \(N=512\) rows at \(t=0.25,0.5\) are at the
\(N_{\rm ref}/2\)-versus-\(N_{\rm ref}\) reference-floor level; the
manuscript's quantitative flower range/trend therefore uses \(N\le256\).
The full \(N=512\) rows remain in `table15b_e7_a5_detail` for transparency.

**Supplementary tables (S1–S4).** The E1/E2/E3/E5 figures and the E1/E2
summary tables are now in the main text. The SM retains Table S1 (E2 full
per-\(N\) errors and orders), Table S2 (E1 \(D_h\) and mesh quality),
Table S3 (`tableS3_constants.md`) and Table S4
(`tableS4_experiment_configurations.md`). The full mapping is in
`A1_Supplementary_Materials.md` §S2b.

## Reproduction

From this folder:

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

E0 is the hard gate. The E0–E6 chain takes about 20–25 minutes on the
reference machine (Python 3.13.12, NumPy 2.4.4, SciPy 1.17.1,
Matplotlib 3.10.8, Windows 11); the Phase-4 additions take about 20–25
minutes more and can be run together with `python -I run_phase4.py`.

## Note on the legacy code

`A1_Legacy_BGN1_Baseline/code/solver/bgn2_powerlaw.py` contains the
prior-work BGN1 and CNLF–EX implementations. It is included only so that the
E6 BGN1 baseline can be regenerated; no claim in the paper uses the CNLF–EX
route.

## Nomenclature note

The manuscript calls the \(L^{2}\) normal-superconvergence hypothesis A5.
Some legacy code comments, table filenames, and diagnostic labels still use
A7 for the same ratio; this is a naming legacy, not a different assumption.

## PDF source

The PDF is generated from the author-only LaTeX source in
`../../03_Revision_Records/SM_Document/`; the Markdown files remain the
archive guide. The PDF is not typeset by MDPI.
