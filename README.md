# mca-4686604-bdf2-taylor

Code and data for the manuscript:

> **A Linearly Implicit BDF2–Taylor Finite Element Method for Nonlocal
> Power-Law Curvature Flow: Conditional Convergence and Computational
> Validation**
>
> Chengde Huoye
>
> *Mathematical and Computational Applications* (MCA), manuscript
> **mca-4686604**

This repository is the persistent code/data archive accompanying the
manuscript's Supplementary Materials. It contains the E0–E9 numerical
experiments, the solver, the raw JSON/CSV outputs, the figure-generation
scripts, and the reviewer-facing Supplementary Materials PDF. It does **not**
contain the manuscript source, the response letter, or the author-only
revision records.

## Contents

| Path | Contents |
|---|---|
| `A1_Numerics/code/` | Self-contained BDF2–Taylor solver, metrics, and experiment drivers E0–E9 |
| `A1_Numerics/data/` | Raw JSON results, CSV/Markdown tables, and `manifest.json` |
| `A1_Numerics/figures/` | Numerical-study figures (PDF and PNG) |
| `A1_Numerics/logs/` | Logs of the Phase-4 experiments (E3b/E7/E8/E9) |
| `paper_figures/` | Paper-specific Fig. 4–7 files and the regeneration script |
| `A1_Legacy_BGN1_Baseline/` | Read-only legacy BGN1 solver used only for the E6 baseline |
| `A1_Supplementary_Materials.pdf` | Reviewer-facing Supplementary Materials (8 pages) |
| `A1_Supplementary_Materials.md` | Supplementary archive guide, table map, and reproduction notes |
| `A1_Supplementary_Materials_README.md` | Original Supplementary Materials README (archive layout) |

The code is the implementation used for the reported experiments. The
manuscript and the Supplementary PDF are the reference documents for the
scheme and for the interpretation of each experiment.

## Quick check

The reference environment is Python 3.13.12, NumPy 2.4.4, SciPy 1.17.1, and
Matplotlib 3.10.8 (Windows 11). Install the pinned dependencies with:

```bash
python -m pip install -r requirements.txt
```

Then verify the archived results without rerunning the experiments:

```bash
cd A1_Numerics/code
python -B -I checks.py
```

Expected output ends with:

```text
checks: PASS
```

`checks.py` verifies the E0 gate, the presence and size of the E0–E9 JSON
files, the figures, and the table exports. It does not modify the archived
JSON data.

## Full reproduction

The complete chain is documented in `A1_Supplementary_Materials_README.md`
and `A1_Numerics/code/README.md`. From `A1_Numerics/code/`:

```bash
python -I experiments/e0_verify.py       # hard gate
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
python -I checks.py
```

E0 is the implementation gate. On the reference machine, E0–E6 take about
20–25 minutes and the Phase-4 additions (E3b/E7/E8/E9) take about 20–25
minutes more. `run_all.py --full` runs E0–E6; `run_phase4.py` runs the
Phase-4 additions only.

The paper-specific figures are regenerated from `paper_figures/` with
`make_paper_figures.py`.

## Data and provenance

- `A1_Numerics/data/*.json` are the raw experiment outputs; the CSV/Markdown
  tables under `A1_Numerics/data/tables/` are derived from them.
- `A1_Numerics/data/manifest.json` records the original generation
  environment, commands, timings, and file hashes. Its absolute paths refer
  to the author's machine and are retained for provenance; they are not
  expected to match a fresh clone.
- `checksums.sha256` at the repository root records SHA-256 hashes for the
  archived files.
- CPU times are machine-dependent; the errors and orders are the
  reproducible quantities.
- No data are synthetic or hand-edited: the JSON files are the outputs of
  the scripts in `A1_Numerics/code/`.

## Notes and limitations

- The legacy `A1_Legacy_BGN1_Baseline/` folder is included only so that the
  E6 first-order BGN1 baseline can be regenerated. It is read-only and is
  not a separate manuscript or submission. The paper does not use the
  legacy CNLF–EX route in its claims.
- The manuscript calls the \(L^2\) normal-superconvergence hypothesis A5;
  some legacy code comments and filenames use A7 for the same ratio. This is
  a naming legacy, not a different assumption.
- The predicted geometry can excite a tangential mesh mode at coarse time
  steps or on strongly non-circular shapes. This is a reported limitation,
  not a hidden failure; E4/E6/E8 document it.
- The archived results are the ones reported in the manuscript; rerunning
  the experiments may produce small platform-dependent timing differences.

## Version and citation

This is release **v1.0.0** (2026-11-17). The release asset is
`mca-4686604-bdf2-taylor-v1.0.0.zip`. If you use this code or data, please
cite the accompanying manuscript; see `CITATION.cff`.

No DOI is claimed for this release here. If a DOI is later minted (for
example through Zenodo), it should be added to the manuscript's Data
Availability statement and to this README.

## License

- **Code** (Python files and commands): MIT License; see `LICENSE`.
- **Data, figures, and Supplementary PDF/Markdown**: CC BY 4.0; see
  `LICENSE-CC-BY-4.0.md`.
