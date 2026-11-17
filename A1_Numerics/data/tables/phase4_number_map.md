# Phase 4 number map (manuscript -> JSON)

| Manuscript item | JSON path | Key(s) |
|---|---|---|
| Table 3 initial perimeters/areas | `data/e7_a5_extension.json`, `solver/shapes.py` | `shapes.build_shape` at N=8192 |
| Table 3 denominator values | `data/e5_cost.json`, `data/e0_verification.json`, `data/e4_stability.json` | `denominator_condition`, `results[min_abs_denominator]`, `safe_tau[].denom_min` |
| Table e3 panel (b) curvature | `data/e3b_startup_curvature.json` | `results[].rows[].k_err_cont`, `k_err_mod` |
| Table e7 A5 summary | `data/e7_a5_extension.json` | `results[].rows` (flower range N<=256; trend N256/N128; N=512 floor-limited) |
| Table e8 stability rows | `data/e8_stability_map.json` | `rows[]` |
| Table e9 work-precision rows | `data/e9_work_precision.json` | `rows[]` |
| Figure 6 E8 heatmap | `data/e8_stability_map.json` | `rows[].psi_max`, `rows[].failed` |
| Figure 7 E9 work-precision | `data/e9_work_precision.json` | `rows[].error`, `rows[].time_median` |
