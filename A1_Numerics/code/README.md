# A1_Numerics/code — BDF2–Taylor solver and experiment drivers

**Scope.** This package implements the discrete scheme analyzed in the
manuscript and produces the numerical evidence reported there. The manuscript
and `A1_Supplementary_Materials.md` are the reference documents for the scheme
and the experiments.

**Flow.** Dimensionless \(L_0=1\), CCW curve, outward normal, \(\kappa>0\) on
the unit circle, \(\kappa\mathbf n=-\partial_{ss}\mathbf X\):

\[
V=(\langle\kappa\rangle^n-\kappa)\mathbf n.
\]

## File map

| file | role |
|---|---|
| `a1n_config.py` | single source of truth: paths, ladders, tolerances, JSON helpers |
| `solver/geometry.py` | \(e_i,h_i,\mathbf n_i,M_i,\mathbf N_i,S\), \(\kappa_h^0\), \(D_h\), diagnostics |
| `solver/bdf2_taylor.py` | BDF2 step (S5.1), Taylor term (S6.2), CN startup (S7), rank-one solve (S8) |
| `solver/scalar_circle.py` | exact finite-\(N\) regular-polygon recurrence, modified/continuum ODE |
| `solver/legacy_adapter.py` | read-only BGN1/CNLF-EX adapter (E6 only) |
| `metrics/shapes.py` | circle, ellipse, flower, superellipse (arc-length equidistributed) |
| `metrics/alignment.py` | arc-length alignment, periodic cubic-spline reference, full \(L^2/H^1\) |
| `metrics/norms.py` | \(L^2\), discrete \(H^1\), curvature, A5 ratio (legacy label A7), orders/fits |
| `experiments/e0_verify.py … e6_compare.py` | E0–E6 |
| `experiments/e3b_startup_curvature.py` | one-step startup curvature error |
| `experiments/e7_a5_extension.py` | extended A5 ratio (circle/ellipse/flower) |
| `experiments/e8_stability_map.py` | two-parameter \((h,\tau)\) stability map |
| `experiments/e9_work_precision.py` | error versus CPU time |
| `make_tables.py` | JSON → CSV/Markdown tables |
| `make_figures.py` | PDF/PNG figures |
| `checks.py` | gate/consistency checks + `data/manifest.json` |
| `run_all.py` | E0→E6 driver |
| `run_phase4.py` | E3b/E7/E8/E9 driver (no E0–E6 rerun) |

## Implementation choices that matter

1. **Predicted geometry, actual masses.**
   All step-\(m+1\) geometry is frozen at
   \(\widetilde{\mathbf X}^{m+1}=2\mathbf X^m-\mathbf X^{m-1}\)
   (S4.1).  The closure predictor uses the **actual** layer means
   \(s_h^m=\sum M^m\kappa^m/L_h^m\) and
   \(s_h^{m-1}=\sum M^{m-1}\kappa^{m-1}/L_h^{m-1}\), not the predicted
   geometry's masses (P1-14).  The new mean \(s_h^{m+1}\) in the Taylor term
   uses the predicted geometry's masses, as in S6.1.

2. **One sparse LU + two back-solves.**
   \(A=A_0+uv^\top\), \(u_i=-\alpha M_i/L_h\) in the normal rows,
   \(v_j=M_j\) in the \(\kappa\) columns,
   \(\alpha=n(s_*)^{n-1}\), \(c=(1-n)(s_*)^n\).
   Solve \(A_0y=b\), \(A_0z=u\), then
   \(U=y-\frac{v\cdot y}{1+v\cdot z}z\).  The denominator is monitored (E5).

3. **CN startup with \(\Gamma^{1/2}\).**
   \(\mathbf V^0=((s_h^0)^n-\kappa_h^0)\mathbf n^0\), with the vertex normal
   \(\mathbf n^0=\mathbf N^0/|\mathbf N^0|\), and
   \(\mathbf X^{1/2}=\mathbf X^0+\frac{\tau}{2}\mathbf V^0\).
   \(G^{1/2}=s_0^n+\frac{\tau}{2}n s_0^{n-1}\dot s_0\),
   \(\dot s_0=-(K_h^0/(L_h^0)^2)(s_0^nK_h^0-\sum M_i^0(\kappa_i^0)^2)\).
   The ablation `predictor='first'` sets \(G^{1/2}=s_0^n\).

4. **Exact circle reduction.**
   On a regular \(N\)-gon, \(c_N=\cos(\pi/N)\), \(\kappa=1/(Rc_N)\), and the
   frozen curvature equation gives \(\kappa^{m+1}=R^{m+1}/(\tilde R^2c_N)\).
   The scalar recurrence is linear:
   \[
   R^{m+1}=\frac{(4R^m-R^{m-1})c_N/(2\tau)-(n-1)s_*^n}
   {3c_N/(2\tau)+(1-ns_*^{n-1})/(\tilde R^2c_N)},\quad
   s_*=\frac{2}{R^mc_N}-\frac{1}{R^{m-1}c_N}.
   \]
   The semi-discrete limit is the modified ODE
   \(dR/dt=[(Rc_N)^{-n}-(Rc_N)^{-1}]/c_N\), not the continuum
   \(dR/dt=R^{-n}-R^{-1}\).  E0 checks both.

5. **Metrics.**
   `full_errors` evaluates the discrete solution's piecewise linear
   interpolant on a fine arc-length grid and compares it with a periodic
   cubic spline through the reference solution.  This is the theorem norm
   (including the interpolation error \(\rho\)).  `nodal_errors` compares
   nodal values only and measures the discrete error \(e_h\).  The spline
   removes the \(O(h_{\rm ref})\) sawtooth of a piecewise linear reference,
   which otherwise contaminates the \(H^1\) measurement.

## Running

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
python -I checks.py --write-manifest
```

`run_phase4.py` runs E3b/E7/E8/E9 only (about 20–25 min on the reference
machine); `run_all.py --full` runs E0–E6 (about 20–25 min) and does not
rerun the Phase-4 additions.

**E8 diagnostics.** `Psi_max` is the maximum mesh ratio over the run.
`min_edge` is the minimum edge over the run. `max_turning_angle` is the
final maximum absolute turning angle between consecutive edges.
`tangential_drift` is `max_i |frac_i - i/N|` (circular), where `frac_i` is
the final normalized cumulative arc length at vertex `i`. `min|d|` is the
minimum Sherman–Morrison denominator over the run. These are a posteriori
diagnostics; `Psi_max` alone is not a predictor of failure.

**E7 reference floor.** The flower reference is \(N_{\rm ref}=1024\) at
\(T=0.5\). At \(t=0.25,0.5\) the \(N=512\) full \(L^2\) error is comparable
to the \(N_{\rm ref}/2\) versus \(N_{\rm ref}\) difference, so the
manuscript's quantitative flower range/trend uses \(N\le256\); the
\(N=512\) rows remain in `table15b` for transparency.

`run_all.py --full` runs E0–E6 in order (about 20–25 min on the reference
machine); `--quick` runs only the E0 gate.

## Dependencies

Python 3.13, numpy 2.4, scipy 1.17, matplotlib 3.10.  The legacy solver
(`A1_Legacy_BGN1_Baseline/code/solver/bgn2_powerlaw.py`) is imported
read-only by `solver/legacy_adapter.py`; its SHA256 is recorded in
`data/manifest.json`.

## Known numerical pitfall (reported, not hidden)

The predicted geometry \(2\mathbf X^m-\mathbf X^{m-1}\) can excite a
**tangential mesh mode**: the shape and area remain accurate, but the node
distribution can cluster (\(\Psi=\max h_i/\min h_i\) grows) for some
\((\tau,h)\) and for strongly non-circular shapes.  The mode is
non-monotonic in \(\tau/h\); it is not a simple CFL condition.  All
convergence experiments use windows where \(\Psi\) stays bounded; E3/E4/E6
document the failures.  Mesh redistribution/stabilisation is outside the
v1.2 scheme and outside the theorem; it is a v1.3 candidate.
