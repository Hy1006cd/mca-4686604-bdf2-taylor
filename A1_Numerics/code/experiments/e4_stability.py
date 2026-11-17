"""
E4 -- long-time stability and structural diagnostics.

Two panels:

  safe_tau   long-time runs in a parameter window where the mesh stays
             quasi-uniform: ellipse (tau=2.5e-4, N=256/512, T=2/5),
             flower (tau=1e-4, N=256/512, T=2; N=256, T=5).
  coarse_tau the fixed coarse tau=1.25e-3 experiment: documents the
             tangential mesh-mode instability (Psi -> infinity) of the
             predicted-geometry BDF2 scheme.

Reported per run: A(T)/A(0), equilibrium area prediction, perimeter,
max/final mesh ratio, min edge, ||kappa||_inf, min s_*, min |Sherman-
Morrison denominator|, failure flag/time, self-intersection count at T.

The experiment is diagnostic: it does not claim area preservation or a
stability theorem; negative results are reported.

Usage: python -I experiments/e4_stability.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import a1n_config as cfg
from experiments import common
from metrics import shapes
from solver import bdf2_taylor as bt
from solver import geometry as geo

N_VALUES_RUN = (1, 2, 3)


def _equilibrium_area(shape_id, n):
    """Continuum equilibrium: circle R=1 for n>=2, same-area circle for n=1."""
    A0 = geo.polygon_area(shapes.build_shape(shape_id, 256))
    return float(A0 if n == 1 else np.pi)


def run_one(sid, n, N, T, tau, label):
    X0, k0 = shapes.build_geometry(sid, N)
    A0 = geo.polygon_area(X0)
    M = int(round(T / tau))
    tau_eff = T / M
    t0 = time.perf_counter()
    try:
        X, k, diags, _, info = bt.solve_bdf2(
            X0, k0, T, M, n, record=True, stop_on_failure=False)
    except Exception as exc:                            # noqa: BLE001
        return dict(label=label, shape=sid, n=n, N=N, T=T, tau=tau_eff, M=M,
                    failed=True, reason=repr(exc),
                    elapsed_s=time.perf_counter() - t0)
    elapsed = time.perf_counter() - t0
    psi_max = float(max(d['mesh_ratio'] for d in diags)) if diags else float('nan')
    denom_min = min((d['denominator_abs'] for d in diags
                     if np.isfinite(d['denominator_abs'])), default=float('nan'))
    s_star_min = min((d['s_star'] for d in diags
                      if np.isfinite(d['s_star'])), default=float('nan'))
    s_star_min_positive = min((d['s_star'] for d in diags
                               if np.isfinite(d['s_star']) and d['s_star'] > 0),
                              default=float('nan'))
    A = geo.polygon_area(X)
    # n=1 preserves area exactly in the continuum; the equilibrium is the
    # same-area circle, so A_eq/A(0)=1 by definition.  For n>=2 the
    # equilibrium is the unit circle, area pi.
    Aeq = A0 if n == 1 else float(np.pi)
    try:
        n_self = geo.self_intersections(X) if N <= 512 else None
    except Exception:                                   # noqa: BLE001
        n_self = None
    return dict(label=label, shape=sid, n=n, N=N, T=T, tau=tau_eff, M=M,
                failed=bool(info.get('failed', False)),
                reason=info.get('reason', ''),
                fail_level=info.get('fail_level'),
                fail_time=info.get('fail_time'),
                area_final=A, area_0=A0,
                area_ratio=A / A0,
                equilibrium_ratio=Aeq / A0,
                area_gap=abs(A - Aeq),
                perimeter_final=geo.polygon_perimeter(X),
                psi_max=psi_max,
                psi_final=float(geo.mesh_ratio(X)),
                min_edge=float(np.min(geo.geometry(X)[1])),
                kappa_inf=float(np.max(np.abs(k))),
                s_star_min=float(s_star_min),
                s_star_min_positive=float(s_star_min_positive),
                denom_min=float(denom_min),
                self_intersections=n_self,
                elapsed_s=elapsed)


def safe_runs():
    specs = []
    for n in N_VALUES_RUN:
        for N in (256, 512):
            for T in (2.0, 5.0):
                specs.append(('ellipse', n, N, T, 2.5e-4))
        for N in (256, 512):
            specs.append(('flower', n, N, 2.0, 1.0e-4))
        specs.append(('flower', n, 256, 5.0, 1.0e-4))
    rows = []
    for sid, n, N, T, tau in specs:
        r = run_one(sid, n, N, T, tau, 'safe_tau')
        rows.append(r)
        print(f"safe  {sid:8s} n={n} N={N:4d} T={T:3.0f} "
              f"failed={r['failed']!s:5s} "
              f"A/A0={r.get('area_ratio', float('nan')):.6f} "
              f"psi_max={r.get('psi_max', float('nan')):.2f} "
              f"t={r.get('elapsed_s', 0):.1f}s")
    return rows


def coarse_runs():
    specs = []
    for n in (1, 2, 3):
        for N in (256, 512):
            specs.append(('ellipse', n, N, 2.0, 1.25e-3))
            specs.append(('flower', n, N, 2.0, 1.25e-3))
    rows = []
    for sid, n, N, T, tau in specs:
        r = run_one(sid, n, N, T, tau, 'coarse_tau')
        rows.append(r)
        print(f"coarse {sid:8s} n={n} N={N:4d} failed={r['failed']!s:5s} "
              f"psi_max={r.get('psi_max', float('nan')):.2f} "
              f"A/A0={r.get('area_ratio', float('nan')):.6f}")
    return rows


def main():
    t_start = time.perf_counter()
    print('=' * 78)
    print('E4 long-time stability and structural diagnostics')
    print('=' * 78)
    safe = safe_runs()
    coarse = coarse_runs()
    payload = dict(experiment='E4', generated=common.now(),
                   environment=cfg.environment_manifest(),
                   setup=dict(safe_tau=dict(ellipse=2.5e-4, flower=1.0e-4),
                              coarse_tau=1.25e-3,
                              n=list(N_VALUES_RUN),
                              ellipse_N=[256, 512], flower_N=[256, 512],
                              T=[2.0, 5.0]),
                   safe_tau=safe, coarse_tau=coarse,
                   elapsed_s=float(time.perf_counter() - t_start))
    common.finish(payload, 'e4_stability.json')
    print(f'\nE4 done in {payload["elapsed_s"]:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
