"""
E8 -- two-parameter (h, tau) stability map for the BDF2-Taylor scheme.

Grid: N = 64,128,256,512 and tau = 5e-3 ... 6.25e-5, T = 0.5, nu = 2,
on the ellipse (2,1) and the flower (2,0.5,8).

For every run the script records completed/failed, the failure reason and
time, Psi_max, the minimum edge over the run, the final maximum turning
angle, the final tangential drift, min|1+v^T z|, and the area ratio.  These
are a posteriori diagnostics; Psi_max alone is not a predictor of failure.

Usage: python -I experiments/e8_stability_map.py
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

T = 0.5
N_VALUES = 2
N_LEVELS = (64, 128, 256, 512)
TAU_LEVELS = (5.0e-3, 2.5e-3, 1.25e-3, 1.0e-3, 5.0e-4,
              2.5e-4, 1.25e-4, 6.25e-5)
SHAPES = ('ellipse', 'flower')


def _max_turning_angle(X):
    X = np.asarray(X, dtype=float)
    e = np.roll(X, -1, axis=0) - X
    en = np.roll(e, -1, axis=0)
    cross = e[:, 0] * en[:, 1] - e[:, 1] * en[:, 0]
    dot = np.sum(e * en, axis=1)
    ang = np.arctan2(cross, dot)
    return float(np.max(np.abs(ang)))


def _tangential_drift(X):
    """max_i |frac_i - i/N| (circular), frac = normalized arc length."""
    X = np.asarray(X, dtype=float)
    e = np.roll(X, -1, axis=0) - X
    le = np.linalg.norm(e, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(le)])
    if cum[-1] <= 0.0:
        return float('nan')
    frac = cum[:-1] / cum[-1]
    idx = np.arange(len(X), dtype=float) / len(X)
    diff = (frac - idx + 0.5) % 1.0 - 0.5
    return float(np.max(np.abs(diff)))


def run_one(shape, N, tau):
    X0, k0 = shapes.build_geometry(shape, N)
    M = int(round(T / tau))
    tau_eff = T / M
    h0 = common.h_of(X0)
    t0 = time.perf_counter()
    try:
        X, k, diags, _, info = bt.solve_bdf2(
            X0, k0, T, M, N_VALUES, record=True, stop_on_failure=False)
    except Exception as exc:                            # noqa: BLE001
        return dict(shape=shape, N=N, tau=tau_eff, M=M,
                    tau_over_h=tau_eff / h0, h0=h0,
                    failed=True, reason=repr(exc),
                    fail_level=None, fail_time=None,
                    psi_max=None, psi_final=None, min_edge=None,
                    max_turning_angle=None, tangential_drift=None,
                    denom_min=None, area_ratio=None,
                    elapsed_s=float(time.perf_counter() - t0))

    psi = [d['mesh_ratio'] for d in diags
           if np.isfinite(d.get('mesh_ratio', float('nan')))]
    edges = [d['min_edge'] for d in diags
             if np.isfinite(d.get('min_edge', float('nan')))]
    denoms = [d['denominator_abs'] for d in diags
              if np.isfinite(d.get('denominator_abs', float('nan')))]
    A0 = geo.polygon_area(X0)
    return dict(
        shape=shape, N=N, tau=tau_eff, M=M,
        tau_over_h=tau_eff / h0, h0=h0,
        failed=bool(info.get('failed', False)),
        reason=info.get('reason', ''),
        fail_level=info.get('fail_level'),
        fail_time=info.get('fail_time'),
        psi_max=float(max(psi)) if psi else None,
        psi_final=float(geo.mesh_ratio(X)),
        min_edge=float(min(edges)) if edges else None,
        max_turning_angle=_max_turning_angle(X),
        tangential_drift=_tangential_drift(X),
        denom_min=float(min(denoms)) if denoms else None,
        area_ratio=float(geo.polygon_area(X) / A0) if A0 > 0 else None,
        elapsed_s=float(time.perf_counter() - t0),
    )


def main():
    t_start = time.perf_counter()
    print('=' * 78)
    print('E8 two-parameter (h,tau) stability map')
    print('=' * 78)
    payload = dict(
        experiment='E8',
        generated=common.now(),
        environment=cfg.environment_manifest(),
        setup=dict(shapes=list(SHAPES), n=N_VALUES, T=T,
                   N_levels=list(N_LEVELS), tau_levels=list(TAU_LEVELS)),
        rows=[],
    )
    for shape in SHAPES:
        for N in N_LEVELS:
            for tau in TAU_LEVELS:
                r = run_one(shape, N, tau)
                payload['rows'].append(r)
                print(f"{shape:8s} N={N:4d} tau={r['tau']:.3e} "
                      f"tau/h={r['tau_over_h']:.4g} "
                      f"{'FAIL' if r['failed'] else 'ok':4s} "
                      f"psi={r['psi_max']} min_edge={r['min_edge']} "
                      f"theta={r['max_turning_angle']}")
    payload['elapsed_s'] = float(time.perf_counter() - t_start)
    common.finish(payload, 'e8_stability_map.json')
    print(f"\nE8 done in {payload['elapsed_s']:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
