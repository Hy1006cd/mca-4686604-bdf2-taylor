"""
E9 -- work-precision comparison of BDF2-Taylor and first-order BGN1.

Fixed N = 512, T = 0.5, nu = 2, tau = 0.1/2^k (k = 0..5), median of three
timed runs.  On the circle the error is |R(T)-R_mod(T)| against the modified
ODE (isolating the time error).  On the ellipse the error is the full L2
error against a reference solution (N_ref = 2048, tau_ref = 6.25e-5).

The script reports both total CPU time and ms/step.  Timing is machine
dependent; the comparison is intended as an observed work-precision curve,
not as a platform-independent complexity claim.

Usage: python -I experiments/e9_work_precision.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import a1n_config as cfg
from experiments import common
from metrics import alignment, shapes
from solver import bdf2_taylor as bt
from solver import geometry as geo
from solver import legacy_adapter as legacy
from solver import scalar_circle as sc

T = 0.5
NVAL = 2
N = 512
TAU0 = 0.1
K_LEVELS = 6
REPEATS = 3
ELL_NREF = 2048
ELL_TAUREF = 6.25e-5
SHAPES = ('circle', 'ellipse')


def _ellipse_reference():
    M_ref = int(round(T / ELL_TAUREF))
    X0, k0 = shapes.build_geometry('ellipse', ELL_NREF)
    Xr, kr, _, _, info = bt.solve_bdf2(
        X0, k0, T, M_ref, NVAL, stop_on_failure=True)
    if info.get('failed', False):
        raise RuntimeError(info.get('reason', 'ellipse reference failed'))
    return Xr, kr, alignment.smooth_spline(Xr, kr), M_ref


def _timed_bdf2(shape, M):
    X0, k0 = shapes.build_geometry(shape, N)
    times = []
    out = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = bt.solve_bdf2(X0, k0, T, M, NVAL, stop_on_failure=True)
        times.append(time.perf_counter() - t0)
    X, k = out[0], out[1]
    return X, k, times


def _timed_bgn1(shape, M):
    X0, k0 = shapes.build_geometry(shape, N)
    X, k, _, tm = legacy.timed_solve(
        legacy.solve_bgn1, X0, k0, T, M, NVAL, repeat=REPEATS,
        record=False)
    return X, k, tm['times']


def _error_with_tau(shape, X, k, tau, ell_ref):
    if shape == 'circle':
        R = sc.radius_from_polygon(X)
        Rmod = sc.modified_radius(1.5, NVAL, N, T)
        return float(abs(R - Rmod))
    Xr, kr, spl, _ = ell_ref
    h = common.h_of(X)
    b = common.error_bundle(X, k, Xr, kr, spl, h, tau, n_fine=32 * N)
    return float(b['full']['E_l2'])


def _target_times(rows, target):
    """Log-log interpolation of time vs error; None outside the data range."""
    pts = [(r['error'], r['time_median']) for r in rows
           if r.get('error') is not None and r.get('time_median') is not None
           and r['error'] > 0.0 and r['time_median'] > 0.0]
    if len(pts) < 2:
        return None
    pts.sort(key=lambda p: p[0])
    errs = np.array([p[0] for p in pts], dtype=float)
    times = np.array([p[1] for p in pts], dtype=float)
    if target < errs[0] or target > errs[-1]:
        return None
    return float(np.exp(np.interp(np.log(target), np.log(errs),
                                  np.log(times))))


def main():
    t_start = time.perf_counter()
    print('=' * 78)
    print('E9 work-precision: BDF2-Taylor vs BGN1')
    print('=' * 78)
    ell_ref = _ellipse_reference()
    payload = dict(
        experiment='E9',
        generated=common.now(),
        environment=cfg.environment_manifest(),
        setup=dict(shapes=list(SHAPES), n=NVAL, N=N, T=T, tau0=TAU0,
                   k_levels=K_LEVELS, repeats=REPEATS,
                   ellipse_ref=dict(N=ELL_NREF, tau=ELL_TAUREF,
                                    M=int(round(T / ELL_TAUREF)))),
        rows=[], time_to_target={},
    )

    for shape in SHAPES:
        for scheme in ('BDF2-Taylor', 'BGN1'):
            for k in range(K_LEVELS):
                tau_target = TAU0 / 2 ** k
                M = int(round(T / tau_target))
                tau = T / M
                row = dict(shape=shape, scheme=scheme, tau=tau, M=M,
                           tau_target=tau_target, failed=False,
                           reason='', error=None,
                           time_median=None, ms_per_step=None,
                           times=None)
                try:
                    if scheme == 'BDF2-Taylor':
                        X, kk, times = _timed_bdf2(shape, M)
                    else:
                        X, kk, times = _timed_bgn1(shape, M)
                    row['times'] = [float(v) for v in times]
                    row['time_median'] = float(np.median(times))
                    row['ms_per_step'] = float(
                        row['time_median'] / M * 1e3)
                    row['error'] = _error_with_tau(shape, X, kk, tau, ell_ref)
                except Exception as exc:               # noqa: BLE001
                    row['failed'] = True
                    row['reason'] = repr(exc)
                payload['rows'].append(row)
                print(f"{shape:7s} {scheme:12s} tau={tau:.3e} "
                      f"M={M:3d} err={row['error']} "
                      f"t={row['time_median']} "
                      f"{'FAIL' if row['failed'] else ''}")

    for shape in SHAPES:
        payload['time_to_target'][shape] = {}
        for scheme in ('BDF2-Taylor', 'BGN1'):
            sub = [r for r in payload['rows']
                   if r['shape'] == shape and r['scheme'] == scheme
                   and not r['failed']]
            payload['time_to_target'][shape][scheme] = {
                f'{target:.0e}': _target_times(sub, target)
                for target in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
            }

    payload['elapsed_s'] = float(time.perf_counter() - t_start)
    common.finish(payload, 'e9_work_precision.json')
    print(f"\nE9 done in {payload['elapsed_s']:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
