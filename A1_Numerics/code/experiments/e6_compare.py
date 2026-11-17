"""
E6 -- prior-work comparison and n-sensitivity (Supplementary).

E6a  circle time order of BDF2-Taylor vs legacy BGN1 / CNLF-EX.
     All three schemes share the same semi-discrete regular-N-gon limit
     (modified ODE), so the circle gives a clean time-order comparison.
E6b  stability/mesh-quality map on the ellipse and flower at n=2,3.
     The legacy routes are stable at coarse tau/h; the BDF2-Taylor scheme
     has a tighter tangential-mesh window (reported honestly).
E6c  n-sensitivity (n=1..6): error constant on the circle and ellipse.

The legacy solver is read-only and used only as a prior-work baseline;
the paper does not describe the legacy CNLF-EX scheme.

Usage: python -I experiments/e6_compare.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import a1n_config as cfg
from experiments import common
from metrics import alignment, norms, shapes
from solver import bdf2_taylor as bt
from solver import geometry as geo
from solver import legacy_adapter as legacy
from solver import scalar_circle as sc

T = 0.5
CIRCLE_N = 512
R0 = 1.5
TAU0 = 0.1
K_LEVELS = 6


# ---------------------------------------------------------------------------
# E6a
# ---------------------------------------------------------------------------

def _circle_error(scheme, n, M):
    tau = T / M
    X0 = sc.regular_polygon(CIRCLE_N, R0)
    k0 = geo.initial_curvature(X0)
    try:
        if scheme == 'BDF2-Taylor':
            X, k, _, _, info = bt.solve_bdf2(X0, k0, T, M, n,
                                             stop_on_failure=True)
        elif scheme == 'BGN1':
            out = legacy.solve_bgn1(X0, k0, T, M, n)
            X = out[0]
        elif scheme == 'CNLF-EX':
            out = legacy.solve_cnlf_ex(X0, k0, T, M, n)
            X = out[0]
        else:
            raise ValueError(scheme)
    except Exception as exc:                            # noqa: BLE001
        return float('nan'), str(exc)
    R = sc.radius_from_polygon(X)
    ref = sc.modified_radius(R0, n, CIRCLE_N, T)
    return abs(R - ref) if n != 1 else 0.0, ''


def circle_orders():
    rows = []
    schemes = ('BDF2-Taylor', 'BGN1', 'CNLF-EX')
    for n in (1, 2, 3):
        for scheme in schemes:
            errs, taus, notes = [], [], []
            for k in range(K_LEVELS):
                M = int(round(T / (TAU0 / 2 ** k)))
                tau = T / M
                e, note = _circle_error(scheme, n, M)
                errs.append(e)
                taus.append(tau)
                notes.append(note)
            ratios = np.array(taus[:-1]) / np.array(taus[1:])
            s = norms.order_summary(errs, ratios, label=scheme)
            rows.append(dict(scheme=scheme, n=n, tau=taus, errors=errs,
                             orders=s['orders'], finest_pair=s['finest_pair'],
                             fit_slope=s['fit_slope'], fit_r2=s['fit_r2'],
                             notes=notes))
            print(f"  circle {scheme:12s} n={n}: orders={s['orders']} "
                  f"finest={s['finest_pair']}")
    return rows


# ---------------------------------------------------------------------------
# E6b
# ---------------------------------------------------------------------------

STAB_RATIOS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)


def _run_scheme(scheme, X0, k0, T_, M, n):
    try:
        if scheme == 'BDF2-Taylor':
            X, k, diags, _, info = bt.solve_bdf2(
                X0, k0, T_, M, n, record=True, stop_on_failure=False)
            psi = max((d['mesh_ratio'] for d in diags), default=float('nan'))
            return X, k, bool(info.get('failed', False)), psi
        elif scheme == 'BGN1':
            out = legacy.solve_bgn1(X0, k0, T_, M, n, record=True)
            X, k, diags = out[0], out[1], out[2]
            psi = max((d['mesh_ratio'] for d in diags), default=float('nan'))
            return X, k, False, psi
        elif scheme == 'CNLF-EX':
            out = legacy.solve_cnlf_ex(X0, k0, T_, M, n, record=True)
            X, k, diags = out[0], out[1], out[2]
            psi = max((d['mesh_ratio'] for d in diags), default=float('nan'))
            return X, k, False, psi
    except Exception as exc:                            # noqa: BLE001
        return None, None, True, float('nan')
    raise ValueError(scheme)


def stability_map():
    rows = []
    for sid in ('ellipse', 'flower'):
        for n in (1, 2, 3):
            for N in (256,):
                X0, k0 = shapes.build_geometry(sid, N)
                h = common.h_of(X0)
                for scheme in ('BDF2-Taylor', 'BGN1', 'CNLF-EX'):
                    for ratio in STAB_RATIOS:
                        M = max(3, int(round(T / (ratio * h))))
                        tau = T / M
                        X, k, failed, psi = _run_scheme(scheme, X0, k0, T, M, n)
                        if X is None or not np.all(np.isfinite(X)):
                            failed = True
                        area_ratio = (geo.polygon_area(X) / geo.polygon_area(X0)
                                      if X is not None else None)
                        rows.append(dict(shape=sid, n=n, N=N, scheme=scheme,
                                         tau_over_h=tau / h, M=M, tau=tau,
                                         failed=bool(failed), psi_max=float(psi),
                                         area_ratio=area_ratio))
    return rows


# ---------------------------------------------------------------------------
# E6c
# ---------------------------------------------------------------------------

def n_sensitivity():
    rows = []
    # circle: exact modified-ODE error
    tau = 2.5e-4
    M = int(round(T / tau))
    for n in (1, 2, 3, 4, 5, 6):
        X0 = sc.regular_polygon(CIRCLE_N, R0)
        k0 = geo.initial_curvature(X0)
        X, k, _, _, info = bt.solve_bdf2(X0, k0, T, M, n,
                                         stop_on_failure=True)
        R = sc.radius_from_polygon(X)
        ref = sc.modified_radius(R0, n, CIRCLE_N, T)
        err = abs(R - ref) if n != 1 else 0.0
        h = common.h_of(X0)
        rows.append(dict(shape='circle', n=n, N=CIRCLE_N, tau=tau, h=h,
                         error=err, C_h2=err / (h * h + tau * tau)
                         if n != 1 else 0.0,
                         failed=bool(info['failed'])))
        print(f"  n-sens circle n={n}: err={err:.3e} "
              f"C={rows[-1]['C_h2']:.3e}")
    # ellipse: reference N=1024, tau_ref=1e-4
    tau_e = 1.0e-4
    M_e = int(round(T / tau_e))
    X0r, k0r = shapes.build_geometry('ellipse', 1024)
    for n in (1, 2, 3, 4, 5, 6):
        Xr, kr, _, _, _ = bt.solve_bdf2(X0r, k0r, T, M_e, n,
                                        stop_on_failure=True)
        spl = alignment.smooth_spline(Xr, kr)
        X0, k0 = shapes.build_geometry('ellipse', 256)
        X, k, _, _, info = bt.solve_bdf2(X0, k0, T, M_e, n,
                                         stop_on_failure=True)
        h = common.h_of(X)
        b = common.error_bundle(X, k, Xr, kr, spl, h, tau_e, n_fine=32 * 256)
        rows.append(dict(shape='ellipse', n=n, N=256, tau=tau_e, h=h,
                         error=b['full']['E_l2'],
                         C_h2=b['full']['E_l2'] / (h * h + tau_e * tau_e),
                         failed=bool(info['failed'])))
        print(f"  n-sens ellipse n={n}: err={b['full']['E_l2']:.3e} "
              f"C={rows[-1]['C_h2']:.3e}")
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t0 = time.perf_counter()
    print('=' * 78)
    print('E6 prior-work comparison and n-sensitivity')
    print('=' * 78)
    payload = dict(experiment='E6', generated=common.now(),
                   environment=cfg.environment_manifest(),
                   legacy_available=bool(legacy.available()),
                   setup=dict(T=T, circle_N=CIRCLE_N, R0=R0, tau0=TAU0,
                              stab_ratios=list(STAB_RATIOS)))
    if legacy.available():
        print('\n[E6a circle time order]')
        payload['circle_orders'] = circle_orders()
        print('\n[E6b stability map]')
        payload['stability_map'] = stability_map()
    else:
        payload['circle_orders'] = []
        payload['stability_map'] = []
        print('legacy solver unavailable; E6a/E6b skipped')
    print('\n[E6c n-sensitivity]')
    payload['n_sensitivity'] = n_sensitivity()
    payload['elapsed_s'] = float(time.perf_counter() - t0)
    common.finish(payload, 'e6_comparison.json')
    print(f'\nE6 done in {payload["elapsed_s"]:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
