"""
E3 -- temporal convergence, startup ablation, and tau-h coupling.

E3a  circle: exact finite-N modified ODE reference; tau ladder;
     both the second-order Taylor startup and the first-order predictor.
E3b  startup-only error at t=tau: order of the one-step CN startup.
E3c  ellipse: joint space-time refinement tau = 0.05 h (non-circular test);
     plus a fixed-N tau study documenting why a clean fixed-N time order is
     not resolvable (the O(h^2) spatial floor and the tau/h mesh window).
E3d  tau-h stability/coupling map on the ellipse.

Usage: python -I experiments/e3_temporal.py
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
from solver import scalar_circle as sc

T = 0.5
TAU0 = 0.1
K_LEVELS = 6
CIRCLE_N = 512
R0 = 1.5
N_VALUES = (1, 2, 3)


# ---------------------------------------------------------------------------
# E3a / E3b -- circle
# ---------------------------------------------------------------------------

def circle_time_order(n, predictor):
    rows = []
    for k in range(K_LEVELS):
        tau_target = TAU0 / 2 ** k
        M = int(round(T / tau_target))
        tau = T / M
        X0 = sc.regular_polygon(CIRCLE_N, R0)
        k0 = geo.initial_curvature(X0)
        X, k, _, _, info = bt.solve_bdf2(X0, k0, T, M, n,
                                         predictor=predictor,
                                         stop_on_failure=True)
        R = sc.radius_from_polygon(X)
        rmod = sc.modified_radius(R0, n, CIRCLE_N, T)
        rcon = sc.continuum_radius(R0, n, T)
        rows.append(dict(tau=tau, M=M, R=R, R_modified=rmod, R_continuum=rcon,
                         err_modified=abs(R - rmod) if n != 1 else 0.0,
                         err_continuum=abs(R - rcon) if n != 1 else 0.0,
                         psi=float(geo.mesh_ratio(X)),
                         failed=bool(info['failed'])))
    taus = [r['tau'] for r in rows]
    ratios = np.array(taus[:-1]) / np.array(taus[1:])
    out = dict(n=n, predictor=predictor, N=CIRCLE_N, T=T, rows=rows,
               taus=taus)
    for key in ('err_modified', 'err_continuum'):
        errs = [r[key] for r in rows]
        out[key] = norms.order_summary(errs, ratios, label=key)
    return out


def startup_order(n, predictor):
    """One-step error at t=tau against the modified ODE."""
    rows = []
    for k in range(K_LEVELS):
        tau = TAU0 / 2 ** k
        X0 = sc.regular_polygon(CIRCLE_N, R0)
        k0 = geo.initial_curvature(X0)
        X, k, _, _, info = bt.solve_bdf2(X0, k0, tau, 1, n,
                                         predictor=predictor,
                                         stop_on_failure=True)
        R = sc.radius_from_polygon(X)
        rmod = sc.modified_radius(R0, n, CIRCLE_N, tau)
        rows.append(dict(tau=tau, R=R, R_modified=rmod,
                         err=abs(R - rmod) if n != 1 else 0.0,
                         failed=bool(info['failed'])))
    taus = [r['tau'] for r in rows]
    ratios = np.array(taus[:-1]) / np.array(taus[1:])
    return dict(n=n, predictor=predictor, N=CIRCLE_N, rows=rows,
                summary=norms.order_summary([r['err'] for r in rows],
                                            ratios, label='startup_err'))


# ---------------------------------------------------------------------------
# E3c -- ellipse joint refinement and fixed-N negative result
# ---------------------------------------------------------------------------

ELL_C = 0.05
ELL_N = (64, 128, 256, 512)
ELL_NREF = 2048
ELL_TAU_REF = 1.0e-4


def build_ellipse_reference(n):
    M_ref = int(round(T / ELL_TAU_REF))
    Xr, kr, _, _, _ = bt.solve_bdf2(
        *shapes.build_geometry('ellipse', ELL_NREF), T, M_ref, n,
        stop_on_failure=True)
    return Xr, kr, alignment.smooth_spline(Xr, kr)


def ellipse_joint(n, ref):
    Xr, kr, spl = ref
    rows = []
    for N in ELL_N:
        X0, k0 = shapes.build_geometry('ellipse', N)
        h = common.h_of(X0)
        tau = ELL_C * h
        M = max(4, int(round(T / tau)))
        tau_eff = T / M
        X, k, _, _, info = bt.solve_bdf2(X0, k0, T, M, n,
                                         stop_on_failure=True)
        h_eff = common.h_of(X)
        b = common.error_bundle(X, k, Xr, kr, spl, h_eff, tau_eff,
                                n_fine=32 * N)
        rows.append(dict(N=N, h=h_eff, tau=tau_eff, M=M,
                         full_L2=b['full']['E_l2'],
                         full_H1=b['full']['E_h1'],
                         nodal_L2=b['nodal']['E_l2'],
                         nodal_H1=b['nodal']['E_h1'],
                         nodal_kappa=b['nodal'].get('E_kappa'),
                         A7=b['A7'], psi=float(geo.mesh_ratio(X))))
    hs = [r['h'] for r in rows]
    orders = {k: norms.order_summary([r[k] for r in rows],
                                     np.array(hs[:-1]) / np.array(hs[1:]), label=k)
              for k in ('full_L2', 'full_H1', 'nodal_L2', 'nodal_H1',
                        'nodal_kappa', 'A7')}
    return dict(n=n, c=ELL_C, N=list(ELL_N), N_ref=ELL_NREF,
                tau_ref=ELL_TAU_REF, M_ref=int(round(T / ELL_TAU_REF)),
                rows=rows, orders=orders)


def ellipse_fixed_N_study(n, ref, N=256):
    """Fixed N, tau ladder inside the quasi-uniform window: shows the floor."""
    Xr, kr, spl = ref
    rows = []
    for k in range(4):
        tau = 0.0025 / 2 ** k
        M = int(round(T / tau))
        X0, k0 = shapes.build_geometry('ellipse', N)
        X, k, _, _, info = bt.solve_bdf2(X0, k0, T, M, n,
                                         stop_on_failure=True)
        h = common.h_of(X)
        b = common.error_bundle(X, k, Xr, kr, spl, h, tau, n_fine=32 * N)
        rows.append(dict(tau=tau, M=M, full_L2=b['full']['E_l2'],
                         full_H1=b['full']['E_h1'],
                         nodal_L2=b['nodal']['E_l2'],
                         nodal_H1=b['nodal']['E_h1'],
                         psi=float(geo.mesh_ratio(X))))
    return dict(n=n, N=N, N_ref=ELL_NREF, tau_ref=ELL_TAU_REF, rows=rows)


# ---------------------------------------------------------------------------
# E3d -- tau-h coupling and stability map
# ---------------------------------------------------------------------------

COUPLING_N = (128, 256, 512)
COUPLING_RATIOS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)


def coupling_map(n, ref):
    Xr, kr, spl = ref
    rows = []
    for N in COUPLING_N:
        X0, k0 = shapes.build_geometry('ellipse', N)
        h = common.h_of(X0)
        for ratio in COUPLING_RATIOS:
            tau_target = ratio * h
            M = max(3, int(round(T / tau_target)))
            tau = T / M
            try:
                X, k, diags, _, info = bt.solve_bdf2(
                    X0, k0, T, M, n, record=True, stop_on_failure=False)
            except Exception as exc:                     # noqa: BLE001
                rows.append(dict(N=N, h=h, tau=tau, tau_over_h=tau / h, M=M,
                                 failed=True, reason=repr(exc),
                                 psi_max=None, area_ratio=None, err_L2=None))
                continue
            psi_max = float(max(d['mesh_ratio'] for d in diags)) if diags else float('nan')
            area_ratio = float(geo.polygon_area(X) / geo.polygon_area(X0))
            if info['failed'] or not np.all(np.isfinite(X)):
                rows.append(dict(N=N, h=h, tau=tau, tau_over_h=tau / h, M=M,
                                 failed=True, reason=info.get('reason', ''),
                                 psi_max=psi_max, area_ratio=area_ratio,
                                 err_L2=None))
                continue
            b = common.error_bundle(X, k, Xr, kr, spl, common.h_of(X), tau,
                                    n_fine=16 * N)
            rows.append(dict(N=N, h=h, tau=tau, tau_over_h=tau / h, M=M,
                             failed=False, reason='',
                             psi_max=psi_max, psi_final=float(geo.mesh_ratio(X)),
                             area_ratio=area_ratio,
                             err_L2=b['full']['E_l2'],
                             err_H1=b['full']['E_h1']))
    return dict(n=n, T=T, ratios=list(COUPLING_RATIOS), rows=rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()
    print('=' * 78)
    print('E3 temporal convergence, startup, tau-h coupling')
    print('=' * 78)
    payload = dict(experiment='E3', generated=common.now(),
                   environment=cfg.environment_manifest(),
                   setup=dict(T=T, tau0=TAU0, k_levels=K_LEVELS,
                              circle_N=CIRCLE_N, R0=R0))

    circle = []
    for n in N_VALUES:
        for pred in ('taylor', 'first'):
            r = circle_time_order(n, pred)
            circle.append(r)
            print(f"circle n={n} {pred:5s}: "
                  f"err_modified orders={r['err_modified']['orders']} "
                  f"finest={r['err_modified']['finest_pair']}")
    payload['circle_time_order'] = circle

    startup = []
    for n in N_VALUES:
        for pred in ('taylor', 'first'):
            r = startup_order(n, pred)
            startup.append(r)
            print(f"startup n={n} {pred:5s}: orders={r['summary']['orders']} "
                  f"finest={r['summary']['finest_pair']}")
    payload['startup'] = startup

    refs = {}
    for n in N_VALUES:
        t0 = time.perf_counter()
        refs[n] = build_ellipse_reference(n)
        print(f'ellipse reference n={n} built in '
              f'{time.perf_counter()-t0:.1f}s')

    joint = []
    for n in N_VALUES:
        r = ellipse_joint(n, refs[n])
        joint.append(r)
        print(f"ellipse joint n={n}: full_L2 finest="
              f"{r['orders']['full_L2']['finest_pair']} full_H1 finest="
              f"{r['orders']['full_H1']['finest_pair']}")
    payload['ellipse_joint'] = joint

    fixed = []
    for n in N_VALUES:
        r = ellipse_fixed_N_study(n, refs[n])
        fixed.append(r)
        print(f"ellipse fixed-N n={n}: L2 errs="
              f"{['%.2e' % x['full_L2'] for x in r['rows']]}")
    payload['ellipse_fixed_N'] = fixed

    coupling = []
    for n in N_VALUES:
        r = coupling_map(n, refs[n])
        coupling.append(r)
        nfail = sum(1 for x in r['rows'] if x['failed'])
        print(f"coupling n={n}: {nfail}/{len(r['rows'])} failed")
    payload['coupling_map'] = coupling

    payload['elapsed_s'] = float(time.perf_counter() - t_start)
    common.finish(payload, 'e3_temporal.json')
    print(f"\nE3 done in {payload['elapsed_s']:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
