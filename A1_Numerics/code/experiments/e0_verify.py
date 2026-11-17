"""
E0 -- implementation and reference-solution verification (hard gate).

Sub-experiments
  E0-a  discrete operator identities and independent geometry cross-check
  E0-b  initial curvature kappa^0 is O(h^2) on a theta-parameterised ellipse
  E0-c  finite-N scalar reduction: 2D solver == exact regular-N-gon recurrence
  E0-d  n=1: regular polygon is an exact steady state
  E0-e  modified ODE vs continuum ODE: O(h^2) (the cos(pi/N) factor)
  E0-f  arc-length alignment self-check
  E0-g  linear-solve residual and closure defect of the initial mesh
  E0-h  exact finite-N circle closure defect 2N tan(pi/N)-2pi

Usage: python -I experiments/e0_verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import a1n_config as cfg
from experiments import common
from metrics import alignment, norms, shapes
from solver import bdf2_taylor as bt
from solver import geometry as geo
from solver import scalar_circle as sc


# ---------------------------------------------------------------------------
# E0-a
# ---------------------------------------------------------------------------

def _indep_geometry(V):
    V = np.asarray(V, dtype=float)
    N = len(V)
    e = np.array([V[(i + 1) % N] - V[i] for i in range(N)])
    le = np.linalg.norm(e, axis=1)
    nrm = np.column_stack([e[:, 1], -e[:, 0]]) / le[:, None]
    Mi = np.array([0.5 * (le[(i - 1) % N] + le[i]) for i in range(N)])
    Ni = np.array([0.5 * (le[(i - 1) % N] * nrm[(i - 1) % N]
                          + le[i] * nrm[i]) for i in range(N)])
    SX = np.array([(V[i] - V[(i - 1) % N]) / le[(i - 1) % N]
                   - (V[(i + 1) % N] - V[i]) / le[i] for i in range(N)])
    return e, le, nrm, Ni, Mi, SX


def e0a():
    rows = []
    shape_Ns = {'circle': (128, 512), 'ellipse': (128, 512),
                'flower': (128, 512), 'superellipse': (128,)}
    for sid, Ns in shape_Ns.items():
        for N in Ns:
            X = shapes.build_shape(sid, N)
            e_u, le_u, Nv_u, M_u = geo.geometry(X)
            SX_u = geo.stiffness_action(X, le_u)
            e_i, le_i, nrm_i, Nv_i, M_i, SX_i = _indep_geometry(X)
            dev = max(float(np.max(np.abs(Nv_u - Nv_i))),
                      float(np.max(np.abs(M_u - M_i))),
                      float(np.max(np.abs(le_u - le_i))),
                      float(np.max(np.abs(SX_u - SX_i))))
            sumN = float(np.max(np.abs(Nv_i.sum(axis=0))))
            sumM = float(abs(M_i.sum() - le_i.sum()))
            rows.append(dict(shape=sid, N=N, geometry_dev=dev,
                             sum_Ni=sumN, sum_Mi_minus_perimeter=sumM,
                             pass_=bool(dev <= cfg.TOL['geometry_crossval']
                                        and sumN <= cfg.TOL['operator_identity']
                                        and sumM <= cfg.TOL['operator_identity'])))
    return dict(name='E0-a operator identities and independent geometry',
                items=rows, pass_=all(r['pass_'] for r in rows))


# ---------------------------------------------------------------------------
# E0-b
# ---------------------------------------------------------------------------

def e0b():
    a, b = 2.0, 1.0
    Ns = (64, 128, 256, 512, 1024)
    errs = []
    for N in Ns:
        V = shapes.ellipse_theta(N, a, b)
        k = geo.initial_curvature(V)
        th = 2.0 * np.pi * np.arange(N) / N
        kex = shapes.ellipse_analytic_curvature(th, a, b)
        M = geo.geometry(V)[3]
        errs.append(float(np.sqrt(np.sum(M * (k - kex) ** 2)
                                  / np.sum(M * kex ** 2))))
    orders = norms.observed_orders(errs)
    lo, hi = cfg.TOL['kappa0_order']
    lead = float(np.nanmean(orders[-2:]))
    return dict(name='E0-b initial curvature kappa^0, theta-ellipse',
                N=list(Ns), errors=errs,
                orders=[float(o) if np.isfinite(o) else None for o in orders],
                representative=lead,
                pass_=bool(lo <= lead <= hi))


# ---------------------------------------------------------------------------
# E0-c
# ---------------------------------------------------------------------------

def e0c():
    rows = []
    T, M = 0.2, 100
    R0 = 1.5
    for n in (1, 2, 3):
        for N in (64, 128, 256, 512):
            X0 = sc.regular_polygon(N, R0)
            k0 = geo.initial_curvature(X0)
            levels = list(range(M + 1))
            X, k, _, snaps, info = bt.solve_bdf2(
                X0, k0, T, M, n, record=False, snapshot_levels=levels,
                stop_on_failure=True)
            seq = sc.scalar_sequence(R0, n, N, M, T)
            R2d = np.array([sc.radius_from_polygon(snaps[m]['X'])
                            for m in range(M + 1)])
            rel = float(np.max(np.abs(R2d - seq['R']) / np.abs(seq['R'])))
            reg, regd = sc.is_regular_polygon(X)
            kref = seq['kappa'][-1]
            kerr = float(abs(np.mean(k) - kref) / abs(kref))
            rows.append(dict(n=n, N=N, T=T, M=M,
                             radius_rel_max=rel, kappa_rel=float(kerr),
                             regular=bool(reg), radius_spread=regd['radius_spread'],
                             edge_spread=regd['edge_spread'],
                             failed=bool(info['failed']),
                             pass_=bool(rel <= cfg.TOL['scalar_reduction_rel']
                                        and reg and not info['failed'])))
    return dict(name='E0-c 2D solver vs exact finite-N scalar reduction',
                items=rows, pass_=all(r['pass_'] for r in rows))


# ---------------------------------------------------------------------------
# E0-d
# ---------------------------------------------------------------------------

def e0d():
    rows = []
    T, M = 0.2, 100
    for N in (64, 128, 256, 512):
        X0 = sc.regular_polygon(N, 1.5)
        k0 = geo.initial_curvature(X0)
        X, k, _, _, info = bt.solve_bdf2(X0, k0, T, M, 1, stop_on_failure=True)
        dx = float(np.max(np.linalg.norm(X - X0, axis=1)))
        reg, _ = sc.is_regular_polygon(X)
        rows.append(dict(N=N, max_dx=dx, regular=bool(reg),
                         failed=bool(info['failed']),
                         pass_=bool(dx <= cfg.TOL['steady_n1_max_dx'] and reg)))
    return dict(name='E0-d n=1 regular polygon is an exact steady state',
                items=rows, pass_=all(r['pass_'] for r in rows))


# ---------------------------------------------------------------------------
# E0-e
# ---------------------------------------------------------------------------

def e0e():
    rows = []
    T = 0.5
    for n in (2, 3):
        errs = []
        for N in (64, 128, 256, 512):
            rc = sc.continuum_radius(1.5, n, T)
            rm = sc.modified_radius(1.5, n, N, T)
            errs.append(float(abs(rm - rc)))
        orders = norms.observed_orders(errs)
        lead = float(orders[-1]) if np.isfinite(orders[-1]) else float('nan')
        rows.append(dict(n=n, N=[64, 128, 256, 512], errors=errs,
                         orders=[float(o) if np.isfinite(o) else None for o in orders],
                         finest_pair=lead,
                         pass_=bool(np.isfinite(lead) and lead >= 1.8)))
    return dict(name='E0-e modified ODE vs continuum ODE (cos(pi/N) defect)',
                items=rows, pass_=all(r['pass_'] for r in rows))


# ---------------------------------------------------------------------------
# E0-f / E0-g / E0-h
# ---------------------------------------------------------------------------

def e0f():
    ok, rows = alignment.selfcheck_alignment(verbose=False)
    return dict(name='E0-f arc-length alignment self-check', items=rows,
                pass_=bool(ok))


def e0g():
    X0, k0 = shapes.build_geometry('ellipse', 256)
    X, k, diags, _, info = bt.solve_bdf2(X0, k0, 0.05, 50, 2, record=True)
    resid = [d['residual'] for d in diags if np.isfinite(d['residual'])]
    denom = [d['denominator_abs'] for d in diags if np.isfinite(d['denominator_abs'])]
    M0 = geo.geometry(X0)[3]
    return dict(name='E0-g linear residual and initial closure',
                max_residual=float(max(resid)),
                min_abs_denominator=float(min(denom)),
                D_h_initial=float(geo.closure_defect(k0, M0)),
                K_h_initial=float(np.dot(M0, k0)),
                pass_=bool(max(resid) <= cfg.TOL['residual_max']
                           and min(denom) >= cfg.TOL['denom_min']))


def e0h():
    rows = []
    for N in (64, 128, 256, 512, 1024, 2048):
        X = sc.regular_polygon(N, 1.5)
        k = geo.initial_curvature(X)
        M = geo.geometry(X)[3]
        D = geo.closure_defect(k, M)
        D_exact = 2.0 * N * np.tan(np.pi / N) - 2.0 * np.pi
        rows.append(dict(N=N, D_h=D, D_exact=float(abs(D_exact)),
                         rel_diff=float(abs(D - abs(D_exact)) / abs(D_exact)),
                         abs_diff=float(abs(D - abs(D_exact))),
                         pass_=bool(abs(D - abs(D_exact)) <= 1e-12)))
    return dict(name='E0-h exact finite-N circle closure defect',
                items=rows, pass_=all(r['pass_'] for r in rows))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print('=' * 78)
    print('E0 implementation and reference verification (hard gate)')
    print('=' * 78)
    results = []
    for fn in (e0a, e0b, e0c, e0d, e0e, e0f, e0g, e0h):
        r = fn()
        results.append(r)
        tag = 'PASS' if r['pass_'] else 'FAIL'
        print(f'\n[{tag}] {r["name"]}')
        if 'items' in r:
            for it in r['items']:
                shown = {k: (f'{v:.3e}' if isinstance(v, float) else v)
                         for k, v in it.items() if k != 'pass_'}
                print(f'   {"OK " if it.get("pass_", True) else "BAD"} {shown}')
        else:
            shown = {k: (f'{v:.3e}' if isinstance(v, float) else v)
                     for k, v in r.items() if k not in ('name', 'pass_')}
            print('   ', shown)
    all_ok = all(r['pass_'] for r in results)
    payload = dict(experiment='E0', generated=common.now(),
                   environment=cfg.environment_manifest(),
                   all_pass=bool(all_ok), results=results)
    common.finish(payload, 'e0_verification.json')
    print('\nE0 gate:', 'PASS' if all_ok else 'FAIL')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
