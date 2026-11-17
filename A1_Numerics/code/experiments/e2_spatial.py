"""
E2 -- spatial convergence in L2/H1/kappa and the A7 ratio R_h^m.

Setup
  shape: ellipse (a=2,b=1), n=1,2,3, T=0.5, tau=2.5e-4 (inside the observed
  quasi-uniform mesh window), N=64,128,256,512.
  reference: N_ref=2048, tau_ref=tau/4 = 6.25e-5.

Metrics
  full L2/H1  : theorem norm, includes the interpolation error rho (S3.2-S3.3)
  nodal L2/H1 : X_i - X_ref(xi_i), i.e. the discrete error e_h
  nodal kappa : kappa_i - kappa_ref(xi_i)
  R_h^m      : |e|_{L2} / (h |e|_{H1} + h^2 + tau^2)  (4.9.5)

The reference is a numerical solution; its own error is estimated by the
N=1024 vs N=2048 difference at the same tau_ref (reported as ref_floor).

Usage: python -I experiments/e2_spatial.py
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

T = 0.5
TAU = 2.5e-4
M = int(round(T / TAU))
TAU_REF = TAU / 4.0
M_REF = int(round(T / TAU_REF))
N_LEVELS = (64, 128, 256, 512)
N_REF = 2048
N_FLOOR = 1024
A7_LEVELS = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
A7_N = (256, 512)


def _one_n(n, ref_snaps, ref_spline, floor_bundle):
    rows = []
    for N in N_LEVELS:
        X0, k0 = shapes.build_geometry('ellipse', N)
        X, k, _, _, info = bt.solve_bdf2(X0, k0, T, M, n,
                                         stop_on_failure=True)
        Xr = ref_snaps['X_final']
        kr = ref_snaps['k_final']
        h = common.h_of(X)
        bundle = common.error_bundle(X, k, Xr, kr, ref_spline, h, TAU)
        rows.append(dict(N=N, h=h, tau=TAU,
                         full_L2=bundle['full']['E_l2'],
                         full_H1=bundle['full']['E_h1'],
                         nodal_L2=bundle['nodal']['E_l2'],
                         nodal_H1=bundle['nodal']['E_h1'],
                         nodal_kappa=bundle['nodal'].get('E_kappa'),
                         A7=bundle['A7'], shift=bundle['delta'],
                         psi=float(geo.mesh_ratio(X)),
                         area_ratio=float(geo.polygon_area(X) / geo.polygon_area(
                             shapes.build_shape('ellipse', N)))))
    hs = [r['h'] for r in rows]

    def block(key):
        errs = [r[key] for r in rows]
        return norms.order_summary(errs, np.array(hs[:-1]) / np.array(hs[1:]),
                                   label=key)

    out = dict(n=n, N=list(N_LEVELS), T=T, tau=TAU, M=M,
               N_ref=N_REF, tau_ref=TAU_REF, M_ref=M_REF,
               ref_floor=floor_bundle,
               rows=rows,
               orders={k: block(k) for k in
                       ('full_L2', 'full_H1', 'nodal_L2', 'nodal_H1',
                        'nodal_kappa', 'A7')})
    # A7 over time for selected N
    a7 = []
    for N in A7_N:
        X0, k0 = shapes.build_geometry('ellipse', N)
        Xv, kv, _, snaps, info = bt.solve_bdf2(
            X0, k0, T, M, n, record=False, snapshot_levels=A7_LEVELS,
            stop_on_failure=True)
        for lev in A7_LEVELS:
            if lev not in snaps:
                continue
            Xs = snaps[lev]['X']
            ks = snaps[lev]['k']
            # reference snapshot at the same physical time (level*4)
            ref_lev = lev * 4
            if ref_lev not in ref_snaps['by_level']:
                continue
            Xr2 = ref_snaps['by_level'][ref_lev]['X']
            kr2 = ref_snaps['by_level'][ref_lev]['k']
            spl2 = alignment.smooth_spline(Xr2, kr2)
            h = common.h_of(Xs)
            b = common.error_bundle(Xs, ks, Xr2, kr2, spl2, h, TAU,
                                    n_fine=32 * N)
            a7.append(dict(N=N, level=lev, t=lev * TAU,
                           A7=b['A7'], full_L2=b['full']['E_l2'],
                           full_H1=b['full']['E_h1'], h=h))
    out['A7_over_time'] = a7
    return out


def main():
    print('=' * 78)
    print('E2 spatial convergence and A7 ratio')
    print('=' * 78)
    results = []
    for n in (1, 2, 3):
        print(f'\n--- n={n}: reference N={N_REF}, M={M_REF} ---')
        t0 = __import__('time').perf_counter()
        X0r, k0r = shapes.build_geometry('ellipse', N_REF)
        ref_levels = sorted({lv * 4 for lv in A7_LEVELS})
        Xr, kr, _, ref_snaps, info = bt.solve_bdf2(
            X0r, k0r, T, M_REF, n, record=False,
            snapshot_levels=ref_levels, stop_on_failure=True)
        spl = alignment.smooth_spline(Xr, kr)
        # reference floor: N=1024 at the same tau_ref
        X0f, k0f = shapes.build_geometry('ellipse', N_FLOOR)
        Xf, kf, _, _, _ = bt.solve_bdf2(X0f, k0f, T, M_REF, n,
                                        stop_on_failure=True)
        _, df, _ = alignment.arclength_aligned_l2(Xf, Xr)
        full_f = alignment.full_errors(Xf, spl, df, n_fine=32 * N_FLOOR)
        nod_f = norms.nodal_errors(Xf, spl, df, kf)
        floor = dict(N=N_FLOOR, N_ref=N_REF, delta=df,
                     full_full_L2=full_f['E_l2'], full_full_H1=full_f['E_h1'],
                     nodal_L2=nod_f['E_l2'], nodal_H1=nod_f['E_h1'],
                     nodal_kappa=nod_f.get('E_kappa'))
        ref_snaps_wrap = dict(by_level=ref_snaps, X_final=Xr, k_final=kr)
        out = _one_n(n, ref_snaps_wrap, spl, floor)
        results.append(out)
        print(f'   elapsed {__import__("time").perf_counter()-t0:.1f}s')
        for k, v in out['orders'].items():
            print(f'   {k:12s} finest={v["finest_pair"]} fit={v["fit_slope"]:.3f} '
                  f'R2={v["fit_r2"]:.4f} orders={v["orders"]}')

    summary = {}
    for r in results:
        o = r['orders']
        summary[str(r['n'])] = dict(
            full_L2_finest=o['full_L2']['finest_pair'],
            full_H1_finest=o['full_H1']['finest_pair'],
            nodal_L2_finest=o['nodal_L2']['finest_pair'],
            nodal_H1_finest=o['nodal_H1']['finest_pair'],
            A7_min=float(min(x['A7'] for x in r['A7_over_time'])) if r['A7_over_time'] else None,
            A7_max=float(max(x['A7'] for x in r['A7_over_time'])) if r['A7_over_time'] else None,
        )
    payload = dict(experiment='E2', generated=common.now(),
                   environment=cfg.environment_manifest(),
                   setup=dict(T=T, tau=TAU, M=M, N_ref=N_REF, tau_ref=TAU_REF,
                              M_ref=M_REF, N_levels=list(N_LEVELS)),
                   results=results, summary=summary)
    common.finish(payload, 'e2_spatial.json')
    print('\nsummary:', summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
