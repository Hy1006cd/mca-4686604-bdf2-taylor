"""
E7 -- extended numerical evidence for A5.

For circle, ellipse and flower, and for nu = 1,2,3, the script measures the
A5 ratio

    R_h^m = ||e||_{L2} / ( h ||e||_{H1} + h^2 + tau^2 )

at t = 0.1, 0.25, 0.5 (and t = 1.0 for circle/ellipse), on N = 64,128,256,512
with tau = 2.5e-4.  Reference solutions use N_ref = 2048 (circle/ellipse) and
N_ref = 1024 (flower, the largest N that completes at T = 0.5); tau_ref =
6.25e-5.  A reference-floor run at N_ref/2 is reported so the ratio can be
interpreted.

All failures and unavailable late-time snapshots are recorded, not hidden.

Usage: python -I experiments/e7_a5_extension.py
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

TAU = 2.5e-4
TAU_REF = 6.25e-5
N_LEVELS = (64, 128, 256, 512)
N_VALUES = (1, 2, 3)

# Per-shape reference ladder: first entry that completes is used.
REF_LADDER = {
    'circle': (
        dict(N=2048, T=1.0, tau=TAU_REF),
        dict(N=1024, T=1.0, tau=TAU_REF),
        dict(N=1024, T=0.5, tau=TAU_REF),
    ),
    'ellipse': (
        dict(N=2048, T=1.0, tau=TAU_REF),
        dict(N=1024, T=1.0, tau=TAU_REF),
        dict(N=1024, T=0.5, tau=TAU_REF),
    ),
    'flower': (
        dict(N=1024, T=0.5, tau=TAU_REF),
        dict(N=512, T=0.5, tau=TAU_REF),
    ),
}

FULL_TIMES = (0.1, 0.25, 0.5, 1.0)
FLOWER_TIMES = (0.1, 0.25, 0.5)


def _levels(times, tau, T):
    vals = sorted({int(round(t / tau)) for t in times
                   if 0.0 < t <= T + 1e-14})
    return [v for v in vals if v >= 1]


def _one_reference(shape, n, spec, times):
    T = float(spec['T'])
    tau = float(spec['tau'])
    N = int(spec['N'])
    M = int(round(T / tau))
    levels = _levels(times, tau, T)
    X0, k0 = shapes.build_geometry(shape, N)
    X, k, _, snaps, info = bt.solve_bdf2(
        X0, k0, T, M, n, record=False, snapshot_levels=levels,
        stop_on_failure=True)
    if info.get('failed', False):
        raise RuntimeError(info.get('reason', 'reference failed'))
    return dict(N=N, T=T, tau=tau, M=M, X=X, k=k, snaps=snaps,
                levels=levels)


def _reference(shape, n, times):
    errors = []
    for spec in REF_LADDER[shape]:
        try:
            ref = _one_reference(shape, n, spec, times)
            ref['ok'] = True
            return ref, errors
        except Exception as exc:                       # noqa: BLE001
            errors.append(dict(N=spec['N'], T=spec['T'], error=repr(exc)))
    return dict(ok=False, N=None, T=None, tau=None, M=None, X=None, k=None,
                snaps={}, levels=[]), errors


def _reference_floor(shape, n, ref):
    Nf = max(32, int(ref['N']) // 2)
    T = float(ref['T'])
    tau = float(ref['tau'])
    M = int(round(T / tau))
    X0, k0 = shapes.build_geometry(shape, Nf)
    Xf, kf, _, _, info = bt.solve_bdf2(
        X0, k0, T, M, n, stop_on_failure=True)
    if info.get('failed', False):
        return dict(N=Nf, failed=True, reason=info.get('reason', ''))
    spl = alignment.smooth_spline(ref['X'], ref['k'])
    h = common.h_of(Xf)
    b = common.error_bundle(Xf, kf, ref['X'], ref['k'], spl, h, tau,
                            n_fine=32 * Nf)
    return dict(N=Nf, failed=False, h=h, tau=tau,
                full_L2=b['full']['E_l2'],
                full_H1=b['full']['E_h1'],
                nodal_L2=b['nodal']['E_l2'],
                nodal_H1=b['nodal']['E_h1'],
                A7=b['A7'])


def _coarse_rows(shape, n, ref, times):
    T = float(ref['T'])
    tau_target = TAU
    M = int(round(T / tau_target))
    tau = T / M
    levels = _levels(times, tau, T)
    # Reference splines at the same physical snapshots.  The reference and
    # coarse runs use different levels for the same time, so match by time.
    ref_list = sorted(ref['snaps'].values(), key=lambda s: s['t'])
    ref_at = {}
    for lv in levels:
        t_coarse = lv * tau
        if not ref_list:
            continue
        rr = min(ref_list, key=lambda s: abs(s['t'] - t_coarse))
        if abs(rr['t'] - t_coarse) > 0.5 * tau:
            continue
        Xr = rr['X']
        kr = rr['k']
        ref_at[lv] = dict(X=Xr, k=kr, spl=alignment.smooth_spline(Xr, kr),
                          t=rr['t'])
    rows = []
    for N in N_LEVELS:
        X0, k0 = shapes.build_geometry(shape, N)
        X, k, _, snaps, info = bt.solve_bdf2(
            X0, k0, T, M, n, record=False, snapshot_levels=levels,
            stop_on_failure=False)
        run_failed = bool(info.get('failed', False))
        h = common.h_of(X)
        for lv in levels:
            if lv not in snaps or lv not in ref_at:
                rows.append(dict(N=N, level=lv, t=(lv * tau),
                                 h=h, tau=tau,
                                 available=False, run_failed=run_failed,
                                 reason=info.get('reason', '')))
                continue
            Xs = snaps[lv]['X']
            ks = snaps[lv]['k']
            rr = ref_at[lv]
            hs = common.h_of(Xs)
            b = common.error_bundle(Xs, ks, rr['X'], rr['k'], rr['spl'],
                                    hs, tau, n_fine=32 * N)
            rows.append(dict(
                N=N, level=lv, t=float(rr['t']), h=hs, tau=tau,
                available=True, run_failed=run_failed,
                reason=info.get('reason', ''),
                full_L2=b['full']['E_l2'],
                full_H1=b['full']['E_h1'],
                nodal_L2=b['nodal']['E_l2'],
                nodal_H1=b['nodal']['E_h1'],
                A7=b['A7'],
                psi=float(geo.mesh_ratio(Xs)),
                area_ratio=float(geo.polygon_area(Xs) /
                                 geo.polygon_area(X0)),
            ))
    return rows, T, tau


def _summary(rows, times):
    avail = [r for r in rows if r.get('available') and r.get('A7') is not None]
    out = dict(n_available=len(avail), n_rows=len(rows))
    if not avail:
        out.update(a7_min=None, a7_max=None, a7_at_N256=None,
                   a7_at_N512=None, trend_512_over_256=None)
        return out
    a7 = [r['A7'] for r in avail]
    out['a7_min'] = float(min(a7))
    out['a7_max'] = float(max(a7))
    for N in (256, 512):
        vals = [r['A7'] for r in avail if r['N'] == N and r['A7'] is not None]
        out[f'a7_at_N{N}'] = float(np.mean(vals)) if vals else None
    if out['a7_at_N512'] is not None and out['a7_at_N256']:
        out['trend_512_over_256'] = float(out['a7_at_N512'] /
                                          out['a7_at_N256'])
    return out


def main():
    t_start = time.perf_counter()
    print('=' * 78)
    print('E7 extended A5 ratio')
    print('=' * 78)
    payload = dict(
        experiment='E7',
        generated=common.now(),
        environment=cfg.environment_manifest(),
        setup=dict(shapes=list(REF_LADDER), n=list(N_VALUES),
                   N_levels=list(N_LEVELS), tau=TAU, tau_ref=TAU_REF,
                   full_times=list(FULL_TIMES),
                   flower_times=list(FLOWER_TIMES),
                   reference_ladder=REF_LADDER),
        results=[],
    )
    for shape in ('circle', 'ellipse', 'flower'):
        times = FULL_TIMES if shape != 'flower' else FLOWER_TIMES
        for n in N_VALUES:
            print(f'\n--- {shape}, nu={n} ---')
            ref, ref_errors = _reference(shape, n, times)
            if not ref['ok']:
                print('  reference unavailable:', ref_errors)
                payload['results'].append(dict(
                    shape=shape, n=n, reference_ok=False,
                    reference_errors=ref_errors, rows=[], summary={}))
                continue
            floor = _reference_floor(shape, n, ref)
            rows, T, tau = _coarse_rows(shape, n, ref, times)
            summary = _summary(rows, times)
            out = dict(shape=shape, n=n, reference_ok=True,
                       N_ref=ref['N'], T_ref=ref['T'],
                       tau_ref=ref['tau'], M_ref=ref['M'],
                       levels=ref['levels'], times=list(times),
                       reference_errors=ref_errors,
                       ref_floor=floor,
                       T_coarse=T, tau_coarse=tau,
                       rows=rows, summary=summary)
            payload['results'].append(out)
            print(f"  ref N={ref['N']} T={ref['T']} tau={ref['tau']:.3e} "
                  f"floor L2={floor.get('full_L2')}")
            print(f"  A5 range={summary.get('a7_min')}..{summary.get('a7_max')} "
                  f"trend512/256={summary.get('trend_512_over_256')} "
                  f"available={summary.get('n_available')}/{summary.get('n_rows')}")

    payload['elapsed_s'] = float(time.perf_counter() - t_start)
    common.finish(payload, 'e7_a5_extension.json')
    print(f"\nE7 done in {payload['elapsed_s']:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
