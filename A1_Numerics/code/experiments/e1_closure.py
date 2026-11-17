"""
E1 -- discrete Gauss-Bonnet closure defect D_h and the exponent q (A5).

Also reports ||kappa_h||_inf (A5b), s_h, and the evolved-shape diagnostics.
The t=T values use the BDF2-Taylor solver with n=2; the t=0 values use the
initial discrete curvature kappa^0.

Usage: python -I experiments/e1_closure.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import a1n_config as cfg
from experiments import common
from metrics import norms, shapes
from solver import bdf2_taylor as bt
from solver import geometry as geo


def run_shape(sid):
    Ns = (64, 128, 256, 512, 1024, 2048)
    T = 0.5
    tau = 1.0e-4
    M = int(round(T / tau))
    rows = []
    for N in Ns:
        X0, k0 = shapes.build_geometry(sid, N)
        M0 = geo.geometry(X0)[3]
        le0 = geo.geometry(X0)[1]
        h0 = float(np.mean(le0))
        D0 = geo.closure_defect(k0, M0)
        k0inf = float(np.max(np.abs(k0)))
        s0 = geo.mean_curvature(k0, M0)
        try:
            X, k, _, _, info = bt.solve_bdf2(X0, k0, T, M, 2,
                                             stop_on_failure=True)
        except bt.SolverFailure as exc:
            rows.append(dict(
                N=N, h=h0, h_final=float('nan'),
                D_h_0=float(D0), D_h_T=None,
                D_over_L_0=float(D0 / np.sum(M0)), D_over_L_T=None,
                kappa_inf_0=k0inf, kappa_inf_T=None,
                s_h_0=float(s0), s_h_T=None,
                psi_T=None, area_ratio=None,
                failed=True, reason=str(exc),
            ))
            continue
        MT = geo.geometry(X)[3]
        leT = geo.geometry(X)[1]
        hT = float(np.mean(leT))
        DT = geo.closure_defect(k, MT)
        kTinf = float(np.max(np.abs(k)))
        sT = geo.mean_curvature(k, MT)
        rows.append(dict(
            N=N, h=h0, h_final=hT, D_h_0=float(D0), D_h_T=float(DT),
            D_over_L_0=float(D0 / np.sum(M0)),
            D_over_L_T=float(DT / np.sum(MT)),
            kappa_inf_0=k0inf, kappa_inf_T=kTinf,
            s_h_0=float(s0), s_h_T=float(sT),
            psi_T=float(geo.mesh_ratio(X)),
            area_ratio=float(geo.polygon_area(X) / geo.polygon_area(X0)),
            failed=bool(info['failed']),
        ))
    hs = [r['h'] for r in rows]
    Ds0 = [r['D_h_0'] for r in rows]
    okT = [r for r in rows if r.get('D_h_T') is not None
           and np.isfinite(r.get('h_final', float('nan')))]
    hsT = [r['h_final'] for r in okT]
    DsT = [r['D_h_T'] for r in okT]

    def block(hs_, Ds_, from_N=None):
        if len(hs_) < 3:
            return dict(errors=Ds_, orders=[], finest_pair=None,
                        fit_all=dict(slope=None, intercept=None, r2=None, n=len(hs_)),
                        fit_asymptotic=dict(slope=None, intercept=None, r2=None,
                                            n=len(hs_), from_N=None),
                        available=False)
        ratios = np.array(hs_[:-1]) / np.array(hs_[1:])
        orders = norms.observed_orders(Ds_, ratios)
        b, a, r2, npts = norms.fit_powerlaw(hs_, Ds_)
        k0 = 2 if len(hs_) > 3 else 1
        b2, a2, r22, npts2 = norms.fit_powerlaw(hs_[k0:], Ds_[k0:])
        return dict(errors=Ds_,
                    orders=[float(o) if np.isfinite(o) else None for o in orders],
                    finest_pair=float(orders[-1]) if np.isfinite(orders[-1]) else None,
                    fit_all=dict(slope=float(b), intercept=float(a),
                                 r2=float(r2), n=int(npts)),
                    fit_asymptotic=dict(slope=float(b2), intercept=float(a2),
                                        r2=float(r22), n=int(npts2),
                                        from_N=from_N[k0] if from_N else None),
                    available=True)
    return dict(shape=sid, N=list(Ns), T=T, tau=tau, M=M, rows=rows,
                n_failed_T=int(sum(1 for r in rows if r.get('failed'))),
                q_t0=block(hs, Ds0, Ns),
                q_T=block(hsT, DsT, [r['N'] for r in okT]))


def main():
    print('=' * 78)
    print('E1 closure defect D_h and exponent q (A5)')
    print('=' * 78)
    results = []
    for sid in ('circle', 'ellipse', 'flower'):
        r = run_shape(sid)
        results.append(r)
        print(f'\n--- {sid} ---')
        for qname in ('q_t0', 'q_T'):
            q = r[qname]
            print(f'  {qname}: orders={q["orders"]} finest={q["finest_pair"]} '
                  f'fit_all={q["fit_all"]["slope"]:.3f} '
                  f'fit_asym={q["fit_asymptotic"]["slope"]:.3f} '
                  f'R2={q["fit_asymptotic"]["r2"]:.4f}')
        for row in r['rows']:
            if row.get('failed'):
                print(f"   N={row['N']:5d} FAILED at T: {row.get('reason','')[:80]}")
                continue
            print(f"   N={row['N']:5d} D0={row['D_h_0']:.3e} DT={row['D_h_T']:.3e} "
                  f"k_inf={row['kappa_inf_T']:.3f} psi={row['psi_T']:.2f}")

    def _pass(q):
        if not q.get('available', False) or q.get('finest_pair') is None:
            return False
        fp = q['finest_pair']
        fa = q['fit_asymptotic']
        return (fp is not None and cfg.TOL['q_order'][0] <= fp <= cfg.TOL['q_order'][1]
                and fa['slope'] is not None
                and cfg.TOL['q_fit'][0] <= fa['slope'] <= cfg.TOL['q_fit'][1]
                and fa['r2'] is not None and fa['r2'] >= cfg.TOL['fit_r2_min'])

    summary = {}
    for r in results:
        summary[r['shape']] = dict(
            q_t0_pass=bool(_pass(r['q_t0'])), q_T_pass=bool(_pass(r['q_T'])),
            q_t0_finest=r['q_t0']['finest_pair'],
            q_T_finest=r['q_T']['finest_pair'],
            q_t0_fit=r['q_t0']['fit_asymptotic']['slope'],
            q_T_fit=r['q_T']['fit_asymptotic']['slope'],
        )
    all_ok = all(v['q_t0_pass'] and v['q_T_pass'] for v in summary.values())
    payload = dict(experiment='E1', generated=common.now(),
                   environment=cfg.environment_manifest(),
                   results=results, summary=summary,
                   all_pass=bool(all_ok))
    common.finish(payload, 'e1_closure.json')
    print('\nE1:', 'PASS' if all_ok else 'CHECK REPORTED VALUES')
    print('summary:', summary)
    return 0 if all_ok else 0     # negative results are still reported


if __name__ == '__main__':
    sys.exit(main())
