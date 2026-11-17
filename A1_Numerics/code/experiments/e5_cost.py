"""
E5 -- cost, Sherman-Morrison denominator, conditioning, and residuals.

Cost panel
  BDF2-Taylor (this package) vs legacy BGN1 and CNLF-EX.
  N=128,256,512,1024; T=0.2; tau=0.01; n=2; median of 3 runs.
  The new scheme uses one sparse LU factorization and two back-substitutions
  per step; the legacy schemes use one sparse solve per step.

Denominator/conditioning panel
  Ellipse; N=64..512; n=1,2,3; T=0.5; tau=2.5e-4.
  Reports min/max |1 + v^T z| and a 1-norm condition estimate of the actual
  perturbed matrix A = A0 + u v^T via onenormest (small N also dense cond).

Usage: python -I experiments/e5_cost.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, onenormest, splu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import a1n_config as cfg
from experiments import common
from metrics import shapes
from solver import bdf2_taylor as bt
from solver import geometry as geo
from solver import legacy_adapter as legacy

T_COST = 0.2
TAU_COST = 0.01
N_COST = (128, 256, 512, 1024)
REPEATS = 3


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------

def cost_new(N=256, n=2, repeats=REPEATS):
    X0, k0 = shapes.build_geometry('ellipse', N)
    M = int(round(T_COST / TAU_COST))
    times = []
    X = k = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        X, k, _, _, _ = bt.solve_bdf2(X0, k0, T_COST, M, n,
                                      stop_on_failure=True)
        times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    tau = T_COST / M
    X1, k1, _ = bt.startup_cn(X0, k0, tau, n)
    nnz = int(bt.assemble_bdf2(X1, X0, k1, k0, tau, n)[0].nnz)
    return dict(scheme='BDF2-Taylor', N=N, M=M, tau=tau,
                times=times, median_s=med, ms_per_step=med / M * 1e3,
                factorizations_per_step=1.0, back_solves_per_step=2.0,
                nnz=nnz)


def cost_legacy(fn, label, N=256, n=2, repeats=REPEATS):
    X0, k0 = shapes.build_geometry('ellipse', N)
    M = int(round(T_COST / TAU_COST))
    X, k, _, tm = legacy.timed_solve(fn, X0, k0, T_COST, M, n,
                                     repeat=repeats, record=False)
    return dict(scheme=label, N=N, M=M, tau=T_COST / M,
                times=tm['times'], median_s=tm['median'],
                ms_per_step=tm['ms_per_step'],
                factorizations_per_step=1.0, back_solves_per_step=1.0,
                nnz=None)


# ---------------------------------------------------------------------------
# denominator / condition
# ---------------------------------------------------------------------------

def _condition_estimate(A0, rhs, u, v, dense=False):
    lu = splu(A0.tocsc())
    luT = splu(A0.T.tocsc())
    nz_u = np.max(np.abs(u)) > 0
    z = lu.solve(u) if nz_u else np.zeros_like(u)
    w = luT.solve(v) if nz_u else np.zeros_like(v)
    denom = 1.0 + float(np.dot(v, z)) if nz_u else 1.0
    n = A0.shape[0]

    def _outer(a, s, W):
        s = np.asarray(s)
        return a * s if W.ndim == 1 else np.outer(a, s)

    def matA(W):
        W = np.asarray(W)
        return A0 @ W + _outer(u, v @ W, W)

    def rmatA(W):
        W = np.asarray(W)
        return A0.T @ W + _outer(v, u @ W, W)

    if nz_u:
        def matAinv(W):
            W = np.asarray(W)
            Y = lu.solve(W)
            return Y - _outer(z, (v @ Y) / denom, W)

        def rmatAinv(W):
            W = np.asarray(W)
            Y = luT.solve(W)
            return Y - _outer(w, (z @ Y) / denom, W)
    else:
        def matAinv(W):
            return lu.solve(np.asarray(W))

        def rmatAinv(W):
            return luT.solve(np.asarray(W))

    opA = LinearOperator((n, n), matvec=matA, rmatvec=rmatA)
    opAinv = LinearOperator((n, n), matvec=matAinv, rmatvec=rmatAinv)
    normA = float(onenormest(opA))
    normInv = float(onenormest(opAinv))
    out = dict(cond_estimate=normA * normInv, norm_A=normA,
               norm_inv=normInv, denom=float(denom), n=int(n))
    if dense:
        Ad = A0.toarray() + np.outer(u, v)
        out['cond_dense_1'] = float(np.linalg.cond(Ad, 1))
        out['cond_dense_2'] = float(np.linalg.cond(Ad, 2))
    return out


def denominator_panel(n, N):
    T = 0.5
    tau = 2.5e-4
    M = int(round(T / tau))
    X0, k0 = shapes.build_geometry('ellipse', N)
    X, k, diags, _, info = bt.solve_bdf2(X0, k0, T, M, n, record=True,
                                         stop_on_failure=True)
    dvals = [d['denominator_abs'] for d in diags
             if np.isfinite(d['denominator_abs'])]
    sstar = [d['s_star'] for d in diags if np.isfinite(d['s_star'])]
    res = [d['residual'] for d in diags if np.isfinite(d['residual'])]
    # first-step condition estimate
    X1, k1, _ = bt.startup_cn(X0, k0, tau, n)
    A0, rhs, u, v, meta = bt.assemble_bdf2(X1, X0, k1, k0, tau, n)
    cond = _condition_estimate(A0, rhs, u, v, dense=(N <= 128))
    return dict(n=n, N=N, T=T, tau=tau, M=M,
                denom_min=float(min(dvals)), denom_max=float(max(dvals)),
                denom_median=float(np.median(dvals)),
                s_star_min=float(min(sstar)), s_star_max=float(max(sstar)),
                residual_max=float(max(res)),
                first_step=cond,
                psi_final=float(geo.mesh_ratio(X)))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    t0 = time.perf_counter()
    print('=' * 78)
    print('E5 cost, Sherman-Morrison denominator, conditioning')
    print('=' * 78)
    payload = dict(experiment='E5', generated=common.now(),
                   environment=cfg.environment_manifest(),
                   setup=dict(T_cost=T_COST, tau_cost=TAU_COST,
                              N_cost=list(N_COST), repeats=REPEATS,
                              denom_T=0.5, denom_tau=2.5e-4))

    print('\n[cost]')
    cost = []
    for N in N_COST:
        r = cost_new(N)
        cost.append(r)
        print(f"  BDF2-Taylor N={N:4d} {r['ms_per_step']:.2f} ms/step "
              f"({r['median_s']:.2f}s, nnz={r['nnz']})")
    if legacy.available():
        for fn, label in ((legacy.solve_bgn1, 'BGN1'),
                          (legacy.solve_cnlf_ex, 'CNLF-EX')):
            for N in N_COST:
                r = cost_legacy(fn, label, N)
                cost.append(r)
                print(f"  {label:10s} N={N:4d} {r['ms_per_step']:.2f} ms/step "
                      f"({r['median_s']:.2f}s)")
    else:
        print('  legacy solver unavailable; skipped legacy cost rows')
    payload['cost'] = cost

    print('\n[denominator / condition]')
    rows = []
    for n in (1, 2, 3):
        for N in (64, 128, 256, 512):
            r = denominator_panel(n, N)
            rows.append(r)
            print(f"  n={n} N={N:4d} min|d|={r['denom_min']:.6f} "
                  f"cond~{r['first_step']['cond_estimate']:.2f} "
                  f"resid={r['residual_max']:.2e} psi={r['psi_final']:.2f}")
    payload['denominator_condition'] = rows
    payload['elapsed_s'] = float(time.perf_counter() - t0)
    common.finish(payload, 'e5_cost.json')
    print(f'\nE5 done in {payload["elapsed_s"]:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
