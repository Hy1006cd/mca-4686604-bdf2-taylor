"""
E3b -- one-step CN startup: curvature error at t = tau.

The E3 startup panel reports the position error |R(tau)-R_mod(tau)|.  A3 also
contains the curvature error ||kappa_h^1 - kappa(t_1)||_{L^2}.  This script
isolates that quantity on the regular-N-gon circle and reports two references:

  k_err_cont : ||kappa_h^1 - 1/R_cont(tau)||_{L^2_h}
               (the A3 quantity against the continuum solution);
  k_err_mod  : ||kappa_h^1 - 1/(R_mod(tau) c_N)||_{L^2_h}
               (against the modified ODE, isolating the time-discretization
               part; the finite-N geometry is removed).

The difference between the two is the O(h^2) finite-N offset at fixed N; it
is part of the C(h+tau^2) bound in A3, not a startup failure.

Usage: python -I experiments/e3b_startup_curvature.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import a1n_config as cfg
from experiments import common
from metrics import norms
from solver import bdf2_taylor as bt
from solver import geometry as geo
from solver import scalar_circle as sc

CIRCLE_N = 512
R0 = 1.5
TAU0 = 0.1
K_LEVELS = 6
N_VALUES = (1, 2, 3)
PREDICTORS = ('taylor', 'first')


def one_run(n, predictor, tau):
    X0 = sc.regular_polygon(CIRCLE_N, R0)
    k0 = geo.initial_curvature(X0)
    X, k, _, _, info = bt.solve_bdf2(
        X0, k0, tau, 1, n, predictor=predictor, stop_on_failure=True)

    _, le, _, M = geo.geometry(X)
    cN = float(np.cos(np.pi / CIRCLE_N))
    R = sc.radius_from_polygon(X)
    R_cont = float(sc.continuum_radius(R0, n, tau))
    R_mod = float(sc.modified_radius(R0, n, CIRCLE_N, tau))

    k_err_cont = float(np.sqrt(np.sum(M * (k - 1.0 / R_cont) ** 2)))
    k_err_mod = float(np.sqrt(np.sum(M * (k - 1.0 / (R_mod * cN)) ** 2)))
    return dict(
        tau=float(tau), M=1,
        R=float(R), R_cont=R_cont, R_mod=R_mod,
        pos_err_cont=float(abs(R - R_cont)),
        pos_err_mod=float(abs(R - R_mod)),
        k_err_cont=k_err_cont,
        k_err_mod=k_err_mod,
        psi=float(geo.mesh_ratio(X)),
        failed=bool(info.get('failed', False)),
    )


def main():
    print('=' * 78)
    print('E3b one-step startup curvature error')
    print('=' * 78)
    t_start = __import__('time').perf_counter()
    payload = dict(
        experiment='E3b',
        generated=common.now(),
        environment=cfg.environment_manifest(),
        setup=dict(shape='circle', R0=R0, N=CIRCLE_N, tau0=TAU0,
                   k_levels=K_LEVELS, n=list(N_VALUES),
                   predictors=list(PREDICTORS)),
        results=[],
    )

    for n in N_VALUES:
        for predictor in PREDICTORS:
            rows = []
            for k in range(K_LEVELS):
                tau = TAU0 / 2 ** k
                rows.append(one_run(n, predictor, tau))
            taus = [r['tau'] for r in rows]
            ratios = np.array(taus[:-1]) / np.array(taus[1:])
            summaries = {}
            for key in ('pos_err_cont', 'pos_err_mod',
                        'k_err_cont', 'k_err_mod'):
                summaries[key] = norms.order_summary(
                    [r[key] for r in rows], ratios, label=key)
            out = dict(n=n, predictor=predictor, N=CIRCLE_N, R0=R0,
                       tau0=TAU0, k_levels=K_LEVELS, rows=rows,
                       summaries=summaries)
            payload['results'].append(out)
            print(f'n={n} {predictor:5s}: '
                  f'k_cont finest={summaries["k_err_cont"]["finest_pair"]:.3f} '
                  f'k_mod finest={summaries["k_err_mod"]["finest_pair"]:.3f} '
                  f'pos_mod finest={summaries["pos_err_mod"]["finest_pair"]:.3f}')

    payload['elapsed_s'] = float(__import__('time').perf_counter() - t_start)
    common.finish(payload, 'e3b_startup_curvature.json')
    print(f'\nE3b done in {payload["elapsed_s"]:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
