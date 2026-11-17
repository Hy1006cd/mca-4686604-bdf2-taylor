"""Shared helpers for the A1_Numerics experiment drivers."""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import a1n_config as cfg
import numpy as _np

from metrics import alignment, norms, shapes
from solver import bdf2_taylor as bt
from solver import geometry as geo


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, float(time.perf_counter() - t0)


def run_solution(shape_id, N, T, M, n, predictor='taylor', record=False,
                 snapshot_levels=None, stop_on_failure=True):
    X0, k0 = shapes.build_geometry(shape_id, N)
    return bt.solve_bdf2(X0, k0, T, M, n, predictor=predictor,
                         record=record, snapshot_levels=snapshot_levels,
                         stop_on_failure=stop_on_failure)


def run_reference(shape_id, N_ref, T, M_ref, n):
    X, k, _, _, info = run_solution(shape_id, N_ref, T, M_ref, n,
                                    stop_on_failure=True)
    return X, k, info


def make_reference_spline(X_ref, k_ref=None):
    return alignment.smooth_spline(X_ref, k_ref)


def error_bundle(X, k, X_ref, k_ref, spl, h, tau, n_fine=None):
    """All E2 metrics for one coarse solution against a reference solution."""
    _, delta, _ = alignment.arclength_aligned_l2(X, X_ref)
    full = alignment.full_errors(X, spl, delta, n_fine=n_fine)
    nodal = norms.nodal_errors(X, spl, delta, k)
    return dict(delta=float(delta), full=full, nodal=nodal,
                A7=float(norms.a7_ratio(full['E_l2'], full['E_h1'], h, tau)),
                h=float(h), tau=float(tau))


def h_of(X):
    return float(geo.polygon_perimeter(X) / len(X))


def summarize(errs, ratios, label=''):
    return norms.order_summary(errs, ratios, label=label)


def finish(payload, name):
    path = cfg.save_json(name, payload)
    print(f'[saved] {path}')
    return path
