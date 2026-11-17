"""
solver/legacy_adapter.py -- read-only adapter to the legacy BGN1 / CNLF-EX
solver, used ONLY for the E6 prior-work comparison.

The legacy file A1_Legacy_BGN1_Baseline/code/solver/bgn2_powerlaw.py
is never modified.  Its BGN2 scheme is the old CNLF + extrapolation route;
the paper does not describe this route, and no new numerical result of the
BDF2-Taylor analysis is taken from it.
"""

from __future__ import annotations

import time

import numpy as np

import a1n_config as cfg

cfg.legacy_path_inject()

try:
    import bgn2_powerlaw as _legacy          # noqa: E402
except Exception as _exc:                    # pragma: no cover
    _legacy = None
    _legacy_import_error = _exc
else:
    _legacy_import_error = None


def available() -> bool:
    return _legacy is not None


def _require():
    if _legacy is None:
        raise RuntimeError(f'legacy solver unavailable: {_legacy_import_error}')
    return _legacy


def solve_bgn1(X0, k0, T, M, n, record=False):
    return _require().solve(X0, k0, T, int(M), n, scheme='A', order=1,
                            startup='cn', record=record)


def solve_cnlf_ex(X0, k0, T, M, n, record=False):
    return _require().solve(X0, k0, T, int(M), n, scheme='A', order=2,
                            startup='cn', record=record)


def solve_cn_taylor(X0, k0, T, M, n, record=False):
    """Legacy scheme B (CN Taylor linearization); prior-work context only."""
    return _require().solve(X0, k0, T, int(M), n, scheme='B', order=2,
                            startup='cn', record=record)


def timed_solve(fn, X0, k0, T, M, n, repeat=3, record=False):
    times = []
    out = None
    for _ in range(int(repeat)):
        t0 = time.perf_counter()
        out = fn(X0, k0, T, M, n, record=record)
        times.append(time.perf_counter() - t0)
    X, k = out[0], out[1]
    diags = out[2] if record else None
    return X, k, diags, dict(times=times, median=float(np.median(times)),
                             ms_per_step=float(np.median(times) / M * 1e3))
