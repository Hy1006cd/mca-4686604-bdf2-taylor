"""
metrics/alignment.py -- common arc-length parameterization and shift search.

Implements the measurement side of Scheme Spec S3.2-S3.3.  The BGN scheme
allows tangential node motion, so a direct vertex comparison is meaningless;
both the discrete solution and the reference are compared as functions of
the arc-length fraction, with an optimal constant shift.

The shift search follows the verified legacy implementation
(A1_Legacy_BGN1_Baseline/code/experiments_code/core/metrics.py):
a coarse grid scan followed by slope bisection, which is robust for the
V-shaped error profile near the optimum.
"""

from __future__ import annotations

import numpy as np


def cumulative_arclength(V):
    V = np.asarray(V, dtype=float)
    P = np.vstack([V, V[:1]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return cum, seg, P


def resample_at_fractions(V, frac):
    """Sample the closed polyline V at arc-length fractions frac in [0,1)."""
    V = np.asarray(V, dtype=float)
    cum, seg, P = cumulative_arclength(V)
    total = cum[-1]
    targets = np.asarray(frac, dtype=float) * total
    n = len(seg)
    safe = np.where(seg > 0.0, seg, 1.0)
    j = np.clip(np.searchsorted(cum, targets, side='right') - 1, 0, n - 1)
    t = (targets - cum[j]) / safe[j]
    return P[j] + t[:, None] * (P[j + 1] - P[j])


def resample_scalar_at_fractions(V, values, frac):
    """Linearly interpolate vertex values along the closed polyline V."""
    V = np.asarray(V, dtype=float)
    values = np.asarray(values, dtype=float)
    cum, seg, _ = cumulative_arclength(V)
    total = cum[-1]
    targets = np.asarray(frac, dtype=float) * total
    n = len(seg)
    safe = np.where(seg > 0.0, seg, 1.0)
    j = np.clip(np.searchsorted(cum, targets, side='right') - 1, 0, n - 1)
    t = (targets - cum[j]) / safe[j]
    v0 = values[j]
    v1 = values[(j + 1) % len(values)]
    return v0 + t * (v1 - v0)


def _fractions(X, M):
    cum, _, _ = cumulative_arclength(X)
    return cum[:-1] / cum[-1], M / np.sum(M)


def _weighted_l2(X, Y, w):
    return float(np.sqrt(np.sum(w * np.sum((X - Y) ** 2, axis=1))))


def arclength_aligned_l2(X, Y, M=None, n_shift=256, refine=True,
                         return_shift=True):
    """min_delta || X - Y(frac_X + delta) ||_M.

    Parameters
    ----------
    X : (N,2) target polygon (defines the arc-length fractions and weights)
    Y : (Nref,2) reference polygon
    M : lumped masses of X (default: geometry(X)[3])

    Returns
    -------
    E, delta, Y_aligned
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if M is None:
        from solver.geometry import geometry
        M = geometry(X)[3]
    frac, w = _fractions(X, M)

    best_e = np.inf
    best_d = 0.0
    for k in range(int(n_shift)):
        d = k / float(n_shift)
        Ya = resample_at_fractions(Y, (frac + d) % 1.0)
        e = _weighted_l2(X, Ya, w)
        if e < best_e:
            best_e, best_d = e, d

    if refine:
        h = 1e-10
        lo = best_d - 1.0 / n_shift
        hi = best_d + 1.0 / n_shift

        def _err(d):
            Ya = resample_at_fractions(Y, (frac + float(d)) % 1.0)
            return _weighted_l2(X, Ya, w)

        for _ in range(160):
            if hi - lo < 1e-15:
                break
            mid = 0.5 * (lo + hi)
            if _err(mid - h) < _err(mid + h):
                hi = mid
            else:
                lo = mid
        for cand in (0.5 * (lo + hi), lo, hi, best_d):
            e = _err(cand)
            if e < best_e:
                best_e, best_d = e, cand % 1.0

    best_d = best_d % 1.0
    Y_aligned = resample_at_fractions(Y, (frac + best_d) % 1.0)
    if return_shift:
        return float(best_e), float(best_d), Y_aligned
    return float(best_e), Y_aligned


def sample_reference(X, Y_ref, k_ref, delta, M=None):
    """Return reference positions and curvature at X's vertices + delta."""
    X = np.asarray(X, dtype=float)
    if M is None:
        from solver.geometry import geometry
        M = geometry(X)[3]
    frac, _ = _fractions(X, M)
    fr = (frac + float(delta)) % 1.0
    Y = resample_at_fractions(Y_ref, fr)
    k = resample_scalar_at_fractions(Y_ref, k_ref, fr)
    return Y, k


# ---------------------------------------------------------------------------
# Smooth reference: periodic cubic spline through the reference nodes
# ---------------------------------------------------------------------------
#
# A comparison against the *piecewise linear* reference polyline leaves an
# O(h_ref) sawtooth in the H1 norm (the reference tangent is only O(h_ref)
# accurate).  A periodic cubic spline through the reference nodes removes
# that artefact while keeping the reference's own scheme error (which is
# O(h_ref^2) in L2 and O(h_ref) in H1).  This is essential for measuring the
# theorem's full H1 error on non-circular shapes.

def smooth_spline(V, values=None):
    """Periodic cubic spline through the closed polyline V.

    Returns dict(fx, fy, fv, total, s) where fx/fy/fv are CubicSpline objects
    parameterised by the polyline arc length s in [0, total].
    """
    from scipy.interpolate import CubicSpline
    V = np.asarray(V, dtype=float)
    cum, _, _ = cumulative_arclength(V)
    s = cum[:-1]
    total = float(cum[-1])
    V2 = np.vstack([V, V[:1]])
    s2 = np.append(s, total)
    fx = CubicSpline(s2, V2[:, 0], bc_type='periodic')
    fy = CubicSpline(s2, V2[:, 1], bc_type='periodic')
    fv = None
    if values is not None:
        v = np.asarray(values, dtype=float)
        v2 = np.append(v, v[0])
        fv = CubicSpline(s2, v2, bc_type='periodic')
    return dict(fx=fx, fy=fy, fv=fv, total=total, s=s)


def eval_spline(spl, frac, shift=0.0, values=False):
    """Evaluate the spline at arc-length fractions + shift."""
    frac = np.asarray(frac, dtype=float)
    t = ((frac + float(shift)) % 1.0) * spl['total']
    if values:
        if spl['fv'] is None:
            raise ValueError('spline has no vertex values')
        return np.asarray(spl['fv'](t), dtype=float)
    return np.column_stack([spl['fx'](t), spl['fy'](t)])


def full_errors(X, spl, delta, n_fine=None):
    """Full (interpolation-included) L2 and H1 errors against a smooth spline.

    The discrete solution X is evaluated through its own piecewise linear
    interpolant at n_fine uniform arc-length fractions; the reference is the
    smooth spline at the same fractions + delta.  This is the theorem's norm
    (including the interpolation error rho), unlike the nodal error.

    Returns dict(E_l2, E_h1, n_fine, ds).
    """
    X = np.asarray(X, dtype=float)
    if n_fine is None:
        n_fine = max(4096, 32 * len(X))
    xi = np.arange(int(n_fine), dtype=float) / float(n_fine)
    Xh = resample_at_fractions(X, xi)
    Y = eval_spline(spl, xi, delta)
    d = Xh - Y
    ds = spl['total'] / float(n_fine)
    dxi = np.roll(d, -1, axis=0) - d
    mass = np.sum(np.sum(d ** 2, axis=1))
    grad = np.sum(np.sum((dxi / ds) ** 2, axis=1))
    return dict(E_l2=float(np.sqrt(mass * ds)),
                E_h1=float(np.sqrt((mass + grad) * ds)),
                n_fine=int(n_fine), ds=float(ds))


def selfcheck_alignment(verbose=False):
    """Identity and cyclic-shift checks (must return ~machine precision)."""
    from metrics.shapes import circle
    rows = []
    for N in (64, 128, 257):
        X = circle(N, 1.0)
        E, d, _ = arclength_aligned_l2(X, X.copy())
        rows.append(dict(N=N, test='identity', value=E,
                         pass_=E <= 1e-13, shift=d))
        Y = np.roll(X, max(1, N // 7), axis=0)
        E, d, _ = arclength_aligned_l2(X, Y)
        rows.append(dict(N=N, test='cyclic_shift', value=E,
                         pass_=E <= 1e-12, shift=d))
    if verbose:
        for r in rows:
            print(f"  [{'PASS' if r['pass_'] else 'FAIL'}] N={r['N']} "
                  f"{r['test']}: {r['value']:.3e} (shift={r['shift']:.3e})")
    return all(r['pass_'] for r in rows), rows
