"""
solver/geometry.py -- discrete polygonal geometry for the BGN framework.

Formulas follow Scheme Spec v1.2 S2.1-S2.9.  They are the same verified
formulas used by the legacy solver (A1_Legacy_BGN1_Baseline/
code/solver/bgn2_powerlaw.py); the Deep Audit v1.1 §8 independently
confirmed the sign convention:

    CCW curve, outward normal n, kappa > 0 on the unit circle,
    kappa n = -d_ss X.

The legacy file is NOT imported: this module is self-contained so that the
numerics package can be reproduced without the old BGN2 code.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def geometry(X):
    """Return (e, le, Nv, M) of a closed polygon X:(N,2).

    e_i   = X_{i+1} - X_i
    le_i  = |e_i|
    Nv_i  = 1/2 (X_{i+1} - X_{i-1})^perp          (lumped normal operator)
    M_i   = 1/2 (le_{i-1} + le_i)                 (lumped mass)
    """
    X = np.asarray(X, dtype=float)
    e = np.roll(X, -1, axis=0) - X
    le = np.linalg.norm(e, axis=1)
    s = 0.5 * (np.roll(e, 1, axis=0) + e)
    Nv = np.column_stack([s[:, 1], -s[:, 0]])       # (x,y) -> (y,-x)
    M = 0.5 * (le + np.roll(le, 1))
    return e, le, Nv, M


def stiffness_action(X, le):
    """(S X)_i = (X_i - X_{i-1})/le_{i-1} - (X_{i+1} - X_i)/le_i."""
    X = np.asarray(X, dtype=float)
    e = np.roll(X, -1, axis=0) - X
    ep = np.roll(e, 1, axis=0)
    lp = np.roll(le, 1)
    return ep / lp[:, None] - e / le[:, None]


def stiffness_coeffs(le):
    """Coefficients of the symmetric stiffness matrix S.

    S_ii = a_i + b_i, S_{i,i-1} = -a_i, S_{i,i+1} = -b_i,
    with a_i = 1/le_{i-1}, b_i = 1/le_i.
    """
    le = np.asarray(le, dtype=float)
    a = 1.0 / np.roll(le, 1)
    b = 1.0 / le
    return a, b


def initial_curvature(X):
    """kappa_i^0 = N_i . (SX)_i / |N_i|^2  (Scheme Spec S7.1)."""
    _, le, Nv, _ = geometry(X)
    SX = stiffness_action(X, le)
    den = np.sum(Nv * Nv, axis=1)
    if np.any(den <= 0.0):
        raise ValueError('degenerate vertex normal in initial_curvature')
    return np.sum(Nv * SX, axis=1) / den


def vertex_normals(Nv):
    """Outward unit vertex normals N_i/|N_i| (used by the CN startup)."""
    nrm = np.linalg.norm(Nv, axis=1)
    if np.any(nrm <= 0.0):
        raise ValueError('degenerate vertex normal')
    return Nv / nrm[:, None]


def mean_curvature(k, M):
    """s_h = sum(M_i k_i) / sum(M_i)."""
    return float(np.dot(M, k) / np.sum(M))


def gauss_bonnet_sum(k, M):
    """K_h = sum_i M_i k_i."""
    return float(np.dot(M, k))


def closure_defect(k, M):
    """D_h = |K_h - 2*pi|  (Scheme Spec S2.9)."""
    return float(abs(np.dot(M, k) - TWO_PI))


def polygon_area(X):
    X = np.asarray(X, dtype=float)
    return float(0.5 * abs(np.sum(X[:, 0] * np.roll(X[:, 1], -1)
                                   - X[:, 1] * np.roll(X[:, 0], -1))))


def signed_area(X):
    X = np.asarray(X, dtype=float)
    return float(0.5 * np.sum(X[:, 0] * np.roll(X[:, 1], -1)
                              - X[:, 1] * np.roll(X[:, 0], -1)))


def polygon_perimeter(X):
    _, le, _, _ = geometry(X)
    return float(np.sum(le))


def mesh_ratio(X):
    _, le, _, _ = geometry(X)
    return float(np.max(le) / np.min(le))


def centroid(X):
    X = np.asarray(X, dtype=float)
    x, y = X[:, 0], X[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    A = 0.5 * np.sum(cr)
    if abs(A) < 1e-14:
        return X.mean(axis=0)
    return np.array([np.sum((x + xn) * cr) / (6.0 * A),
                     np.sum((y + yn) * cr) / (6.0 * A)])


def vertex_radii(X):
    """Mean and spread of |X_i - mean(X)| (regular-polygon diagnostics)."""
    V = np.asarray(X, dtype=float)
    c = V.mean(axis=0)
    r = np.linalg.norm(V - c, axis=1)
    rm = float(r.mean())
    spread = float((r.max() - r.min()) / rm) if rm > 0 else float('nan')
    return rm, spread


def edge_spread(X):
    _, le, _, _ = geometry(X)
    return float((le.max() - le.min()) / le.mean())


def regular_polygon_curvature(R, N):
    """kappa = 1/(R cos(pi/N)) for a regular N-gon on a circle of radius R."""
    return 1.0 / (R * np.cos(np.pi / N))


def is_finite_state(X, k, denom=None, max_kappa=1e8, max_psi=1e4):
    """Basic admissibility test used by the stability experiments."""
    X = np.asarray(X, dtype=float)
    k = np.asarray(k, dtype=float)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(k)):
        return False, 'nonfinite'
    if X.shape[0] < 3:
        return False, 'too_few_vertices'
    _, le, _, _ = geometry(X)
    if np.min(le) <= 1e-14:
        return False, 'degenerate_edge'
    if np.max(le) / np.min(le) > max_psi:
        return False, 'mesh_ratio'
    if polygon_area(X) <= 1e-14:
        return False, 'nonpositive_area'
    if np.max(np.abs(k)) > max_kappa:
        return False, 'curvature_blowup'
    if denom is not None and abs(denom) < 1e-14:
        return False, 'sm_denominator'
    return True, 'ok'


def self_intersections(X, max_edges=None):
    """Number of proper crossings between non-adjacent edges (diagnostic).

    O(N^2) with vectorised per-edge tests; intended for N <= ~1024 and only
    at selected snapshots, not every time step.
    """
    X = np.asarray(X, dtype=float)
    N = len(X)
    A = X
    B = np.roll(X, -1, axis=0)
    count = 0
    for i in range(N):
        idx = np.arange(N)
        mask = (idx != i) & (idx != (i + 1) % N) & (idx != (i - 1) % N)
        j = idx[mask]
        if max_edges is not None:
            j = j[:max_edges]
        if len(j) == 0:
            continue
        p, q = A[i], B[i]
        r, s = A[j], B[j]
        d1 = (q[0] - p[0]) * (r[:, 1] - p[1]) - (q[1] - p[1]) * (r[:, 0] - p[0])
        d2 = (q[0] - p[0]) * (s[:, 1] - p[1]) - (q[1] - p[1]) * (s[:, 0] - p[0])
        d3 = (s[:, 0] - r[:, 0]) * (p[1] - r[:, 1]) - (s[:, 1] - r[:, 1]) * (p[0] - r[:, 0])
        d4 = (s[:, 0] - r[:, 0]) * (q[1] - r[:, 1]) - (s[:, 1] - r[:, 1]) * (q[0] - r[:, 0])
        cross = (d1 * d2 < 0.0) & (d3 * d4 < 0.0)
        count += int(np.count_nonzero(cross))
    return count // 2
