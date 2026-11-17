"""
solver/scalar_circle.py -- exact finite-N scalar reduction of the BDF2-Taylor
scheme on a regular N-gon, plus the modified and continuum circle ODEs.

Why this module matters
-----------------------
On a regular N-gon with radius R, every node has the same curvature
    kappa = 1/(R c_N),   c_N = cos(pi/N),
and s_h = kappa.  The scheme therefore preserves the regular-polygon
ansatz exactly; the 2D linear system reduces to a scalar recurrence.  This
gives an exact reference for E0 (implementation) and E3 (time order) that
contains no interpolation or alignment error.

Two different limits must not be confused:

  * continuum ODE:       dR/dt = R^{-n} - R^{-1};
  * semi-discrete limit: dR/dt = [ (R c_N)^{-n} - (R c_N)^{-1} ] / c_N.

They differ by O(N^{-2}) = O(h^2) in the velocity.  Using the continuum ODE
as the reference on a finite N-gon leaves an O(h^2) floor; E0 measures it.

The BDF2 recurrence below uses the *predicted* geometry in the curvature
equation: kappa^{m+1} = R^{m+1}/(Rtilde^2 c_N), with Rtilde = 2 R^m - R^{m-1}.
This is the exact reduction of Scheme Spec S5.1, not of a simplified scheme.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Exact finite-N recurrence
# ---------------------------------------------------------------------------

def scalar_sequence(R0, n, N, M, T, predictor='taylor'):
    """Exact regular-N-gon recurrence of the BDF2-Taylor scheme.

    Returns dict(R=..., kappa=..., tau=..., cN=..., sdot0=..., G_half=...).
    """
    R0 = float(R0)
    n = int(n)
    N = int(N)
    M = int(M)
    tau = float(T) / M
    cN = float(np.cos(np.pi / N))

    k0 = 1.0 / (R0 * cN)
    s0 = k0

    # ---- CN startup ----
    R_half = R0 + 0.5 * tau * (s0 ** n - k0)
    if predictor == 'taylor':
        # dot s0 = kappa^2 (kappa - kappa^n) on a regular N-gon.
        sdot0 = k0 ** 2 * (k0 - k0 ** n)
        G_half = s0 ** n + 0.5 * tau * n * (s0 ** (n - 1)) * sdot0
    elif predictor == 'first':
        sdot0 = 0.0
        G_half = s0 ** n
    else:
        raise ValueError(f'unknown predictor {predictor!r}')

    # cN (R1-R0)/tau + (k1+k0)/2 - G = 0,   k1+k0 = (R1+R0)/(R_half^2 cN)
    A = cN / tau + 1.0 / (2.0 * R_half ** 2 * cN)
    B = R0 * cN / tau - R0 / (2.0 * R_half ** 2 * cN) + G_half
    R1 = B / A
    k1 = (R1 + R0) / (R_half ** 2 * cN) - k0

    Rs = [R0, R1]
    ks = [k0, k1]

    # ---- BDF2 main steps ----
    for m in range(1, M):
        Rm = Rs[-1]
        Rm1 = Rs[-2]
        R_pred = 2.0 * Rm - Rm1
        s_star = 2.0 / (Rm * cN) - 1.0 / (Rm1 * cN)

        num = (4.0 * Rm - Rm1) * cN / (2.0 * tau) - (n - 1) * s_star ** n
        den = (3.0 * cN / (2.0 * tau)
               + (1.0 - n * s_star ** (n - 1)) / (R_pred ** 2 * cN))
        Rnext = num / den
        knext = Rnext / (R_pred ** 2 * cN)
        Rs.append(Rnext)
        ks.append(knext)

    return dict(R=np.asarray(Rs), kappa=np.asarray(ks), tau=float(tau),
                cN=float(cN), sdot0=float(sdot0), G_half=float(G_half),
                R_half=float(R_half))


# ---------------------------------------------------------------------------
# ODE references
# ---------------------------------------------------------------------------

def continuum_rhs(t, R, n):
    return R ** (-float(n)) - 1.0 / R


def modified_rhs(t, R, n, cN):
    return ((R * cN) ** (-float(n)) - 1.0 / (R * cN)) / cN


def _integrate(rhs, R0, T, rtol=1e-13, atol=1e-15, dense=True):
    if T == 0.0:
        return float(R0), None
    sol = solve_ivp(rhs, (0.0, float(T)), [float(R0)], method='DOP853',
                    rtol=rtol, atol=atol, dense_output=dense)
    if not sol.success:
        raise RuntimeError(f'ODE integration failed: {sol.message}')
    return float(sol.sol(float(T))[0]), sol


def continuum_radius(R0, n, T):
    return _integrate(lambda t, R: continuum_rhs(t, R, n), R0, T)[0]


def modified_radius(R0, n, N, T):
    cN = float(np.cos(np.pi / N))
    return _integrate(lambda t, R: modified_rhs(t, R, n, cN), R0, T)[0]


def continuum_radius_curve(R0, n, T, n_pts=200):
    _, sol = _integrate(lambda t, R: continuum_rhs(t, R, n), R0, T)
    ts = np.linspace(0.0, float(T), int(n_pts))
    return ts, np.array([float(sol.sol(t)[0]) for t in ts])


def modified_radius_curve(R0, n, N, T, n_pts=200):
    cN = float(np.cos(np.pi / N))
    _, sol = _integrate(lambda t, R: modified_rhs(t, R, n, cN), R0, T)
    ts = np.linspace(0.0, float(T), int(n_pts))
    return ts, np.array([float(sol.sol(t)[0]) for t in ts])


# ---------------------------------------------------------------------------
# Regular polygon helpers
# ---------------------------------------------------------------------------

def regular_polygon(N, R):
    th = 2.0 * np.pi * np.arange(int(N)) / int(N)
    return R * np.column_stack([np.cos(th), np.sin(th)])


def regular_polygon_curvature(R, N):
    return 1.0 / (R * np.cos(np.pi / N))


def radius_from_polygon(X):
    """Mean vertex radius of a polygon about its vertex mean (regular check)."""
    V = np.asarray(X, dtype=float)
    c = V.mean(axis=0)
    return float(np.linalg.norm(V - c, axis=1).mean())


def is_regular_polygon(X, tol=1e-8):
    V = np.asarray(X, dtype=float)
    N = len(V)
    c = V.mean(axis=0)
    r = np.linalg.norm(V - c, axis=1)
    le = np.linalg.norm(np.roll(V, -1, axis=0) - V, axis=1)
    rad_spread = float((r.max() - r.min()) / r.mean()) if r.mean() > 0 else np.inf
    edge_spread = float((le.max() - le.min()) / le.mean()) if le.mean() > 0 else np.inf
    return (rad_spread <= tol) and (edge_spread <= tol), dict(
        radius_spread=rad_spread, edge_spread=edge_spread,
        radius_mean=float(r.mean()))
