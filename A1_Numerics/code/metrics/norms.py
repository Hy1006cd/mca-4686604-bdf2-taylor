"""
metrics/norms.py -- error norms and order estimates.

Primary norms (Scheme Spec S3.2-S3.6, Proof Chain 1.9-1.11):
    E_L2   = sqrt( sum_i M_i |X_i - Y_i|^2 )
    E_H1   = sqrt( sum_i M_i |d_i|^2 + sum_i h_i |(d_{i+1}-d_i)/h_i|^2 )
    E_kappa= sqrt( sum_i M_i (k_i - k_ref_i)^2 )
    E_energy = E_H1 + E_kappa       (L0 = 1; equivalent up to sqrt(2))
    R_h    = E_L2 / (h E_H1 + h^2 + tau^2)   (A7 ratio, (4.9.5))
"""

from __future__ import annotations

import numpy as np

from solver import geometry as geo


def l2_error(X, Y, M=None):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if M is None:
        M = geo.geometry(X)[3]
    return float(np.sqrt(np.sum(M * np.sum((X - Y) ** 2, axis=1))))


def h1_error(X, Y, M=None, le=None):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if M is None or le is None:
        _, le0, _, M0 = geo.geometry(X)
        M = le0 if False else (M0 if M is None else M)
        le = le0 if le is None else le
    d = X - Y
    mass_term = np.sum(M * np.sum(d ** 2, axis=1))
    dd = np.roll(d, -1, axis=0) - d
    grad_term = np.sum(le * np.sum((dd / le[:, None]) ** 2, axis=1))
    return float(np.sqrt(mass_term + grad_term))


def curvature_error(k, k_ref, M=None):
    k = np.asarray(k, dtype=float)
    k_ref = np.asarray(k_ref, dtype=float)
    if M is None:
        # caller must supply weights for a meaningful value
        M = np.ones_like(k)
    return float(np.sqrt(np.sum(M * (k - k_ref) ** 2)))


def energy_error(X, Y, k, k_ref, M=None, le=None):
    e_l2 = l2_error(X, Y, M)
    e_h1 = h1_error(X, Y, M, le)
    e_k = curvature_error(k, k_ref, M)
    return dict(E_l2=e_l2, E_H1=e_h1, E_kappa=e_k,
                E_energy=e_h1 + e_k,
                E_energy_l2=float(np.sqrt(e_h1 ** 2 + e_k ** 2)))


def a7_ratio(e_l2, e_h1, h, tau):
    den = h * e_h1 + h * h + tau * tau
    return float(e_l2 / den) if den > 0 else float('nan')


def closure_defect(k, M):
    return float(abs(np.dot(M, k) - geo.TWO_PI))


def observed_orders(errs, ratios=None):
    e = np.asarray(errs, dtype=float)
    if ratios is None:
        ratios = np.full(len(e) - 1, 2.0)
    ratios = np.asarray(ratios, dtype=float)
    out = []
    for i in range(len(e) - 1):
        if e[i] <= 0 or e[i + 1] <= 0 or not np.isfinite(e[i]) or not np.isfinite(e[i + 1]):
            out.append(float('nan'))
        else:
            out.append(float(np.log(e[i] / e[i + 1]) / np.log(ratios[i])))
    return out


def fit_powerlaw(x, y):
    """log-log fit y = a x^b; returns (b, a, R^2, n_pts)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return float('nan'), float('nan'), float('nan'), int(len(x))
    lx, ly = np.log(x), np.log(y)
    b, a = np.polyfit(lx, ly, 1)
    pred = a + b * lx
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(b), float(np.exp(a)), float(r2), int(len(x))


def finest_pair_order(errs, ratios=None):
    o = observed_orders(errs, ratios)
    return float(o[-1]) if o and np.isfinite(o[-1]) else float('nan')


def order_summary(errs, ratios=None, label=''):
    o = observed_orders(errs, ratios)
    x = np.arange(len(errs), dtype=float)
    if ratios is None:
        ratios = np.full(len(errs) - 1, 2.0)
    # log-log fit against the actual h/tau ladder (cumulative ratios)
    levels = np.concatenate([[1.0], np.cumprod(np.asarray(ratios, dtype=float))])
    b, a, r2, npts = fit_powerlaw(levels, errs)
    return dict(label=label, errors=[float(e) for e in errs],
                orders=[float(v) if np.isfinite(v) else None for v in o],
                finest_pair=finest_pair_order(errs, ratios),
                fit_slope=float(b), fit_r2=float(r2), fit_n=int(npts))


def relative_error(err, scale):
    return float(err / scale) if scale > 0 else float('nan')


def nodal_errors(X, spl, delta, k=None, M=None, le=None):
    """Nodal (discrete) errors against a smooth spline reference.

    E_l2/E_h1 use the nodal values X_i - X_ref(xi_i); this measures the
    scheme's discrete error e_h, not the full error e = e_h + rho.  E_kappa
    uses the spline curvature at the same fractions.
    """
    from metrics import alignment
    X = np.asarray(X, dtype=float)
    if M is None or le is None:
        _, le0, _, M0 = geo.geometry(X)
        M = M0 if M is None else M
        le = le0 if le is None else le
    frac, _ = alignment._fractions(X, M)
    Y = alignment.eval_spline(spl, frac, delta)
    out = dict(E_l2=l2_error(X, Y, M), E_h1=h1_error(X, Y, M, le))
    if k is not None and spl.get('fv') is not None:
        k_ref = alignment.eval_spline(spl, frac, delta, values=True)
        out['E_kappa'] = curvature_error(k, k_ref, M)
    return out
