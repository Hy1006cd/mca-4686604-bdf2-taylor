"""
solver/bdf2_taylor.py -- BDF2 + second-order Taylor linearization + rank-one solve.

Implements Scheme Spec v1.2:
    S5.1  BDF2 system on the predicted geometry Gamma-tilde^{m+1}
    S6.1-S6.2  s_*^{m+1} = 2 s_h^m - s_h^{m-1},
               g_lin = (s_*)^n + n (s_*)^{n-1} (s_h^{m+1} - s_*)
    S7    one-step CN startup with Gamma^{1/2} and G^{1/2}
    S8    A = A0 + u v^T; one sparse LU + two back-substitutions

Important implementation choices (documented in the numerics report):
  * s_h^m and s_h^{m-1} are computed on the ACTUAL polygons Gamma^m and
    Gamma^{m-1} with their own masses (not on the predicted geometry).
  * All other step-m+1 geometry (h, n, M, N, d_s) is frozen at Gamma-tilde.
  * The CN startup uses the predicted midpoint polygon Gamma^{1/2} and the
    vertex normal n^0 = N^0/|N^0|.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from . import geometry as geo


class SolverFailure(RuntimeError):
    """Raised when a time step produces a non-admissible state."""

    def __init__(self, message, state=None):
        super().__init__(message)
        self.state = state or {}


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------

def _assemble_curvature_rows(rows, cols, vals, rhs, N, Nv, le):
    """Append the two curvature rows (x and y) with the frozen geometry."""
    idx = np.arange(N)
    ip = (idx + 1) % N
    im = (idx - 1) % N
    a, b = geo.stiffness_coeffs(le)

    # x-component row N+i:  N_i^x kappa_i - (S X)_i^x
    rows.append(N + idx)
    cols.append(2 * N + idx)
    vals.append(Nv[:, 0])

    # -S block in the x-columns
    rows.append(N + idx)
    cols.append(idx)
    vals.append(-(a + b))
    rows.append(N + idx)
    cols.append(im)
    vals.append(a)
    rows.append(N + idx)
    cols.append(ip)
    vals.append(b)

    # y-component row 2N+i: N_i^y kappa_i - (S X)_i^y
    rows.append(2 * N + idx)
    cols.append(2 * N + idx)
    vals.append(Nv[:, 1])

    rows.append(2 * N + idx)
    cols.append(N + idx)
    vals.append(-(a + b))
    rows.append(2 * N + idx)
    cols.append(N + im)
    vals.append(a)
    rows.append(2 * N + idx)
    cols.append(N + ip)
    vals.append(b)


def assemble_bdf2(Xm, Xm1, km, km1, tau, n, use_predicted=True):
    """Assemble A0, rhs, u, v and metadata for one BDF2 step.

    use_predicted=True  : Scheme Spec S5.1 (geometry frozen at Gamma-tilde).
    use_predicted=False : diagnostic variant (geometry frozen at Gamma^m);
                          used only to locate the source of tangential drift.
    """
    Xm = np.asarray(Xm, dtype=float)
    Xm1 = np.asarray(Xm1, dtype=float)
    km = np.asarray(km, dtype=float)
    km1 = np.asarray(km1, dtype=float)
    N = len(Xm)

    # Predicted geometry (frozen for the whole step).
    Xt = 2.0 * Xm - Xm1 if use_predicted else Xm.copy()
    e, le, Nv, M = geo.geometry(Xt)
    Lh = float(np.sum(M))

    # Actual-layer means s_h^m, s_h^{m-1} (own masses).
    _, le_m, _, M_m = geo.geometry(Xm)
    _, le_m1, _, M_m1 = geo.geometry(Xm1)
    sh_m = geo.mean_curvature(km, M_m)
    sh_m1 = geo.mean_curvature(km1, M_m1)
    s_star = 2.0 * sh_m - sh_m1

    alpha = n * (s_star ** (n - 1))
    c = (1.0 - n) * (s_star ** n)

    idx = np.arange(N)
    rows, cols, vals = [], [], []

    # ---- first (normal) row block ----
    coef = 3.0 / (2.0 * tau)
    rows.append(idx)
    cols.append(idx)
    vals.append(Nv[:, 0] * coef)
    rows.append(idx)
    cols.append(N + idx)
    vals.append(Nv[:, 1] * coef)
    rows.append(idx)
    cols.append(2 * N + idx)
    vals.append(M)

    rhs = np.zeros(3 * N)
    rhs[:N] = ((4.0 * Xm - Xm1) * Nv).sum(axis=1) / (2.0 * tau) + M * c

    # ---- curvature rows ----
    _assemble_curvature_rows(rows, cols, vals, rhs, N, Nv, le)

    data = np.concatenate([np.asarray(v, dtype=float).ravel() for v in vals])
    rr = np.concatenate([np.asarray(r).ravel() for r in rows])
    cc = np.concatenate([np.asarray(c_).ravel() for c_ in cols])
    A0 = coo_matrix((data, (rr, cc)), shape=(3 * N, 3 * N)).tocsr()

    u = np.zeros(3 * N)
    u[:N] = -alpha * M / Lh
    v = np.zeros(3 * N)
    v[2 * N:3 * N] = M

    meta = dict(
        N=N, tau=float(tau), n=int(n),
        s_h_m=float(sh_m), s_h_m1=float(sh_m1), s_star=float(s_star),
        alpha=float(alpha), c=float(c),
        L_h_pred=float(Lh), nnz=int(A0.nnz),
        n_negative_mass=int(np.count_nonzero(M <= 0.0)),
        s_star_positive=bool(s_star > 0.0),
    )
    return A0, rhs, u, v, meta


# ---------------------------------------------------------------------------
# One BDF2 step
# ---------------------------------------------------------------------------

def step_bdf2(Xm, Xm1, km, km1, tau, n, alpha_abs_floor=1e-14,
              use_predicted=True):
    """Advance one BDF2-Taylor step. Returns (X_new, kappa_new, diag)."""
    t0 = time.perf_counter()
    A0, rhs, u, v, meta = assemble_bdf2(Xm, Xm1, km, km1, tau, n,
                                        use_predicted=use_predicted)
    t_assemble = time.perf_counter() - t0

    t1 = time.perf_counter()
    try:
        lu = splu(A0.tocsc())
    except RuntimeError as exc:                     # singular factorization
        raise SolverFailure(f'splu failed: {exc}', dict(meta)) from exc
    y = lu.solve(rhs)
    if np.max(np.abs(u)) > 0.0:
        z = lu.solve(u)
        denom = 1.0 + float(np.dot(v, z))
        if abs(denom) < alpha_abs_floor:
            raise SolverFailure(
                f'Sherman-Morrison denominator too small: {denom:.3e}',
                dict(meta, denom=float(denom)))
        U = y - (float(np.dot(v, y)) / denom) * z
    else:
        z = np.zeros_like(u)
        denom = 1.0
        U = y
    t_solve = time.perf_counter() - t1

    X_new = np.column_stack([U[:meta['N']], U[meta['N']:2 * meta['N']]])
    k_new = U[2 * meta['N']:]

    # Residual of the full perturbed system (without forming A).
    resid_vec = A0 @ U + u * float(np.dot(v, U)) - rhs
    rhs_norm = float(np.linalg.norm(rhs))
    residual = float(np.linalg.norm(resid_vec) / rhs_norm) if rhs_norm > 0 else float('nan')

    ok, reason = geo.is_finite_state(X_new, k_new, denom)
    if not ok:
        raise SolverFailure(f'non-admissible state after BDF2 step: {reason}',
                            dict(meta, denom=denom, residual=residual))

    diag = dict(meta)
    diag.update(
        denom=float(denom),
        denominator_abs=float(abs(denom)),
        residual=residual,
        t_assemble=float(t_assemble),
        t_factor_solve=float(t_solve),
        step_time=float(time.perf_counter() - t0),
        z_norm=float(np.linalg.norm(z)),
        u_norm=float(np.linalg.norm(u)),
        v_norm=float(np.linalg.norm(v)),
    )
    diag.update(_state_diagnostics(X_new, k_new, tau, n))
    return X_new, k_new, diag


def _state_diagnostics(X, k, tau, n):
    _, le, _, M = geo.geometry(X)
    Lh = float(np.sum(M))
    sh = geo.mean_curvature(k, M)
    return dict(
        t_step=float(tau),
        L_h=float(Lh),
        s_h=float(sh),
        K_h=float(np.dot(M, k)),
        D_h=float(abs(np.dot(M, k) - geo.TWO_PI)),
        area=geo.polygon_area(X),
        perimeter=float(np.sum(le)),
        mesh_ratio=geo.mesh_ratio(X),
        min_edge=float(np.min(le)),
        max_edge=float(np.max(le)),
        kappa_inf=float(np.max(np.abs(k))),
        kappa_min=float(np.min(k)),
        kappa_max=float(np.max(k)),
        s_h_positive=bool(sh > 0.0),
        centroid_x=float(geo.centroid(X)[0]),
        centroid_y=float(geo.centroid(X)[1]),
        finite=bool(np.all(np.isfinite(X)) and np.all(np.isfinite(k))),
    )


# ---------------------------------------------------------------------------
# CN startup
# ---------------------------------------------------------------------------

def _taylor_g0(X0, k0, tau, n):
    """G^{1/2} and the diagnostic scalars (Scheme Spec S7.4-S7.5)."""
    _, le0, Nv0, M0 = geo.geometry(X0)
    L0 = float(np.sum(M0))
    K0 = float(np.dot(M0, k0))
    s0 = K0 / L0
    k2 = float(np.dot(M0, k0 ** 2))
    sdot = -(K0 / (L0 ** 2)) * ((s0 ** n) * K0 - k2)
    G = (s0 ** n) + 0.5 * tau * n * (s0 ** (n - 1)) * sdot
    return float(G), float(s0), float(sdot), float(K0), float(L0)


def startup_cn(X0, k0, tau, n, predictor='taylor'):
    """One-step CN startup (Scheme Spec S7.2-S7.3).

    predictor = 'taylor' : G^{1/2} = s0^n + (tau/2) n s0^{n-1} sdot0
    predictor = 'first'  : G^{1/2} = s0^n   (ablation only)
    """
    X0 = np.asarray(X0, dtype=float)
    k0 = np.asarray(k0, dtype=float)
    N = len(X0)

    _, le0, Nv0, M0 = geo.geometry(X0)
    Lh0 = float(np.sum(M0))
    s0 = geo.mean_curvature(k0, M0)

    if predictor == 'taylor':
        G, s0, sdot, K0, L0 = _taylor_g0(X0, k0, tau, n)
    elif predictor == 'first':
        G = float(s0 ** n)
        sdot = 0.0
    else:
        raise ValueError(f'unknown predictor {predictor!r}')

    # Predicted midpoint geometry.
    n_hat = geo.vertex_normals(Nv0)
    V0 = ((s0 ** n) - k0)[:, None] * n_hat
    Xhalf = X0 + 0.5 * tau * V0
    _, leh, Nvh, Mh = geo.geometry(Xhalf)

    idx = np.arange(N)
    rows, cols, vals = [], [], []

    # normal row: N_i^{1/2} . X_i^1 / tau + M_i^{1/2} kappa_i^1 / 2
    rows.append(idx)
    cols.append(idx)
    vals.append(Nvh[:, 0] / tau)
    rows.append(idx)
    cols.append(N + idx)
    vals.append(Nvh[:, 1] / tau)
    rows.append(idx)
    cols.append(2 * N + idx)
    vals.append(Mh / 2.0)

    rhs = np.zeros(3 * N)
    rhs[:N] = (Nvh * X0).sum(axis=1) / tau - Mh * k0 / 2.0 + Mh * G

    # curvature rows multiplied by 2:
    #   N_i kappa_i^1 - (S X^1)_i = -N_i kappa_i^0 + (S X^0)_i
    _assemble_curvature_rows(rows, cols, vals, rhs, N, Nvh, leh)
    SX0 = geo.stiffness_action(X0, leh)
    rhs[N:2 * N] = -Nvh[:, 0] * k0 + SX0[:, 0]
    rhs[2 * N:3 * N] = -Nvh[:, 1] * k0 + SX0[:, 1]

    data = np.concatenate([np.asarray(v, dtype=float).ravel() for v in vals])
    rr = np.concatenate([np.asarray(r).ravel() for r in rows])
    cc = np.concatenate([np.asarray(c_).ravel() for c_ in cols])
    A0 = coo_matrix((data, (rr, cc)), shape=(3 * N, 3 * N)).tocsr()

    t0 = time.perf_counter()
    try:
        lu = splu(A0.tocsc())
    except RuntimeError as exc:
        raise SolverFailure(f'CN startup splu failed: {exc}') from exc
    U = lu.solve(rhs)
    t_solve = time.perf_counter() - t0

    X1 = np.column_stack([U[:N], U[N:2 * N]])
    k1 = U[2 * N:]
    ok, reason = geo.is_finite_state(X1, k1)
    if not ok:
        raise SolverFailure(f'non-admissible state after CN startup: {reason}')

    resid = A0 @ U - rhs
    residual = float(np.linalg.norm(resid) / np.linalg.norm(rhs)) if np.linalg.norm(rhs) > 0 else float('nan')

    diag = dict(
        stage='startup', predictor=predictor, N=N, tau=float(tau), n=int(n),
        G_half=float(G), s0=float(s0), sdot0=float(sdot),
        L_h=float(np.sum(Mh)), R_half=float('nan'),
        denom=1.0, denominator_abs=1.0, residual=residual,
        t_factor_solve=float(t_solve), step_time=float(t_solve),
        nnz=int(A0.nnz), z_norm=float('nan'), u_norm=0.0, v_norm=0.0,
        s_h_m=float(s0), s_h_m1=float(s0), s_star=float(s0),
        alpha=float(n * s0 ** (n - 1)), c=float((1 - n) * s0 ** n),
    )
    diag.update(_state_diagnostics(X1, k1, tau, n))
    return X1, k1, diag


# ---------------------------------------------------------------------------
# Time loop
# ---------------------------------------------------------------------------

def solve_bdf2(X0, k0, T, M, n, startup='cn', predictor='taylor',
               record=False, snapshot_levels=None, stop_on_failure=True,
               use_predicted=True):
    """March the BDF2-Taylor scheme from t=0 to T.

    Parameters
    ----------
    X0, k0 : initial polygon and discrete curvature
    T : final time
    M : number of time steps (tau = T/M)
    n : power
    startup : 'taylor' (CN startup) or 'bgn1' (legacy-style one-step, E6 only)
    predictor : 'taylor' or 'first' (startup ablation)
    record : store per-step diagnostics
    snapshot_levels : iterable of coarse levels m at which to store (t, X, k)
    stop_on_failure : if False, return the last admissible state and mark failure

    Returns
    -------
    X, k, diags, snaps, info
        snaps = {level: dict(t=..., X=..., k=...)}
        info  = dict(failed=..., reason=..., fail_level=..., fail_time=...)
    """
    X0 = np.asarray(X0, dtype=float)
    k0 = np.asarray(k0, dtype=float)
    M = int(M)
    tau = float(T) / M
    if tau <= 0:
        raise ValueError('tau must be positive')

    snap_levels = set(int(s) for s in (snapshot_levels or []))
    snaps = {}
    diags = [] if record else None

    def _record(level, t, X, k, d):
        if record:
            dd = dict(d)
            dd.update(level=int(level), t=float(t))
            diags.append(dd)
        if level in snap_levels:
            snaps[level] = dict(level=int(level), t=float(t),
                                X=X.copy(), k=k.copy())

    info = dict(failed=False, reason='', fail_level=None, fail_time=None)

    _record(0, 0.0, X0, k0,
            dict(stage='initial', level=0, t=0.0,
                 s_h_m=float('nan'), s_h_m1=float('nan'),
                 s_star=float('nan'), alpha=float('nan'), c=float('nan'),
                 s_star_positive=True, denominator_abs=float('nan'),
                 denom=float('nan'), residual=float('nan'),
                 step_time=0.0, t_assemble=0.0, t_factor_solve=0.0,
                 nnz=0, u_norm=0.0, v_norm=0.0, z_norm=0.0,
                 L_h_pred=float('nan'),
                 **_state_diagnostics(X0, k0, tau, n)))

    try:
        if startup == 'bgn1':
            # Legacy-style explicit first step, used only for E6 comparison.
            # The old BGN1 step is provided by solver/legacy_adapter.py; here
            # we simply reject to avoid duplicating the legacy scheme.
            raise ValueError("startup='bgn1' is handled by solver/legacy_adapter.py")
        elif startup in ('cn', 'taylor'):
            X1, k1, d1 = startup_cn(X0, k0, tau, n, predictor=predictor)
        else:
            raise ValueError(f'unknown startup {startup!r}')
        _record(1, tau, X1, k1, d1)
    except SolverFailure as exc:
        if stop_on_failure:
            raise
        info.update(failed=True, reason=str(exc), fail_level=1, fail_time=tau)
        return X0, k0, diags, snaps, info

    Xm1, km1 = X0, k0
    Xm, km = X1, k1
    for m in range(1, M):
        try:
            Xn, kn, dn = step_bdf2(Xm, Xm1, km, km1, tau, n,
                                   use_predicted=use_predicted)
        except SolverFailure as exc:
            if stop_on_failure:
                raise
            info.update(failed=True, reason=str(exc), fail_level=m + 1,
                        fail_time=(m + 1) * tau)
            break
        Xm1, km1 = Xm, km
        Xm, km = Xn, kn
        _record(m + 1, (m + 1) * tau, Xm, km, dn)

    return Xm, km, diags, snaps, info


def solve_bdf2_to_snapshots(X0, k0, T, M, n, levels, predictor='taylor'):
    """Convenience wrapper: run and return snapshots at the requested levels."""
    X, k, _, snaps, info = solve_bdf2(
        X0, k0, T, M, n, predictor=predictor, record=False,
        snapshot_levels=levels, stop_on_failure=True)
    if info['failed']:
        raise SolverFailure(info['reason'], info)
    return snaps
