"""
metrics/shapes.py -- initial shapes, arc-length equidistributed.

The constructions follow the project's verified shapes.py (read-only legacy
utility); they are reproduced here so that A1_Numerics is self-contained.
"""

from __future__ import annotations

import numpy as np

import a1n_config as cfg


def _resample_by_arclength(fun, N, n_fine=200001):
    rho = np.linspace(0.0, 1.0, n_fine, endpoint=False)
    P = fun(rho)
    d = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    target = np.linspace(0.0, total, N, endpoint=False)
    rho_new = np.interp(target, s[:-1], rho)
    return fun(rho_new)


def circle(N, R=1.0):
    th = 2.0 * np.pi * np.arange(N) / N
    return R * np.column_stack([np.cos(th), np.sin(th)])


def ellipse_theta(N, a=2.0, b=1.0):
    th = 2.0 * np.pi * np.arange(N) / N
    return np.column_stack([a * np.cos(th), b * np.sin(th)])


def ellipse(N, a=2.0, b=1.0):
    return _resample_by_arclength(lambda r: ellipse_theta(len(r), a, b), N)


def flower(N, a=2.0, b=0.5, k=8, resample=True):
    def fun(rho):
        th = 2.0 * np.pi * rho
        r = a + b * np.cos(k * th)
        return np.column_stack([r * np.cos(th), r * np.sin(th)])
    return _resample_by_arclength(fun, N) if resample else fun(np.arange(N) / N)


def superellipse(N, p=8.0, size=1.0):
    def fun(rho):
        th = 2.0 * np.pi * rho
        c, s = np.cos(th), np.sin(th)
        ex = 2.0 / p
        x = size * np.sign(c) * np.abs(c) ** ex
        y = size * np.sign(s) * np.abs(s) ** ex
        return np.column_stack([x, y])
    V = _resample_by_arclength(fun, N)
    if np.sum(V[:, 0] * np.roll(V[:, 1], -1)
              - V[:, 1] * np.roll(V[:, 0], -1)) < 0:
        V = V[::-1].copy()
    return V


def build_shape(shape_id, N):
    if shape_id not in cfg.SHAPES:
        raise KeyError(f'unknown shape {shape_id!r}; known: {list(cfg.SHAPES)}')
    spec = cfg.SHAPES[shape_id]
    kind, prm = spec['kind'], spec['params']
    if kind == 'circle':
        return circle(N, prm['R'])
    if kind == 'ellipse':
        return ellipse(N, prm['a'], prm['b'])
    if kind == 'flower':
        return flower(N, prm['a'], prm['b'], prm['k'])
    if kind == 'superellipse':
        return superellipse(N, prm['p'], prm['size'])
    raise KeyError(kind)


def build_geometry(shape_id, N):
    """Return (X0, kappa0) using Scheme Spec S7.1."""
    from solver.geometry import initial_curvature
    X0 = build_shape(shape_id, N)
    return X0, initial_curvature(X0)


def ellipse_analytic_curvature(theta, a=2.0, b=1.0):
    """Signed curvature of x=a cos(theta), y=b sin(theta) (positive CCW)."""
    return a * b / (a * a * np.sin(theta) ** 2 + b * b * np.cos(theta) ** 2) ** 1.5
