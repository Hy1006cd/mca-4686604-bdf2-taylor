"""
shapes.py  ——  初始形状构造
================================================================================

所有形状都返回 (N,2) 的逆时针闭合多边形顶点，并且是 **按弧长等分布** 的
（参考论文 Jiang-Su-Zhang 用的就是 "an equidistributed polygon X^0 with N vertices"）。

提供：
    circle(N, R)              圆
    ellipse(N, a, b)          椭圆（按弧长等分布）
    ellipse_theta(N, a, b)    椭圆（按参数 θ 等分，便于和解析曲率对照）
    flower(N, a, b, k)        花形  r(θ) = a + b cos(kθ)
    square(N, size)           近似正方形
"""

import numpy as np


def _resample_by_arclength(fun, N, n_fine=200001):
    """给定参数曲线 fun(ρ) -> (N,2)（ρ ∈ [0,1)），按弧长重采样成 N 个等距点。

    做法：先在很细的网格上算出弧长函数 s(ρ)，再用线性插值反解出等弧长处对应的 ρ。
    """
    rho = np.linspace(0.0, 1.0, n_fine, endpoint=False)
    P = fun(rho)
    d = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    target = np.linspace(0.0, total, N, endpoint=False)
    rho_new = np.interp(target, s[:-1], rho)
    return fun(rho_new)


def circle(N, R=1.0):
    """半径为 R 的圆（按弧长等分布 = 按 θ 等分）。"""
    th = 2.0 * np.pi * np.arange(N) / N
    return R * np.column_stack([np.cos(th), np.sin(th)])


def ellipse_theta(N, a=2.0, b=1.0):
    """椭圆 x=a cosθ, y=b sinθ，按参数 θ 等分（**不是**按弧长等分布）。"""
    th = 2.0 * np.pi * np.arange(N) / N
    return np.column_stack([a * np.cos(th), b * np.sin(th)])


def ellipse(N, a=2.0, b=1.0):
    """椭圆 x²/a² + y²/b² = 1，按弧长等分布（论文 Shape 1 / Shape 2）。"""
    return _resample_by_arclength(lambda r: ellipse_theta(len(r), a, b), N)


def flower(N, a=2.0, b=0.5, k=8, resample=True):
    """花形 r(θ) = a + b cos(kθ)，θ = 2πρ（论文 Shape 3）。"""
    def fun(rho):
        th = 2.0 * np.pi * rho
        r = a + b * np.cos(k * th)
        return np.column_stack([r * np.cos(th), r * np.sin(th)])
    return _resample_by_arclength(fun, N) if resample else fun(np.arange(N) / N)


def square(N, size=2.0):
    """近似正方形（论文 Shape 2 的另一种取法）：把周长等分成 N 段。"""
    h = size / 2.0
    corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=float)
    edges = np.roll(corners, -1, axis=0) - corners
    L = np.linalg.norm(edges, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    target = np.linspace(0.0, cum[-1], N, endpoint=False)
    j = np.clip(np.searchsorted(cum, target, side='right') - 1, 0, 3)
    t = (target - cum[j]) / L[j]
    return corners[j] + t[:, None] * edges[j]


#: 论文正文用到的命名，便于脚本里按名字取形状
REGISTRY = {
    'circle':  lambda N: circle(N, 1.0),
    'ellipse': lambda N: ellipse(N, 2.0, 1.0),
    'flower':  lambda N: flower(N, 2.0, 0.5, 8),
    'square':  lambda N: square(N, 2.0),
}
