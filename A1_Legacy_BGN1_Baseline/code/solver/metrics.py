"""
metrics.py  ——  两条闭合曲线之间的四种误差度量
================================================================================

参考论文 Jiang-Su-Zhang (JCP 514 (2024) 113220) 的核心方法学结论是：
**BGN 类格式在 L² / L∞ 顶点范数下根本测不出阶**（其 Table 3 中 BGN2 的 L² 阶
只有 0.04 / 0.00 / 0.02，完全停滞），原因是 BGN 允许切向运动使节点等分布，
参数化变了但形状没变。必须改用 **形状度量**：

    L² 范数        ‖X − Y‖_{L²}       —— 顶点位置差的加权 L²（保留，用于对照说明其失效）
    L∞ 范数        ‖X − Y‖_{L∞}       —— 顶点位置差的最大值（同上）
    流形距离 M     |Ω_X △ Ω_Y|         —— 两个区域对称差的面积
    豪斯多夫距离 H  max{H̃(X,Y), H̃(Y,X)} —— 曲线作为点集的最大偏离

--------------------------------------------------------------------------------
实现要点（这两点是本项目实测出来必须避免的坑）
--------------------------------------------------------------------------------
* **流形距离不用径向插值。** 直接对折线做 r(θ) 线性插值会引入 O(h²) 的锯齿噪声，
  在 N=256、误差 ~1e-4 量级时会把收敛阶打乱成 3.85 / −4.91 / 7.23 这种跳变。
  本实现利用「星形多边形可以用极坐标精确表示」这一事实：

      Ω = {(r, φ) : 0 ≤ r ≤ r(φ)},   |Ω_X △ Ω_Y| = ½ ∫ |r_X(φ)² − r_Y(φ)²| dφ

  其中 r(φ) 对每条边都是解析的（射线与线段求交），积分在两条折线的**全部角断点**
  分段上用 Gauss-Legendre 求积 —— 每个子区间上被积函数解析，故没有离散化噪声。

* **豪斯多夫距离按折线精确计算。** 折线之间的豪斯多夫距离一定在某个顶点处取到，
  所以 H̃(X,Y) = max_i dist(x_i, Y)，其中 dist 是点到折线（各线段）的最小距离。

两种度量都要求两条曲线关于同一点是星形的（本项目所有测试形状都满足）。
若不满足，manifold_distance 会抛出明确异常，改用 hausdorff_distance 即可。
"""

import numpy as np


# ============================================================================
# 1. 顶点范数（保留，用于说明它们为什么不适合 BGN 类格式）
# ============================================================================

def l2_vertex(X, Y, M=None):
    """集中质量加权的顶点 L² 误差：sqrt(Σ_i |X_i − Y_i|² M_i)。"""
    diff = np.linalg.norm(X - Y, axis=1) ** 2
    if M is None:
        return float(np.sqrt(np.sum(diff)))
    return float(np.sqrt(np.sum(diff * M)))


def linf_vertex(X, Y):
    """顶点 L∞ 误差：max_i |X_i − Y_i|。"""
    return float(np.max(np.linalg.norm(X - Y, axis=1)))


# ============================================================================
# 2. 流形距离（对称差面积）
# ============================================================================

def _signed_area(V):
    return 0.5 * np.sum(V[:, 0] * np.roll(V[:, 1], -1) - V[:, 1] * np.roll(V[:, 0], -1))


def _polar_edges(V, center):
    """把多边形表示成以 center 为极点的极坐标：返回 (W, th)。

    W  : 平移后的顶点坐标 (N,2)
    th : (N+1,) 单调递增的极角断点，th[-1] = th[0] + 2π

    要求多边形关于 center 是星形且逆时针。
    """
    W = V - center
    if _signed_area(W) < 0:
        raise ValueError("manifold_distance 要求逆时针多边形")
    th = np.unwrap(np.arctan2(W[:, 1], W[:, 0]))
    # 星形 ⟺ 绕中心的极角严格单调递增、且总增量不超过 2π。
    # 注意：最后一个顶点的极角通常略小于 th[0]+2π（回到起点的那条边还没走完），
    # 这是正常的，闭合区间 [th[-1], th[0]+2π) 正对应「最后一个顶点 -> 第一个顶点」那条边。
    if not np.all(np.diff(th) > 0) or th[-1] >= th[0] + 2 * np.pi:
        raise ValueError("极角不单调：多边形关于给定中心不是星形，请改用 hausdorff_distance")
    return W, np.append(th, th[0] + 2 * np.pi)


def _radius(W, th, phi):
    """精确计算多边形在极角 phi 处的径向函数 r(φ)（射线与各边求交）。"""
    N = len(W)
    tt = np.mod(phi - th[0], 2 * np.pi) + th[0]
    j = np.clip(np.searchsorted(th, tt, side='right') - 1, 0, N - 1)
    A = W[j]
    B = W[(j + 1) % N]
    d = B - A
    u = np.column_stack([np.cos(tt), np.sin(tt)])
    cr_A = A[:, 0] * u[:, 1] - A[:, 1] * u[:, 0]        # cross(A, u)
    cr_d = d[:, 0] * u[:, 1] - d[:, 1] * u[:, 0]        # cross(d, u)
    if np.any(np.abs(cr_d) < 1e-14):
        raise ValueError("射线与边平行：多边形关于给定中心不是星形")
    t = -cr_A / cr_d
    Phit = A + t[:, None] * d
    return np.sum(Phit * u, axis=1)


def manifold_distance(X, Y, center=None, n_gauss=12):
    """流形距离 |Ω_X △ Ω_Y| = |Ω_X| + |Ω_Y| − 2|Ω_X ∩ Ω_Y|（精确积分，无插值噪声）。

    center : 两条曲线共同的内点；缺省取两条曲线面积质心的中点。
    """
    if center is None:
        center = 0.5 * (_centroid(X) + _centroid(Y))
    WX, thX = _polar_edges(X, center)
    WY, thY = _polar_edges(Y, center)

    # 合并两条折线的全部角断点
    bx = np.mod(thX[:-1] - thX[0], 2 * np.pi)
    by = np.mod(thY[:-1] - thY[0], 2 * np.pi)
    bp = np.unique(np.concatenate([[0.0, 2 * np.pi], bx, by]))

    gx, gw = np.polynomial.legendre.leggauss(n_gauss)
    lo, hi = bp[:-1], bp[1:]
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    phi = (mid[:, None] + half[:, None] * gx[None, :]).ravel()      # 全部求积节点
    w = (half[:, None] * gw[None, :]).ravel()

    phi_glob = phi + thX[0]
    rX = _radius(WX, thX, phi_glob)
    rY = _radius(WY, thY, phi_glob)
    return float(0.5 * np.sum(np.abs(rX ** 2 - rY ** 2) * w))


def _centroid(V):
    """多边形的面积质心。"""
    x, y = V[:, 0], V[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    A = 0.5 * np.sum(cr)
    if abs(A) < 1e-14:
        return V.mean(axis=0)
    cx = np.sum((x + xn) * cr) / (6.0 * A)
    cy = np.sum((y + yn) * cr) / (6.0 * A)
    return np.array([cx, cy])


# ============================================================================
# 3. 豪斯多夫距离（折线之间，精确）
# ============================================================================

def point_to_polyline(P, V):
    """P:(m,2) 中每个点到闭合折线 V:(n,2) 的最小距离，返回 (m,)。"""
    A = V
    B = np.roll(V, -1, axis=0)
    AB = B - A
    L2 = np.sum(AB * AB, axis=1)
    L2 = np.where(L2 > 0.0, L2, 1.0)
    # 逐点算，避免 m×n 距离矩阵过大
    out = np.empty(len(P))
    for k, p in enumerate(P):
        t = np.clip(np.sum((p - A) * AB, axis=1) / L2, 0.0, 1.0)
        proj = A + t[:, None] * AB
        out[k] = np.min(np.linalg.norm(p - proj, axis=1))
    return out


def hausdorff_distance(X, Y):
    """对称豪斯多夫距离 max{H̃(X,Y), H̃(Y,X)}（折线之间的精确值）。"""
    hXY = float(np.max(point_to_polyline(X, Y)))
    hYX = float(np.max(point_to_polyline(Y, X)))
    return max(hXY, hYX)


# ============================================================================
# 4. 统一接口
# ============================================================================

#: 参考论文表里用的四行度量
METRICS = ('L2', 'Linf', 'manifold', 'hausdorff')

#: 论文式报告里各度量的显示名
METRIC_LABEL = {
    'L2':        r'$L^2$-norm',
    'Linf':      r'$L^\infty$-norm',
    'manifold':  'Manifold distance',
    'hausdorff': 'Hausdorff distance',
}


def all_metrics(X, Y, M=None):
    """一次算出四种度量，返回 dict。"""
    return {
        'L2':        l2_vertex(X, Y, M),
        'Linf':      linf_vertex(X, Y),
        'manifold':  manifold_distance(X, Y),
        'hausdorff': hausdorff_distance(X, Y),
    }


def convergence_order(errs, ratios=None):
    """二阶收敛阶：order_k = log(E_k / E_{k+1}) / log(ratio_k)。

    errs   : 误差序列（从粗到细）
    ratios : 相邻两级的 τ 之比，缺省全为 2（每次 τ 减半）
    """
    errs = np.asarray(errs, dtype=float)
    if ratios is None:
        ratios = np.full(len(errs) - 1, 2.0)
    return [float(np.log(errs[i] / errs[i + 1]) / np.log(ratios[i]))
            for i in range(len(errs) - 1)]
