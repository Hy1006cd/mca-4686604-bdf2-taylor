"""
bgn2_powerlaw.py  ——  幂律保面积曲线缩短流 (AP-CSF) 的 BGN 格式求解器
================================================================================

本文件实现论文
    "A Second-Order BGN Parametric Finite Element Method for Power-Law Curve
     Shortening Flow with Area Control"
中幂律 AP-CSF 的 BGN1 / BGN2 格式，并提供两种幂律驱动项的处理方式：

  方案 A（论文现印格式） ： G^m = 2(<κ^m>)^p - (<κ^{m-1}>)^p        （外推，常数）
  方案 B（改进格式）     ： G^m = (1-p)(<κ^m>)^p + p(<κ^m>)^{p-1} <κ̄>
                           （在 <κ^m> 处对 <κ>^p 做 Taylor 线性化）

两者共用同一个主格式（冻结 Γ^m 的 Crank-Nicolson leap-frog）：

    ( (X^{m+1}-X^{m-1})/(2τ), φ n^m )^h_Γm + ( κ̄ - G^m, φ )^h_Γm = 0
    ( κ̄, n^m·ω )^h_Γm - ( (∂_s X^{m+1}+∂_s X^{m-1})/2, ∂_s ω )_Γm = 0

其中 κ̄ = (κ^{m+1}+κ^{m-1})/2。

--------------------------------------------------------------------------------
记号与离散算子（全部经解析验证，见 crossval.py）
--------------------------------------------------------------------------------
X        : (N,2) 数组，闭合多边形顶点，逆时针方向
e_i      : e_i = X_{i+1} - X_i          第 i 条边向量
|e_i|    : 第 i 条边长度
n_i      : 第 i 条边的单位外法向 = (e_i^y, -e_i^x)/|e_i|
M_i      : 集中质量 = ½(|e_{i-1}| + |e_i|)，满足 Σ_i M_i = 周长
N_i      : 集中法向算子 = ½(|e_{i-1}| n_{i-1} + |e_i| n_i) = ½ (X_{i+1}-X_{i-1})^⊥
           —— 就是离散面积泛函的梯度 ∂A/∂X_i，且 Σ_i N_i ≡ 0（精确）
(S X)_i  : 刚度作用 = (X_i - X_{i-1})/|e_{i-1}| - (X_{i+1} - X_i)/|e_i|
<κ>      : (Σ_i M_i κ_i) / (Σ_i M_i)

⚠ 关键：集中质量内积 (v·n, φ_i)^h 的正确权重是 N_i，**不是** M_i n_i。
   老代码（`代码/` 目录）用的是 M_i n_i，法向配对错位，会破坏离散面积守恒。
   判据：Σ_i N_i ≈ 0（机器精度），而 Σ_i M_i n_i ≠ 0。

--------------------------------------------------------------------------------
线性方程组
--------------------------------------------------------------------------------
未知量 u = [X^{new}_x (N); X^{new}_y (N); κ^{new} (N)]，共 3N 个。

  第 1 组（i = 0..N-1，动量/法向方程）：
      N_i · X^{new}_i / tw  +  M_i h κ^{new}_i  -  M_i G^m  =  N_i · X^{prev}_i / tw
                                                             -  M_i h κ^{prev}_i
  第 2、3 组（i = 0..N-1，曲率-弧长方程，x / y 分量各 N 条）：
      h M_i κ^{new}_i n_i^a  -  h (S X^{new})_i^a  =  - h M_i κ^{prev}_i n_i^a
                                                      + h (S X^{prev})_i^a

  时间权重：
      BGN1  : tw = τ,   h = 1,   X^{prev}=X^m, κ^{prev}=0, 且第二组右端不加 (S X^{prev})
      CN 启动: tw = τ,   h = 1/2, X^{prev}=X^0, κ^{prev}=κ^0
      BGN2  : tw = 2τ,  h = 1/2, X^{prev}=X^{m-1}, κ^{prev}=κ^{m-1}

方案 B 的 G^m 含 Σ_j M_j κ^{new}_j（未知量），故系数矩阵多一个秩一项，
用 Sherman-Morrison 公式求解（仍是每步一个 3N×3N 稀疏系统，不需要 Newton）。
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


# ============================================================================
# 1. 离散几何算子
# ============================================================================

def geometry(X):
    """输入顶点 X:(N,2)，返回该多边形的所有离散几何量。

    返回
    ----
    e  : (N,2)  边向量 e_i = X_{i+1} - X_i
    le : (N,)   边长 |e_i|
    Nv : (N,2)  集中法向算子 N_i（见文件头注释）
    M  : (N,)   集中质量 M_i
    """
    e = np.roll(X, -1, axis=0) - X             # e_i = X_{i+1} - X_i
    le = np.linalg.norm(e, axis=1)             # |e_i|
    s = 0.5 * (np.roll(e, 1, axis=0) + e)      # ½(e_{i-1} + e_i)
    Nv = np.column_stack([s[:, 1], -s[:, 0]])  # 逆时针旋转 90°: (x,y) -> (y,-x)
    M = 0.5 * (le + np.roll(le, 1))            # M_i = ½(|e_{i-1}| + |e_i|)
    return e, le, Nv, M


def stiffness_action(X, le):
    """刚度矩阵作用 (S X)_i = (X_i - X_{i-1})/|e_{i-1}| - (X_{i+1} - X_i)/|e_i|。"""
    e = np.roll(X, -1, axis=0) - X
    ep = np.roll(e, 1, axis=0)                 # e_{i-1}
    lp = np.roll(le, 1)                        # |e_{i-1}|
    return ep / lp[:, None] - e / le[:, None]


def initial_curvature(X):
    """初始曲率 κ^0（最小二乘有限元 + 集中质量）。

    由论文 eq.(eq:init-kappa)  κ^0 = (N_0ᵀ N_0)^{-1} N_0ᵀ A_0 X^0
    加上集中质量（此时 N_0ᵀ N_0 退化为对角阵）后得闭式：

        κ_i^0 = N_i · (S X^0)_i / |N_i|^2

    ⚠ 老代码写成 A_0 @ X0.T.flatten()，坐标被打乱，κ^0 变成 O(1e2) 的垃圾值。
      正确写法是 X0.flatten()（(2,N) 行主序 = [x 行; y 行]），本式即其集中质量特例。
    """
    _, le, Nv, _ = geometry(X)
    SX = stiffness_action(X, le)               # (N,2)
    return np.sum(Nv * SX, axis=1) / np.sum(Nv * Nv, axis=1)


def mean_curvature(k, M):
    """集中质量意义下的平均曲率 <κ> = (Σ M_i κ_i)/(Σ M_i)。"""
    return float(np.dot(M, k) / np.sum(M))


def taylor_g(X, k, p, dt):
    """二阶 Taylor 预测 G ≈ (<κ(t_0+dt)>)^p，用初始/当前层的几何与曲率。

    由 <κ> = K/L、K = Σ M_i κ_i、L = Σ M_i，以及闭合曲线上的
        d<κ>/dt = -(K/L^2) [ (<κ>)^p K - Σ M_i κ_i^2 ]
    得
        G = (<κ>)^p + dt · p (<κ>)^{p-1} d<κ>/dt.
    该预测只依赖给定层的数据，不需要 Sherman-Morrison。
    """
    _, _, _, M = geometry(X)
    K = float(np.dot(M, k))
    L = float(np.sum(M))
    s = K / L
    k2 = float(np.dot(M, k ** 2))
    dkbar = -(K / L ** 2) * (s ** p * K - k2)
    return s ** p + dt * p * s ** (p - 1) * dkbar


def mesh_ratio(X):
    """网格比 Ψ = max|h|/min|h|。"""
    le = np.linalg.norm(np.roll(X, -1, axis=0) - X, axis=1)
    return float(le.max() / le.min())


def polygon_area(X):
    """鞋带公式面积。"""
    return 0.5 * abs(np.sum(X[:, 0] * np.roll(X[:, 1], -1)
                            - X[:, 1] * np.roll(X[:, 0], -1)))


def polygon_perimeter(X):
    return float(np.sum(np.linalg.norm(np.roll(X, -1, axis=0) - X, axis=1)))


def arclength_redistribute(X, N=None):
    """保形的弧长重分布：沿同一条折线按等弧长重新取节点。

    只沿曲线切向移动节点，**不改变曲线的形状**，因此不会污染时间收敛阶。
    收敛性实验默认关闭网格正则化（见 convergence.py 的说明）。
    """
    if N is None:
        N = len(X)
    P = np.vstack([X, X[:1]])                        # 闭合
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, cum[-1], N, endpoint=False)
    j = np.clip(np.searchsorted(cum, targets, side='right') - 1, 0, len(seg) - 1)
    t = (targets - cum[j]) / np.where(seg[j] > 0, seg[j], 1.0)
    return P[j] + t[:, None] * (P[j + 1] - P[j])


# ============================================================================
# 2. 单步组装与求解
# ============================================================================

def step(Xg, Xprev, kprev, km, tau, p, scheme='B', order=2, startup=False,
         sm_prev=None, sm_prev2=None, g_override=None):
    """推进一个时间步，返回 (X_new, kappa_new)。

    参数
    ----
    Xg     : 冻结几何 Γ^m 的顶点 X^m
    Xprev  : 时间导数里减去的状态（BGN2: X^{m-1}；启动步: X^0；BGN1 忽略）
    kprev  : 与 Xprev 配套的曲率（BGN1 忽略）
    km     : 计算 <κ^m> 用的曲率，恒为 κ^m
    startup: 'taylor' 时用二阶 Taylor 预测 G^0；True 时保留旧 CN 启动（s^p）
    sm_prev, sm_prev2 : 格式 A 主步的滞后外推所需的前两层平均
                        （分别用 Γ^{m-1}、Γ^{m-2} 的质量计算）
    g_override : 若给出，直接作为格式 A 的 G^m（用于 G^1 的 Taylor 预测）
    """
    N = len(Xg)
    _, le, Nv, M = geometry(Xg)
    Msum = float(np.sum(M))

    # ---- 时间权重（见文件头「时间权重」表） ----
    if order == 1:
        tw, h = tau, 1.0
        Xp, kp = Xg, np.zeros_like(km)         # BGN1 的时间导数用 X^m，没有 κ^m 项
        use_sx_rhs = False                     # BGN1 第二组右端为 0
    else:
        tw = tau if startup else 2.0 * tau
        h = 0.5
        Xp, kp = Xprev, kprev
        use_sx_rhs = True

    idx = np.arange(N)
    ip, im = (idx + 1) % N, (idx - 1) % N
    a = 1.0 / le[im]                           # 1/|e_{i-1}|
    b = 1.0 / le                               # 1/|e_i|

    # ---- 幂律驱动项 G^m ----
    s = mean_curvature(km, M)                  # <κ^m>
    sm = mean_curvature(kp, M)                 # <κ^{m-1}>
    if order == 1:
        # BGN1：驱动项 = (<κ^m>)^p，显式常数
        src_const, alpha = s ** p, np.zeros(N)
    elif scheme == 'A':
        # 方案 A：
        #   startup='taylor' -> G^0 用二阶 Taylor 预测（t_{1/2}）
        #   g_override       -> G^1 的 Taylor 预测（t_1）
        #   sm_prev/sm_prev2 -> 主步的滞后外推 2<s_{m-1}>^p - <s_{m-2}>^p
        #   否则保留旧口径 2<s_m>^p - <s_{m-1}>^p（仅用于对照）
        if startup == 'taylor':
            src_const = taylor_g(Xg, km, p, 0.5 * tau)
        elif startup:
            src_const = s ** p
        elif g_override is not None:
            src_const = float(g_override)
        elif sm_prev is not None and sm_prev2 is not None:
            src_const = 2.0 * sm_prev ** p - sm_prev2 ** p
        else:
            src_const = 2.0 * s ** p - sm ** p
        alpha = np.zeros(N)
    else:
        # 方案 B：G = (1-p) s^p + p s^{p-1} <κ̄>
        #   <κ̄> = (Σ_j M_j κ^new_j + Σ_j M_j κ^prev_j) / (2 Msum)
        src_const = (1.0 - p) * s ** p
        alpha = M * p * s ** (p - 1) / (2.0 * Msum)

    rows, cols, vals = [], [], []

    def put(r, c, v):
        rows.append(np.asarray(r).ravel())
        cols.append(np.asarray(c).ravel())
        vals.append(np.broadcast_to(np.asarray(v, dtype=float), np.shape(r)).ravel())

    rhs = np.zeros(3 * N)

    # ---- 第 1 组：动量 / 法向方程 ----
    put(idx, idx, Nv[:, 0] / tw)
    put(idx, N + idx, Nv[:, 1] / tw)
    put(idx, 2 * N + idx, M * h)
    rhs[idx] = (Nv[:, 0] * Xp[:, 0] + Nv[:, 1] * Xp[:, 1]) / tw
    if order != 1:
        rhs[idx] -= M * h * kp
    rhs[idx] += M * src_const + alpha * float(np.dot(M, kp))
    # 注意：方案 B 的秩一项 -alpha_i * M_j **不在这里组装**，
    #       而是交给下面的 Sherman-Morrison 处理（否则会重复计入两次）。

    # ---- 第 2、3 组：曲率-弧长方程（x 与 y 分量） ----
    put(N + idx, 2 * N + idx, Nv[:, 0] * h)
    put(2 * N + idx, 2 * N + idx, Nv[:, 1] * h)
    if order != 1:
        rhs[N + idx] = -Nv[:, 0] * h * kp
        rhs[2 * N + idx] = -Nv[:, 1] * h * kp

    # 离散曲率关系是 N_i κ_i = (S X)_i（与 initial_curvature 一致），
    # 所以第 2、3 组的刚度项必须以 **负号** 出现在左端：
    #     h N_i κ^new_i  −  h (S X^new)_i  =  − h N_i κ^prev_i  +  h (S X^prev)_i
    put(N + idx, idx, -h * (a + b))
    put(N + idx, ip, h * b)
    put(N + idx, im, h * a)
    put(2 * N + idx, N + idx, -h * (a + b))
    put(2 * N + idx, N + ip, h * b)
    put(2 * N + idx, N + im, h * a)

    if use_sx_rhs:
        SXp = stiffness_action(Xp, le)
        rhs[N + idx] += h * SXp[:, 0]
        rhs[2 * N + idx] += h * SXp[:, 1]

    A = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(3 * N, 3 * N)).tocsr()

    if np.any(alpha != 0.0):
        # Sherman-Morrison：A = A0 + u vᵀ
        #   x = A0⁻¹b - (A0⁻¹u)(vᵀA0⁻¹b)/(1 + vᵀA0⁻¹u)
        u = np.zeros(3 * N)
        u[:N] = -alpha
        v = np.zeros(3 * N)
        v[2 * N:3 * N] = M
        y = spsolve(A, rhs)
        z = spsolve(A, u)
        sol = y - z * (float(np.dot(v, y)) / (1.0 + float(np.dot(v, z))))
    else:
        sol = spsolve(A, rhs)

    return np.column_stack([sol[:N], sol[N:2 * N]]), sol[2 * N:]


# ============================================================================
# 3. 时间推进
# ============================================================================

def solve(X0, k0, T, n_steps, p, scheme='B', order=2, startup='cn',
          record=False, n_snapshots=0):
    """把格式从 t=0 推到 t=T。

    X0, k0  : 初值（顶点与初始曲率）
    T       : 终止时刻
    n_steps : 总步数 M，时间步 tau = T/M
    p       : 幂指数 n
    scheme  : 'A'（论文现印外推）或 'B'（Taylor 线性化改进）
    order   : 1（BGN1）或 2（BGN2）
    startup : BGN2 启动方式：
              'cn'     旧 Crank-Nicolson 启动（G^0 = (<κ^0>)^p，一阶非局部项）
              'taylor' 二阶 Taylor 启动（G^0 在 t_{1/2}，G^1 在 t_1），
                       并启用格式 A 的滞后外推主循环
              'bgn1'   跑一步 BGN1（一阶）
    record  : True 时返回逐步诊断列表
    n_snapshots : >0 时额外返回 n_snapshots 个等间隔的曲线快照（含 t=0）

    返回 (X_M, kappa_M, diagnostics, snapshots)
          snapshots 为 [(t, X), ...]，未开启时为空列表
    """
    tau = T / n_steps
    diags = [] if record else None
    snaps = []
    snap_at = set()
    if n_snapshots > 0:
        snap_at = set(np.unique(np.linspace(
            0, n_steps - 1, min(n_snapshots, n_steps)).round().astype(int)).tolist())

    def log(m, X, k, kappa_hist=None):
        if diags is not None:
            M = geometry(X)[3]
            diags.append(dict(m=m, t=m * tau,
                              perimeter=polygon_perimeter(X),
                              area=polygon_area(X),
                              mesh_ratio=mesh_ratio(X),
                              mean_curvature=mean_curvature(k, M)))
        if m in snap_at:
            snaps.append((m * tau, X.copy()))

    if 0 in snap_at:
        snaps.append((0.0, X0.copy()))

    # ---------- 启动 ----------
    if order == 1 or startup == 'bgn1':
        Xcur, kcur = step(X0, None, None, k0, tau, p, scheme, 1)
    elif startup == 'taylor':
        Xcur, kcur = step(X0, X0, k0, k0, tau, p, scheme, 2,
                          startup='taylor')
    else:
        # 旧口径：Crank-Nicolson 单步，G^0 = (<κ^0>)^p
        Xcur, kcur = step(X0, X0, k0, k0, tau, p, scheme, 2, startup=True)
    log(1, Xcur, kcur)

    # ---------- 主循环 ----------
    if order == 1:
        for m in range(1, n_steps):
            Xcur, kcur = step(Xcur, None, None, kcur, tau, p, scheme, 1)
            log(m + 1, Xcur, kcur)
    else:
        Xprev, kprev = X0, k0
        Xprev2, kprev2 = X0, k0
        # 格式 A + Taylor 启动：主步用滞后外推
        #   G^m = 2(<κ^{m-1}>)^p - (<κ^{m-2}>)^p,  m >= 2,
        # 其中前两层的平均分别用各自几何的质量计算。
        use_lagged = (scheme == 'A' and startup == 'taylor')
        for m in range(1, n_steps):
            if use_lagged:
                if m == 1:
                    # 第一主步的 G^1：在 t_1 处的二阶 Taylor 预测
                    g1 = taylor_g(X0, k0, p, tau)
                    Xn, kn = step(Xcur, Xprev, kprev, kcur, tau, p,
                                  scheme, 2, g_override=g1)
                else:
                    Mp = geometry(Xprev)[3]
                    Mp2 = geometry(Xprev2)[3]
                    sm_prev = mean_curvature(kprev, Mp)
                    sm_prev2 = mean_curvature(kprev2, Mp2)
                    Xn, kn = step(Xcur, Xprev, kprev, kcur, tau, p,
                                  scheme, 2, sm_prev=sm_prev,
                                  sm_prev2=sm_prev2)
            else:
                Xn, kn = step(Xcur, Xprev, kprev, kcur, tau, p, scheme, 2)
            Xprev2, kprev2 = Xprev, kprev
            Xprev, kprev, Xcur, kcur = Xcur, kcur, Xn, kn
            log(m + 1, Xcur, kcur)

    return Xcur, kcur, diags, snaps
