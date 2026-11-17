"""
a1n_config.py -- single source of truth for the A1 Stage-3 numerics.

All experiment scripts read parameters from here; no experiment hard-codes
its own ladder.  The code implements Scheme Spec v1.2 (S2-S9) and is used
only for the numerical evidence of Proof Chain v1.2 (A1-A7).

Directory layout (A1_文件组织规范 §3.1)
---------------------------------------
    A1_Numerics/
      A1_BDF2_Taylor_Numerics.md
      A1_BDF2_Taylor_Numerics_中文翻译.md
      data/         JSON / CSV raw output
      figures/      PDF + PNG
      code/         this file and the package below
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_DIR = os.path.dirname(CODE_DIR)                 # A1_Numerics/
PROJECT_DIR = os.path.dirname(NUM_DIR)              # project8.1.7/
DATA_DIR = os.path.join(NUM_DIR, 'data')
TABLE_DIR = os.path.join(DATA_DIR, 'tables')
FIG_DIR = os.path.join(NUM_DIR, 'figures')

# Legacy solver: read-only, used ONLY for the E6 prior-work comparison.
LEGACY_SOLVER_DIR = os.path.join(
    PROJECT_DIR, 'A1_Legacy_BGN1_Baseline', 'code', 'solver')
LEGACY_FILES = ('bgn2_powerlaw.py', 'metrics.py', 'shapes.py')

for _d in (DATA_DIR, TABLE_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)


def package_path_inject() -> str:
    """Put the A1_Numerics/code directory on sys.path (for `import solver...`)."""
    if CODE_DIR not in sys.path:
        sys.path.insert(0, CODE_DIR)
    return CODE_DIR


def legacy_path_inject() -> str:
    """Put the legacy solver directory on sys.path (read-only E6 comparison)."""
    if LEGACY_SOLVER_DIR not in sys.path:
        sys.path.insert(0, LEGACY_SOLVER_DIR)
    return LEGACY_SOLVER_DIR


# ---------------------------------------------------------------------------
# Fixed numerical parameters
# ---------------------------------------------------------------------------
SEED = 20261020
L0 = 1.0                      # nondimensionalization (Scheme Spec S0)
TWO_PI = 2.0 * np.pi

# Shape registry: id -> (kind, params, role)
SHAPES = {
    'circle':   dict(kind='circle',  params=dict(R=1.5),
                     role='circle; finite-N scalar reduction / exact modified ODE'),
    'circle1':  dict(kind='circle',  params=dict(R=1.0),
                     role='near-equilibrium circle R=1; closure/steady probes'),
    'ellipse':  dict(kind='ellipse', params=dict(a=2.0, b=1.0),
                     role='main non-circular convergence shape'),
    'flower':   dict(kind='flower',  params=dict(a=2.0, b=0.5, k=8),
                     role='strongly non-circular supplementary shape'),
    'superellipse': dict(kind='superellipse', params=dict(p=8.0, size=1.0),
                         role='curvature-discontinuity diagnostic (supplementary)'),
}

# Spatial ladders
N_E0 = (64, 128, 256, 512)
N_E1 = (64, 128, 256, 512, 1024, 2048)
N_E2 = (64, 128, 256, 512)
N_E2_SELF = (64, 128, 256, 512, 1024)
N_REF_E2 = 4096
N_REF_PROBE = 8192

# Time ladders
TAU0 = 0.1
TAU_LEVELS = 6                 # tau = TAU0 / 2^k, k=0..5
T_ORDER = 0.5
T_LONG = (2.0, 5.0)
TAU_SMALL = 1.0e-4
TAU_REF_FACTOR = 4             # tau_ref = tau / 4 in E2/E3 references

# Exponent values
N_VALUES = (1, 2, 3)
N_VALUES_SUPP = (1, 2, 3, 4, 5, 6)

# E3 tau-h coupling
TAU_OVER_H = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
N_COUPLING = (128, 256, 512)
T_COUPLING = 0.5

# E4 long-time
N_LONG = (256, 512)
TAU_LONG = 0.01                # within the expected stable window

# E5 cost
N_COST = (128, 256, 512, 1024)
T_COST = 0.2
COST_REPEATS = 3

# ---------------------------------------------------------------------------
# Tolerances / acceptance criteria (reported honestly if missed)
# ---------------------------------------------------------------------------
TOL = dict(
    operator_identity=1.0e-14,
    geometry_crossval=1.0e-14,
    scalar_reduction_rel=1.0e-10,
    steady_n1_max_dx=1.0e-8,
    kappa0_order=(1.9, 2.1),
    q_order=(1.8, 2.2),
    q_fit=(1.9, 2.1),
    fit_r2_min=0.98,
    order2=(1.8, 2.2),
    order1=(0.8, 1.2),
    order_match=0.25,
    alignment_selfcheck=1.0e-12,
    denom_min=1.0e-12,
    residual_max=1.0e-10,
)

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def np_safe(o):
    """Recursively convert numpy scalars/arrays to JSON-serializable objects."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (list, tuple)):
        return [np_safe(v) for v in o]
    if isinstance(o, dict):
        return {str(k): np_safe(v) for k, v in o.items()}
    return o


def save_json(name: str, payload, subdir: str = '') -> str:
    path = os.path.join(DATA_DIR, subdir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(np_safe(payload), f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


def load_json(name: str, subdir: str = ''):
    path = os.path.join(DATA_DIR, subdir, name)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def environment_manifest() -> dict:
    import platform
    import scipy
    import matplotlib
    return dict(
        date=time.strftime('%Y-%m-%d %H:%M:%S'),
        python=sys.version.split()[0],
        platform=platform.platform(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        matplotlib=matplotlib.__version__,
        seed=SEED,
        cwd=os.getcwd(),
        code_dir=CODE_DIR,
        project_dir=PROJECT_DIR,
        legacy_solver_dir=LEGACY_SOLVER_DIR,
    )
