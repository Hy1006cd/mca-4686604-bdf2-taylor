"""
run_all.py -- run the A1_Numerics experiment chain E0 -> E6.

Usage
    python -I run_all.py --full     # all experiments (long; ~20-25 min)
    python -I run_all.py --quick    # E0 gate only + check existing JSONs

The scripts are deterministic; only BLAS/sparse-solver timing (E5) depends
on the machine.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a1n_config as cfg
from experiments import (e0_verify, e1_closure, e2_spatial, e3_temporal,
                         e4_stability, e5_cost, e6_compare)


def main(full=True):
    mods = [('E0', e0_verify)] if not full else [
        ('E0', e0_verify), ('E1', e1_closure), ('E2', e2_spatial),
        ('E3', e3_temporal), ('E4', e4_stability), ('E5', e5_cost),
        ('E6', e6_compare)]
    t_all = time.perf_counter()
    status = {}
    for name, mod in mods:
        t0 = time.perf_counter()
        print('\n' + '#' * 78)
        print(f'# {name}')
        print('#' * 78)
        try:
            rc = mod.main()
            status[name] = dict(rc=rc, seconds=time.perf_counter() - t0,
                                ok=True)
        except Exception as exc:                         # noqa: BLE001
            status[name] = dict(rc=None, seconds=time.perf_counter() - t0,
                                ok=False, error=repr(exc))
            print(f'{name} FAILED: {exc!r}')
    print('\n' + '=' * 78)
    for name, s in status.items():
        print(f'{name}: {"ok" if s["ok"] else "FAILED"} '
              f'({s["seconds"]:.1f}s)')
    print(f'total {time.perf_counter() - t_all:.1f}s')
    return 0 if all(s['ok'] for s in status.values()) else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', default=True)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    sys.exit(main(full=not a.quick))
