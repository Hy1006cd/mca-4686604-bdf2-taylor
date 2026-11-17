"""
run_phase4.py -- run the Phase-4 additions only (E3b, E7, E8, E9).

This avoids rerunning the already frozen E0-E6 JSONs.  The experiments are
deterministic except for the wall-clock timing in E9.

Usage: python -I run_phase4.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments import e3b_startup_curvature, e7_a5_extension
from experiments import e8_stability_map, e9_work_precision


def main():
    t_all = time.perf_counter()
    status = {}
    for name, mod in (('E3b', e3b_startup_curvature),
                      ('E7', e7_a5_extension),
                      ('E8', e8_stability_map),
                      ('E9', e9_work_precision)):
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
    sys.exit(main())
