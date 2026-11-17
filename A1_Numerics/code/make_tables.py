"""
make_tables.py -- export the experiment JSON to CSV and Markdown tables.

Every number quoted in the report is copied from these tables.  The script
is idempotent: rerun it after any experiment.

Usage: python -I make_tables.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a1n_config as cfg


def _load(name):
    return cfg.load_json(name)


def _num(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def f(x, fmt='%.4e'):
    if x is None:
        return '--'
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return '--'
    return fmt % x


def i(x, fmt='%.3f'):
    return f(x, fmt)


def write_csv(name, header, rows):
    path = os.path.join(cfg.TABLE_DIR, name + '.csv')
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(['' if v is None else v for v in r])
    return path


def write_md(name, header, rows, title=''):
    lines = []
    if title:
        lines += [f'### {title}', '']
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('|' + '|'.join(['---'] * len(header)) + '|')
    for r in rows:
        lines.append('| ' + ' | '.join('' if v is None else str(v) for v in r) + ' |')
    path = os.path.join(cfg.TABLE_DIR, name + '.md')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return path


# ---------------------------------------------------------------------------
# Table 1 -- E0
# ---------------------------------------------------------------------------

def table_e0():
    d = _load('e0_verification.json')
    rows = []
    for r in d['results']:
        rows.append([r['name'], 'PASS' if r['pass_'] else 'FAIL'])
    write_csv('table1_e0_gate', ['E0 sub-experiment', 'result'], rows)
    write_md('table1_e0_gate', ['E0 sub-experiment', 'result'], rows,
             'Table 1. E0 implementation gate')


# ---------------------------------------------------------------------------
# Table 2 -- E1
# ---------------------------------------------------------------------------

def table_e1():
    d = _load('e1_closure.json')
    rows = []
    for r in d['results']:
        for tname, key in (('t=0', 'q_t0'), ('t=T', 'q_T')):
            q = r[key]
            rows.append([
                r['shape'], tname,
                f(q.get('finest_pair'), '%.3f'),
                f(q['fit_asymptotic'].get('slope'), '%.3f'),
                f(q['fit_asymptotic'].get('r2'), '%.4f'),
                f(q['fit_all'].get('slope'), '%.3f'),
                f(q['fit_all'].get('r2'), '%.4f'),
                q.get('available'),
            ])
    write_csv('table2_e1_closure',
              ['shape', 'time', 'q_finest', 'q_fit_asym', 'R2_asym',
               'q_fit_all', 'R2_all', 'fit_available'], rows)
    write_md('table2_e1_closure',
             ['shape', 'time', 'q finest', 'q fit (asym.)', 'R² (asym.)',
              'q fit (all)', 'R² (all)', 'available'], rows,
             'Table 2. E1 closure defect D_h ~ h^q (A5)')
    # detailed D_h table
    rows2 = []
    for r in d['results']:
        for x in r['rows']:
            rows2.append([r['shape'], x['N'], f(x['h'], '%.4e'),
                          f(x['D_h_0'], '%.4e'), f(x.get('D_h_T'), '%.4e'),
                          f(x.get('kappa_inf_T'), '%.4f'),
                          f(x.get('psi_T'), '%.2f'),
                          'FAIL' if x.get('failed') else 'ok'])
    write_csv('table2b_e1_Dh', ['shape', 'N', 'h', 'D_h(0)', 'D_h(T)',
                                '||kappa||_inf(T)', 'Psi(T)', 'status'], rows2)
    write_md('table2b_e1_Dh', ['shape', 'N', 'h', 'D_h(0)', 'D_h(T)',
                               '||kappa||∞(T)', 'Ψ(T)', 'status'], rows2,
             'Table 2b. E1 closure defect and mesh quality')


# ---------------------------------------------------------------------------
# Table 3 -- E2
# ---------------------------------------------------------------------------

def table_e2():
    d = _load('e2_spatial.json')
    rows = []
    for r in d['results']:
        o = r['orders']
        for idx, x in enumerate(r['rows']):
            def order(key):
                arr = o[key]['orders']
                return f(arr[idx], '%.3f') if idx < len(arr) else '--'
            rows.append([
                r['n'], x['N'], f(x['h'], '%.4e'),
                f(x['full_L2'], '%.4e'), order('full_L2'),
                f(x['full_H1'], '%.4e'), order('full_H1'),
                f(x['nodal_L2'], '%.4e'), order('nodal_L2'),
                f(x['nodal_H1'], '%.4e'), order('nodal_H1'),
                f(x['nodal_kappa'], '%.4e'), order('nodal_kappa'),
                f(x['A7'], '%.4f'), f(x['psi'], '%.2f'),
            ])
    header = ['n', 'N', 'h', 'E_L2 full', 'p', 'E_H1 full', 'p',
              'E_L2 nodal', 'p', 'E_H1 nodal', 'p', 'E_kappa nodal', 'p',
              'R_h (A7)', 'Psi']
    write_csv('table3_e2_spatial', header, rows)
    write_md('table3_e2_spatial', header, rows,
             'Table 3. E2 spatial convergence and A7 ratio')
    # summary
    rows2 = []
    for r in d['results']:
        o = r['orders']
        rows2.append([r['n'],
                      f(o['full_L2']['finest_pair'], '%.3f'),
                      f(o['full_H1']['finest_pair'], '%.3f'),
                      f(o['nodal_L2']['finest_pair'], '%.3f'),
                      f(o['nodal_H1']['finest_pair'], '%.3f'),
                      f(o['nodal_kappa']['finest_pair'], '%.3f'),
                      f(abs(o['full_L2']['fit_slope']), '%.3f'),
                      f(abs(o['full_H1']['fit_slope']), '%.3f'),
                      f(o['full_L2']['fit_r2'], '%.4f'),
                      f(o['full_H1']['fit_r2'], '%.4f')])
    write_csv('table3b_e2_summary',
              ['n', 'full L2 finest', 'full H1 finest', 'nodal L2 finest',
               'nodal H1 finest', 'nodal kappa finest',
               'full L2 fit order', 'full H1 fit order', 'R2 L2', 'R2 H1'], rows2)
    write_md('table3b_e2_summary',
             ['n', 'full L2 finest', 'full H1 finest', 'nodal L2 finest',
              'nodal H1 finest', 'nodal κ finest', 'full L2 fit order',
              'full H1 fit order', 'R² L2', 'R² H1'], rows2,
             'Table 3b. E2 finest-pair orders and fits')
    # A7 over time range
    rows3 = []
    for r in d['results']:
        a = r['A7_over_time']
        for N in (256, 512):
            sub = [x for x in a if x['N'] == N]
            if sub:
                rows3.append([r['n'], N,
                              f(min(x['A7'] for x in sub), '%.4f'),
                              f(max(x['A7'] for x in sub), '%.4f'),
                              f(sub[0]['A7'], '%.4f'),
                              f(sub[-1]['A7'], '%.4f')])
    write_csv('table3c_e2_A7_time', ['n', 'N', 'R_h min', 'R_h max',
                                     'R_h first', 'R_h last'], rows3)
    write_md('table3c_e2_A7_time', ['n', 'N', 'R_h min', 'R_h max',
                                    'R_h first', 'R_h last'], rows3,
             'Table 3c. E2 A7 ratio over time')


# ---------------------------------------------------------------------------
# Table 4/5/6 -- E3
# ---------------------------------------------------------------------------

def table_e3():
    d = _load('e3_temporal.json')
    # circle
    rows = []
    for r in d['circle_time_order']:
        o = r['err_modified']
        for idx, x in enumerate(r['rows']):
            p = o['orders'][idx] if idx < len(o['orders']) else None
            rows.append([r['n'], r['predictor'], x['M'], f(x['tau'], '%.4e'),
                         f(x['err_modified'], '%.4e'), f(p, '%.3f'),
                         f(x['err_continuum'], '%.4e'), f(x['psi'], '%.3f')])
    header = ['n', 'predictor', 'M', 'tau', '|R-R_mod|', 'p_tau',
              '|R-R_cont|', 'Psi']
    write_csv('table4_e3_circle', header, rows)
    write_md('table4_e3_circle', header, rows,
             'Table 4. E3 circle time order (modified ODE reference)')
    # startup
    rows = []
    for r in d['startup']:
        s = r['summary']
        rows.append([r['n'], r['predictor'],
                     f(s['orders'][-1] if s['orders'] else None, '%.3f'),
                     f(s['finest_pair'], '%.3f'),
                     '; '.join(f(x['err'], '%.3e') for x in r['rows'])])
    write_csv('table4b_e3_startup',
              ['n', 'predictor', 'p_startup', 'finest', 'errors'], rows)
    write_md('table4b_e3_startup',
             ['n', 'predictor', 'p startup', 'finest', 'errors'], rows,
             'Table 4b. E3 one-step startup error at t=tau')
    # ellipse joint
    rows = []
    for r in d['ellipse_joint']:
        o = r['orders']
        for idx, x in enumerate(r['rows']):
            def order(key):
                arr = o[key]['orders']
                return f(arr[idx], '%.3f') if idx < len(arr) else '--'
            rows.append([r['n'], x['N'], f(x['h'], '%.4e'),
                         f(x['tau'], '%.4e'), x['M'],
                         f(x['full_L2'], '%.4e'), order('full_L2'),
                         f(x['full_H1'], '%.4e'), order('full_H1'),
                         f(x['nodal_L2'], '%.4e'), order('nodal_L2'),
                         f(x['psi'], '%.2f')])
    header = ['n', 'N', 'h', 'tau', 'M', 'E_L2 full', 'p', 'E_H1 full', 'p',
              'E_L2 nodal', 'p', 'Psi']
    write_csv('table5_e3_ellipse_joint', header, rows)
    write_md('table5_e3_ellipse_joint', header, rows,
             'Table 5. E3 ellipse joint refinement tau = 0.05 h')
    # fixed-N
    rows = []
    for r in d['ellipse_fixed_N']:
        for x in r['rows']:
            rows.append([r['n'], r['N'], f(x['tau'], '%.4e'), x['M'],
                         f(x['full_L2'], '%.4e'), f(x['full_H1'], '%.4e'),
                         f(x['nodal_L2'], '%.4e'), f(x['psi'], '%.2f')])
    write_csv('table5b_e3_fixedN',
              ['n', 'N', 'tau', 'M', 'E_L2 full', 'E_H1 full',
               'E_L2 nodal', 'Psi'], rows)
    write_md('table5b_e3_fixedN',
             ['n', 'N', 'tau', 'M', 'E_L2 full', 'E_H1 full',
              'E_L2 nodal', 'Psi'], rows,
             'Table 5b. E3 fixed-N tau study (spatial-floor negative result)')
    # coupling compact: group actual tau/h by the nearest target ratio
    ratios = d['coupling_map'][0]['ratios'] if d['coupling_map'] else []
    rows = []
    for r in d['coupling_map']:
        by = {}
        for x in r['rows']:
            target = min(ratios, key=lambda q: abs(q - x['tau_over_h']))
            by.setdefault((r['n'], x['N']), {})[round(target, 3)] = x
        for (n, N), sub in sorted(by.items()):
            cells = []
            for ratio in ratios:
                x = sub.get(round(ratio, 3))
                if x is None:
                    cells.append('--')
                elif x['failed']:
                    cells.append('F')
                else:
                    cells.append(f(x['psi_max'], '%.1f'))
            rows.append([n, N] + cells)
    header = ['n', 'N'] + [f'{r:g}' for r in ratios]
    write_csv('table6_e3_coupling', header, rows)
    write_md('table6_e3_coupling', header, rows,
             'Table 6. E3 tau/h coupling: max Psi (F = failed)')


# ---------------------------------------------------------------------------
# Table 7/8 -- E4
# ---------------------------------------------------------------------------

def table_e4():
    d = _load('e4_stability.json')
    rows = []
    for r in d['safe_tau']:
        rows.append([r['shape'], r['n'], r['N'], f(r['T'], '%.0f'),
                     f(r['tau'], '%.1e'), r['M'],
                     f(r['area_ratio'], '%.6f'),
                     f(r['equilibrium_ratio'], '%.6f'),
                     f(r['area_gap'], '%.3e'),
                     f(r['perimeter_final'], '%.6f'),
                     f(r['psi_max'], '%.2f'),
                     f(r['kappa_inf'], '%.4f'),
                     f(r['s_star_min'], '%.4f'),
                     f(r['denom_min'], '%.6f'),
                     r.get('self_intersections')])
    header = ['shape', 'n', 'N', 'T', 'tau', 'M', 'A(T)/A(0)', 'A_eq/A(0)',
              '|A-A_eq|', 'perimeter', 'Psi max', '||kappa||inf',
              'min s_*', 'min |d|', 'self-int']
    write_csv('table7_e4_safe', header, rows)
    write_md('table7_e4_safe', header, rows,
             'Table 7. E4 long-time diagnostics in the safe-tau window')
    rows = []
    for r in d['coarse_tau']:
        rows.append([r['shape'], r['n'], r['N'], f(r['T'], '%.0f'),
                     f(r['tau'], '%.1e'), r['M'],
                     'FAIL' if r['failed'] else 'ok',
                     f(r.get('psi_max'), '%.1f'),
                     f(r.get('area_ratio'), '%.6f'),
                     f(r.get('fail_time'), '%.3f'),
                     (r.get('reason') or '')[:60]])
    write_csv('table8_e4_coarse',
              ['shape', 'n', 'N', 'T', 'tau', 'M', 'status', 'Psi max',
               'A/A0', 'fail time', 'reason'], rows)
    write_md('table8_e4_coarse',
             ['shape', 'n', 'N', 'T', 'tau', 'M', 'status', 'Ψ max',
              'A/A0', 'fail t', 'reason'], rows,
             'Table 8. E4 coarse-tau failure panel')


# ---------------------------------------------------------------------------
# Table 9/10 -- E5
# ---------------------------------------------------------------------------

def table_e5():
    d = _load('e5_cost.json')
    rows = []
    for r in d['cost']:
        rows.append([r['scheme'], r['N'], r['M'], f(r['tau'], '%.4f'),
                     f(r['ms_per_step'], '%.3f'), f(r['median_s'], '%.3f'),
                     r.get('factorizations_per_step'),
                     r.get('back_solves_per_step'), r.get('nnz')])
    write_csv('table9_e5_cost',
              ['scheme', 'N', 'M', 'tau', 'ms/step', 'median s',
               'LU/step', 'backsolves/step', 'nnz'], rows)
    write_md('table9_e5_cost',
             ['scheme', 'N', 'M', 'tau', 'ms/step', 'median s',
              'LU/step', 'backsolves/step', 'nnz'], rows,
             'Table 9. E5 cost per step')
    rows = []
    for r in d['denominator_condition']:
        c = r['first_step']
        rows.append([r['n'], r['N'], f(r['denom_min'], '%.6f'),
                     f(r['denom_max'], '%.6f'),
                     f(r['s_star_min'], '%.4f'), f(r['s_star_max'], '%.4f'),
                     f(c.get('cond_estimate'), '%.3e'),
                     f(c.get('cond_dense_1'), '%.3e'),
                     f(r['residual_max'], '%.2e'), f(r['psi_final'], '%.2f')])
    header = ['n', 'N', 'min|d|', 'max|d|', 'min s_*', 'max s_*',
              'cond~', 'cond dense (1-norm)', 'residual', 'Psi']
    write_csv('table10_e5_condition', header, rows)
    write_md('table10_e5_condition', header, rows,
             'Table 10. E5 Sherman-Morrison denominator and conditioning')


# ---------------------------------------------------------------------------
# Table 11/12/13 -- E6
# ---------------------------------------------------------------------------

def table_e6():
    d = _load('e6_comparison.json')
    rows = []
    for r in d['circle_orders']:
        rows.append([r['scheme'], r['n'], f(r['finest_pair'], '%.3f'),
                     f(abs(r['fit_slope']), '%.3f'), f(r['fit_r2'], '%.4f'),
                     '; '.join(f(x, '%.2e') for x in r['errors'])])
    write_csv('table11_e6_circle',
              ['scheme', 'n', 'p finest', '|fit slope|', 'R2', 'errors'], rows)
    write_md('table11_e6_circle',
             ['scheme', 'n', 'p finest', '|fit slope|', 'R²', 'errors'], rows,
             'Table 11. E6 circle time order, prior-work comparison')
    # stability compact: group actual tau/h by the nearest target ratio
    targets = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
    rows = []
    for sid in ('ellipse', 'flower'):
        for n in (1, 2, 3):
            for scheme in ('BDF2-Taylor', 'BGN1', 'CNLF-EX'):
                sub = [r for r in d['stability_map']
                       if r['shape'] == sid and r['n'] == n
                       and r['scheme'] == scheme]
                by = {}
                for r in sub:
                    target = min(targets, key=lambda q: abs(q - r['tau_over_h']))
                    by[round(target, 3)] = r
                cells = []
                for ratio in targets:
                    x = by.get(round(ratio, 3))
                    if x is None:
                        cells.append('--')
                    elif x['failed']:
                        cells.append('F')
                    else:
                        cells.append(f(x['psi_max'], '%.1f'))
                rows.append([sid, n, scheme] + cells)
    header = ['shape', 'n', 'scheme'] + [f'{r:g}' for r in targets]
    write_csv('table12_e6_stability', header, rows)
    write_md('table12_e6_stability', header, rows,
             'Table 12. E6 stability map: max Psi (F = failed)')
    rows = []
    for r in d['n_sensitivity']:
        rows.append([r['shape'], r['n'], r['N'], f(r['tau'], '%.1e'),
                     f(r['error'], '%.3e'), f(r['C_h2'], '%.3e'),
                     'FAIL' if r.get('failed') else 'ok'])
    write_csv('table13_e6_n_sensitivity',
              ['shape', 'n', 'N', 'tau', 'error', 'C=err/(h^2+tau^2)',
               'status'], rows)
    write_md('table13_e6_n_sensitivity',
             ['shape', 'n', 'N', 'tau', 'error', 'C=err/(h²+τ²)', 'status'],
             rows, 'Table 13. E6 n-sensitivity')


# ---------------------------------------------------------------------------
# Tables 14-17 -- Phase 4 (E3b, E7, E8, E9)
# ---------------------------------------------------------------------------

def table_e3b():
    d = _load('e3b_startup_curvature.json')
    rows = []
    for r in d['results']:
        for x in r['rows']:
            rows.append([r['n'], r['predictor'], f(x['tau'], '%.4e'),
                         f(x['pos_err_cont'], '%.4e'),
                         f(x['pos_err_mod'], '%.4e'),
                         f(x['k_err_cont'], '%.4e'),
                         f(x['k_err_mod'], '%.4e'),
                         'FAIL' if x.get('failed') else 'ok'])
    header = ['nu', 'predictor', 'tau', 'pos_err_cont', 'pos_err_mod',
              'k_err_cont', 'k_err_mod', 'status']
    write_csv('table14_e3b_curvature', header, rows)
    write_md('table14_e3b_curvature', header, rows,
             'Table 14. E3b one-step startup curvature error')


def table_e7():
    d = _load('e7_a5_extension.json')
    rows = []
    for r in d['results']:
        if not r.get('reference_ok'):
            rows.append([r['shape'], r['n'], '--', '--', '--', '--',
                         '--', '--', 'reference unavailable', '--', '--', '--'])
            continue
        shape = r['shape']
        max_n = 256 if shape == 'flower' else 512
        vals = [x['A7'] for x in r['rows']
                if x.get('available') and x['N'] <= max_n
                and x.get('A7') is not None]
        if shape == 'flower':
            n_hi, n_lo = 256, 128
            range_basis = 'N<=256'
            trend_basis = 'N256/N128'
        else:
            n_hi, n_lo = 512, 256
            range_basis = 'N<=512'
            trend_basis = 'N512/N256'
        hi = [x['A7'] for x in r['rows']
              if x.get('available') and x['N'] == n_hi
              and x.get('A7') is not None]
        lo = [x['A7'] for x in r['rows']
              if x.get('available') and x['N'] == n_lo
              and x.get('A7') is not None]
        trend = (float(np.mean(hi) / np.mean(lo)) if hi and lo else None)
        floor = (r.get('ref_floor') or {}).get('full_L2')
        note = ('N=512 at reference floor; excluded' if shape == 'flower'
                else '')
        rows.append([shape, r['n'], ','.join(str(t) for t in r['times']),
                     f(min(vals), '%.4f') if vals else '--',
                     f(max(vals), '%.4f') if vals else '--',
                     f(trend, '%.4f'), range_basis, trend_basis,
                     note, f"{sum(1 for x in r['rows'] if x.get('available'))}/"
                     f"{len(r['rows'])}", r.get('N_ref'), f(floor, '%.4e')])
    header = ['shape', 'nu', 'times', 'A7 min', 'A7 max', 'trend',
              'range_basis', 'trend_basis', 'note', 'available', 'N_ref',
              'floor_full_L2']
    write_csv('table15_e7_a5_summary', header, rows)
    write_md('table15_e7_a5_summary', header, rows,
             'Table 15. E7 extended A5 ratio (summary; flower quantitative '
             'range uses N<=256 because N=512 is at the reference floor)')
    rows2 = []
    for r in d['results']:
        if not r.get('reference_ok'):
            continue
        for x in r['rows']:
            rows2.append([r['shape'], r['n'], x['N'], f(x['t'], '%.3f'),
                          f(x['h'], '%.4e'), f(x['tau'], '%.4e'),
                          f(x.get('A7'), '%.5f'), f(x.get('full_L2'), '%.4e'),
                          f(x.get('full_H1'), '%.4e'), 'FAIL' if x.get('run_failed') else 'ok'])
    header2 = ['shape', 'nu', 'N', 't', 'h', 'tau', 'A7',
               'full_L2', 'full_H1', 'run_status']
    write_csv('table15b_e7_a5_detail', header2, rows2)
    write_md('table15b_e7_a5_detail', header2, rows2,
             'Table 15b. E7 extended A5 ratio (detail)')


def table_e8():
    d = _load('e8_stability_map.json')
    rows = []
    for r in d['rows']:
        rows.append([r['shape'], r['N'], f(r['tau'], '%.4e'),
                     f(r['tau_over_h'], '%.4e'),
                     'FAIL' if r['failed'] else 'ok',
                     f(r.get('psi_max'), '%.4e'),
                     f(r.get('min_edge'), '%.4e'),
                     f(r.get('max_turning_angle'), '%.4e'),
                     f(r.get('tangential_drift'), '%.4e'),
                     f(r.get('denom_min'), '%.6f'),
                     f(r.get('area_ratio'), '%.6f')])
    header = ['shape', 'N', 'tau', 'tau/h', 'status', 'Psi_max',
              'min_edge', 'max_turning_angle', 'tangential_drift',
              'min_abs_denominator', 'area_ratio']
    write_csv('table16_e8_stability', header, rows)
    write_md('table16_e8_stability', header, rows,
             'Table 16. E8 two-parameter (h,tau) stability map')


def table_e9():
    d = _load('e9_work_precision.json')
    rows = []
    for r in d['rows']:
        rows.append([r['shape'], r['scheme'], f(r['tau'], '%.5e'), r['M'],
                     f(r.get('error'), '%.4e'),
                     f(r.get('time_median'), '%.6e'),
                     f(r.get('ms_per_step'), '%.4f'),
                     'FAIL' if r['failed'] else 'ok'])
    header = ['shape', 'scheme', 'tau', 'M', 'error', 'time_median_s',
              'ms_per_step', 'status']
    write_csv('table17_e9_work_precision', header, rows)
    write_md('table17_e9_work_precision', header, rows,
             'Table 17. E9 work-precision')


def phase4_number_map():
    lines = [
        '# Phase 4 number map (manuscript -> JSON)',
        '',
        '| Manuscript item | JSON path | Key(s) |',
        '|---|---|---|',
        '| Table 3 initial perimeters/areas | `data/e7_a5_extension.json`, `solver/shapes.py` | `shapes.build_shape` at N=8192 |',
        '| Table 3 denominator values | `data/e5_cost.json`, `data/e0_verification.json`, `data/e4_stability.json` | `denominator_condition`, `results[min_abs_denominator]`, `safe_tau[].denom_min` |',
        '| Table e3 panel (b) curvature | `data/e3b_startup_curvature.json` | `results[].rows[].k_err_cont`, `k_err_mod` |',
        '| Table e7 A5 summary | `data/e7_a5_extension.json` | `results[].rows` (flower range N<=256; trend N256/N128; N=512 floor-limited) |',
        '| Table e8 stability rows | `data/e8_stability_map.json` | `rows[]` |',
        '| Table e9 work-precision rows | `data/e9_work_precision.json` | `rows[]` |',
        '| Figure 6 E8 heatmap | `data/e8_stability_map.json` | `rows[].psi_max`, `rows[].failed` |',
        '| Figure 7 E9 work-precision | `data/e9_work_precision.json` | `rows[].error`, `rows[].time_median` |',
    ]
    path = os.path.join(cfg.TABLE_DIR, 'phase4_number_map.md')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return path


def main():
    table_e0()
    table_e1()
    table_e2()
    table_e3()
    table_e4()
    table_e5()
    table_e6()
    table_e3b()
    table_e7()
    table_e8()
    table_e9()
    phase4_number_map()
    print('tables written to', cfg.TABLE_DIR)


if __name__ == '__main__':
    main()
