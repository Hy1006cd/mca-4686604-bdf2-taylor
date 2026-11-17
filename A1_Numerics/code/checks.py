"""
checks.py -- consistency checks and reproduction manifest.

Checks
  * E0 gate passed
  * all E0-E6 JSON files exist and are non-empty
  * figures (PDF + PNG) and table exports exist
  * English/Chinese report structural consistency (headings, Table/Fig
    references, formula tags)
  * writes data/manifest.json with the environment, commands, file hashes,
    experiment timings and legacy-solver hash

Usage: python -I checks.py [--write-manifest]
"""

from __future__ import annotations

import argparse
import hashlib
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a1n_config as cfg

JSON_FILES = ('e0_verification.json', 'e1_closure.json', 'e2_spatial.json',
              'e3_temporal.json', 'e3b_startup_curvature.json',
              'e4_stability.json', 'e5_cost.json', 'e6_comparison.json',
              'e7_a5_extension.json', 'e8_stability_map.json',
              'e9_work_precision.json')
FIG_NAMES = ('fig1_closure', 'fig2_spatial', 'fig3_temporal',
             'fig4_stability', 'fig5_cost', 'fig6_n_sensitivity',
             'fig6_e8_stability', 'fig7_e9_work_precision')
REPORT_EN = 'A1_BDF2_Taylor_Numerics.md'
REPORT_ZH = 'A1_BDF2_Taylor_Numerics_中文翻译.md'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def file_rows(patterns, base):
    rows = []
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(base, pat), recursive=True)):
            if os.path.isfile(p):
                rows.append(dict(path=os.path.relpath(p, cfg.NUM_DIR),
                                 size=os.path.getsize(p),
                                 sha256=sha256(p)))
    return rows


def check_structure():
    en = os.path.join(cfg.NUM_DIR, REPORT_EN)
    zh = os.path.join(cfg.NUM_DIR, REPORT_ZH)
    out = dict(en_exists=os.path.exists(en), zh_exists=os.path.exists(zh))
    if not (out['en_exists'] and out['zh_exists']):
        return out
    se = open(en, encoding='utf-8').read()
    sz = open(zh, encoding='utf-8').read()
    out['en_headings'] = len(re.findall(r'^##+ ', se, flags=re.M))
    out['zh_headings'] = len(re.findall(r'^##+ ', sz, flags=re.M))
    out['en_tables'] = len(re.findall(r'\bTable\s+\d', se))
    out['zh_tables'] = len(re.findall(r'\bTable\s+\d', sz))
    out['en_figs'] = len(re.findall(r'\bFig(?:ure)?\.?\s+\d', se))
    out['zh_figs'] = len(re.findall(r'\bFig(?:ure)?\.?\s+\d', sz))
    out['en_tags'] = len(re.findall(r'tag\{', se))
    out['zh_tags'] = len(re.findall(r'tag\{', sz))
    out['headings_match'] = out['en_headings'] == out['zh_headings']
    out['tables_match'] = out['en_tables'] == out['zh_tables']
    out['figs_match'] = out['en_figs'] == out['zh_figs']
    out['tags_match'] = out['en_tags'] == out['zh_tags']
    return out


def build_manifest():
    env = cfg.environment_manifest()
    timings = {}
    for name in JSON_FILES:
        p = os.path.join(cfg.DATA_DIR, name)
        if os.path.exists(p):
            d = json.load(open(p, encoding='utf-8'))
            timings[name] = dict(
                generated=d.get('generated'),
                elapsed_s=d.get('elapsed_s'),
                size_bytes=os.path.getsize(p))
    legacy = {}
    for name in cfg.LEGACY_FILES:
        p = os.path.join(cfg.LEGACY_SOLVER_DIR, name)
        if os.path.exists(p):
            legacy[name] = dict(size=os.path.getsize(p), sha256=sha256(p))
    manifest = dict(
        generated=cfg.environment_manifest()['date'],
        environment=env,
        working_directory=cfg.CODE_DIR,
        commands=[
            'python -I experiments/e0_verify.py',
            'python -I experiments/e1_closure.py',
            'python -I experiments/e2_spatial.py',
            'python -I experiments/e3_temporal.py',
            'python -I experiments/e4_stability.py',
            'python -I experiments/e5_cost.py',
            'python -I experiments/e6_compare.py',
            'python -I experiments/e3b_startup_curvature.py',
            'python -I experiments/e7_a5_extension.py',
            'python -I experiments/e8_stability_map.py',
            'python -I experiments/e9_work_precision.py',
            'python -I run_phase4.py',
            'python -I make_tables.py',
            'python -I make_figures.py',
            'python -I checks.py --write-manifest',
        ],
        experiment_timings=timings,
        legacy_solver=dict(path=cfg.LEGACY_SOLVER_DIR, files=legacy),
        code=file_rows(['**/*.py', 'README.md'], cfg.CODE_DIR),
        data=[r for r in file_rows(['*.json'], cfg.DATA_DIR)
              if not r['path'].endswith('manifest.json')],
        tables=file_rows(['tables/*.csv', 'tables/*.md'], cfg.DATA_DIR),
        figures=file_rows(['*.pdf', '*.png'], cfg.FIG_DIR),
        reports=file_rows(['A1_BDF2_Taylor_Numerics*.md',
                           'A1_Numerics_Evidence_Interpretation*.md'],
                          cfg.NUM_DIR),
        notes=dict(flow='V=(<kappa>^n-kappa)n, dimensionless L0=1',
                   seed=cfg.SEED,
                   scheme='BDF2 + second-order Taylor + predicted geometry',
                   legacy_use='E6 prior-work comparison only, read-only'),
    )
    path = os.path.join(cfg.DATA_DIR, 'manifest.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return path


def main(write_manifest=False):
    print('=' * 78)
    print('A1_Numerics checks')
    print('=' * 78)
    ok = True
    e0_path = os.path.join(cfg.DATA_DIR, 'e0_verification.json')
    if os.path.exists(e0_path):
        e0 = json.load(open(e0_path, encoding='utf-8'))
        print('E0 gate:', 'PASS' if e0.get('all_pass') else 'FAIL')
        ok = ok and bool(e0.get('all_pass'))
    else:
        print('E0 gate: MISSING')
        ok = False
    for name in JSON_FILES:
        p = os.path.join(cfg.DATA_DIR, name)
        good = os.path.exists(p) and os.path.getsize(p) > 100
        print(f'  JSON {name:24s} {"ok" if good else "MISSING"}')
        ok = ok and good
    for name in FIG_NAMES:
        for ext in ('pdf', 'png'):
            p = os.path.join(cfg.FIG_DIR, f'{name}.{ext}')
            good = os.path.exists(p)
            print(f'  FIG  {name}.{ext:4s} {"ok" if good else "MISSING"}')
            ok = ok and good
    n_tables = len(glob.glob(os.path.join(cfg.TABLE_DIR, '*.csv')))
    print(f'  tables: {n_tables} CSV files')
    ok = ok and n_tables >= 10
    st = check_structure()
    print('  report structure:', st)
    if st.get('en_exists') and st.get('zh_exists'):
        ok = ok and all(st.get(k, False) for k in
                        ('headings_match', 'tables_match', 'figs_match',
                         'tags_match'))
    if write_manifest:
        p = build_manifest()
        print('manifest:', p)
    print('\nchecks:', 'PASS' if ok else 'ATTENTION NEEDED')
    return 0 if ok else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--write-manifest', action='store_true')
    a = ap.parse_args()
    sys.exit(main(write_manifest=a.write_manifest))
