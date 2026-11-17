"""
make_figures.py -- generate the report figures (PDF + PNG).

Black/white distinguishable line styles and markers; no colour-only encoding.
Usage: python -I make_figures.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import a1n_config as cfg

STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
MARKERS = ['o', 's', '^', 'D', 'v', '<']
LW = 1.4
FS = 9


def _save(fig, name):
    pdf = os.path.join(cfg.FIG_DIR, name + '.pdf')
    png = os.path.join(cfg.FIG_DIR, name + '.png')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', pdf, 'and', png)


def _loglog_axes(ax, xlabel, ylabel, title=None):
    from matplotlib.ticker import LogLocator
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.tick_params(labelsize=FS)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.tick_params(axis='both', which='minor', labelbottom=False,
                   labelleft=False)
    ax.grid(True, which='both', ls=':', lw=0.4, alpha=0.5)
    if title:
        ax.set_title(title, fontsize=FS)


def _ref_line(ax, x, y0, slope, label):
    x = np.asarray(x, dtype=float)
    y = y0 * (x / x[0]) ** slope
    ax.plot(x, y, 'k:', lw=1.0, label=label)


# ---------------------------------------------------------------------------
# Fig 1 -- E1 closure
# ---------------------------------------------------------------------------

def fig1():
    d = cfg.load_json('e1_closure.json')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for j, (key, ttl) in enumerate((('D_h_0', 't = 0'), ('D_h_T', 't = T'))):
        ax = axes[j]
        for k, r in enumerate(d['results']):
            xs = [x['h'] for x in r['rows'] if x.get(key) is not None]
            ys = [x[key] for x in r['rows'] if x.get(key) is not None]
            if len(xs) < 2:
                continue
            ax.plot(xs, ys, marker=MARKERS[k], ls=STYLES[k], lw=LW,
                    ms=3.5, label=r['shape'])
        # h^2 reference anchored to the ellipse at its smallest h
        ell = d['results'][1]['rows']
        x0 = min(x['h'] for x in ell)
        y0 = [x[key] for x in ell if x['h'] == x0][0]
        xs = sorted(x['h'] for x in ell)
        ax.plot(xs, [y0 * (x / x0) ** 2 for x in xs], 'k:', lw=1.0,
                label=r'$h^2$ (anchored)')
        _loglog_axes(ax, r'$h$', r'$D_h$', ttl)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle('E1: discrete Gauss-Bonnet defect $D_h$', fontsize=FS + 1)
    fig.tight_layout()
    _save(fig, 'fig1_closure')


# ---------------------------------------------------------------------------
# Fig 2 -- E2 spatial
# ---------------------------------------------------------------------------

def fig2():
    d = cfg.load_json('e2_spatial.json')
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    r = [x for x in d['results'] if x['n'] == 2][0]
    hs = [x['h'] for x in r['rows']]
    ax.plot(hs, [x['full_L2'] for x in r['rows']], marker='o', ls='-',
            lw=LW, ms=4, label=r'full $L^2$ (theorem norm)')
    ax.plot(hs, [x['full_H1'] for x in r['rows']], marker='s', ls='--',
            lw=LW, ms=4, label=r'full $H^1$ (theorem norm)')
    ax.plot(hs, [x['nodal_L2'] for x in r['rows']], marker='^', ls='-.',
            lw=1.1, ms=3.5, label=r'nodal $L^2$ ($e_h$)')
    ax.plot(hs, [x['nodal_H1'] for x in r['rows']], marker='D', ls=':',
            lw=1.1, ms=3.5, label=r'nodal $H^1$ ($e_h$)')
    x0 = hs[0]
    y0 = r['rows'][0]['full_L2']
    y1 = r['rows'][0]['full_H1']
    _ref_line(ax, hs, y0, 2.0, r'$h^2$')
    _ref_line(ax, hs, y1, 1.0, r'$h$')
    _loglog_axes(ax, r'$h$', 'error', r'E2: ellipse, $\nu=2$')
    ax.legend(fontsize=6.8, frameon=False)
    fig.tight_layout()
    _save(fig, 'fig2_spatial')


# ---------------------------------------------------------------------------
# Fig 3 -- E3 circle time order
# ---------------------------------------------------------------------------

def fig3():
    d = cfg.load_json('e3_temporal.json')
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    k = 0
    for r in d['circle_time_order']:
        if r['n'] == 1:
            continue
        errs = [x['err_modified'] for x in r['rows']]
        taus = [x['tau'] for x in r['rows']]
        ax.plot(taus, errs, marker=MARKERS[k], ls=STYLES[k], lw=LW, ms=3.5,
                label=rf"$\nu={r['n']}$ {r['predictor']}")
        k += 1
    tau0 = [x['tau'] for x in d['circle_time_order'][2]['rows']]
    _ref_line(ax, tau0, 1e-3, 2.0, r'$\tau^2$')
    _ref_line(ax, tau0, 1e-5, 3.0, r'$\tau^3$')
    _loglog_axes(ax, r'$\tau$', r'$|R(T)-R_{\rm mod}(T)|$',
                 'E3: circle time order (modified ODE)')
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    _save(fig, 'fig3_temporal')


# ---------------------------------------------------------------------------
# Fig 4 -- stability / coupling
# ---------------------------------------------------------------------------

def fig4():
    e3 = cfg.load_json('e3_temporal.json')
    e6 = cfg.load_json('e6_comparison.json')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    fail_y = 2.0e4

    def mark_failed(ax, xs, failed, label='failed'):
        xs = np.asarray([x for x, fl in zip(xs, failed) if fl], dtype=float)
        if len(xs):
            ax.plot(xs, np.full_like(xs, fail_y), 'kx', ms=6, mew=1.2,
                    label=label)

    ax = axes[0]
    k = 0
    for r in e3['coupling_map']:
        if r['n'] != 2:
            continue
        for N in (256, 512):
            sub = [x for x in r['rows'] if x['N'] == N]
            xs = [x['tau_over_h'] for x in sub]
            ys = [x['psi_max'] for x in sub]
            ax.plot(xs, ys, marker=MARKERS[k], ls=STYLES[k], lw=LW, ms=3.5,
                    label=f'N={N}')
            mark_failed(ax, xs, [x['failed'] for x in sub],
                        label='failed' if k == 0 else '_nolegend_')
            k += 1
    ax.axhline(1e4, color='k', ls=':', lw=1.0)
    _loglog_axes(ax, r'$\tau/h$', r'$\Psi_{\max}$', r'E3: BDF2-Taylor, $\nu=2$')
    ax.set_ylim(top=5e4)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    k = 0
    for scheme in ('BDF2-Taylor', 'BGN1', 'CNLF-EX'):
        sub = [r for r in e6['stability_map']
               if r['shape'] == 'flower' and r['n'] == 2 and r['scheme'] == scheme]
        xs = [r['tau_over_h'] for r in sub]
        ys = [r['psi_max'] for r in sub]
        ax.plot(xs, ys, marker=MARKERS[k], ls=STYLES[k], lw=LW, ms=3.5,
                label=scheme)
        mark_failed(ax, xs, [r['failed'] for r in sub],
                    label='failed' if k == 0 else '_nolegend_')
        k += 1
    ax.axhline(1e4, color='k', ls=':', lw=1.0)
    _loglog_axes(ax, r'$\tau/h$', r'$\Psi_{\max}$', r'E6: flower, $\nu=2$')
    ax.set_ylim(top=5e4)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle('Tangential mesh quality: max mesh ratio (x = failed)',
                 fontsize=FS + 1)
    fig.tight_layout()
    _save(fig, 'fig4_stability')


# ---------------------------------------------------------------------------
# Fig 5 -- cost / conditioning
# ---------------------------------------------------------------------------

def fig5():
    d = cfg.load_json('e5_cost.json')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    for k, scheme in enumerate(('BDF2-Taylor', 'BGN1', 'CNLF-EX')):
        sub = [r for r in d['cost'] if r['scheme'] == scheme]
        sub.sort(key=lambda r: r['N'])
        ax.plot([r['N'] for r in sub], [r['ms_per_step'] for r in sub],
                marker=MARKERS[k], ls=STYLES[k], lw=LW, ms=3.5, label=scheme)
    _loglog_axes(ax, r'$N$', 'ms / step', 'E5: cost per step')
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for n in (1, 2, 3):
        sub = [r for r in d['denominator_condition'] if r['n'] == n]
        ax.plot([r['N'] for r in sub], [r['denom_min'] for r in sub],
                marker=MARKERS[n - 1], ls=STYLES[n - 1], lw=LW, ms=3.5,
                label=rf'$\nu={n}$')
    ax.set_xscale('log')
    ax.set_xlabel(r'$N$', fontsize=FS)
    ax.set_ylabel(r'$\min |1+v^\top z|$', fontsize=FS)
    ax.set_title('E5: Sherman-Morrison denominator', fontsize=FS)
    ax.tick_params(labelsize=FS)
    ax.grid(True, which='both', ls=':', lw=0.4, alpha=0.5)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, 'fig5_cost')


# ---------------------------------------------------------------------------
# Fig 6 -- n-sensitivity
# ---------------------------------------------------------------------------

def fig6():
    d = cfg.load_json('e6_comparison.json')
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for k, sid in enumerate(('circle', 'ellipse')):
        sub = [r for r in d['n_sensitivity'] if r['shape'] == sid]
        ax.semilogy([r['n'] for r in sub], [r['C_h2'] for r in sub],
                    marker=MARKERS[k], ls=STYLES[k], lw=LW, ms=3.5,
                    label=sid)
    ax.set_xlabel(r'$\nu$', fontsize=FS)
    ax.set_ylabel(r'$E/(h^2+\tau^2)$', fontsize=FS)
    ax.set_title(r'E6: $\nu$-sensitivity of the error constant', fontsize=FS)
    ax.tick_params(labelsize=FS)
    ax.grid(True, which='both', ls=':', lw=0.4, alpha=0.5)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, 'fig6_n_sensitivity')


def main():
    print('figures ->', cfg.FIG_DIR)
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()


if __name__ == '__main__':
    main()
