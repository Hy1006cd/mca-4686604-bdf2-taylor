"""
make_paper_figures.py -- paper-specific versions of Fig. 4 and Fig. 5.

The numerical report figures include the legacy CNLF-EX baseline.  The paper
compares only with the first-order BGN baseline, so these two figures are
regenerated from the same JSON data with CNLF-EX removed.

Usage: python -I make_paper_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'A1_Numerics', 'data')
FIG = HERE

STYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'D']
LW = 1.4
FS = 9


def load(name):
    with open(os.path.join(DATA, name), encoding='utf-8') as f:
        return json.load(f)


def loglog(ax, xlabel, ylabel, title=None):
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


def mark_failed(ax, xs, failed, y=2.0e4, label='failed'):
    xs = np.asarray([x for x, fl in zip(xs, failed) if fl], dtype=float)
    if len(xs):
        ax.plot(xs, np.full_like(xs, y), 'kx', ms=6, mew=1.2,
                label=label)


def fig4():
    e3 = load('e3_temporal.json')
    e6 = load('e6_comparison.json')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    fail_y = 2.0e4

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
            mark_failed(ax, xs, [x['failed'] for x in sub], fail_y,
                        label='failed' if k == 0 else '_nolegend_')
            k += 1
    ax.axhline(1e4, color='k', ls=':', lw=1.0)
    loglog(ax, r'$\tau/h$', r'$\Psi_{\max}$', r'BDF2-Taylor, $\nu=2$')
    ax.set_ylim(top=5e4)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for k, scheme in enumerate(('BDF2-Taylor', 'BGN1')):
        sub = [r for r in e6['stability_map']
               if r['shape'] == 'flower' and r['n'] == 2
               and r['scheme'] == scheme]
        xs = [r['tau_over_h'] for r in sub]
        ys = [r['psi_max'] for r in sub]
        ax.plot(xs, ys, marker=MARKERS[k], ls=STYLES[k], lw=LW, ms=3.5,
                label=scheme)
        mark_failed(ax, xs, [r['failed'] for r in sub], fail_y,
                    label='failed' if k == 0 else '_nolegend_')
    ax.axhline(1e4, color='k', ls=':', lw=1.0)
    loglog(ax, r'$\tau/h$', r'$\Psi_{\max}$', r'flower, $\nu=2$')
    ax.set_ylim(top=5e4)
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle('Tangential mesh quality: max mesh ratio (x = failed)',
                 fontsize=FS + 1)
    fig.tight_layout()
    out = os.path.join(FIG, 'fig4_stability_paper.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


def fig5():
    d = load('e5_cost.json')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    for k, scheme in enumerate(('BDF2-Taylor', 'BGN1')):
        sub = [r for r in d['cost'] if r['scheme'] == scheme]
        sub.sort(key=lambda r: r['N'])
        ax.plot([r['N'] for r in sub], [r['ms_per_step'] for r in sub],
                marker=MARKERS[k], ls=STYLES[k], lw=LW, ms=3.5, label=scheme)
    loglog(ax, r'$N$', 'ms / step', 'cost per step')
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
    ax.set_title('Sherman-Morrison denominator', fontsize=FS)
    ax.tick_params(labelsize=FS)
    ax.grid(True, which='both', ls=':', lw=0.4, alpha=0.5)
    ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    out = os.path.join(FIG, 'fig5_cost_paper.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


def _save(fig, name):
    out = os.path.join(FIG, name + '.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


def fig6():
    d = load('e8_stability_map.json')
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, shape in zip(axes, ('ellipse', 'flower')):
        sub = [r for r in d['rows'] if r['shape'] == shape]
        Ns = sorted({r['N'] for r in sub})
        taus = sorted({r['tau'] for r in sub})
        Z = np.full((len(Ns), len(taus)), np.nan)
        F = np.zeros_like(Z, dtype=bool)
        for r in sub:
            i = Ns.index(r['N'])
            j = taus.index(r['tau'])
            if r.get('psi_max') is not None:
                Z[i, j] = np.log10(max(float(r['psi_max']), 1e-12))
            F[i, j] = bool(r['failed'])
        im = ax.imshow(Z, origin='lower', aspect='auto', cmap='viridis',
                       vmin=0.0, vmax=4.0)
        ax.set_xticks(range(len(taus)))
        ax.set_xticklabels([f'{t:.2e}' for t in taus], rotation=45,
                           ha='right', fontsize=6.5)
        ax.set_yticks(range(len(Ns)))
        ax.set_yticklabels([str(n) for n in Ns], fontsize=7)
        for i in range(len(Ns)):
            for j in range(len(taus)):
                if F[i, j]:
                    ax.plot(j, i, 'kx', ms=6, mew=1.2)
        ax.set_xlabel(r'$\tau$', fontsize=FS)
        ax.set_ylabel(r'$N$', fontsize=FS)
        ax.set_title(shape, fontsize=FS)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r'$\log_{10}\Psi_{\max}$', fontsize=FS)
        cbar.ax.tick_params(labelsize=6.5)
    fig.suptitle(r'E8: $(h,\tau)$ stability map ($\nu=2$, $T=0.5$; '
                 r'$\times$ = failed)', fontsize=FS + 1)
    fig.tight_layout()
    _save(fig, 'fig6_e8_stability')


def fig7():
    d = load('e9_work_precision.json')
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, shape in zip(axes, ('circle', 'ellipse')):
        for k, scheme in enumerate(('BDF2-Taylor', 'BGN1')):
            sub = [r for r in d['rows']
                   if r['shape'] == shape and r['scheme'] == scheme
                   and not r['failed'] and r.get('error') is not None
                   and r.get('time_median') is not None]
            sub.sort(key=lambda r: r['time_median'])
            ax.loglog([r['time_median'] for r in sub],
                      [r['error'] for r in sub],
                      marker=MARKERS[k], ls=STYLES[k], lw=LW, ms=4,
                      label=scheme)
        ax.set_xlabel('CPU time (s)', fontsize=FS)
        ax.set_ylabel('error', fontsize=FS)
        ax.set_title(shape, fontsize=FS)
        ax.tick_params(labelsize=FS)
        ax.grid(True, which='both', ls=':', lw=0.4, alpha=0.5)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(r'E9: work--precision ($N=512$, $T=0.5$, $\nu=2$)',
                 fontsize=FS + 1)
    fig.tight_layout()
    _save(fig, 'fig7_e9_work_precision')


if __name__ == '__main__':
    fig4()
    fig5()
    fig6()
    fig7()
