"""
=============================================================================
ГРАФИКИ ДЛЯ ЭТАПОВ ПОДГОТОВКИ ДАННЫХ И ПОКАЗАТЕЛЕЙ
Оформление по общим требованиям: запятая как десятичный разделитель,
палитра проекта, 300 DPI при сохранении, без лишних элементов.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.dates as mdates

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PALETTE, cf, comma_fmt, style_axes

REGION_COLORS = {'НСО': PALETTE[0], 'Омская обл.': PALETTE[1]}


def _clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def completeness_heatmap(comp, var, names=None, figsize=(12, 8)):
    """Полнота рядов (%) по станциям и годам для одной переменной."""
    d = comp[comp['Переменная'] == var]
    t = d.pivot_table(index=['Регион', 'Станция'], columns='Год', values='Полнота, %')
    fig, ax = plt.subplots(figsize=(figsize[0], max(4, 0.22 * len(t) + 1.5)))
    cmap = plt.get_cmap('viridis')
    im = ax.imshow(t.values, aspect='auto', cmap=cmap, vmin=50, vmax=100)
    ax.set_xticks(range(t.shape[1]))
    ax.set_xticklabels(t.columns, fontsize=9, rotation=0)
    lab = [f'{names.get(s, s) if names else s}' for _, s in t.index]
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(lab, fontsize=7)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            if np.isfinite(v) and v < 99.5:
                ax.text(j, i, cf(v, 0), ha='center', va='center', fontsize=6,
                        color='white' if v < 80 else 'black')
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label('Полнота, %', fontsize=10)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: cf(v, 0)))
    # разделитель регионов
    regs = [r for r, _ in t.index]
    for i in range(1, len(regs)):
        if regs[i] != regs[i - 1]:
            ax.axhline(i - 0.5, color='white', lw=2)
    ax.set_title(f'Полнота рядов {var} по станциям и годам (значения < 100 % подписаны)',
                 fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()
    return fig


def qc_series(df, comp, st, log=None, clean=None, filled_log=None, title='', ylabel='', figsize=(10, 5),
              period=None):
    """Ряд одной станции: исходные значения, ошибочные, требующие проверки, исправленные и восстановленные."""
    d = df[df['st'] == st].set_index('date')[comp]
    if period:
        d = d[period[0]:period[1]]
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(d.index, d.values, color='#9AA5B1', lw=0.7, zorder=1, label='Исходные данные')
    if clean is not None:
        c = clean[clean['st'] == st].set_index('date')[comp]
        c = c.reindex(d.index)
        ax.plot(c.index, c.values, color=PALETTE[0], lw=0.9, zorder=2, label='После очистки')
    handles = [Line2D([0], [0], color='#9AA5B1', lw=1.2, label='Исходные данные')]
    if clean is not None:
        handles.append(Line2D([0], [0], color=PALETTE[0], lw=1.2, label='После очистки'))
    if log is not None and not log.empty:
        L = log[(log['Станция'] == st) & (log['Компонент'] == comp) & log['Основное']]
        L = L[pd.to_datetime(L['Дата']).isin(d.index)]
        for act, col, mk in [('ошибочное', PALETTE[3], 'x'), ('требует проверки', PALETTE[1], 'o'),
                             ('исправленное', PALETTE[4], 's')]:
            s = L[L['Действие'] == act]
            if len(s):
                ax.scatter(pd.to_datetime(s['Дата']), s['Значение'], color=col, marker=mk, s=40,
                           zorder=5, lw=1.5 if mk == 'x' else 0.8, edgecolors='k' if mk != 'x' else None)
                handles.append(Line2D([0], [0], color=col, marker=mk, ls='', label=f'{act.capitalize()} ({len(s)})'))
    if filled_log is not None and not filled_log.empty:
        F = filled_log[(filled_log['st'] == st) & (filled_log['comp'] == comp)]
        F = F[F['date'].isin(d.index)]
        if len(F):
            ax.scatter(F['date'], F['value'], color=PALETTE[2], s=10, zorder=4)
            handles.append(Line2D([0], [0], color=PALETTE[2], marker='o', ls='', ms=4,
                                  label=f'Восстановленное ({len(F)})'))
    ax.yaxis.set_major_formatter(comma_fmt(0))
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(handles=handles, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=min(5, len(handles)), frameon=False)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    _clean(ax)
    fig.tight_layout()
    return fig


def cv_bars(cv, comp, figsize=(8, 5)):
    """RMSE и MAE методов восстановления пропусков по перекрёстной проверке."""
    d = cv[cv['Компонент'] == comp].dropna(subset=['RMSE']).sort_values('RMSE')
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(d))
    best = d['RMSE'].idxmin()
    cols = [PALETTE[1] if i == best else PALETTE[0] for i in d.index]
    ax.barh(y, d['RMSE'], color=cols, height=0.6)
    for yi, (r, m, cov) in enumerate(zip(d['RMSE'], d['MAE'], d['Восстановлено, %'])):
        ax.text(r, yi, f'  RMSE {cf(r, 2)}; MAE {cf(m, 2)}; восст. {cf(cov, 0)} %', va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(d['Описание'], fontsize=10)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(comma_fmt(1))
    ax.set_xlim(0, d['RMSE'].max() * 1.9)
    ax.set_xlabel('RMSE на скрытых значениях', fontsize=12)
    ax.set_title(f'Точность методов восстановления: {comp}', fontsize=13, fontweight='bold', pad=12)
    _clean(ax)
    fig.tight_layout()
    return fig


def index_series(ann, code, label, regions=None, trend=True, figsize=(8, 5)):
    """Средний по станциям ряд показателя для каждого региона, с наклоном Сена."""
    from .trends import sen_slope, mann_kendall
    fig, ax = plt.subplots(figsize=figsize)
    a = ann[ann['Показатель'] == code]
    regions = regions or sorted(a['region'].dropna().unique())
    for i, rg in enumerate(regions):
        s = a[a['region'] == rg].groupby('year')['Значение'].mean()
        if s.empty:
            continue
        col = REGION_COLORS.get(rg, PALETTE[i % len(PALETTE)])
        ax.plot(s.index, s.values, 'o-', color=col, lw=1.8, ms=5, label=rg)
        if trend and len(s) >= 5:
            b, a0 = sen_slope(s.values, s.index.values)
            mk = mann_kendall(s.values)
            ax.plot(s.index, a0 + b * s.index.values, '--', color='#E63946' if len(regions) == 1 else col, lw=2,
                    alpha=0.9)
            ax.text(s.index[-1], a0 + b * s.index[-1], f'  {cf(b * 10, 2)} за 10 лет'
                    f'{" *" if mk["p"] < 0.05 else ""}', fontsize=8, color=col, va='center')
    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.yaxis.set_major_formatter(comma_fmt(1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if len(regions) > 1:
        ax.legend(fontsize=10, frameon=False)
    ax.set_title(label.split(',')[0], fontsize=13, fontweight='bold', pad=12)
    _clean(ax)
    fig.tight_layout()
    return fig


def spell_classes_bar(reg_table, title='', figsize=(8, 5)):
    """Повторяемость периодов по градациям продолжительности, % — по регионам."""
    cols = [c for c in reg_table.columns if c.endswith('%')]
    fig, ax = plt.subplots(figsize=figsize)
    n = len(reg_table)
    w = 0.8 / max(n, 1)
    x = np.arange(len(cols))
    for i, (_, r) in enumerate(reg_table.iterrows()):
        vals = r[cols].values.astype(float)
        col = REGION_COLORS.get(r['region'], PALETTE[i % len(PALETTE)])
        b = ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.95, color=col, label=r['region'])
        for xi, v in zip(x + (i - (n - 1) / 2) * w, vals):
            ax.text(xi, v + 0.8, cf(v, 1), ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' дн., %', '') for c in cols], fontsize=11)
    ax.set_xlabel('Продолжительность периода, дни', fontsize=12)
    ax.set_ylabel('Повторяемость, %', fontsize=12)
    ax.yaxis.set_major_formatter(comma_fmt(0))
    if n > 1:
        ax.legend(fontsize=10, frameon=False)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    _clean(ax)
    fig.tight_layout()
    return fig


def period_compare_bar(cmp, code, label, figsize=(8, 5)):
    d = cmp[cmp['Показатель'] == code]
    if d.empty:
        return None
    c1 = [c for c in d.columns if c.startswith('Среднее')][:2]
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(d))
    for i, c in enumerate(c1):
        ax.bar(x + (i - 0.5) * 0.38, d[c], 0.36, color=PALETTE[i], label=c.replace('Среднее ', ''))
        for xi, v in zip(x + (i - 0.5) * 0.38, d[c]):
            ax.text(xi, v / 2, cf(v, 1), ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    key = d.columns[0]
    ax.set_xticks(x)
    ax.set_xticklabels(d[key].astype(str), fontsize=10)
    ax.set_ylabel(label, fontsize=12)
    ax.yaxis.set_major_formatter(comma_fmt(1))
    ax.legend(fontsize=10, frameon=False)
    ax.set_title(f'{label.split(",")[0]}: сравнение периодов', fontsize=13, fontweight='bold', pad=12)
    _clean(ax)
    fig.tight_layout()
    return fig


def qc_summary_bar(summary, figsize=(8, 5)):
    """Число ошибочных, исправленных и требующих проверки значений по переменным."""
    t = summary.groupby(['Переменная', 'Действие'])['Значений'].sum().unstack(fill_value=0)
    order = [c for c in ['ошибочное', 'исправленное', 'требует проверки'] if c in t.columns]
    t = t[order]
    fig, ax = plt.subplots(figsize=figsize)
    cols = {'ошибочное': PALETTE[3], 'исправленное': PALETTE[4], 'требует проверки': PALETTE[1]}
    left = np.zeros(len(t))
    for c in order:
        ax.barh(t.index, t[c], left=left, color=cols[c], label=c.capitalize(), height=0.6)

        for i, (l, v) in enumerate(zip(left, t[c])):
            if v > t.values.sum() * 0.02:
                ax.text(l + v / 2, i, f'{int(v)}', ha='center', va='center', color='white', fontsize=9,
                        fontweight='bold')
        left += t[c].values
    ax.set_xlabel('Число значений', fontsize=12)
    ax.legend(fontsize=10, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax.set_title('Результаты контроля качества по переменным', fontsize=13, fontweight='bold', pad=12)
    ax.invert_yaxis()
    _clean(ax)
    fig.tight_layout()
    return fig
