"""
=============================================================================
КОНФИГУРАЦИЯ: палитры, форматирование чисел, стиль графиков
Единая точка настройки внешнего вида — используется всеми модулями.
=============================================================================
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ─── Палитры (из Промпта 1) ──────────────────────────────────────────────
PALETTE = ['#2E86AB', '#F18F01', '#6A994E', '#E63946', '#A06CD5']
CMAP_GRADIENT = plt.cm.viridis
CMAP_DIVERGING = plt.cm.coolwarm

# ─── Размеры графиков (дюймы) ────────────────────────────────────────────
FIGSIZE_SIMPLE = (8, 5)
FIGSIZE_MATRIX = (10, 8)
FIGSIZE_MULTI = (12, 8)
FIGSIZE_LARGE = (14, 12)
DPI = 300

# ─── rcParams по умолчанию ───────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'figure.dpi': 150,       # экранный dpi ниже для скорости отрисовки в Streamlit
    'savefig.dpi': DPI,      # dpi при сохранении файла — 300, как требуется
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})


def cf(x, dec=2):
    """Форматирование числа с запятой как разделителем (европейский стандарт)."""
    if x is None:
        return '—'
    try:
        import numpy as np
        if isinstance(x, float) and np.isnan(x):
            return '—'
    except Exception:
        pass
    return f'{x:.{dec}f}'.replace('.', ',')


def comma_fmt(dec=2):
    """Formatter для осей matplotlib с запятой вместо точки."""
    return FuncFormatter(lambda x, p: cf(x, dec))


def pfmt(p):
    """Форматирование p-value: '< 0,001' для очень малых значений."""
    if p is None:
        return '—'
    return '< 0,001' if p < 0.001 else cf(p, 3)


def sig_stars(p):
    """Звёздочки значимости: * p<0.05, ** p<0.01, *** p<0.001."""
    if p is None:
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def style_axes(ax):
    """Убирает верхнюю и правую рамку, добавляет мягкую сетку (правило Тафти)."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')


def df_to_comma_strings(df, decimals=2, exclude_cols=None):
    """
    Возвращает копию DataFrame, где числовые столбцы отформатированы
    строками с запятой — для сохранения в Excel в требуемом виде.
    """
    import pandas as pd
    exclude_cols = exclude_cols or []
    out = df.copy()
    for col in out.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].apply(lambda v: cf(v, decimals))
    return out
