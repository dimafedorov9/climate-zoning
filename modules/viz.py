"""
=============================================================================
МОДУЛЬ ВИЗУАЛИЗАЦИИ
Графики строго по стилю из ТЗ (Промпт 1): запятые, палитра, DPI=300,
правило Тафти, минималистичный дизайн для публикаций Q1.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy import stats as sstats

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PALETTE, comma_fmt, cf, style_axes, DPI


CHART_DESCRIPTIONS = {
    'hist_kde': ('Гистограмма с кривой плотности (KDE)',
                 'Показывает форму распределения значений: где сосредоточена основная '
                 'масса наблюдений, есть ли асимметрия или несколько «пиков». '
                 'Вертикальные линии — среднее и медиана; их расхождение указывает '
                 'на асимметрию распределения.'),
    'boxplot': ('Ящик с усами (боксплот) по группам',
                'Сравнивает распределение показателя между группами (станциями, '
                'регионами, периодами): медиана, межквартильный размах, выбросы. '
                'Красный ромб — среднее, чёрная линия — медиана.'),
    'heatmap': ('Тепловая карта корреляций',
                'Показывает силу и направление парных связей между признаками: '
                'синий — отрицательная корреляция, красный — положительная. '
                'Чем ближе значение к ±1, тем сильнее связь.'),
    'scatter': ('Диаграмма рассеяния с линией тренда',
                'Показывает связь между двумя переменными: по расположению точек и '
                'наклону линии тренда видно, растёт ли одна величина вместе с другой '
                'и насколько тесно (r в заголовке).'),
    'timeseries': ('Динамика показателя во времени',
                   'Изменение значения показателя по годам для выбранной станции, '
                   'региона или в среднем по всем станциям. Позволяет увидеть тренд, '
                   'колебания и переломные периоды.'),
    'timeseries_combo': ('Совмещённая динамика двух показателей',
                         'Два показателя (например температура и осадки) на одном '
                         'графике с двумя осями Y — удобно для визуального '
                         'сопоставления их динамики за один и тот же период.'),
    'pettitt': ('Тест Петтитта на однородность ряда',
               'Верхняя панель — временной ряд с отмеченной точкой разрыва '
               'однородности; нижняя — профиль статистики U_t, пик которой '
               'указывает год наиболее вероятного изменения режима.'),
    'importance': ('Важность признаков (комбинированная оценка)',
                   'Ранжирует признаки по совокупной важности для предсказания '
                   'целевой переменной, объединяя несколько методов (RF, GB, '
                   'permutation importance, MI) в одну нормированную оценку.'),
}

LEGEND_LOC_OPTIONS = {
    'Авто (best)': 'best',
    'Сверху справа': 'upper right',
    'Сверху слева': 'upper left',
    'Снизу справа': 'lower right',
    'Снизу слева': 'lower left',
    'Снаружи справа': 'center left',  # используется вместе с bbox_to_anchor=(1.02, 0.5)
    'Под графиком': 'lower center',
}


def _apply_legend(ax, loc_label, ncol=1, outside=False):
    """Применяет позицию легенды по человекочитаемому названию из LEGEND_LOC_OPTIONS."""
    loc = LEGEND_LOC_OPTIONS.get(loc_label, 'best')
    if loc_label == 'Снаружи справа':
        ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=ncol)
    elif loc_label == 'Под графиком':
        ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=ncol)
    else:
        ax.legend(fontsize=9, loc=loc, ncol=ncol)


def hist_kde(series, title='', xlabel='', figsize=(8, 5), color=None, legend_loc='Авто (best)'):
    """Гистограмма с KDE: частота слева, плотность справа (красная линия)."""
    color = color or PALETTE[0]
    x = pd.to_numeric(series, errors='coerce').dropna().values
    fig, ax1 = plt.subplots(figsize=figsize)

    counts, bins, patches = ax1.hist(x, bins='auto', color=color,
                                      alpha=0.75, edgecolor='black', linewidth=0.6)
    ax1.set_ylabel('Частота', fontweight='bold', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    if len(x) > 1 and np.std(x) > 0:
        kde = sstats.gaussian_kde(x)
        xs = np.linspace(x.min(), x.max(), 300)
        ax2.plot(xs, kde(xs), color='#E63946', lw=2.2, label='Плотность (KDE)')
    ax2.set_ylabel('Плотность', fontweight='bold', color='#E63946')
    ax2.tick_params(axis='y', labelcolor='#E63946')

    mean_v, med_v = np.mean(x), np.median(x)
    ax1.axvline(mean_v, color='#333', ls='--', lw=1.8,
                label=f'Среднее: {cf(mean_v)}')
    ax1.axvline(med_v, color='#666', ls=':', lw=1.8,
                label=f'Медиана: {cf(med_v)}')

    ax1.set_xlabel(xlabel, fontweight='bold')
    ax1.set_title(title, fontweight='bold', pad=14)
    ax1.xaxis.set_major_formatter(comma_fmt(1))
    style_axes(ax1)
    ax2.spines['top'].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='best')
    _apply_legend(ax1, legend_loc)

    fig.tight_layout()
    return fig


def boxplot_groups(df, value_col, group_col, title='', ylabel='', figsize=(8, 5),
                    colors=None, legend_loc='Авто (best)', annotate_minmax=False):
    """
    Ящики с усами по группам: цветные заливки, среднее — красный ромб,
    медиана — чёрная жирная линия, легенда без дублирования (Line2D).
    Группы без данных (пустые после отбрасывания NaN) исключаются заранее —
    matplotlib на некоторых версиях падает на пустых массивах.

    annotate_minmax: если True — подписи "Мин/Макс" рисуются прямо на графике
    (может накладываться при многих группах); если False (по умолчанию) —
    подписи не рисуются на графике, сводка выводится отдельно текстом/таблицей
    через boxplot_minmax_summary().
    """
    groups_all = sorted(df[group_col].dropna().unique())
    groups, data = [], []
    for g in groups_all:
        d = pd.to_numeric(df.loc[df[group_col] == g, value_col],
                           errors='coerce').dropna().values
        if len(d) > 0:
            groups.append(g)
            data.append(d)

    fig, ax = plt.subplots(figsize=figsize)
    if not data:
        ax.text(0.5, 0.5, 'Нет данных для построения графика',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                     medianprops={'color': 'black', 'lw': 2.2},
                     whiskerprops={'lw': 1.3}, capprops={'lw': 1.3})
    # Подписи групп задаём отдельно (не через 'labels'/'tick_labels' —
    # имя параметра менялось между версиями matplotlib)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([str(g) for g in groups],
                        rotation=45 if len(groups) > 8 else 0,
                        ha='right' if len(groups) > 8 else 'center')

    colors = colors or (PALETTE * (len(groups) // len(PALETTE) + 1))[:len(groups)]
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    for i, d in enumerate(data, start=1):
        m = np.mean(d)
        ax.scatter(i, m, color='red', marker='D', s=55, zorder=5)
        if annotate_minmax:
            mx, mn = np.max(d), np.min(d)
            ax.text(i, mx, f'Макс: {cf(mx)}', ha='center', va='bottom', fontsize=8)
            ax.text(i, mn, f'Мин: {cf(mn)}', ha='center', va='top', fontsize=8)

    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=14)
    ax.yaxis.set_major_formatter(comma_fmt(1))
    style_axes(ax)

    legend_elems = [Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
                            markersize=8, label='Среднее'),
                    Line2D([0], [0], color='black', lw=2.2, label='Медиана')]
    loc = LEGEND_LOC_OPTIONS.get(legend_loc, 'best')
    if legend_loc == 'Снаружи справа':
        ax.legend(handles=legend_elems, fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5))
    elif legend_loc == 'Под графиком':
        ax.legend(handles=legend_elems, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    else:
        ax.legend(handles=legend_elems, fontsize=9, loc=loc)

    fig.tight_layout()
    return fig


def correlation_heatmap(corr_df, title='', figsize=(10, 8), mask_upper=True, cmap='coolwarm'):
    """Тепловая карта корреляций: маска верхнего треугольника, аннотации с запятой."""
    import seaborn as sns

    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr_df, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    hm = sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                      center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                      cbar_kws={'label': 'Коэффициент корреляции'}, ax=ax)

    for text in ax.texts:
        text.set_text(text.get_text().replace('.', ','))

    # Запятые вместо точек и на шкале цветов (colorbar) справа
    cbar = hm.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(comma_fmt(1))

    ax.set_title(title, fontweight='bold', pad=16)
    fig.tight_layout()
    return fig


def scatter_with_trend(df, x_col, y_col, title='', xlabel='', ylabel='', figsize=(8, 5),
                        color=None, legend_loc='Авто (best)'):
    """Диаграмма рассеяния: точки alpha=0.6 с окантовкой, линия тренда, r и p в заголовке."""
    color = color or PALETTE[0]
    x = pd.to_numeric(df[x_col], errors='coerce')
    y = pd.to_numeric(df[y_col], errors='coerce')
    mask = x.notna() & y.notna()
    x, y = x[mask].values, y[mask].values

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, alpha=0.6, s=60, color=color, edgecolors='black', linewidth=0.6)

    if len(x) > 2:
        sl, inter, r, p, _ = sstats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, sl * xs + inter, 'r--', lw=2, label='Линия тренда')
        p_txt = '< 0,001' if p < 0.001 else cf(p, 3)
        ax.set_title(f'{title}\nr = {cf(r, 2)}, p {p_txt}', fontweight='bold', pad=14)
        _apply_legend(ax, legend_loc)
    else:
        ax.set_title(title, fontweight='bold', pad=14)

    ax.set_xlabel(xlabel or x_col, fontweight='bold')
    ax.set_ylabel(ylabel or y_col, fontweight='bold')
    ax.xaxis.set_major_formatter(comma_fmt(1))
    ax.yaxis.set_major_formatter(comma_fmt(1))
    style_axes(ax)
    fig.tight_layout()
    return fig


def time_series_line(yearly_series, title='', ylabel='', figsize=(8, 5), color=None,
                      show_trend=True, show_smooth=False, legend_loc='Авто (best)'):
    """
    График динамики показателя по годам. yearly_series — pandas Series,
    индекс — год, значения — показатель (см. data_io.yearly_series).
    """
    color = color or PALETTE[0]
    years = yearly_series.index.values
    vals = yearly_series.values

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(years, vals, color=color, lw=1.6, marker='o', ms=4,
            alpha=0.85, label='Годовые значения', zorder=3)

    if show_smooth and len(years) >= 11:
        smth = pd.Series(vals, index=years).rolling(11, center=True, min_periods=3).mean()
        ax.plot(years, smth.values, color='#333', lw=2.3, label='11-летнее скользящее', zorder=4)

    if show_trend and len(years) > 2:
        sl, inter, r, p, _ = sstats.linregress(years, vals)
        p_txt = '< 0,001' if p < 0.001 else cf(p, 3)
        sig = '' if p >= 0.05 else (' ***' if p < 0.001 else (' **' if p < 0.01 else ' *'))
        ax.plot(years, sl * years + inter, 'r--', lw=2, zorder=5,
                label=f'Тренд: {cf(sl, 3)}/год{sig} (p {p_txt})')

    ax.set_xlabel('Год', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=14)
    ax.yaxis.set_major_formatter(comma_fmt(1))
    style_axes(ax)
    _apply_legend(ax, legend_loc)
    fig.tight_layout()
    return fig


def time_series_combo(series1, series2, label1, label2, title='', figsize=(9, 5),
                       color1=None, color2=None, legend_loc='Авто (best)'):
    """
    Совмещённый график динамики двух показателей (например температура и
    осадки) с двумя осями Y. series1, series2 — pandas Series (индекс — год).
    """
    color1 = color1 or PALETTE[3]
    color2 = color2 or PALETTE[0]

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(series1.index, series1.values, color=color1, lw=2, marker='o', ms=4,
              label=label1, zorder=3)
    ax1.set_ylabel(label1, fontweight='bold', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.yaxis.set_major_formatter(comma_fmt(1))

    ax2 = ax1.twinx()
    ax2.plot(series2.index, series2.values, color=color2, lw=2, marker='s', ms=4,
              ls='--', label=label2, zorder=3)
    ax2.set_ylabel(label2, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.yaxis.set_major_formatter(comma_fmt(1))

    ax1.set_xlabel('Год', fontweight='bold')
    ax1.set_title(title, fontweight='bold', pad=14)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.grid(alpha=0.3, linestyle='--')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    loc = LEGEND_LOC_OPTIONS.get(legend_loc, 'best')
    if legend_loc == 'Снаружи справа':
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
                    loc='center left', bbox_to_anchor=(1.08, 0.5))
    elif legend_loc == 'Под графиком':
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
                    loc='upper center', bbox_to_anchor=(0.5, -0.15))
    else:
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc=loc)

    fig.tight_layout()
    return fig


def boxplot_minmax_summary(df, value_col, group_col):
    """
    Табличная сводка Мин/Макс/Среднее/Медиана по группам — для вывода над
    графиком вместо подписей на самом боксплоте (которые накладываются
    при большом числе групп).
    """
    rows = []
    for g in sorted(df[group_col].dropna().unique()):
        d = pd.to_numeric(df.loc[df[group_col] == g, value_col],
                           errors='coerce').dropna().values
        if len(d) == 0:
            continue
        rows.append({
            'Группа': g, 'N': len(d), 'Минимум': np.min(d), 'Максимум': np.max(d),
            'Среднее': np.mean(d), 'Медиана': np.median(d),
        })
    return pd.DataFrame(rows)


def pettitt_chart(yearly_series, pettitt_result, title='', figsize=(9, 7), color=None):
    """
    Двухпанельный график: временной ряд с отмеченной точкой разрыва (сверху)
    и профиль статистики Петтитта U_t (снизу).
    """
    color = color or PALETTE[0]
    s = yearly_series.dropna().sort_index()
    years, vals = s.index.values, s.values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(years, vals, color=color, lw=1.4, alpha=0.8, marker='o', ms=3, zorder=2)
    if len(years) >= 11:
        smth = pd.Series(vals, index=years).rolling(11, center=True, min_periods=3).mean()
        ax1.plot(years, smth.values, color='#333', lw=2.2, label='11-летнее скользящее', zorder=3)

    cy = pettitt_result.get('change_year')
    if cy is not None:
        ax1.axvline(cy, color='red', ls='--', lw=1.8,
                    label=f'Точка разрыва: {cy}')
    ax1.set_ylabel(title or 'Значение', fontweight='bold')
    ax1.set_title('Временной ряд с точкой разрыва однородности', fontweight='bold', pad=10)
    ax1.yaxis.set_major_formatter(comma_fmt(1))
    style_axes(ax1)
    ax1.legend(fontsize=9)

    if pettitt_result.get('U') is not None:
        U = pettitt_result['U']
        ax2.plot(years, U, color=PALETTE[3], lw=2, zorder=3)
        ax2.fill_between(years, 0, U, where=(np.array(U) < 0), alpha=0.25, color=PALETTE[3])
        ax2.axhline(0, color='gray', lw=0.8)
        if cy is not None:
            ax2.axvline(cy, color='red', ls='--', lw=1.5,
                        label=f'max|U_t|={cf(pettitt_result["K"],0)}')
            ax2.legend(fontsize=9)
    ax2.set_xlabel('Год', fontweight='bold')
    ax2.set_ylabel('Статистика U_t', fontweight='bold')
    ax2.yaxis.set_major_formatter(comma_fmt(0))
    style_axes(ax2)

    fig.tight_layout()
    return fig


def feature_importance_bar(imp_df, score_col='Комбинированная оценка', title='',
                            figsize=(8, 6), color=None, top_n=None):
    """Горизонтальный бар-чарт важности признаков (по убыванию сверху вниз)."""
    color = color or PALETTE[0]
    d = imp_df.copy()
    if top_n:
        d = d.head(top_n)
    d = d.iloc[::-1]  # для горизонтального бара — самый важный сверху

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(d['Признак'], d[score_col], color=color, edgecolor='black', lw=0.6)
    for bar, v in zip(bars, d[score_col]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                cf(v, 3), va='center', fontsize=9)

    ax.set_xlabel(score_col, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=14)
    ax.xaxis.set_major_formatter(comma_fmt(2))
    ax.set_xlim(0, max(d[score_col].max() * 1.15, 0.1))
    style_axes(ax)
    fig.tight_layout()
    return fig


def silhouette_plot(scores, best_k=None, title='', figsize=(8, 5), color=None):
    """График силуэтного коэффициента по числу кластеров k, с отметкой лучшего k."""
    color = color or PALETTE[0]
    ks = sorted(scores.keys())
    vals = [scores[k] for k in ks]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ks, vals, 'o-', color=color, lw=2, ms=8)
    if best_k is not None and best_k in scores:
        ax.scatter([best_k], [scores[best_k]], s=200, marker='*', color='red',
                   zorder=5, label=f'Оптимум k={best_k}')
        ax.legend(fontsize=9)
    ax.set_xlabel('Число кластеров (k)', fontweight='bold')
    ax.set_ylabel('Силуэтный коэффициент', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=14)
    ax.set_xticks(ks)
    ax.yaxis.set_major_formatter(comma_fmt(3))
    style_axes(ax)
    fig.tight_layout()
    return fig


def pca_biplot(Xpca, pca_model, labels, feature_names, explained_var, title='',
               figsize=(9, 7), cluster_colors=None):
    """
    Биплот PCA: точки станций, окрашенные по кластеру, + векторы нагрузок
    признаков на первые две главные компоненты.
    """
    cluster_colors = cluster_colors or PALETTE
    unique_labels = sorted(set(labels))

    fig, ax = plt.subplots(figsize=figsize)
    for i, lbl in enumerate(unique_labels):
        mask = np.array(labels) == lbl
        c = cluster_colors[i % len(cluster_colors)]
        ax.scatter(Xpca[mask, 0], Xpca[mask, 1], c=c, s=80, edgecolors='k',
                   lw=0.6, label=f'Кластер {lbl}', zorder=3)

    scale = np.max(np.abs(Xpca[:, :2])) * 0.75
    for fi, fname in enumerate(feature_names):
        vx = pca_model.components_[0, fi] * scale
        vy = pca_model.components_[1, fi] * scale
        ax.annotate('', xy=(vx, vy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.1))
        ax.text(vx * 1.12, vy * 1.12, fname, fontsize=8, color='#333', ha='center')

    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel(f'ГК1 ({explained_var[0]*100:.1f}%)', fontweight='bold')
    ax.set_ylabel(f'ГК2 ({explained_var[1]*100:.1f}%)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=14)
    style_axes(ax)
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    return fig


def dendrogram_plot(Z, labels_list, threshold=None, title='', figsize=(12, 5), cluster_colors=None):
    """
    Дендрограмма иерархической кластеризации, ветви окрашены по кластерам
    (через color_threshold scipy) — линии выше порога серые, ниже —
    раскрашены по палитре PALETTE.
    """
    import scipy.cluster.hierarchy as sch

    cluster_colors = cluster_colors or PALETTE
    fig, ax = plt.subplots(figsize=figsize)

    if threshold is not None:
        sch.set_link_color_palette(list(cluster_colors))
        sch.dendrogram(Z, ax=ax, labels=labels_list, leaf_rotation=90, leaf_font_size=7,
                        color_threshold=threshold, above_threshold_color='#999')
        sch.set_link_color_palette(None)  # сброс на дефолт для следующих вызовов
        ax.axhline(threshold, color='red', lw=1.5, ls='--',
                   label=f'Порог отсечения: {cf(threshold, 2)}')
        ax.legend(fontsize=9)
    else:
        sch.dendrogram(Z, ax=ax, labels=labels_list, leaf_rotation=90, leaf_font_size=7,
                        color_threshold=0, above_threshold_color='#555')

    ax.set_title(title, fontweight='bold', pad=14)
    ax.set_xlabel('Станции', fontweight='bold')
    ax.set_ylabel('Евклидово расстояние (Уорд)', fontweight='bold')
    ax.yaxis.set_major_formatter(comma_fmt(1))
    style_axes(ax)
    fig.tight_layout()
    return fig


def k_selection_plot(sil_scores, ari_scores=None, chosen_k=None, title='',
                      figsize=(8, 5)):
    """
    График выбора числа кластеров: силуэт (и опционально ARI), с отметкой
    выбранного k звёздочкой — в стиле сравнения нескольких периодов.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ks = sorted(sil_scores.keys())
    ax.plot(ks, [sil_scores[k] for k in ks], 'o-', color=PALETTE[0], lw=2, ms=8,
            label='Силуэт')
    if ari_scores:
        ax.plot(ks, [ari_scores.get(k, np.nan) for k in ks], 's--', color=PALETTE[1],
                lw=2, ms=8, label='ARI')
    if chosen_k is not None and chosen_k in sil_scores:
        ax.scatter([chosen_k], [sil_scores[chosen_k]], s=220, marker='*', color='red',
                   zorder=5, label=f'Оптимум k={chosen_k}')
    ax.set_xlabel('Количество кластеров (k)', fontweight='bold')
    ax.set_ylabel('Значение метрики', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=14)
    ax.set_xticks(ks)
    ax.yaxis.set_major_formatter(comma_fmt(3))
    style_axes(ax)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def map_scatter(df, lat_col='Lat', lon_col='Lon', cluster_col=None, label_col=None,
                 title='', figsize=(9, 7), cluster_colors=None, show_labels=True):
    """
    Минималистичная карта станций: точки на плоскости широта/долгота,
    окрашенные по кластеру (если указан cluster_col), с подписями (опц.).
    Без реального картографического слоя — простая схематичная проекция.
    """
    import matplotlib.patheffects as pe
    cluster_colors = cluster_colors or PALETTE

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#EEF4FB')

    if cluster_col and cluster_col in df.columns:
        for i, c in enumerate(sorted(df[cluster_col].dropna().unique())):
            sub = df[df[cluster_col] == c]
            ax.scatter(sub[lon_col], sub[lat_col], s=140,
                       color=cluster_colors[i % len(cluster_colors)],
                       edgecolors='k', lw=0.7, zorder=4,
                       label=f'Кластер {c} (n={len(sub)})')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
    else:
        ax.scatter(df[lon_col], df[lat_col], s=110, color=PALETTE[0],
                   edgecolors='k', lw=0.7, zorder=4)

    if show_labels and label_col and label_col in df.columns:
        for _, row in df.iterrows():
            ax.annotate(str(row[label_col])[:10], (row[lon_col], row[lat_col]),
                        textcoords='offset points', xytext=(4, 4), fontsize=6,
                        color='#222', zorder=5,
                        path_effects=[pe.withStroke(linewidth=1.3, foreground='white')])

    ax.set_xlabel('Долгота, °в.д.', fontweight='bold')
    ax.set_ylabel('Широта, °с.ш.', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=14)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{cf(x, 1)}°'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{cf(x, 1)}°'))
    ax.grid(alpha=0.3, linestyle='--', zorder=0)
    style_axes(ax)
    fig.tight_layout()
    return fig


def _add_north_arrow(ax, x=0.06, y=0.92, size=0.05):
    """Стрелка на север в верхнем левом углу карты (координаты — доли осей)."""
    ax.annotate('', xy=(x, y), xytext=(x, y - size),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='black', lw=2, mutation_scale=18))
    ax.text(x, y + 0.015, 'N', transform=ax.transAxes, fontsize=12, fontweight='bold',
            ha='center', va='bottom')


def _add_scale_bar(ax, mean_lat, x_range, y_range):
    """
    Масштабная линейка (км) в нижнем левом углу. Длина сегмента подбирается
    "круглым" числом км исходя из ширины карты, переводится в градусы
    долготы по широте mean_lat.
    """
    km_per_deg_lon = 111.32 * np.cos(np.radians(mean_lat))
    width_km = x_range * km_per_deg_lon
    # Подбор "круглой" длины линейки (~1/4 ширины карты)
    candidates = [10, 20, 25, 50, 100, 200, 250, 500, 1000]
    target = width_km * 0.25
    bar_km = min(candidates, key=lambda c: abs(c - target))

    bar_deg = bar_km / km_per_deg_lon
    x0 = ax.get_xlim()[0] + x_range * 0.06
    y0 = ax.get_ylim()[0] + y_range * 0.05
    seg = bar_deg / 2

    for i, col in enumerate(['black', 'white']):
        ax.add_patch(plt.Rectangle((x0 + i * seg, y0), seg, y_range * 0.012,
                                    facecolor=col, edgecolor='black', lw=0.8,
                                    transform=ax.transData, zorder=6))
    ax.text(x0, y0 + y_range * 0.02, '0', fontsize=7, ha='center', zorder=6)
    ax.text(x0 + seg, y0 + y_range * 0.02, str(bar_km // 2), fontsize=7, ha='center', zorder=6)
    ax.text(x0 + 2 * seg, y0 + y_range * 0.02, f'{bar_km} км', fontsize=7, ha='center', zorder=6)


def choropleth_map(polygons_with_cluster, stations_df=None, lat_col='Lat', lon_col='Lon',
                    cluster_col='Кластер', title='', figsize=(10, 8), cluster_colors=None,
                    show_station_points=True, show_district_labels=False,
                    show_north_arrow=True, show_scale_bar=True):
    """
    Раскрашенная карта районов по преобладающему кластеру (choropleth), в
    стиле стандартного ГИС-оформления: стрелка на север, масштабная
    линейка, координатная сетка, белый фон за пределами территории.
    polygons_with_cluster — результат mapping.assign_clusters_to_polygons():
    список dict {'name', 'geometry' (shapely), 'cluster', 'n_stations_inside'}.
    """
    from matplotlib.patches import Polygon as MplPolygon
    import matplotlib.patheffects as pe

    cluster_colors = cluster_colors or PALETTE
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    clusters_present = sorted({p['cluster'] for p in polygons_with_cluster
                                if p['cluster'] is not None})
    color_map = {c: cluster_colors[i % len(cluster_colors)]
                 for i, c in enumerate(clusters_present)}

    all_x, all_y = [], []
    for poly in polygons_with_cluster:
        geom = poly['geometry']
        color = color_map.get(poly['cluster'], '#CCCCCC')
        geoms = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
        for g in geoms:
            xs, ys = g.exterior.xy
            all_x.extend(xs)
            all_y.extend(ys)
            patch = MplPolygon(list(zip(xs, ys)), closed=True, facecolor=color,
                               edgecolor='#222', lw=0.6, zorder=2)
            ax.add_patch(patch)
        if show_district_labels:
            c = geom.centroid
            ax.text(c.x, c.y, poly['name'][:14], fontsize=6, ha='center', va='center',
                    color='#111', zorder=4,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

    if show_station_points and stations_df is not None:
        for i, c in enumerate(clusters_present):
            sub = stations_df[stations_df[cluster_col] == c]
            ax.scatter(sub[lon_col], sub[lat_col], s=50, color=color_map[c],
                       edgecolors='k', lw=0.9, zorder=5)

    legend_elems = [Line2D([0], [0], marker='s', color='w', markerfacecolor=color_map[c],
                            markersize=14, label=f'Кластер {c}') for c in clusters_present]
    ax.legend(handles=legend_elems, fontsize=10, loc='lower right', framealpha=0.95,
              edgecolor='#333')

    if all_x and all_y:
        pad_x = (max(all_x) - min(all_x)) * 0.04
        pad_y = (max(all_y) - min(all_y)) * 0.04
        x_min, x_max = min(all_x) - pad_x, max(all_x) + pad_x
        y_min, y_max = min(all_y) - pad_y, max(all_y) + pad_y
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if show_scale_bar:
            _add_scale_bar(ax, mean_lat=(y_min + y_max) / 2,
                           x_range=x_max - x_min, y_range=y_max - y_min)

    if show_north_arrow:
        _add_north_arrow(ax)

    ax.set_title(title, fontweight='bold', pad=14)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{cf(x, 3)}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{cf(x, 3)}'))
    ax.grid(alpha=0.4, linestyle='-', linewidth=0.5, color='#999', zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#333')
        spine.set_linewidth(1.0)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def fig_to_bytes(fig, fmt='png'):
    """Сериализует matplotlib figure в байты для скачивания (PNG или PDF)."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=DPI, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()
