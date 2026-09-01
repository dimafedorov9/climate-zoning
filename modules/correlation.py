"""
=============================================================================
МОДУЛЬ КОРРЕЛЯЦИОННОГО АНАЛИЗА
Пирсон (линейная связь), Спирмен (монотонная связь), p-value, звёздочки
значимости, ранжирование признаков по силе связи с целевой переменной.
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats


def sig_stars(p):
    if p is None or pd.isna(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def correlation_matrix(df, columns, method='pearson'):
    """
    Матрица корреляций (Пирсон или Спирмен) + матрица p-value для тех же пар.
    Возвращает (corr_df, pval_df).
    """
    num_df = df[columns].apply(pd.to_numeric, errors='coerce')
    n = len(columns)
    corr = pd.DataFrame(np.eye(n), index=columns, columns=columns)
    pval = pd.DataFrame(np.zeros((n, n)), index=columns, columns=columns)

    func = stats.pearsonr if method == 'pearson' else stats.spearmanr

    for i in range(n):
        for j in range(i + 1, n):
            a = num_df.iloc[:, i]
            b = num_df.iloc[:, j]
            mask = a.notna() & b.notna()
            if mask.sum() < 3:
                r, p = np.nan, np.nan
            else:
                r, p = func(a[mask], b[mask])
            corr.iloc[i, j] = corr.iloc[j, i] = r
            pval.iloc[i, j] = pval.iloc[j, i] = p

    return corr, pval


def target_correlations(df, target_col, feature_cols, method='both'):
    """
    Корреляции всех признаков с целевой переменной — ранжированный список
    (не "Топ-N", а "Ранжированные N признаков" по требованию ТЗ).
    method: 'pearson', 'spearman', 'both'.
    """
    y = pd.to_numeric(df[target_col], errors='coerce')
    rows = []
    for col in feature_cols:
        if col == target_col:
            continue
        x = pd.to_numeric(df[col], errors='coerce')
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            continue
        row = {'Признак': col, 'N': int(mask.sum())}
        if method in ('pearson', 'both'):
            r_p, p_p = stats.pearsonr(x[mask], y[mask])
            row['r Пирсона'] = r_p
            row['p Пирсона'] = p_p
            row['Значимость (Пирсон)'] = sig_stars(p_p)
        if method in ('spearman', 'both'):
            r_s, p_s = stats.spearmanr(x[mask], y[mask])
            row['ρ Спирмена'] = r_s
            row['p Спирмена'] = p_s
            row['Значимость (Спирмен)'] = sig_stars(p_s)
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    sort_col = 'r Пирсона' if 'r Пирсона' in result.columns else 'ρ Спирмена'
    result = result.reindex(
        result[sort_col].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    result.insert(0, 'Ранг', range(1, len(result) + 1))
    return result


def auto_flag_multicollinearity(df, columns, threshold=0.8, method='pearson'):
    """
    Автоматическая проверка мультиколлинеарности: находит пары признаков
    с |r| выше порога и жадно формирует список признаков "на удаление"
    (из каждой конфликтной пары оставляется первый по порядку в списке
    columns, второй помечается кандидатом на исключение).

    Возвращает (flagged_pairs_df, keep_list, drop_list).
    """
    corr, pval = correlation_matrix(df, columns, method=method)
    pairs = []
    to_drop = set()
    cols = list(columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > threshold:
                pairs.append({
                    'Признак 1': cols[i],
                    'Признак 2': cols[j],
                    'r': r,
                    '|r|': abs(r),
                })
                # Оставляем тот, что раньше встретился в списке; второй — кандидат на удаление
                if cols[i] not in to_drop:
                    to_drop.add(cols[j])

    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values('|r|', ascending=False).drop(columns='|r|')

    drop_list = sorted(to_drop)
    keep_list = [c for c in cols if c not in to_drop]
    return pairs_df, keep_list, drop_list


def significant_correlations_text(ranked_df, alpha=0.05, max_items=7,
                                   r_strong=0.7, r_moderate=0.4):
    """
    Формирует список словесных описаний значимых корреляций для
    научного текста (Промпт 2, раздел 5). Если значимых связей много,
    берёт top max_items по силе; если мало — все значимые.
    """
    if ranked_df.empty:
        return []
    p_col = 'p Пирсона' if 'p Пирсона' in ranked_df.columns else 'p Спирмена'
    r_col = 'r Пирсона' if 'r Пирсона' in ranked_df.columns else 'ρ Спирмена'
    sig = ranked_df[ranked_df[p_col] < alpha].copy()
    if sig.empty:
        return []
    sig = sig.reindex(sig[r_col].abs().sort_values(ascending=False).index)
    sig = sig.head(max_items)

    lines = []
    for _, row in sig.iterrows():
        r = row[r_col]
        strength = ('сильную' if abs(r) >= r_strong else
                    'умеренную' if abs(r) >= r_moderate else 'слабую')
        direction = 'положительную' if r > 0 else 'отрицательную'
        p_txt = '< 0,001' if row[p_col] < 0.001 else f'{row[p_col]:.3f}'.replace('.', ',')
        r_txt = f'{r:.2f}'.replace('.', ',')
        lines.append(
            f'{row["Признак"]} демонстрирует {strength} {direction} корреляцию '
            f'(r = {r_txt}, p {p_txt})'
        )
    return lines
