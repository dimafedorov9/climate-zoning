"""
=============================================================================
МОДУЛЬ СТАТИСТИЧЕСКИХ ТЕСТОВ ДЛЯ ВРЕМЕННЫХ РЯДОВ
Тест Петтитта (точка разрыва однородности) и Innovative Trend Analysis
(Şen, 2012) — используются для обоснованной рекомендации о разделении
временного ряда на подвыборки (ранний/современный период и т.п.).
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats


def pettitt_test(series):
    """
    Непараметрический тест Петтитта на точку разрыва однородности ряда.
    series — pandas Series, индекс — год (или любой упорядоченный ключ).
    Возвращает dict: K (статистика), p (p-value), change_year (год разрыва),
    U (профиль статистики Петтитта), years (соответствующие годы).
    """
    s = series.dropna().sort_index()
    x = s.values.astype(float)
    years = s.index.values
    n = len(x)
    if n < 8:
        return dict(K=np.nan, p=np.nan, change_year=None, U=None, years=years, n=n)

    ranks = stats.rankdata(x)
    U = np.zeros(n)
    for t in range(1, n + 1):
        U[t - 1] = 2 * ranks[:t].sum() - t * (n + 1)

    K = np.max(np.abs(U))
    t_idx = int(np.argmax(np.abs(U)))
    p = 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2))
    change_year = years[t_idx]

    return dict(K=K, p=p, change_year=int(change_year), U=U, years=years, n=n)


def ita(series):
    """
    Innovative Trend Analysis (Şen, 2012). Ряд делится на две равные
    хронологические половины, каждая сортируется по возрастанию.
    Возвращает dict: s (наклон), ci (95% доверительный интервал),
    signif (bool), s1, s2 (отсортированные половины), mean1, mean2, n.
    """
    x = np.asarray(series.dropna().sort_index().values, dtype=float)
    n = len(x)
    if n < 10:
        return None
    if n % 2 != 0:
        mid = n // 2
        x = np.delete(x, mid)
        n = len(x)
    h = n // 2
    x1, x2 = x[:h], x[h:]
    s1, s2 = np.sort(x1), np.sort(x2)
    mean1, mean2 = x1.mean(), x2.mean()
    s = 2.0 * (mean2 - mean1) / n
    sigma = x.std(ddof=1)
    rho = np.corrcoef(x1, x2)[0, 1]
    rho = 0.0 if np.isnan(rho) else rho
    sigma_s = (2.0 * np.sqrt(2) / (n * np.sqrt(n))) * sigma * np.sqrt(max(1 - rho, 0))
    ci = 1.96 * sigma_s
    signif = abs(s) > ci
    return dict(s=s, ci=ci, signif=signif, s1=s1, s2=s2,
                mean1=mean1, mean2=mean2, n=n)


def pfmt(p):
    if p is None or pd.isna(p):
        return '—'
    return '< 0,001' if p < 0.001 else f'{p:.3f}'.replace('.', ',')


def cf(x, dec=3):
    if x is None or pd.isna(x):
        return '—'
    return f'{x:.{dec}f}'.replace('.', ',')


def recommend_periods(yearly_series, alpha=0.05):
    """
    Формирует текстовую рекомендацию о целесообразности разделения ряда
    на подвыборки на основе теста Петтитта и ITA.
    Возвращает dict: pettitt, ita, recommend_split (bool), text (markdown).
    """
    pett = pettitt_test(yearly_series)
    ita_res = ita(yearly_series)

    lines = []
    recommend_split = False

    if pett['change_year'] is not None:
        sig = pett['p'] < alpha
        lines.append(
            f'**Тест Петтитта:** точка разрыва однородности — **{pett["change_year"]} год** '
            f'(K = {cf(pett["K"], 0)}, p {pfmt(pett["p"])}) — '
            f'{"статистически значимо" if sig else "незначимо"}.'
        )
        if sig:
            recommend_split = True
    else:
        lines.append('**Тест Петтитта:** недостаточно данных для расчёта '
                      '(нужно не менее 8 точек ряда).')

    if ita_res is not None:
        sig2 = ita_res['signif']
        lines.append(
            f'**ITA (Şen, 2012):** наклон s = {cf(ita_res["s"], 4)} за весь период '
            f'(ДИ ±{cf(ita_res["ci"], 4)}) — {"значим" if sig2 else "незначим"}.'
        )
        if sig2:
            recommend_split = True
    else:
        lines.append('**ITA:** недостаточно данных для расчёта (нужно не менее 10 точек).')

    if recommend_split and pett['change_year'] is not None:
        y_min = int(yearly_series.dropna().index.min())
        y_max = int(yearly_series.dropna().index.max())
        cy = pett['change_year']
        lines.append(
            f'**Рекомендация:** ряд статистически неоднороден — стоит рассмотреть '
            f'разделение на подвыборки, например «Ранний период» ({y_min}–{cy}) и '
            f'«Современный период» ({cy + 1}–{y_max}).'
        )
    else:
        lines.append('**Рекомендация:** выраженной неоднородности не выявлено — '
                      'ряд можно анализировать как единый период.')

    return dict(pettitt=pett, ita=ita_res, recommend_split=recommend_split,
                text='\n\n'.join(lines))
