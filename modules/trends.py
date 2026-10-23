"""
=============================================================================
ТРЕНДЫ И СРАВНЕНИЕ ПЕРИОДОВ
  • тест Манна — Кендалла (классический и модифицированный Хамеда — Рао
    с поправкой на автокорреляцию), оценка наклона Сена;
  • тест Петтитта (точка смены режима) — из climate-zoning;
  • сравнение двух периодов: критерий Манна — Уитни, бутстреп-интервал
    разности средних; сравнение с базовыми значениями (например, отчёт
    2016 г. за 1985–2014 гг.).
По рекомендации ВМО выводы о трендах делаются по рядам ≥ 30 лет; для
более коротких рядов результат помечается как предварительный.
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats

from .tests_stat import pettitt_test  # noqa: F401  (переиспользуется из climate-zoning)


def sen_slope(y, x=None):
    y = np.asarray(y, float)
    x = np.arange(len(y)) if x is None else np.asarray(x, float)
    ok = np.isfinite(y)
    y, x = y[ok], x[ok]
    if len(y) < 3:
        return np.nan, np.nan
    i, j = np.triu_indices(len(y), 1)
    s = (y[j] - y[i]) / (x[j] - x[i])
    slope = np.median(s)
    return slope, np.median(y - slope * x)


def mann_kendall(y, modified=True, alpha=0.05):
    """
    Тест Манна — Кендалла. modified=True — поправка Хамеда — Рао на
    автокорреляцию (по остаткам после удаления наклона Сена).
    Возвращает dict: S, z, p, tau, тренд ('рост'/'снижение'/'нет').
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 5:
        return dict(n=n, S=np.nan, z=np.nan, p=np.nan, tau=np.nan, trend='—')
    i, j = np.triu_indices(n, 1)
    S = np.sign(y[j] - y[i]).sum()
    _, counts = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18
    if modified and n >= 8:
        slope, _ = sen_slope(y)
        r = y - slope * np.arange(n)
        rk = stats.rankdata(r)
        acf = [np.corrcoef(rk[:-k], rk[k:])[0, 1] for k in range(1, n // 3)]
        sig = 1.96 / np.sqrt(n)
        corr = sum((n - k) * (n - k - 1) * (n - k - 2) * a for k, a in enumerate(acf, 1) if abs(a) > sig)
        nns = 1 + 2 / (n * (n - 1) * (n - 2)) * corr
        var *= max(nns, 1e-6)
    z = (S - np.sign(S)) / np.sqrt(var) if S != 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = S / (n * (n - 1) / 2)
    trend = ('рост' if z > 0 else 'снижение') if p < alpha else 'нет'
    return dict(n=n, S=S, z=z, p=p, tau=tau, trend=trend)


def trend_table(ann, codes=None, by='region', min_years=30):
    """
    Тренды показателей по средним рядам региона (by='region') или по
    станциям (by='st'). Наклон Сена — в единицах показателя за 10 лет.
    """
    rows = []
    a = ann if codes is None else ann[ann['Показатель'].isin(codes)]
    for (key, code), g in a.groupby([by, 'Показатель']):
        s = g.groupby('year')['Значение'].mean().dropna()
        if len(s) < 5:
            continue
        mk = mann_kendall(s.values)
        slope, _ = sen_slope(s.values, s.index.values)
        pt = pettitt_test(s)
        rows.append({by if by != 'st' else 'Станция': key, 'Показатель': code, 'Лет': len(s),
                     'Период': f'{s.index.min()}–{s.index.max()}',
                     'Наклон Сена за 10 лет': slope * 10, 'Z (МК)': mk['z'], 'p (МК)': mk['p'],
                     'Тренд': mk['trend'], 'Точка Петтитта': pt['change_year'], 'p (Петтитт)': pt['p'],
                     'Оценка': 'предварительная (< 30 лет)' if len(s) < min_years else ''})
    t = pd.DataFrame(rows)
    return t.rename(columns={'region': 'Регион'})


def compare_periods(ann, p1, p2, codes=None, by='region', n_boot=2000, seed=1):
    """
    Сравнение средних значений показателей в двух периодах: разность,
    95 % бутстреп-интервал, p критерия Манна — Уитни (по годовым средним
    значениям региона или станции).
    """
    rng = np.random.default_rng(seed)
    rows = []
    a = ann if codes is None else ann[ann['Показатель'].isin(codes)]
    for (key, code), g in a.groupby([by, 'Показатель']):
        s = g.groupby('year')['Значение'].mean()
        x1 = s[(s.index >= p1[0]) & (s.index <= p1[1])].dropna().values
        x2 = s[(s.index >= p2[0]) & (s.index <= p2[1])].dropna().values
        if len(x1) < 3 or len(x2) < 3:
            continue
        diff = x2.mean() - x1.mean()
        bs = [rng.choice(x2, len(x2)).mean() - rng.choice(x1, len(x1)).mean() for _ in range(n_boot)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = stats.mannwhitneyu(x1, x2).pvalue
        rows.append({'Регион' if by == 'region' else 'Станция': key, 'Показатель': code,
                     f'Среднее {p1[0]}–{p1[1]}': x1.mean(), f'Среднее {p2[0]}–{p2[1]}': x2.mean(),
                     'Разность': diff, '95 % ДИ: от': lo, '95 % ДИ: до': hi, 'p (Манн — Уитни)': p})
    return pd.DataFrame(rows)


def compare_with_base(current, base):
    """
    Сравнение текущих значений с базовыми (введёнными вручную, например из
    отчёта 2016 г.). current, base — dict {характеристика: значение}.
    """
    keys = [k for k in base if k in current]
    return pd.DataFrame({'Характеристика': keys,
                         'Базовый период': [base[k] for k in keys],
                         'Текущий период': [current[k] for k in keys],
                         'Изменение': [current[k] - base[k] if isinstance(current[k], (int, float)) and
                                       isinstance(base[k], (int, float)) else np.nan for k in keys]})
