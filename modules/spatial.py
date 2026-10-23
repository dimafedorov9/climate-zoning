"""
=============================================================================
ОБЩИЕ ПРОСТРАНСТВЕННЫЕ ОПЕРАЦИИ: расстояния, соседние станции, матрицы
«дата × станция», аномалии относительно сезонного хода.
Используется контролем качества, заполнением пропусков и интерполяцией.
=============================================================================
"""

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """Расстояние по дуге большого круга, км (векторно)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def distance_matrix(stations):
    """Матрица расстояний между станциями (DataFrame st×st), NaN при отсутствии координат."""
    s = stations.set_index('st')
    lat, lon = s['lat'].values.astype(float), s['lon'].values.astype(float)
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    return pd.DataFrame(D, index=s.index, columns=s.index)


def to_matrix(df, comp):
    """Длинная таблица [st, date, comp] → матрица date × st."""
    M = df.pivot_table(index='date', columns='st', values=comp, aggfunc='first')
    # общая сетка для всех компонентов: полный ряд дат и все станции файла,
    # иначе матрицы Tмин/Tмакс и т. п. различаются строками и не сравниваются
    full = pd.date_range(df['date'].min(), df['date'].max())
    sts = sorted(df['st'].unique())
    return M.reindex(index=full, columns=sts)


def seasonal_anomalies(M, window=15):
    """
    Аномалии относительно сглаженного сезонного хода каждой станции:
    климатическая норма для дня года = среднее по годам, сглаженное
    скользящим окном (циклически). Возвращает (A, clim) — аномалии и норма
    с тем же индексом, что у M.
    """
    doy = M.index.dayofyear.values
    doy = np.where(doy == 366, 365, doy)
    clim_doy = M.groupby(doy).mean()
    clim_doy = clim_doy.reindex(range(1, 366))
    ext = pd.concat([clim_doy.iloc[-window:], clim_doy, clim_doy.iloc[:window]])
    sm = ext.rolling(2 * window + 1, center=True, min_periods=5).mean().iloc[window:-window]
    sm.index = range(1, 366)
    clim = pd.DataFrame(sm.loc[doy].values, index=M.index, columns=M.columns)
    return M - clim, clim


def neighbors(stations, A=None, n=5, max_km=150.0, same_region=False, min_r=None):
    """
    Подбирает соседей для каждой станции: в пределах max_km (если есть
    координаты), упорядочены по корреляции аномалий A (если передана), иначе
    по расстоянию. При отсутствии координат соседи выбираются только по
    корреляции. Возвращает dict {st: [(сосед, вес, r, км), ...]}.
    """
    s = stations.set_index('st')
    ids = [i for i in s.index if A is None or i in A.columns]
    has_xy = s[['lat', 'lon']].notna().all(axis=1)
    D = distance_matrix(stations) if has_xy.any() else None
    C = A[ids].corr(min_periods=60) if A is not None else None
    out = {}
    for st in ids:
        cand = []
        for o in ids:
            if o == st:
                continue
            if same_region and 'region' in s and s.loc[o, 'region'] != s.loc[st, 'region']:
                continue
            km = D.loc[st, o] if D is not None and has_xy.get(st, False) and has_xy.get(o, False) else np.nan
            r = C.loc[st, o] if C is not None else np.nan
            if not np.isnan(km) and km > max_km and (np.isnan(r) or r < 0.9):
                continue
            if min_r is not None and (np.isnan(r) or r < min_r):
                continue
            cand.append((o, r, km))
        if C is not None:
            cand.sort(key=lambda t: -(t[1] if not np.isnan(t[1]) else -1))
        else:
            cand.sort(key=lambda t: t[2])
        cand = cand[:n]
        res = []
        for o, r, km in cand:
            w = (max(r, 0.01) ** 2) if not np.isnan(r) else 1.0 / max(km, 1.0)
            res.append((o, w, r, km))
        out[st] = res
    return out


def neighbor_prediction(A, nbrs):
    """
    Оценка аномалии каждой станции по соседям (взвешенное среднее аномалий
    соседей с весами r²). Возвращает (pred, n_used) — матрицы как A.
    """
    pred = pd.DataFrame(np.nan, index=A.index, columns=A.columns)
    nused = pd.DataFrame(0, index=A.index, columns=A.columns)
    for st, nb in nbrs.items():
        if not nb:
            continue
        cols = [o for o, *_ in nb]
        w = np.array([wt for _, wt, *_ in nb])
        X = A[cols].values
        m = ~np.isnan(X)
        ws = (m * w).sum(1)
        val = np.where(ws > 0, np.nansum(np.nan_to_num(X) * w, 1) / np.where(ws > 0, ws, 1), np.nan)
        pred[st] = val
        nused[st] = m.sum(1)
    return pred, nused


def robust_z(resid, by_month=True):
    """
    z-оценка по медиане и MAD для каждой станции. При by_month=True масштаб
    оценивается отдельно по каждому месяцу: зимой расхождения между
    станциями (инверсии) заметно больше, чем летом.
    """
    if not by_month:
        med = resid.median()
        mad = ((resid - med).abs().median() * 1.4826).replace(0, np.nan)
        return (resid - med) / mad
    g = resid.groupby(resid.index.month)
    med = g.transform('median')
    mad = (resid - med).abs().groupby(resid.index.month).transform('median') * 1.4826
    return (resid - med) / mad.replace(0, np.nan)


def in_periods(dates, periods):
    """Булев массив: дата попадает в один из периодов [(начало, конец), ...]."""
    dates = pd.DatetimeIndex(dates)
    m = np.zeros(len(dates), bool)
    for a, b in periods:
        m |= (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
    return m
