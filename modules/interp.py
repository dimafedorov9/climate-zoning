"""
=============================================================================
ПРОСТРАНСТВЕННАЯ ИНТЕРПОЛЯЦИЯ ПОКАЗАТЕЛЕЙ МЕЖДУ СТАНЦИЯМИ
  • IDW — обратно-взвешенное расстояние;
  • регрессионный кригинг: линейная регрессия показателя на широту,
    долготу и (при наличии) высоту, затем обычный кригинг остатков
    (экспоненциальная или сферическая вариограмма);
  • ближайшая станция (полигоны Тиссена).
Качество проверяется перекрёстной проверкой с поочерёдным исключением
станций (LOO): MAE, RMSE, R².
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .spatial import haversine_km


def make_grid(bbox, step=0.05):
    """Регулярная сетка (долгота, широта) в пределах bbox=(lon0, lon1, lat0, lat1)."""
    lon = np.arange(bbox[0], bbox[1] + step / 2, step)
    lat = np.arange(bbox[2], bbox[3] + step / 2, step)
    LON, LAT = np.meshgrid(lon, lat)
    return LON, LAT


def _dist(lat1, lon1, lat2, lon2):
    return haversine_km(lat1[:, None], lon1[:, None], lat2[None, :], lon2[None, :])


def idw(lat, lon, val, glat, glon, power=2.0, max_km=None):
    lat, lon, val = map(np.asarray, (lat, lon, val))
    d = _dist(glat.ravel(), glon.ravel(), lat, lon)
    d = np.maximum(d, 0.5)
    w = 1.0 / d ** power
    if max_km:
        w[d > max_km] = 0
    est = (w * val[None, :]).sum(1) / np.where(w.sum(1) > 0, w.sum(1), np.nan)
    return est.reshape(glat.shape)


def nearest(lat, lon, val, glat, glon):
    d = _dist(glat.ravel(), glon.ravel(), np.asarray(lat), np.asarray(lon))
    return np.asarray(val)[np.argmin(d, axis=1)].reshape(glat.shape)


# ─── Вариограмма и кригинг ───────────────────────────────────────────────
def _vmodel(h, nugget, sill, rng, kind):
    if kind == 'spherical':
        g = np.where(h < rng, nugget + sill * (1.5 * h / rng - 0.5 * (h / rng) ** 3), nugget + sill)
    else:
        g = nugget + sill * (1 - np.exp(-3 * h / rng))
    return np.where(h == 0, 0, g)


def fit_variogram(lat, lon, resid, kind='exponential', nbins=10):
    """Подбор параметров вариограммы по эмпирической полудисперсии (МНК)."""
    D = _dist(np.asarray(lat), np.asarray(lon), np.asarray(lat), np.asarray(lon))
    i, j = np.triu_indices(len(resid), 1)
    h = D[i, j]
    g = 0.5 * (resid[i] - resid[j]) ** 2
    edges = np.quantile(h, np.linspace(0, 1, nbins + 1))
    hb, gb = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (h >= a) & (h <= b)
        if m.sum() >= 3:
            hb.append(h[m].mean())
            gb.append(g[m].mean())
    hb, gb = np.array(hb), np.array(gb)
    var = np.var(resid) if np.var(resid) > 0 else 1e-6
    x0 = [0.1 * var, var, np.median(h) if len(h) else 100]
    if len(hb) < 3:
        return dict(nugget=x0[0], sill=x0[1], range=x0[2], kind=kind, emp=(hb, gb))
    f = lambda p: _vmodel(hb, abs(p[0]), abs(p[1]), max(abs(p[2]), 1.0), kind) - gb
    r = least_squares(f, x0, bounds=([0, 1e-9, 1.0], [np.inf, np.inf, 5 * h.max()]))
    nug, sill, rg = r.x
    return dict(nugget=nug, sill=sill, range=rg, kind=kind, emp=(hb, gb))


def ordinary_kriging(lat, lon, val, glat, glon, vg):
    lat, lon, val = map(np.asarray, (lat, lon, val))
    n = len(val)
    D = _dist(lat, lon, lat, lon)
    K = np.ones((n + 1, n + 1))
    K[:n, :n] = _vmodel(D, vg['nugget'], vg['sill'], vg['range'], vg['kind'])
    K[n, n] = 0
    Kinv = np.linalg.pinv(K)
    d = _dist(glat.ravel(), glon.ravel(), lat, lon)
    k = np.ones((d.shape[0], n + 1))
    k[:, :n] = _vmodel(d, vg['nugget'], vg['sill'], vg['range'], vg['kind'])
    w = k @ Kinv.T
    est = w[:, :n] @ val
    return est.reshape(glat.shape)


def _design(lat, lon, alt=None):
    cols = [np.ones_like(lat), lat, lon]
    if alt is not None:
        cols.append(alt)
    return np.column_stack(cols)


def regression_kriging(lat, lon, val, glat, glon, alt=None, galt=None, kind='exponential'):
    """
    Регрессионный кригинг. Если высоты станций (alt) и узлов сетки (galt)
    заданы — в регрессию входит высота. Возвращает (поле, информация).
    """
    lat, lon, val = map(lambda x: np.asarray(x, float), (lat, lon, val))
    use_alt = alt is not None and galt is not None and np.isfinite(alt).all()
    X = _design(lat, lon, np.asarray(alt, float) if use_alt else None)
    beta, *_ = np.linalg.lstsq(X, val, rcond=None)
    trend_st = X @ beta
    resid = val - trend_st
    vg = fit_variogram(lat, lon, resid, kind)
    Xg = _design(glat.ravel(), glon.ravel(), np.asarray(galt, float).ravel() if use_alt else None)
    trend_g = (Xg @ beta).reshape(glat.shape)
    if use_alt:
        trend_g = np.where(np.isfinite(trend_g), trend_g, (_design(glat.ravel(), glon.ravel(), np.full(glat.size, np.nanmean(alt))) @ beta).reshape(glat.shape))
    res_g = ordinary_kriging(lat, lon, resid, glat, glon, vg)
    ss = ((val - val.mean()) ** 2).sum()
    info = dict(beta=beta, vg=vg, use_alt=use_alt,
                r2_trend=1 - (resid ** 2).sum() / ss if ss > 0 else np.nan)
    return trend_g + res_g, info


def loo_cv(lat, lon, val, alt=None, methods=('idw', 'rk', 'nearest')):
    """Перекрёстная проверка с поочерёдным исключением станций. Возвращает таблицу метрик."""
    lat, lon, val = map(lambda x: np.asarray(x, float), (lat, lon, val))
    n = len(val)
    rows = []
    for m in methods:
        pred = np.full(n, np.nan)
        for i in range(n):
            k = np.arange(n) != i
            g1, g2 = np.array([[lat[i]]]), np.array([[lon[i]]])
            try:
                if m == 'idw':
                    pred[i] = idw(lat[k], lon[k], val[k], g1, g2)[0, 0]
                elif m == 'nearest':
                    pred[i] = nearest(lat[k], lon[k], val[k], g1, g2)[0, 0]
                else:
                    a = None if alt is None else np.asarray(alt, float)
                    pred[i] = regression_kriging(lat[k], lon[k], val[k], g1, g2,
                                                 alt=None if a is None else a[k],
                                                 galt=None if a is None else np.array([[a[i]]]))[0][0, 0]
            except Exception:
                pass
        ok = np.isfinite(pred)
        e = pred[ok] - val[ok]
        ss = ((val[ok] - val[ok].mean()) ** 2).sum()
        rows.append(dict(Метод={'idw': 'IDW', 'rk': 'Регрессионный кригинг', 'nearest': 'Ближайшая станция'}[m],
                         MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                         **{'R²': 1 - (e ** 2).sum() / ss if ss > 0 else np.nan}))
    return pd.DataFrame(rows)


def mask_by_geometry(glat, glon, geom):
    """Маска узлов сетки внутри полигона (shapely)."""
    import shapely
    return shapely.contains_xy(geom, glon, glat)


def zonal_mean(field, glat, glon, polygons):
    """Среднее значение поля внутри каждого полигона района. Возвращает список значений."""
    import shapely
    out = []
    for p in polygons:
        m = shapely.contains_xy(p['geometry'], glon, glat)
        v = field[m]
        if m.any() and np.isfinite(v).any():
            out.append(np.nanmean(v))
        else:
            # малый район без узлов сетки — значение в ближайшем узле к его внутренней точке
            c = p['geometry'].representative_point()
            k = np.nanargmin((glon - c.x) ** 2 + (glat - c.y) ** 2)
            out.append(np.ravel(field)[k])
    return out
