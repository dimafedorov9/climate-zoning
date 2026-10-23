"""
=============================================================================
МОДУЛЬ ВОССТАНОВЛЕНИЯ ПРОПУСКОВ

Методы (выбираются пользователем или автоматически по перекрёстной проверке):
  interp   — линейная интерполяция по времени (только короткие разрывы,
             ≤ FILL_MAX_INTERP_DAYS суток; для температурных рядов);
  regress  — регрессия аномалий станции на аномалии соседних станций:
             для каждого соседа строится парная регрессия, оценки
             объединяются с весами r² (по доступным в эти сутки соседям);
  idw      — обратно-взвешенные по расстоянию аномалии соседей;
  ratio    — для осадков и ветра: значения соседей × отношение средних
             месячных величин станции и соседа, веса r² (осадки < 0,1 мм → 0);
  knn      — k ближайших соседей в пространстве «сутки × станции»
             (KNNImputer по матрице аномалий);
  rf       — Random Forest: признаки — аномалии соседних станций и день года;
  clim     — климатическая норма дня года (только как резерв).

Проверка точности: из имеющихся значений случайно скрывается доля
FILL_CV_SHARE, они восстанавливаются каждым методом и сравниваются с
фактическими: MAE, RMSE, смещение, R² (для осадков — также доля верно
определённых суток «с осадками / без осадков»).

Пропуски длиннее FILL_MAX_GAP_DAYS и сутки, когда нет данных ни на одной
соседней станции, не заполняются. Каждое восстановленное значение
записывается в журнал с указанием метода.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as P
from .spatial import to_matrix, seasonal_anomalies, neighbors, distance_matrix

METHODS = {
    'interp': 'Линейная интерполяция (короткие разрывы)',
    'regress': 'Регрессия по соседним станциям',
    'idw': 'Обратно-взвешенное расстояние (IDW)',
    'ratio': 'Отношение к соседним станциям (осадки, ветер)',
    'knn': 'k ближайших соседей (KNN)',
    'rf': 'Random Forest по соседним станциям',
    'clim': 'Климатическая норма дня года',
}
NONNEG = ('rr', 'fm')
_FUT = int(pd.__version__.split('.')[0]) >= 2 and 'future_stack' in pd.DataFrame.stack.__code__.co_varnames
FILL_COMPS = {
    'TX': ['tmin', 'tmax', 'tmean'],
    'TT': ['tt00', 'tt03', 'tt06', 'tt09', 'tt12', 'tt15', 'tt18', 'tt21'],
    'RS': ['td00', 'td03', 'td06', 'td09', 'td12', 'td15', 'td18', 'td21'],
    'TS': ['ts00', 'ts12'],
    'RR': ['rr_s'],
    'FM': ['fm_n', 'fm_d', 'fm_s'],
}
DEFAULT_METHODS = {'TX': 'auto', 'TT': 'regress', 'RS': 'regress', 'TS': 'auto', 'RR': 'auto', 'FM': 'auto'}
CANDIDATES_T = ['interp', 'regress', 'idw', 'knn', 'rf']
CANDIDATES_NN = ['ratio', 'regress', 'knn', 'rf']


def _full_matrix(df, comp):
    M = to_matrix(df, comp)
    full = pd.date_range(M.index.min(), M.index.max())
    return M.reindex(full)


def _gap_length(M):
    """Длина серии пропусков, к которой относится каждая ячейка (0 для имеющихся значений)."""
    L = pd.DataFrame(0, index=M.index, columns=M.columns)
    for c in M.columns:
        na = M[c].isna()
        g = (na != na.shift()).cumsum()
        L[c] = na.groupby(g).transform('sum').where(na, 0)
    return L


# ─── Методы ──────────────────────────────────────────────────────────────
def m_interp(M, **kw):
    return M.interpolate(limit=P.FILL_MAX_INTERP_DAYS, limit_area='inside')


def m_regress(M, A, clim, nbrs, **kw):
    out = M.copy()
    for st, nb in nbrs.items():
        if st not in M.columns or not nb:
            continue
        y = A[st]
        est_num = pd.Series(0.0, index=M.index)
        est_den = pd.Series(0.0, index=M.index)
        for o, w, r, km in nb:
            x = A[o]
            ok = x.notna() & y.notna()
            if ok.sum() < 60:
                continue
            xm, ym = x[ok].mean(), y[ok].mean()
            vx = ((x[ok] - xm) ** 2).sum()
            if vx == 0:
                continue
            b = ((x[ok] - xm) * (y[ok] - ym)).sum() / vx
            a = ym - b * xm
            rr = np.corrcoef(x[ok], y[ok])[0, 1]
            if np.isnan(rr) or rr < 0.3:
                continue
            e = a + b * x
            wt = rr ** 2
            est_num = est_num.add((e * wt).fillna(0))
            est_den = est_den.add(x.notna() * wt)
        est = (est_num / est_den.replace(0, np.nan)) + clim[st]
        out[st] = M[st].fillna(est)
    return out


def m_idw(M, A, clim, stations, **kw):
    D = distance_matrix(stations[stations['st'].isin(M.columns)])
    out = M.copy()
    for st in M.columns:
        if st not in D.index:
            continue
        d = D.loc[st].drop(st).dropna()
        d = d[d <= P.NEIGHBOR_MAX_KM * 1.5]
        if d.empty:
            continue
        w = 1.0 / np.maximum(d.values, 1.0) ** 2
        X = A[d.index].values
        m = ~np.isnan(X)
        ws = (m * w).sum(1)
        est = np.where(ws > 0, np.nansum(np.nan_to_num(X) * w, 1) / np.where(ws > 0, ws, 1), np.nan)
        out[st] = M[st].fillna(pd.Series(est, index=M.index) + clim[st])
    return out


def m_ratio(M, nbrs, **kw):
    """Для неотрицательных величин: значения соседей × отношение месячных средних."""
    out = M.copy()
    mon = M.index.month
    mm = M.groupby(mon).mean()
    for st, nb in nbrs.items():
        if st not in M.columns or not nb:
            continue
        num = pd.Series(0.0, index=M.index)
        den = pd.Series(0.0, index=M.index)
        for o, w, r, km in nb:
            ratio = (mm[st] / mm[o].replace(0, np.nan)).reindex(mon).values
            e = M[o] * ratio
            wt = (r ** 2) if not np.isnan(r) else w
            num = num.add((e * wt).fillna(0))
            den = den.add(e.notna() * wt)
        est = num / den.replace(0, np.nan)
        out[st] = M[st].fillna(est)
    return out


def m_knn(M, A, clim, k=5, **kw):
    """
    KNN по матрице аномалий «сутки × станции». Сутки-доноры подбираются
    внутри того же календарного года (быстро и с учётом сезона); строки, где
    данных нет ни на одной станции, не обрабатываются.
    """
    from sklearn.impute import KNNImputer
    X = A.values
    Xf = X.copy()
    years = A.index.year.values
    for y in np.unique(years):
        r = np.where(years == y)[0]
        sub = X[r]
        rows_ok = ~np.all(np.isnan(sub), axis=1)
        cols_ok = ~np.all(np.isnan(sub[rows_ok]), axis=0) if rows_ok.any() else np.zeros(sub.shape[1], bool)
        if rows_ok.sum() <= k or cols_ok.sum() < 2 or not np.isnan(sub[np.ix_(rows_ok, cols_ok)]).any():
            continue
        imp = KNNImputer(n_neighbors=k, weights='distance')
        part = imp.fit_transform(sub[np.ix_(rows_ok, cols_ok)])
        blk = Xf[r]
        blk[np.ix_(rows_ok, cols_ok)] = part
        Xf[r] = blk
    est = pd.DataFrame(Xf, index=A.index, columns=A.columns) + clim
    return M.fillna(est)


def m_rf(M, A, clim, nbrs, **kw):
    from sklearn.ensemble import RandomForestRegressor
    out = M.copy()
    doy = M.index.dayofyear.values
    sin, cos = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for st, nb in nbrs.items():
        if st not in M.columns or not nb or M[st].isna().sum() == 0:
            continue
        cols = [o for o, *_ in nb]
        F = A[cols].copy()
        avail = F.notna().sum(axis=1)
        F = F.apply(lambda c: c.fillna(F.mean(axis=1)))     # пропуск соседа — среднее доступных соседей
        F['sin'], F['cos'] = sin, cos
        y = A[st]
        tr = y.notna() & (avail >= 1)
        te = y.isna() & (avail >= 1)
        if tr.sum() < 200 or te.sum() == 0:
            continue
        rf = RandomForestRegressor(n_estimators=60, min_samples_leaf=5, max_samples=0.5,
                                   n_jobs=-1, random_state=42)
        rf.fit(F[tr].fillna(0).values, y[tr].values)
        pred = pd.Series(np.nan, index=M.index)
        pred[te] = rf.predict(F[te].fillna(0).values)
        out[st] = M[st].fillna(pred + clim[st])
    return out


def m_clim(M, clim, **kw):
    return M.fillna(clim)


def _apply_method(method, M, ctx):
    fn = {'interp': m_interp, 'regress': m_regress, 'idw': m_idw, 'ratio': m_ratio,
          'knn': m_knn, 'rf': m_rf, 'clim': m_clim}[method]
    return fn(M, **ctx)


def _context(M, stations, comp):
    A, clim = seasonal_anomalies(M)
    nbrs = neighbors(stations[stations['st'].isin(M.columns)], A, n=P.NEIGHBOR_N,
                     max_km=P.NEIGHBOR_MAX_KM * 1.5, same_region=P.NEIGHBOR_SAME_REGION)
    return dict(A=A, clim=clim, nbrs=nbrs, stations=stations)


def _postprocess(Mf, comp):
    if comp.startswith(NONNEG):
        Mf = Mf.clip(lower=0)
        if comp.startswith('rr'):
            Mf = Mf.mask((Mf < 0.1) & (Mf > 0), 0.0)
        if comp.startswith('fm'):
            Mf = Mf.round()
    return Mf


# ─── Перекрёстная проверка ───────────────────────────────────────────────
def cross_validate(df, comp, stations, methods=None, share=P.FILL_CV_SHARE, seed=42):
    """
    Скрывает долю имеющихся значений, восстанавливает каждым методом и
    сравнивает с фактом. Возвращает таблицу метрик по методам.
    """
    M = _full_matrix(df, comp)
    nonneg = comp.startswith(NONNEG)
    methods = methods or (CANDIDATES_NN if nonneg else CANDIDATES_T)
    rng = np.random.default_rng(seed)
    known = M.notna().values
    # скрываем одиночные сутки, чтобы проверка была сопоставима для интерполяции
    hide = known & (rng.random(M.shape) < share)
    Mh = M.mask(hide)
    truth = M.values[hide]
    rows = []
    ctx = _context(Mh, stations, comp)
    for meth in methods:
        try:
            est = _postprocess(_apply_method(meth, Mh, ctx), comp)
        except Exception as e:  # noqa
            rows.append(dict(Компонент=comp, Метод=meth, Описание=METHODS[meth], Ошибка=str(e)))
            continue
        pred = est.values[hide]
        ok = ~np.isnan(pred)
        if ok.sum() == 0:
            continue
        err = pred[ok] - truth[ok]
        ss = ((truth[ok] - truth[ok].mean()) ** 2).sum()
        row = dict(Компонент=comp, Метод=meth, Описание=METHODS[meth],
                   **{'Восстановлено, %': 100 * ok.mean(), 'MAE': np.abs(err).mean(),
                      'RMSE': np.sqrt((err ** 2).mean()), 'Смещение': err.mean(),
                      'R²': 1 - (err ** 2).sum() / ss if ss > 0 else np.nan, 'N': int(ok.sum())})
        if comp.startswith('rr'):
            row['Верно «есть/нет осадков», %'] = 100 * np.mean((pred[ok] >= 1.0) == (truth[ok] >= 1.0))
        rows.append(row)
    res = pd.DataFrame(rows)
    return res


def best_method(cv):
    """Лучший метод: максимальная доля восстановления ≥ 90 %, затем минимальная RMSE."""
    c = cv.dropna(subset=['RMSE'])
    if c.empty:
        return None
    full = c[c['Восстановлено, %'] >= 90]
    c = full if not full.empty else c
    return c.sort_values('RMSE').iloc[0]['Метод']


# ─── Заполнение ──────────────────────────────────────────────────────────
def fill_component(df, comp, stations, method='auto', cv=None):
    """
    Восстанавливает пропуски одного компонента. При method='auto' выбирается
    лучший по перекрёстной проверке; интерполяция всегда применяется первой
    к коротким разрывам (для температурных рядов). Возвращает (M_filled,
    log_rows, cv_table, method_used).
    """
    M = _full_matrix(df, comp)
    nonneg = comp.startswith(NONNEG)
    if method == 'auto':
        cv = cv if cv is not None else cross_validate(df, comp, stations)
        method = best_method(cv) or ('ratio' if nonneg else 'regress')
    gaps = _gap_length(M)
    fill_ok = (gaps > 0) & (gaps <= P.FILL_MAX_GAP_DAYS)
    ctx = _context(M, stations, comp)
    used = pd.DataFrame('', index=M.index, columns=M.columns)
    cur = M.copy()
    if not nonneg and method != 'interp':
        step = m_interp(cur)
        new = step.notna() & cur.isna() & fill_ok
        cur = cur.where(~new, step)
        used = used.mask(new, 'interp')
    step = _apply_method(method, cur, ctx)
    step = _postprocess(step, comp)
    new = step.notna() & cur.isna() & fill_ok
    cur = cur.where(~new, step)
    used = used.mask(new, method)
    log = used.stack()
    log = log[log != ''].reset_index()
    log.columns = ['date', 'st', 'method']
    log['comp'] = comp
    log['value'] = [cur.at[d, s] for d, s in zip(log['date'], log['st'])]
    return cur, log, cv, method


def fill_dataset(data, stations, methods=None, comps=None, progress=None):
    """
    Восстанавливает пропуски во всех переменных. methods — dict var → метод
    ('auto' или ключ METHODS). Возвращает (filled_data, fill_log, cv_tables).
    """
    methods = {**DEFAULT_METHODS, **(methods or {})}
    filled = {v: d.copy() for v, d in data.items()}
    logs, cvs = [], []
    todo = [(v, c) for v, cs in FILL_COMPS.items() if v in data and methods.get(v) != 'none'
            for c in (comps or {}).get(v, cs) if c in data[v].columns]
    # каждый регион обрабатывается отдельно: свои соседи, своя проверка методов
    for k, (var, comp) in enumerate(todo):
        if progress:
            progress(k / max(1, len(todo)), f'{var}: {comp}')
        longs = []
        for rg, dfr in data[var].groupby('region', dropna=False):
            str_ = stations[stations['st'].isin(dfr['st'].unique())]
            if dfr[comp].notna().sum() == 0:
                continue
            Mf, log, cv, used = fill_component(dfr, comp, str_, methods.get(var, 'auto'))
            if cv is not None and not cv.empty:
                cv = cv.copy()
                cv.insert(0, 'Переменная', var)
                cv.insert(0, 'Регион', rg)
                cv['Выбран'] = cv['Метод'] == used
                cvs.append(cv)
            log.insert(0, 'var', var)
            log['region'] = rg
            logs.append(log)
            long = Mf.stack(future_stack=True).rename(comp).reset_index() if _FUT else \
                Mf.stack(dropna=False).rename(comp).reset_index()
            long.columns = ['date', 'st', comp]
            longs.append(long)
        if not longs:
            continue
        long = pd.concat(longs, ignore_index=True)
        base = filled[var].drop(columns=[comp])
        region = filled[var].groupby('st')['region'].first()
        base = base.merge(long, on=['st', 'date'], how='outer')
        base['region'] = base['st'].map(region)
        filled[var] = base.sort_values(['st', 'date']).reset_index(drop=True)
    # согласование TX: Tмин ≤ Tср ≤ Tмакс для восстановленных значений
    if 'TX' in filled:
        t = filled['TX']
        lo = t[['tmin', 'tmean']].min(axis=1)
        hi = t[['tmax', 'tmean']].max(axis=1)
        t['tmin'] = np.where(t['tmin'] > t['tmean'], lo, t['tmin'])
        t['tmax'] = np.where(t['tmax'] < t['tmean'], hi, t['tmax'])
    if 'FM' in filled and {'fm_n', 'fm_d', 'fm_s'} <= set(filled['FM'].columns):
        f = filled['FM']
        f['fm_s'] = np.where(f['fm_n'].notna() & f['fm_d'].notna(), f[['fm_n', 'fm_d']].max(axis=1), f['fm_s'])
    fill_log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    cv_all = pd.concat(cvs, ignore_index=True) if cvs else pd.DataFrame()
    return filled, fill_log, cv_all


def fill_summary(fill_log, data):
    """Сводка: сколько значений восстановлено по переменным, компонентам и методам."""
    if fill_log is None or fill_log.empty:
        return pd.DataFrame()
    keys = (['region'] if 'region' in fill_log else []) + ['var', 'comp', 'method']
    t = fill_log.groupby(keys).size().rename('Восстановлено').reset_index()
    t['Метод (описание)'] = t['method'].map(METHODS)
    return t.rename(columns={'region': 'Регион', 'var': 'Переменная', 'comp': 'Компонент', 'method': 'Метод'})
