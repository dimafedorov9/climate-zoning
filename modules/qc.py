"""
=============================================================================
МОДУЛЬ КОНТРОЛЯ КАЧЕСТВА СУТОЧНЫХ ДАННЫХ

Виды проверок названы по Руководству по климатологической практике
(ВМО-№ 100, 2018, раздел 3.3): проверка допустимости, проверка
внутренней, временной и пространственной согласованности.

Категории значений после контроля:
  «ошибочное»          — значение исключается (заменяется пропуском);
  «требует проверки»   — значение остаётся в ряду; пользователь решает,
                         исключать ли такие значения по каждому правилу;
  «исправленное»       — значение заменено исправленным (ошибка известного
                         происхождения: перепутанные единицы, лишняя цифра,
                         суточное значение ≠ max(ночь, день));
  «восстановленное»    — пропуск заполнен (этап восстановления пропусков).

Коды правил:
  L    проверка допустимости: значение вне диапазона, возможного для
       величины в регионе (пределы — в params.LIMITS);
  N    пространственная согласованность: отклонение от медианы значений
       станций региона в тот же день;
  S    пространственная согласованность: расхождение с оценкой по
       соседним станциям (робастная z-оценка остатка по месяцам);
  P    временная согласованность: одно и то же ненулевое значение
       ≥ 7 суток подряд;
  TX1–TX3, RS1, RH1, RR4, FM5 — внутренняя согласованность значений
       одних суток (Tмин ≤ Tмакс, Td ≤ T, ночь + день = сутки …);
  TS1, FM1 — коды и массовые сбои вместо значений;
  RR1–RR3, FM2–FM4 — проверки допустимости и пространственной
       согласованности для осадков и ветра;
  IF   Isolation Forest — многомерный поиск нетипичных сочетаний
       (Liu et al., 2008), только «требует проверки».
Периоды повышенной строгости (params.STRICT_PERIODS): пороги ×0,6.
Каждое действие записывается в журнал: что, где, правило, было → стало.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as P
from .spatial import (to_matrix, seasonal_anomalies, neighbors, neighbor_prediction,
                      robust_z, in_periods)

REJECT, SUSPECT, CORRECT = 'ошибочное', 'требует проверки', 'исправленное'

RULES = {
    'L': 'Проверка допустимости: вне допустимых пределов',
    'N': 'Пространственная согласованность: отклонение от медианы по сети',
    'S': 'Пространственная согласованность: расхождение с соседними станциями',
    'P': 'Временная согласованность: неизменное значение ≥ 7 суток',
    'TX1': 'Внутренняя согласованность: Tмин > Tмакс',
    'TX2': 'Внутренняя согласованность: Tмакс − Tср > 15 °C',
    'TX3': 'Внутренняя согласованность: вероятная ошибка знака',
    'TX4': 'Внутренняя согласованность: Tср вне интервала [Tмин; Tмакс]',
    'RS1': 'Внутренняя согласованность: точка росы выше температуры',
    'RH1': 'Проверка допустимости: влажность > 100 %',
    'TS1': 'Код вместо значения (±90…99)',
    'RR1': 'Проверка допустимости: суточная сумма выше порога сезона',
    'RR2': 'Пространственная согласованность: большая сумма при малых осадках у соседей',
    'RR3': 'Временная согласованность: одинаковые большие суммы подряд',
    'RR4': 'Внутренняя согласованность: ночь + день ≠ сутки',
    'FM1': 'Массовый сбой: круглые значения на многих станциях в одну дату',
    'FM2': 'Завышение (две цифры вместо одной), деление на 10',
    'FM3': 'Скорость в км/ч вместо м/с',
    'FM4': 'Пространственная согласованность: одиночный выброс ветра',
    'FM5': 'Внутренняя согласованность: сутки ≠ max(ночь, день)',
    'IF': 'Многомерная проверка (Isolation Forest): нетипичное сочетание',
}

# Пояснения к правилам (выводятся в программе как сноски)
RULE_HELP = {
    'L': 'Значение невозможно для данной величины в регионе: например, Tмин = −62,5 °C или Tмакс = 90,3 °C. '
         'Пределы задаются в параметрах (params.LIMITS). Соответствует «проверке допустимости» (ВМО-№ 100, 3.3.7).',
    'N': 'Значение отличается от медианы по станциям того же региона в тот же день больше чем на заданный порог '
         '(по умолчанию 20 °C). Для TS и Td — «требует проверки»: зимой на северных станциях такие различия бывают реальными.',
    'S': 'Аномалия значения относительно сезонной нормы сравнивается со взвешенной оценкой по 5 наиболее связанным '
         'соседним станциям. Расхождение нормируется на его типичный разброс в этом месяце (робастная z-оценка по медиане '
         'и MAD). |z| > 4 — требует проверки; |z| > 6 и расхождение > 8 °C — ошибочное.',
    'P': 'Одно и то же ненулевое значение повторяется 7 и более суток подряд — признак «зависшего» значения.',
    'TX1': 'Tмин и Tмакс в TX определены по полусуткам, поэтому при резкой смене погоды Tмин может немного превышать '
           'Tмакс. Превышение до 5 °C — требует проверки, больше — ошибочное (исключается компонент с большим расхождением '
           'с соседями).',
    'TX2': 'Tмакс превышает среднюю за сутки больше чем на 15 °C (летом такие разности практически не встречаются).',
    'TX3': 'Зимой Tмакс > +15 °C при отрицательной средней — вероятно, потерян знак «минус».',
    'TX4': 'Средняя суточная температура ниже минимальной или выше максимальной более чем на 1 °C. Обычно связано с '
           'разными интервалами осреднения (экстремумы — за полусутки, средняя — по восьми срокам); значение остаётся '
           'в ряду с признаком «требует проверки».',
    'RS1': 'Точка росы не может быть выше температуры воздуха в тот же срок (допуск 0,5 °C).',
    'RH1': 'Относительная влажность больше 100 %.',
    'TS1': 'Значения −90…−99 и +90…+99 в температуре поверхности почвы — коды, а не измерения.',
    'RR1': 'Суточная сумма > 100 мм вне лета или ≥ 150 мм летом исключается (пороги в параметрах).',
    'RR2': 'Сумма ≥ 50 мм за сутки, тогда как на всех соседних станциях выпало < 20 мм.',
    'RR3': 'Одинаковые суммы ≥ 30 мм два дня подряд — признак дублирования записи.',
    'RR4': 'Сумма ночной и дневной половин отличается от суточной более чем на 1 мм.',
    'FM1': 'Круглые значения 30, 40 … 90 и 98 м/с в одну дату у многих станций региона (по умолчанию ≥ 5) — '
           'сбой передачи или базы данных; значения исключаются как пропуск.',
    'FM2': 'Для станций, указанных в параметрах, завышенные значения (≥ 30 м/с и более чем в 2,5 раза выше максимума у '
           'соседей) делятся на 10: вместо одной цифры передано две. Такие значения следует использовать с осторожностью.',
    'FM3': 'Значение ≥ 30 м/с совпадает с целой скоростью в м/с, переведённой в км/ч (round(n × 3,6): 32, 36, 40, 43, 47, '
           '50, 54, 58, 61, 65, 68, 72, 76 …), а у соседей ветер < 20 м/с. Если включён пересчёт и после деления на 3,6 '
           'значение согласуется с соседями (не больше их максимума + 8 м/с), оно исправляется; иначе — ошибочное. '
           'Значения вне этого ряда (например 69, 71, 91, 92) нельзя объяснить переводом единиц, поэтому они проверяются '
           'правилом FM4.',
    'FM4': 'Ветер ≥ 30 м/с при ветре у всех соседних станций < 20 м/с — одиночный выброс.',
    'FM5': 'Суточный максимум должен равняться большему из ночного и дневного; значение пересчитывается.',
    'IF': 'Алгоритм Isolation Forest (Liu et al., 2008) ищет сутки с нетипичным сочетанием аномалий Tмин, Tмакс, Tср '
          'и их расхождений с соседями; результат — только «требует проверки».',
}

VAR_COMPS = {
    'TX': ['tmin', 'tmax', 'tmean'],
    'TT': ['tt00', 'tt03', 'tt06', 'tt09', 'tt12', 'tt15', 'tt18', 'tt21'],
    'RS': ['td00', 'td03', 'td06', 'td09', 'td12', 'td15', 'td18', 'td21'],
    'RH': ['rh00', 'rh03', 'rh06', 'rh09', 'rh12', 'rh15', 'rh18', 'rh21'],
    'TS': ['ts00', 'ts12'],
    'RR': ['rr_n', 'rr_d', 'rr_s'],
    'FM': ['fm_n', 'fm_d', 'fm_s'],
}
LIM_KEY = {'tmin': 'tmin', 'tmax': 'tmax', 'tmean': 'tmean'}


def _lim(var, comp):
    if comp in LIM_KEY:
        return P.LIMITS[LIM_KEY[comp]]
    return P.LIMITS[{'TT': 'tt', 'RS': 'td', 'RH': 'rh', 'TS': 'ts', 'RR': 'rr', 'FM': 'fm'}[var]]


class Log:
    """Журнал контроля качества."""

    def __init__(self):
        self.rows = []

    def add_mask(self, var, comp, M, mask, action, rule, new=None, note=''):
        """Добавляет записи для всех True в mask (матрица date × st)."""
        if mask is None:
            return
        mask = mask.fillna(False) if hasattr(mask, 'fillna') else mask
        idx = np.argwhere(np.asarray(mask, bool))
        if len(idx) == 0:
            return
        dates, sts = M.index, M.columns
        vals = M.values
        newv = None if new is None else np.asarray(new.values if hasattr(new, 'values') else new, float)
        for i, j in idx:
            self.rows.append((var, comp, sts[j], dates[i], vals[i, j],
                              np.nan if newv is None else newv[i, j], action, rule, note))

    def frame(self):
        df = pd.DataFrame(self.rows, columns=['Переменная', 'Компонент', 'Станция', 'Дата',
                                              'Значение', 'Новое значение', 'Действие',
                                              'Правило', 'Комментарий'])
        if df.empty:
            return df
        df['Описание правила'] = df['Правило'].map(RULES)
        # при нескольких правилах для одного значения — приоритет: исправленное > ошибочное > требует проверки
        pr = {CORRECT: 0, REJECT: 1, SUSPECT: 2}
        df['_p'] = df['Действие'].map(pr)
        df = df.sort_values(['Переменная', 'Компонент', 'Станция', 'Дата', '_p'])
        df['Основное'] = ~df.duplicated(['Переменная', 'Компонент', 'Станция', 'Дата'])
        return df.drop(columns='_p').reset_index(drop=True)


# ─── Общие проверки ──────────────────────────────────────────────────────
def _strict_matrix(M):
    s = in_periods(M.index, P.STRICT_PERIODS)
    return pd.DataFrame(np.repeat(s[:, None], M.shape[1], 1), index=M.index, columns=M.columns)


def _region_groups(stations, cols):
    s = stations.set_index('st')['region']
    g = {}
    for c in cols:
        g.setdefault(s.get(c, '—'), []).append(c)
    return g


def check_limits(var, comp, M, log):
    lo, hi = _lim(var, comp)
    mask = (M < lo) | (M > hi)
    log.add_mask(var, comp, M, mask, REJECT, 'L', note=f'пределы [{lo}; {hi}]'.replace('.', ','))
    return mask


def check_network_median(var, comp, M, stations, log, thr, action=REJECT):
    strict = _strict_matrix(M)
    th = np.where(strict, thr * P.STRICT_FACTOR, thr)
    total = pd.DataFrame(False, index=M.index, columns=M.columns)
    for rg, cols in _region_groups(stations, M.columns).items():
        if len(cols) < 4:
            continue
        sub = M[cols]
        med = sub.median(axis=1)
        dev = sub.sub(med, axis=0).abs()
        mask = dev > th[:, [M.columns.get_loc(c) for c in cols]]
        total.loc[:, cols] = mask
    log.add_mask(var, comp, M, total, action, 'N', note=f'порог {thr:g} °C'.replace('.', ','))
    return total


def spatial_check(var, comp, M, stations, log, nbrs=None, z_s=P.SPATIAL_Z_SUSPECT,
                  z_r=P.SPATIAL_Z_REJECT, allow_reject=True):
    """Проверка по соседям. Возвращает (z, nbrs)."""
    A, _ = seasonal_anomalies(M)
    if nbrs is None:
        nbrs = neighbors(stations, A, n=P.NEIGHBOR_N, max_km=P.NEIGHBOR_MAX_KM, same_region=P.NEIGHBOR_SAME_REGION)
    strict = _strict_matrix(M)
    fac = np.where(strict, P.STRICT_FACTOR, 1.0)
    # два прохода: во втором соседние значения с грубыми отклонениями не участвуют в оценке
    pred, nused = neighbor_prediction(A, nbrs)
    z = robust_z(A - pred).where(nused >= 2)
    A2 = A.mask(z.abs() > z_r * fac)
    pred, nused = neighbor_prediction(A2, nbrs)
    resid = A - pred
    z = robust_z(resid).where(nused >= 2)
    # брак — только при большом z и большом абсолютном расхождении с оценкой по соседям
    big = resid.abs() > np.where(strict, P.SPATIAL_MIN_ABS * P.STRICT_FACTOR, P.SPATIAL_MIN_ABS)
    rej = (z.abs() > z_r * fac) & big & allow_reject
    sus = (z.abs() > z_s * fac) & ~rej
    log.add_mask(var, comp, M, rej, REJECT, 'S', note='|z| по соседям > ' + str(z_r))
    log.add_mask(var, comp, M, sus, SUSPECT, 'S', note='|z| по соседям > ' + str(z_s))
    return z, nbrs, rej


def check_persistence(var, comp, M, log, ndays=7):
    """Одно и то же ненулевое значение ≥ ndays суток подряд — требует проверки."""
    mask = pd.DataFrame(False, index=M.index, columns=M.columns)
    for c in M.columns:
        x = M[c]
        g = (x != x.shift()).cumsum()
        size = x.groupby(g).transform('size')
        mask[c] = (size >= ndays) & x.notna() & x.ne(0)
    log.add_mask(var, comp, M, mask, SUSPECT, 'P', note=f'≥ {ndays} суток подряд')
    return mask


# ─── Переменные ──────────────────────────────────────────────────────────
def qc_tx(df, stations, log, opts):
    M = {c: to_matrix(df, c) for c in VAR_COMPS['TX']}
    bad = {c: check_limits('TX', c, M[c], log) for c in M}
    z = {}
    nb = None
    for c in M:
        Mc = M[c].mask(bad[c])
        nm = check_network_median('TX', c, Mc, stations, log, P.NETWORK_MEDIAN_DEV_T)
        Mc = Mc.mask(nm)
        z[c], nb, _ = spatial_check('TX', c, Mc, stations, log, nbrs=nb)
        if opts.get('persistence', True):
            check_persistence('TX', c, Mc, log)
    tmin, tmax, tmean = M['tmin'], M['tmax'], M['tmean']
    # Tмин > Tмакс: бракуется компонент с большим |z|; при неопределённости — оба
    # Tмин и Tмакс в TX определены по полусуткам, поэтому небольшое превышение Tмин над Tмакс
    # возможно при резкой смене погоды — «требует проверки»; превышение > 5 °C — ошибочное
    diff = tmin - tmax
    inv = diff > P.TEMP_ORDER_TOL
    gross = diff > P.TX_ORDER_REJECT
    zmin, zmax = z['tmin'].abs().fillna(0), z['tmax'].abs().fillna(0)
    log.add_mask('TX', 'tmin', tmin, gross & (zmin >= zmax), REJECT, 'TX1')
    log.add_mask('TX', 'tmax', tmax, gross & (zmax > zmin), REJECT, 'TX1')
    log.add_mask('TX', 'tmin', tmin, inv & ~gross, SUSPECT, 'TX1', note='возможно из-за полусуточных экстремумов')
    # Tмакс − Tср > 15
    big = (tmax - tmean) > P.TMAX_MINUS_TMEAN_MAX
    log.add_mask('TX', 'tmax', tmax, big, REJECT, 'TX2', note='Tмакс − Tср > 15 °C')
    out_mean = ((tmin > tmean + 1) | (tmean > tmax + 1)) & ~(tmin > tmax)
    log.add_mask('TX', 'tmean', tmean, out_mean, SUSPECT, 'TX4', note='Tср вне [Tмин; Tмакс] более чем на 1 °C')
    # ошибка знака: зимой Tмакс > +15 при оценке по соседям ниже 0
    winter = pd.DataFrame(np.repeat(np.isin(tmax.index.month, P.season_months(opts.get('region'), 'cold'))[:, None], tmax.shape[1], 1),
                          index=tmax.index, columns=tmax.columns)
    sign = winter & (tmax > 15) & (tmean < 0)
    if opts.get('correct_sign', False):
        log.add_mask('TX', 'tmax', tmax, sign, CORRECT, 'TX3', new=-tmax, note='знак изменён')
    else:
        log.add_mask('TX', 'tmax', tmax, sign, REJECT, 'TX3')
    sign_min = winter & (tmin > 15) & (tmean < 0)
    log.add_mask('TX', 'tmin', tmin, sign_min, REJECT, 'TX3')
    return M


def qc_temperature_terms(var, df, stations, log, opts, tt_df=None):
    comps = [c for c in VAR_COMPS[var] if c in df.columns]
    nb = None
    for c in comps:
        Mc = to_matrix(df, c)
        if var == 'TS':
            code = (Mc.abs() >= 89.5) & (Mc.abs() <= 99.5)
            log.add_mask(var, c, Mc, code, REJECT, 'TS1')
            Mc = Mc.mask(code)
        bad = check_limits(var, c, Mc, log)
        Mc = Mc.mask(bad)
        if var == 'TS':
            check_network_median(var, c, Mc, stations, log, P.NETWORK_MEDIAN_DEV_T, action=SUSPECT)
            spatial_check(var, c, Mc, stations, log, nbrs=nb, allow_reject=False)
            continue
        if var == 'RH':
            over = Mc > 100
            log.add_mask(var, c, Mc, over, REJECT, 'RH1')
            continue
        if var == 'RS' and tt_df is not None:
            h = c[-2:]
            if f'tt{h}' in tt_df.columns:
                T = to_matrix(tt_df, f'tt{h}').reindex(index=Mc.index, columns=Mc.columns)
                over = Mc > T + 0.5
                log.add_mask(var, c, Mc, over, REJECT, 'RS1')
                Mc = Mc.mask(over)
        nm = check_network_median(var, c, Mc, stations, log, P.NETWORK_MEDIAN_DEV_T,
                                  action=REJECT if var == 'TT' else SUSPECT)
        if var == 'TT':
            Mc = Mc.mask(nm)
        _, nb, _ = spatial_check(var, c, Mc, stations, log, nbrs=nb, allow_reject=(var == 'TT'))


def _neighbor_max(M, nbrs_list):
    out = pd.DataFrame(np.nan, index=M.index, columns=M.columns)
    for st, nb in nbrs_list.items():
        cols = [o for o, *_ in nb if o in M.columns]
        if cols:
            out[st] = M[cols].max(axis=1, skipna=True)
    return out


def _distance_neighbors(stations, cols, n=P.NEIGHBOR_N, max_km=P.NEIGHBOR_MAX_KM, M=None):
    st = stations[stations['st'].isin(cols)]
    A = None
    if st[['lat', 'lon']].isna().any(axis=1).all() and M is not None:
        A = M  # без координат — по корреляции рядов
    return neighbors(st, A, n=n, max_km=max_km, same_region=P.NEIGHBOR_SAME_REGION)


def qc_rr(df, stations, log, opts):
    M = {c: to_matrix(df, c) for c in VAR_COMPS['RR']}
    s = M['rr_s']
    nb = _distance_neighbors(stations, s.columns, M=s)
    months = s.index.month
    warm = pd.DataFrame(np.repeat(np.isin(months, P.season_months(opts.get('region'), 'summer'))[:, None], s.shape[1], 1),
                        index=s.index, columns=s.columns)
    for c in M:
        check_limits('RR', c, M[c], log)
        thr = warm * P.RR_MAX_WARM + (~warm) * P.RR_MAX_COLD
        over = (M[c] >= thr) & warm | (M[c] > thr) & ~warm
        log.add_mask('RR', c, M[c], over, REJECT, 'RR1',
                     note=f'> {P.RR_MAX_COLD:g} мм вне летних месяцев / ≥ {P.RR_MAX_WARM:g} мм в летние месяцы')
    # если суточная сумма забракована — полусуточные значения тоже
    over_s = ((s >= P.RR_MAX_WARM) & warm) | ((s > P.RR_MAX_COLD) & ~warm)
    for c in ('rr_n', 'rr_d'):
        log.add_mask('RR', c, M[c], over_s & M[c].notna() & ~(((M[c] >= P.RR_MAX_WARM) & warm) |
                                                              ((M[c] > P.RR_MAX_COLD) & ~warm)),
                     REJECT, 'RR1', note='забракована суточная сумма')
    nmax = _neighbor_max(s, nb)
    sus = (s >= P.RR_NEIGHBOR_SUSPECT) & (nmax < P.RR_NEIGHBOR_LOW) & ~over_s
    log.add_mask('RR', 'rr_s', s, sus, SUSPECT, 'RR2',
                 note='≥ 50 мм при соседях < 20 мм' + (' (холодный сезон)' if False else ''))
    rep = (s >= 30) & (s.diff() == 0)
    rep = rep | rep.shift(-1, fill_value=False)
    log.add_mask('RR', 'rr_s', s, rep & ~over_s, SUSPECT, 'RR3')
    both = M['rr_n'].notna() & M['rr_d'].notna() & s.notna()
    incons = both & ((M['rr_n'] + M['rr_d'] - s).abs() > 1.05)
    log.add_mask('RR', 'rr_s', s, incons & ~over_s, SUSPECT, 'RR4')


def qc_fm(df, stations, log, opts):
    M = {c: to_matrix(df, c) for c in VAR_COMPS['FM']}
    s = M['fm_s']
    nb = _distance_neighbors(stations, s.columns, M=s)
    groups = _region_groups(stations, s.columns)
    # FM1: круглые значения одновременно на многих станциях
    for c in M:
        Mc = M[c]
        is_round = Mc.isin(P.FM_ROUND_VALUES)
        mask = pd.DataFrame(False, index=Mc.index, columns=Mc.columns)
        for rg, cols in groups.items():
            cnt = is_round[cols].sum(axis=1)
            day_bad = cnt >= P.FM_ROUND_MIN_STATIONS
            mask.loc[day_bad, cols] = is_round.loc[day_bad, cols]
        log.add_mask('FM', c, Mc, mask, REJECT, 'FM1', note='вероятный сбой сети/базы, считается пропуском')
        M[c] = Mc.mask(mask)
    for c in M:
        Mc = M[c]
        nmax = _neighbor_max(Mc, nb)
        # FM2: станции из списка «делить на 10»
        corr_mask = pd.DataFrame(False, index=Mc.index, columns=Mc.columns)
        for st, y0 in P.FM_DIVIDE_BY_10.items():
            if st in Mc.columns:
                infl = (Mc[st] >= P.FM_SPIKE) & (Mc.index.year >= y0) & ((Mc[st] > 2.5 * nmax[st]) | nmax[st].isna())
                corr_mask[st] = infl
        log.add_mask('FM', c, Mc, corr_mask, CORRECT, 'FM2', new=Mc / 10.0,
                     note='исправлено делением на 10; использовать с осторожностью')
        rest = ~corr_mask
        # FM3: целые м/с, переведённые в км/ч (round(n·3,6)), при спокойных соседях
        kmh = Mc.isin(P.FM_KMH_SERIES) & (Mc >= P.FM_SPIKE) & (nmax < P.FM_SPIKE_NEIGHBORS) & rest
        conv = (Mc / 3.6).round()
        plausible = conv <= nmax + 8
        if opts.get('fm_kmh_convert', False):
            log.add_mask('FM', c, Mc, kmh & plausible, CORRECT, 'FM3', new=conv,
                         note='пересчёт из км/ч в м/с (÷ 3,6)')
            log.add_mask('FM', c, Mc, kmh & ~plausible, REJECT, 'FM3',
                         note='после пересчёта не согласуется с соседями')
        else:
            log.add_mask('FM', c, Mc, kmh, REJECT, 'FM3')
        # FM4: одиночные выбросы
        spike = (Mc >= P.FM_SPIKE) & (nmax < P.FM_SPIKE_NEIGHBORS) & rest & ~kmh
        log.add_mask('FM', c, Mc, spike, REJECT, 'FM4')
        check_limits('FM', c, Mc, log)
    # FM5: суточное = max(ночь, день)
    n, d = M['fm_n'], M['fm_d']
    mx = np.fmax(n, d)
    diff = s.notna() & n.notna() & d.notna() & (s != mx)
    log.add_mask('FM', 'fm_s', s, diff, CORRECT, 'FM5', new=mx, note='сутки = max(ночь, день)')


# ─── Isolation Forest ────────────────────────────────────────────────────
def isolation_forest_tx(df, stations, log, contamination=P.ISOFOREST_CONTAMINATION):
    """Многомерный поиск нетипичных суток в TX: аномалии Tмин/Tмакс/Tср, амплитуда, остатки по соседям."""
    from sklearn.ensemble import IsolationForest
    feats = {}
    nb = None
    for c in ['tmin', 'tmax', 'tmean']:
        M = to_matrix(df, c)
        A, _ = seasonal_anomalies(M)
        if nb is None:
            nb = neighbors(stations, A, n=P.NEIGHBOR_N, max_km=P.NEIGHBOR_MAX_KM, same_region=P.NEIGHBOR_SAME_REGION)
        pred, _ = neighbor_prediction(A, nb)
        feats[f'a_{c}'] = A.stack()
        feats[f'r_{c}'] = (A - pred).stack()
    X = pd.DataFrame(feats).dropna()
    if len(X) < 500:
        return
    X['amp'] = X['a_tmax'] - X['a_tmin']
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    lab = model.fit_predict(X.values)
    flagged = X.index[lab == -1]
    Mt = to_matrix(df, 'tmean')
    mask = pd.DataFrame(False, index=Mt.index, columns=Mt.columns)
    for d, st in flagged:
        mask.loc[d, st] = True
    log.add_mask('TX', 'tmean', Mt, mask, SUSPECT, 'IF', note='сутки в целом')


# ─── Запуск и применение ─────────────────────────────────────────────────
def _run_region(data, stations, opts, log, progress, k0, ntot):
    steps = [v for v in ['TX', 'TT', 'RS', 'RH', 'TS', 'RR', 'FM'] if v in data and not data[v].empty]
    for k, var in enumerate(steps):
        if progress:
            progress(min(1.0, (k0 + k) / ntot), var)
        df = data[var]
        if var == 'TX':
            qc_tx(df, stations, log, opts)
            if opts.get('isolation_forest', True):
                isolation_forest_tx(df, stations, log)
        elif var in ('TT', 'RS', 'RH', 'TS'):
            qc_temperature_terms(var, df, stations, log, opts, tt_df=data.get('TT'))
        elif var == 'RR':
            qc_rr(df, stations, log, opts)
        elif var == 'FM':
            qc_fm(df, stations, log, opts)
    return len(steps)


def run_qc(data, stations, opts=None, progress=None):
    """
    Выполняет все проверки отдельно для каждого региона (соседние станции,
    медиана по сети и многомерная проверка — внутри региона). opts — dict:
    persistence, correct_sign, fm_kmh_convert, isolation_forest.
    Возвращает журнал (DataFrame) с колонкой «Регион».
    """
    opts = opts or {}
    regions = sorted({r for d in data.values() for r in d['region'].dropna().unique()})
    frames = []
    ntot = max(1, 7 * len(regions))
    k0 = 0
    for rg in regions:
        log = Log()
        sub = {v: d[d['region'] == rg] for v, d in data.items() if v != 'DY'}
        st_r = stations[stations['region'] == rg] if 'region' in stations else stations
        k0 += _run_region(sub, st_r, {**opts, 'region': rg}, log, progress, k0, ntot)
        f = log.frame()
        if not f.empty:
            f.insert(0, 'Регион', rg)
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else Log().frame()


def apply_qc(data, log, reject_suspect_rules=()):
    """
    Применяет журнал к копии данных: «ошибочное» → NaN, «исправленное» → новое
    значение; «требует проверки» → NaN только для правил из reject_suspect_rules.
    Для забракованной суточной суммы RR признаки следов сбрасываются.
    Возвращает (clean_data, qc_flags) — qc_flags: dict var → DataFrame
    [st, date, comp, flag] для учёта на последующих этапах.
    """
    clean = {v: d.copy() for v, d in data.items()}
    flags = {}
    if log is None or log.empty:
        return clean, flags
    main = log[log['Основное']]
    for var, sub in main.groupby('Переменная'):
        if var not in clean:
            continue
        df = clean[var].set_index(['st', 'date'])
        rows = []
        for comp, s2 in sub.groupby('Компонент'):
            idx = pd.MultiIndex.from_arrays([s2['Станция'].values, pd.to_datetime(s2['Дата']).values])
            act = s2['Действие'].values
            rule = s2['Правило'].values
            rej = (act == REJECT) | ((act == SUSPECT) & np.isin(rule, list(reject_suspect_rules)))
            cor = act == CORRECT
            if rej.any():
                df.loc[idx[rej], comp] = np.nan
            if cor.any():
                df.loc[idx[cor], comp] = s2['Новое значение'].values[cor]
            fl = np.where(rej, 'R', np.where(cor, 'C', 'S'))
            rows.append(pd.DataFrame({'st': s2['Станция'].values, 'date': pd.to_datetime(s2['Дата']).values,
                                      'comp': comp, 'flag': fl}))
        clean[var] = df.reset_index()
        flags[var] = pd.concat(rows, ignore_index=True)
    # RR: если забракована суточная сумма — удалить и полусуточные значения
    if 'RR' in clean:
        r = clean['RR']
        gone = r['rr_s'].isna() & r[['rr_n', 'rr_d']].notna().any(axis=1) & \
            ~(r['rr_n'].notna() & r['rr_d'].notna())
        r.loc[gone, ['rr_n', 'rr_d']] = np.nan
    return clean, flags


def qc_summary(log):
    """Сводка журнала: число значений по переменным, правилам и действиям."""
    if log is None or log.empty:
        return pd.DataFrame()
    m = log[log['Основное']]
    keys = (['Регион'] if 'Регион' in m else []) + ['Переменная', 'Правило', 'Описание правила', 'Действие']
    t = m.groupby(keys).size().rename('Значений').reset_index()
    return t.sort_values(keys[:-3] + ['Значений'], ascending=[True] * (len(keys) - 3) + [False])
