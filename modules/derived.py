"""
=============================================================================
ПРОИЗВОДНЫЕ ВЕЛИЧИНЫ ПО СРОЧНЫМ ДАННЫМ
Относительная влажность по T и Td (формула Магнуса), дефицит насыщения,
индекс жары NWS (регрессия Ротфуса, в °C), Humidex, температура
смоченного термометра (Stull, 2011).
=============================================================================
"""

import numpy as np
import pandas as pd

HOURS = ['00', '03', '06', '09', '12', '15', '18', '21']


def e_sat(t):
    """Давление насыщенного пара над водой, гПа (Магнус, коэффициенты Сонтага)."""
    return 6.112 * np.exp(17.62 * t / (243.12 + t))


def rh_from_td(t, td):
    """Относительная влажность, %, по температуре и точке росы."""
    rh = 100.0 * e_sat(td) / e_sat(t)
    return np.clip(rh, 0, 100)


def vpd(t, td):
    """Дефицит насыщения, гПа."""
    return e_sat(t) - e_sat(td)


def heat_index_c(t, rh):
    """
    Индекс жары NWS (регрессия Ротфуса) в °C. Определён при T ≥ 27 °C и
    RH ≥ 40 %; вне этих условий возвращается NaN.
    """
    t = np.asarray(t, float)
    rh = np.asarray(rh, float)
    f = t * 9 / 5 + 32
    hi = (-42.379 + 2.04901523 * f + 10.14333127 * rh - 0.22475541 * f * rh
          - 6.83783e-3 * f ** 2 - 5.481717e-2 * rh ** 2 + 1.22874e-3 * f ** 2 * rh
          + 8.5282e-4 * f * rh ** 2 - 1.99e-6 * f ** 2 * rh ** 2)
    hi_c = (hi - 32) * 5 / 9
    ok = (t >= 27) & (rh >= 40)
    return np.where(ok, hi_c, np.nan)


def humidex(t, td):
    e = e_sat(td)
    return t + 0.5555 * (e - 10.0)


def wet_bulb_stull(t, rh):
    """Температура смоченного термометра, °C (Stull, 2011), при RH 5–99 %, T −20…50 °C."""
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh)
            - np.arctan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def build_terms(tt, rs, fm=None):
    """
    Объединяет TT и RS в таблицу по срокам: [st, date, region, hour, T, Td,
    RH, VPD, HI, Humidex, Tw]. Влажность рассчитывается по T и Td.
    """
    m = tt.merge(rs.drop(columns=['region'], errors='ignore'), on=['st', 'date'], how='left')
    parts = []
    for h in HOURS:
        if f'tt{h}' not in m.columns:
            continue
        T = m[f'tt{h}'].values
        Td = m[f'td{h}'].values if f'td{h}' in m.columns else np.full(len(m), np.nan)
        Td = np.where(Td > T + 0.5, np.nan, np.minimum(Td, T))
        rh = rh_from_td(T, Td)
        p = pd.DataFrame({'st': m['st'].values, 'date': m['date'].values, 'region': m['region'].values,
                          'hour': h, 'T': T, 'Td': Td, 'RH': rh, 'VPD': vpd(T, Td),
                          'HI': heat_index_c(T, rh), 'Humidex': humidex(T, Td),
                          'Tw': wet_bulb_stull(T, rh)})
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def daily_humidity(terms):
    """Суточные характеристики влажности: минимальная RH, средний дефицит насыщения, максимум индекса жары."""
    g = terms.groupby(['st', 'date'])
    d = pd.DataFrame({'rh_min': g['RH'].min(), 'rh_mean': g['RH'].mean(),
                      'vpd_mean': g['VPD'].mean(), 'hi_max': g['HI'].max(),
                      'humidex_max': g['Humidex'].max(), 'tw_max': g['Tw'].max()}).reset_index()
    return d
