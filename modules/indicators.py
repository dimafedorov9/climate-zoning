"""
=============================================================================
МОДУЛЬ РАСЧЁТА КЛИМАТИЧЕСКИХ ПОКАЗАТЕЛЕЙ
Из "сырых" месячных данных температуры и осадков считает производные
характеристики по станциям: сезонные T/P, индекс континентальности
Конрада, индексы аридности/увлажнения, ГТК, линейные тренды.
Формулы перенесены из рабочего скрипта nso_climate_analysis.py.
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats

MONTHS = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
          'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']

# Индексы месяцев (0=Январь ... 11=Декабрь) для сезонов
WINTER_IDX = [11, 0, 1]   # Дек, Янв, Фев
SPRING_IDX = [2, 3, 4]    # Мар, Апр, Май
SUMMER_IDX = [5, 6, 7]    # Июн, Июл, Авг
AUTUMN_IDX = [8, 9, 10]   # Сен, Окт, Ноя

INDICATOR_FORMULAS = {
    'T_год': 'Среднее из 12 среднемесячных температур за период',
    'T_зим/T_вес/T_лет/T_ос': 'Среднее температур за 3 месяца сезона (дек-фев / мар-май / июн-авг / сен-ноя)',
    'A_T': 'Годовая амплитуда: T(самый тёплый месяц) − T(самый холодный месяц)',
    'P_год': 'Сумма 12 среднемесячных сумм осадков за период',
    'P_зим/P_вес/P_лет/P_ос': 'Сумма осадков за 3 месяца сезона',
    'K_Конрад': 'Индекс континентальности Конрада: K = 1,7·A_T/sin(широта) − 20,4',
    'I_аридн': 'Индекс аридности де Мартона: Ia = P_год / (T_год + 10). '
               'Ia < 20 — засушливо, 20–30 — умеренно увлажнено',
    'I_увл': 'Индекс увлажнения Высоцкого-Иванова: Iy = P_год / (0,0018·(T_год+25)²)',
    'ГТК': 'Упрощённый гидротермический коэффициент Селянинова по летним месяцам',
    'Тренд_T_дек': 'Линейный тренд среднегодовой температуры, °C/декада (по годам периода)',
    'Тренд_P_дек': 'Линейный тренд годовой суммы осадков, мм/декада (по годам периода)',
}


def _pivot_station_months(long_df, tip, station_col='Станция'):
    """
    Из длинного формата делает средние по месяцам за период на станцию:
    возвращает DataFrame (index=Станция, columns=MONTHS) — средние значения
    месяца, усреднённые по всем годам в выборке.
    """
    d = long_df[long_df['Тип'] == tip]
    pv = d.pivot_table(index=station_col, columns='Месяц', values='Значение', aggfunc='mean')
    pv = pv.reindex(columns=MONTHS)
    return pv


def _yearly_station_series(long_df, tip, agg, station_col='Станция'):
    """Годовой ряд (Станция, Год) -> агрегированное значение (mean для T, sum для P)."""
    d = long_df[long_df['Тип'] == tip]
    return d.groupby([station_col, 'Год'])['Значение'].agg(agg)


def compute_indicators(long_df, temp_tip, precip_tip, lat_map=None):
    """
    Рассчитывает климатические показатели по станциям на основе объединённой
    длинной таблицы (см. data_io.combine_typed_datasets / merge_station_reference).

    temp_tip, precip_tip — значения столбца 'Тип', соответствующие температуре
    и осадкам (пользователь выбирает их явно в интерфейсе).
    lat_map — словарь {Станция: широта}; если None или станции нет в словаре,
    K_Конрада для неё не считается (NaN) — индекс требует широты.

    Возвращает DataFrame: одна строка на станцию + метаданные
    (Название/Регион/Lat/Lon, если есть в исходной таблице).
    """
    lat_map = lat_map or {}
    t_pivot = _pivot_station_months(long_df, temp_tip)
    p_pivot = _pivot_station_months(long_df, precip_tip)

    t_yearly = _yearly_station_series(long_df, temp_tip, 'mean')
    p_yearly = _yearly_station_series(long_df, precip_tip, 'sum')

    stations = sorted(set(t_pivot.index) & set(p_pivot.index))
    records = []
    for stn in stations:
        t_mon = t_pivot.loc[stn].values.astype(float)
        p_mon = p_pivot.loc[stn].values.astype(float)
        if np.all(np.isnan(t_mon)) or np.all(np.isnan(p_mon)):
            continue

        T_god = np.nanmean(t_mon)
        P_god = np.nansum(p_mon)
        T_zim = np.nanmean(t_mon[WINTER_IDX])
        T_ves = np.nanmean(t_mon[SPRING_IDX])
        T_let = np.nanmean(t_mon[SUMMER_IDX])
        T_os = np.nanmean(t_mon[AUTUMN_IDX])
        P_zim = np.nansum(p_mon[WINTER_IDX])
        P_ves = np.nansum(p_mon[SPRING_IDX])
        P_let = np.nansum(p_mon[SUMMER_IDX])
        P_os = np.nansum(p_mon[AUTUMN_IDX])
        A_T = np.nanmax(t_mon) - np.nanmin(t_mon)

        lat = lat_map.get(stn)
        if lat is not None and not pd.isna(lat):
            K = (1.7 * A_T / np.sin(np.radians(lat))) - 20.4
        else:
            K = np.nan

        Ia = P_god / (T_god + 10) if T_god > -10 else np.nan
        Iy = P_god / (0.0018 * (T_god + 25) ** 2) if abs(T_god + 25) > 0.1 else np.nan
        summer_sum = np.nansum(t_mon[SUMMER_IDX])
        gtk = P_let / (0.1 * max(summer_sum * 30 * 24 / 1000, 1))

        # Тренды по годовым рядам данной станции
        if stn in t_yearly.index.get_level_values(0):
            ty = t_yearly.loc[stn].dropna()
            sl_t = stats.linregress(ty.index.values, ty.values).slope if len(ty) > 5 else np.nan
        else:
            sl_t = np.nan
        if stn in p_yearly.index.get_level_values(0):
            py = p_yearly.loc[stn].dropna()
            sl_p = stats.linregress(py.index.values, py.values).slope if len(py) > 5 else np.nan
        else:
            sl_p = np.nan

        records.append({
            'Станция': stn,
            'T_год': T_god, 'T_зим': T_zim, 'T_вес': T_ves, 'T_лет': T_let, 'T_ос': T_os,
            'A_T': A_T, 'P_год': P_god, 'P_зим': P_zim, 'P_вес': P_ves, 'P_лет': P_let, 'P_ос': P_os,
            'K_Конрад': K, 'I_аридн': Ia, 'I_увл': Iy, 'ГТК': gtk,
            'Тренд_T_дек': sl_t * 10 if pd.notna(sl_t) else np.nan,
            'Тренд_P_дек': sl_p * 10 if pd.notna(sl_p) else np.nan,
        })

    result = pd.DataFrame(records)

    # Присоединяем метаданные станций (Название/Регион/Lat/Lon), если есть
    meta_cols = [c for c in ['Название', 'Регион', 'Lat', 'Lon'] if c in long_df.columns]
    if meta_cols and not result.empty:
        meta = long_df[['Станция'] + meta_cols].drop_duplicates(subset='Станция')
        result = result.merge(meta, on='Станция', how='left')
        cols_order = ['Станция'] + meta_cols + [c for c in result.columns
                                                  if c not in meta_cols + ['Станция']]
        result = result[cols_order]

    return result


def indicator_feature_columns(df):
    """Список числовых столбцов-показателей (исключая идентификаторы/координаты)."""
    exclude = {'Станция', 'Название', 'Регион', 'Lat', 'Lon'}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
