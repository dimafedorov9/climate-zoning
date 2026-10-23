"""
=============================================================================
МОДУЛЬ КЛИМАТИЧЕСКИХ ПОКАЗАТЕЛЕЙ ДЛЯ ПЛАНОВ АДАПТАЦИИ

Показатели рассчитываются по очищенным (и, при выборе, восстановленным)
суточным данным для каждой станции и года, далее сводятся по периодам.

  Температура: индексы ETCCDI (TXx, TNn, TXn, TNx, SU25, SU30, TR20, FD, ID,
    DTR); периоды «аномально жаркой погоды» (АЖ) и «сильной жары» (СЖ) по
    критериям УГМС; сильный мороз и аномально холодная погода; заморозки;
    даты устойчивого перехода через 0, 5, 10 °C (метод накопленных
    отклонений), продолжительность вегетационного периода, суммы
    активных температур > 10 °C; градусо-сутки отопительного периода.
  Осадки: индексы ETCCDI (PRCPTOT, R10mm, R20mm, RX1day, RX5day, CDD, CWD,
    SDII, R95p); осадки ≥ 35 и ≥ 50 мм за 12 ч; продолжительный сильный
    дождь ≥ 100 мм за 12–48 ч; очень сильный снег ≥ 20 мм за 12 ч;
    бездождные периоды по градациям раздельно для тёплого и холодного
    сезонов; периоды без снегопада; атмосферная засуха; ГТК Селянинова.
  Ветер: число дней с ветром > 10, ≥ 15, ≥ 25 м/с; годовой максимум.
  Влажность и жара: индекс жары NWS, суховеи.
  Пожарная опасность: дни с классами ≥ 4 и 5, длительность периодов,
    сроки пожароопасного сезона.
  Экстремумы: значения 0,95 обеспеченности, уровни повторяемости раз в
    N лет (распределение Гумбеля), распределение Пуассона для числа
    периодов АЖ.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as P

# Подписи показателей: код → (название, единицы, группа)
INDEX_INFO = {
    'TXx': ('Максимальная Tмакс за год', '°C', 'Температура'),
    'TNn': ('Минимальная Tмин за год', '°C', 'Температура'),
    'TXn': ('Минимальная Tмакс за год', '°C', 'Температура'),
    'TNx': ('Максимальная Tмин за год', '°C', 'Температура'),
    'Tср_год': ('Средняя годовая температура', '°C', 'Температура'),
    'DTR': ('Средняя суточная амплитуда', '°C', 'Температура'),
    'SU25': ('Дней с Tмакс > 25 °C', 'дни', 'Температура'),
    'SU30': ('Дней с Tмакс ≥ 30 °C', 'дни', 'Температура'),
    'TR20': ('Ночей с Tмин > 20 °C', 'дни', 'Температура'),
    'FD': ('Дней с Tмин < 0 °C', 'дни', 'Температура'),
    'ID': ('Дней с Tмакс < 0 °C', 'дни', 'Температура'),
    'АЖ_случаев': ('Случаев аномально жаркой погоды', 'случаи', 'Жара'),
    'АЖ_дней': ('Дней в периодах АЖ', 'дни', 'Жара'),
    'СЖ_случаев': ('Случаев сильной жары', 'случаи', 'Жара'),
    'НЯ_жара_дней': ('Дней с высокой температурой (НЯ)', 'дни', 'Жара'),
    'Мороз_случаев': ('Случаев сильного мороза', 'случаи', 'Холод'),
    'АХ_случаев': ('Случаев аномально холодной погоды', 'случаи', 'Холод'),
    'Заморозок_весна': ('Последний весенний заморозок (день года)', 'день года', 'Заморозки'),
    'Заморозок_осень': ('Первый осенний заморозок (день года)', 'день года', 'Заморозки'),
    'Безморозный': ('Безморозный период', 'дни', 'Заморозки'),
    'Заморозок_почва_весна': ('Последний весенний заморозок на почве (день года)', 'день года', 'Заморозки'),
    'Заморозок_почва_осень': ('Первый осенний заморозок на почве (день года)', 'день года', 'Заморозки'),
    'Переход0_весна': ('Переход через 0 °C весной (день года)', 'день года', 'Агроклимат'),
    'Переход5_весна': ('Переход через 5 °C весной (день года)', 'день года', 'Агроклимат'),
    'Переход10_весна': ('Переход через 10 °C весной (день года)', 'день года', 'Агроклимат'),
    'Переход10_осень': ('Переход через 10 °C осенью (день года)', 'день года', 'Агроклимат'),
    'Переход5_осень': ('Переход через 5 °C осенью (день года)', 'день года', 'Агроклимат'),
    'Переход0_осень': ('Переход через 0 °C осенью (день года)', 'день года', 'Агроклимат'),
    'Вегетация5': ('Продолжительность периода с T > 5 °C', 'дни', 'Агроклимат'),
    'Сумма10': ('Сумма активных температур > 10 °C', '°C', 'Агроклимат'),
    'ГТК': ('ГТК Селянинова', '', 'Увлажнение'),
    'ГСОП': ('Градусо-сутки отопительного периода', '°C·сут', 'ЖКХ'),
    'Отоп_период': ('Продолжительность отопительного периода', 'дни', 'ЖКХ'),
    'PRCPTOT': ('Годовая сумма осадков (дни ≥ 1 мм)', 'мм', 'Осадки'),
    'R10mm': ('Дней с осадками ≥ 10 мм', 'дни', 'Осадки'),
    'R20mm': ('Дней с осадками ≥ 20 мм', 'дни', 'Осадки'),
    'RX1day': ('Максимальная суточная сумма', 'мм', 'Осадки'),
    'RX5day': ('Максимальная 5-суточная сумма', 'мм', 'Осадки'),
    'SDII': ('Средняя интенсивность осадков', 'мм/сут', 'Осадки'),
    'R95p': ('Сумма осадков в дни > 95-го процентиля', 'мм', 'Осадки'),
    'CDD': ('Наибольшая продолжительность сухого периода', 'дни', 'Осадки'),
    'CWD': ('Наибольшая продолжительность влажного периода', 'дни', 'Осадки'),
    'R35_12ч': ('Случаев ≥ 35 мм за 12 ч (НЯ)', 'случаи', 'Осадки'),
    'R50_12ч': ('Случаев ≥ 50 мм за 12 ч (ОЯ)', 'случаи', 'Осадки'),
    'R100_48ч': ('Случаев ≥ 100 мм за 12–48 ч (ОЯ)', 'случаи', 'Осадки'),
    'Снег20_12ч': ('Случаев очень сильного снега ≥ 20 мм за 12 ч', 'случаи', 'Осадки'),
    'Атм_засуха': ('Случаев атмосферной засухи', 'случаи', 'Засуха'),
    'Атм_засуха_дней': ('Дней в периодах атмосферной засухи', 'дни', 'Засуха'),
    'Суховей_случаев': ('Случаев суховея', 'случаи', 'Засуха'),
    'Ветер10': ('Дней с ветром > 10 м/с', 'дни', 'Ветер'),
    'Ветер15': ('Дней с ветром ≥ 15 м/с', 'дни', 'Ветер'),
    'Ветер25': ('Дней с ветром ≥ 25 м/с (ОЯ)', 'дни', 'Ветер'),
    'FMx': ('Годовой максимум ветра', 'м/с', 'Ветер'),
    'ИЖ27': ('Дней с индексом жары ≥ 27 °C', 'дни', 'Жара'),
    'ИЖ32': ('Дней с индексом жары ≥ 32 °C', 'дни', 'Жара'),
    'ИЖ41': ('Дней с индексом жары ≥ 41 °C', 'дни', 'Жара'),
    'ИЖ_макс': ('Максимум индекса жары', '°C', 'Жара'),
    'Снег_дней': ('Дней со снежным покровом', 'дни', 'Снег'),
    'Снег_макс': ('Наибольшая высота снежного покрова', 'см', 'Снег'),
    'Метель_дней': ('Дней с метелью', 'дни', 'Явления'),
    'Гроза_дней': ('Дней с грозой', 'дни', 'Явления'),
    'Туман_дней': ('Дней с туманом', 'дни', 'Явления'),
    'Град_дней': ('Дней с градом', 'дни', 'Явления'),
    'Гололёд_дней': ('Дней с гололёдом', 'дни', 'Явления'),
    'Порыв15_дней': ('Дней с порывом ветра ≥ 15 м/с', 'дни', 'Ветер'),
    'Порыв_макс': ('Максимальный порыв ветра за год', 'м/с', 'Ветер'),
    'Пож4': ('Дней с классом пожарной опасности ≥ 4', 'дни', 'Пожарная опасность'),
    'Пож5': ('Дней с 5-м классом пожарной опасности', 'дни', 'Пожарная опасность'),
    'Пож4_макс_длит': ('Наибольшая длительность периода с классом ≥ 4', 'дни', 'Пожарная опасность'),
    'Пож_начало': ('Начало пожароопасного сезона (класс ≥ 2), день года', 'день года', 'Пожарная опасность'),
    'Пож_конец': ('Окончание пожароопасного сезона (класс ≥ 2), день года', 'день года', 'Пожарная опасность'),
}


# Определения показателей (сноски в программе)
INDEX_HELP = {
    'TXx': 'Наибольшая за год суточная Tмакс (индекс ETCCDI).',
    'TNn': 'Наименьшая за год суточная Tмин (ETCCDI).',
    'TXn': 'Наименьшая за год суточная Tмакс (ETCCDI).',
    'TNx': 'Наибольшая за год суточная Tмин (ETCCDI).',
    'DTR': 'Средняя за год разность Tмакс − Tмин (ETCCDI).',
    'SU25': 'Число дней с Tмакс > 25 °C («летние дни», ETCCDI).',
    'SU30': 'Число дней с Tмакс ≥ 30 °C.',
    'TR20': 'Число ночей с Tмин > 20 °C («тропические ночи», ETCCDI).',
    'FD': 'Число дней с Tмин < 0 °C (дни с морозом, ETCCDI).',
    'ID': 'Число дней с Tмакс < 0 °C (дни без оттепели, ETCCDI).',
    'АЖ_случаев': 'Периоды Tмакс не ниже порога ОЯ «аномально жаркая погода» подряд не менее заданного числа суток '
                  '(критерии УГМС региона).',
    'СЖ_случаев': 'Периоды Tмакс не ниже порога ОЯ «сильная жара» подряд не менее заданного числа суток.',
    'Сумма10': 'Сумма средних суточных температур выше 10 °C между датами устойчивого перехода через 10 °C '
               'весной и осенью.',
    'Переход10_весна': 'Дата устойчивого перехода средней суточной температуры через 10 °C (метод накопленных '
                       'отклонений: день после минимума суммы (T − 10) в первой половине года).',
    'ГТК': 'Гидротермический коэффициент Селянинова: 10 · ΣR / ΣT за период с T > 10 °C.',
    'ГСОП': 'Градусо-сутки отопительного периода: Σ(18 − Tср) по дням с Tср ≤ 8 °C (отопительный год региона, по умолчанию июль–июнь).',
    'PRCPTOT': 'Годовая сумма осадков в дни с осадками ≥ 1 мм (ETCCDI).',
    'R10mm': 'Число дней с осадками ≥ 10 мм (ETCCDI).',
    'R20mm': 'Число дней с осадками ≥ 20 мм (ETCCDI).',
    'RX1day': 'Наибольшая суточная сумма осадков за год (ETCCDI).',
    'RX5day': 'Наибольшая сумма осадков за 5 суток подряд (ETCCDI).',
    'SDII': 'Средняя сумма осадков в день с осадками ≥ 1 мм (ETCCDI).',
    'R95p': 'Сумма осадков в дни с осадками выше 95-го процентиля станции (ETCCDI).',
    'CDD': 'Наибольшее число сухих дней подряд (осадки ниже порога сухого дня, ETCCDI).',
    'CWD': 'Наибольшее число дней подряд с осадками ≥ 1 мм (ETCCDI).',
    'R35_12ч': 'Число случаев с осадками не менее критерия НЯ «сильный дождь» за 12 ч (ночь или день).',
    'R50_12ч': 'Число случаев с осадками не менее критерия ОЯ «очень сильный дождь» за 12 ч.',
    'R100_48ч': 'Число случаев ОЯ «продолжительный сильный дождь»: сумма за 2–4 полусуток подряд не менее критерия.',
    'Атм_засуха': 'Не менее 30 суток подряд в месяцы оценки засухи (по умолчанию май–сентябрь, задаются по региону) без осадков > 5 мм при средней за период Tмакс > 25 °C.',
    'Суховей_случаев': 'Не менее 3 суток подряд: ветер ≥ 7 м/с, Tмакс > 25 °C, наименьшая влажность ≤ 30 %.',
    'ИЖ27': 'Число дней летних месяцев региона (по умолчанию июнь–август), когда индекс жары NWS (регрессия Ротфуса, по T и влажности из Td) хотя бы в '
            'один срок достиг 27 °C.',
    'Пож4': 'Число дней пожароопасного сезона региона (по умолчанию апрель–октябрь) с классом пожарной опасности по условиям погоды ≥ 4.',
    'Снег_дней': 'Число дней с высотой снежного покрова > 0 см.',
    'Ветер15': 'Число дней с максимальной скоростью ветра ≥ 15 м/с.',
}
REFERENCES = [
    'ВМО-№ 100. Руководство по климатологической практике. Издание 2018 г. — разделы 3.3 (контроль качества) и '
    '5.9 (оценивание данных).',
    'Zhang X. et al. Indices for monitoring changes in extremes based on daily temperature and precipitation data // '
    'WIREs Climate Change. 2011. Vol. 2. P. 851–870 (индексы ETCCDI).',
    'Hamed K.H., Rao A.R. A modified Mann-Kendall trend test for autocorrelated data // J. Hydrology. 1998. '
    'Vol. 204. P. 182–196.',
    'Sen P.K. Estimates of the regression coefficient based on Kendall\'s tau // JASA. 1968. Vol. 63. P. 1379–1389.',
    'Pettitt A.N. A non-parametric approach to the change-point problem // Applied Statistics. 1979. Vol. 28. '
    'P. 126–135.',
    'Liu F.T., Ting K.M., Zhou Z.-H. Isolation Forest // Proc. 8th IEEE ICDM. 2008. P. 413–422.',
    'Rothfusz L.P. The heat index equation. NWS Technical Attachment SR 90-23. 1990.',
]


def label(code):
    info = INDEX_INFO.get(code)
    return f'{info[0]}, {info[1]}' if info and info[1] else (info[0] if info else code)


# ─── Подготовка суточной таблицы ─────────────────────────────────────────
def build_daily(data, humidity=None):
    """Сводит переменные в одну суточную таблицу [st, date, region, tmin, tmax, …]."""
    parts = []
    keep = {'TX': ['tmin', 'tmax', 'tmean'], 'RR': ['rr_n', 'rr_d', 'rr_s'],
            'FM': ['fm_n', 'fm_d', 'fm_s'], 'TS': ['ts00', 'ts12'], 'FR': ['fr']}
    for var, cols in keep.items():
        if var in data:
            d = data[var]
            parts.append(d[['st', 'date'] + [c for c in cols if c in d.columns]].set_index(['st', 'date']))
    if humidity is not None and not humidity.empty:
        parts.append(humidity.set_index(['st', 'date']))
    if 'DY' in data:
        # прочие суточные характеристики из таблиц Excel (влажность, снег, явления, порыв)
        parts.append(data['DY'].drop(columns=['region'], errors='ignore').set_index(['st', 'date']))
    if not parts:
        return pd.DataFrame()
    D = pd.concat(parts, axis=1)
    if D.columns.duplicated().any():
        # одинаковые показатели из разных источников (например, RH по срокам и из таблиц):
        # берётся первое непустое значение
        merged = {}
        for c in dict.fromkeys(D.columns):
            sub = D.loc[:, D.columns == c]
            merged[c] = sub.bfill(axis=1).iloc[:, 0] if sub.shape[1] > 1 else sub.iloc[:, 0]
        D = pd.DataFrame(merged, index=D.index)
    D = D.reset_index()
    region = pd.concat([d.groupby('st')['region'].first() for d in data.values()]).groupby(level=0).first()
    D['region'] = D['st'].map(region)
    # непрерывный ряд дат для каждой станции
    out = []
    for st, g in D.groupby('st'):
        full = pd.date_range(g['date'].min(), g['date'].max())
        g = g.set_index('date').reindex(full)
        g['st'] = st
        g['region'] = region.get(st)
        out.append(g.rename_axis('date').reset_index())
    D = pd.concat(out, ignore_index=True)
    D['year'] = D['date'].dt.year
    D['month'] = D['date'].dt.month
    D['doy'] = D['date'].dt.dayofyear
    return D


def runs(mask):
    """Серии подряд идущих True. Возвращает массив [начало, конец] (индексы, включительно)."""
    m = np.asarray(mask, bool)
    if not m.any():
        return np.empty((0, 2), int)
    d = np.diff(np.r_[0, m.astype(int), 0])
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return np.c_[starts, ends]


def _crit(region):
    return P.criteria_for(region)


# ─── События (периоды) ───────────────────────────────────────────────────
def heat_events(D):
    """
    Периоды АЖ (Tмакс ≥ порога не менее N суток подряд) и СЖ. Возвращает
    таблицу событий [st, region, Тип, Начало, Конец, Дней, Tмакс_макс].
    Пропуск данных прерывает период.
    """
    rows = []
    for st, g in D.groupby('st'):
        c = _crit(g['region'].iloc[0])
        t = g['tmax'].values
        dates = g['date'].values
        for typ, thr, nmin in [('АЖ', c['АЖ_порог'], c['АЖ_дней']), ('СЖ', c['СЖ_порог'], c['СЖ_дней'])]:
            for a, b in runs(t >= thr):
                if b - a + 1 >= nmin:
                    rows.append(dict(st=st, region=g['region'].iloc[0], Тип=typ, Начало=pd.Timestamp(dates[a]),
                                     Конец=pd.Timestamp(dates[b]), Дней=b - a + 1,
                                     **{'Tмакс_макс': np.nanmax(t[a:b + 1])}))
    return pd.DataFrame(rows)


def cold_events(D):
    rows = []
    for st, g in D.groupby('st'):
        c = _crit(g['region'].iloc[0])
        t = g['tmin'].values
        dates = g['date'].values
        for typ, thr, nmin in [('Мороз', c['Мороз_порог'], c['Мороз_дней']), ('АХ', c['АХ_порог'], c['АХ_дней'])]:
            for a, b in runs(t <= thr):
                if b - a + 1 >= nmin:
                    rows.append(dict(st=st, region=g['region'].iloc[0], Тип=typ, Начало=pd.Timestamp(dates[a]),
                                     Конец=pd.Timestamp(dates[b]), Дней=b - a + 1,
                                     **{'Tмин_мин': np.nanmin(t[a:b + 1])}))
    return pd.DataFrame(rows)


def drought_events(D):
    """Атмосферная засуха: ≥ 30 суток подряд без осадков > 5 мм в месяцы оценки засухи региона при средней Tмакс > 25 °C."""
    a_ = P.ATMOS_DROUGHT
    rows = []
    for st, g in D.groupby('st'):
        warm = g['month'].isin(P.season_months(g['region'].iloc[0], 'drought')).values
        rr = g['rr_s'].values
        ok = warm & (rr <= a_['effective_rr'])
        for a, b in runs(ok):
            n = b - a + 1
            if n >= a_['days']:
                tm = np.nanmean(g['tmax'].values[a:b + 1])
                if tm > a_['tmax']:
                    rows.append(dict(st=st, region=g['region'].iloc[0], Тип='Атм. засуха',
                                     Начало=g['date'].iloc[a], Конец=g['date'].iloc[b], Дней=n,
                                     **{'Tмакс_средняя': tm}))
    return pd.DataFrame(rows)


def sukhovey_events(D):
    """Суховей: ветер ≥ 7 м/с, Tмакс > 25 °C, минимальная влажность ≤ 30 % не менее 3 суток подряд."""
    s = P.SUKHOVEY
    if 'rh_min' not in D.columns or 'fm_d' not in D.columns:
        return pd.DataFrame()
    rows = []
    for st, g in D.groupby('st'):
        ok = (g['fm_d'] >= s['wind']) & (g['tmax'] > s['tmax']) & (g['rh_min'] <= s['rh'])
        for a, b in runs(ok.values):
            if b - a + 1 >= s['days']:
                rows.append(dict(st=st, region=g['region'].iloc[0], Тип='Суховей',
                                 Начало=g['date'].iloc[a], Конец=g['date'].iloc[b], Дней=b - a + 1))
    return pd.DataFrame(rows)


def heavy_precip_events(D):
    """Сильные осадки по полусуточным суммам: НЯ/ОЯ дождя, очень сильный снег, продолжительный дождь."""
    rows = []
    for st, g in D.groupby('st'):
        c = _crit(g['region'].iloc[0])
        halves = np.c_[g['rr_n'].values, g['rr_d'].values].ravel()      # хронологический ряд полусуток
        tmax2 = np.repeat(g['tmax'].values, 2)
        dates2 = np.repeat(g['date'].values, 2)
        part = np.tile(['ночь', 'день'], len(g))
        snow = tmax2 <= P.SNOW_TMAX
        for i in np.where(halves >= c['Дождь_НЯ_12ч'])[0]:
            typ = 'Очень сильный дождь (ОЯ)' if halves[i] >= c['Дождь_ОЯ_12ч'] else 'Сильный дождь (НЯ)'
            if snow[i]:
                continue
            rows.append(dict(st=st, region=g['region'].iloc[0], Тип=typ, Начало=pd.Timestamp(dates2[i]),
                             Полусутки=part[i], **{'Сумма, мм': halves[i]}))
        for i in np.where((halves >= c['Снег_ОЯ_12ч']) & snow)[0]:
            rows.append(dict(st=st, region=g['region'].iloc[0], Тип='Очень сильный снег (ОЯ)',
                             Начало=pd.Timestamp(dates2[i]), Полусутки=part[i], **{'Сумма, мм': halves[i]}))
        # продолжительный: сумма за 2–4 полусуток подряд ≥ 100 мм (без перекрытия)
        h = np.nan_to_num(halves)
        cs = pd.Series(h).rolling(4, min_periods=2).sum().values
        i = 0
        while i < len(cs):
            if cs[i] >= c['Дождь_прод_ОЯ'] and not snow[i]:
                rows.append(dict(st=st, region=g['region'].iloc[0], Тип='Продолжительный сильный дождь (ОЯ)',
                                 Начало=pd.Timestamp(dates2[max(0, i - 3)]), Полусутки='12–48 ч',
                                 **{'Сумма, мм': cs[i]}))
                i += 4
            else:
                i += 1
    return pd.DataFrame(rows)


def _season_bounds(season, region):
    return P.seasons_for(region)['warm' if season == 'тёплый' else 'cold']


def _season_of(dates, season, region=None):
    """Принадлежность суток сезону и год сезона (переход через новый год — год окончания)."""
    m = dates.dt.month
    a, b = _season_bounds(season, region)
    if a <= b:
        sel = m.between(a, b)
        key = dates.dt.year
    else:
        sel = (m >= a) | (m <= b)
        key = dates.dt.year + (m >= a).astype(int)      # зима 2016/17 → 2017
    return sel, key


def dry_spells(D, season='тёплый', dry_mm=None, kind='dry'):
    """
    Бездождные периоды (kind='dry': сутки с осадками < dry_mm) или периоды
    без снегопада (kind='nosnow': холодный сезон, нет суток с осадками ≥
    dry_mm при Tмакс ≤ 0 °C). Период относится к сезону, в котором начался.
    Сезоны с долей пропусков > SEASON_MAX_MISSING исключаются.
    Возвращает (spells, seasons_ok): события и перечень учтённых сезонов.
    """
    dry_mm = P.DRY_DAY_MM if dry_mm is None else dry_mm
    rows, ok_rows = [], []
    for st, g in D.groupby('st'):
        g = g.reset_index(drop=True)
        rg0 = g['region'].iloc[0]
        sel, key = _season_of(g['date'], season, rg0)
        rr = g['rr_s']
        if kind == 'dry':
            cond = (rr < dry_mm)
        else:
            snowday = (rr >= dry_mm) & (g['tmax'] <= P.SNOW_TMAX)
            cond = ~snowday & rr.notna()
        cond = cond & rr.notna()
        # сезонная полнота
        for sk, sub in g[sel].groupby(key[sel]):
            a_, b_ = _season_bounds(season, rg0)
            y0 = sk if a_ <= b_ else sk - 1
            exp_days = (pd.Timestamp(sk, b_, 1) + pd.offsets.MonthEnd(0) - pd.Timestamp(y0, a_, 1)).days + 1
            miss = 1 - sub['rr_s'].notna().sum() / exp_days
            ok_rows.append(dict(st=st, Сезон=sk, Пропусков=miss, Учтён=miss <= P.SEASON_MAX_MISSING))
        okset = {r['Сезон'] for r in ok_rows if r['st'] == st and r['Учтён']}
        for a, b in runs(cond.values):
            if not sel.iloc[a]:
                continue
            sk = key.iloc[a]
            if sk not in okset:
                continue
            rows.append(dict(st=st, region=g['region'].iloc[0], Сезон=sk, Начало=g['date'].iloc[a],
                             Конец=g['date'].iloc[b], Дней=b - a + 1))
    return pd.DataFrame(rows), pd.DataFrame(ok_rows)


def spell_class_table(spells, seasons_ok, classes=None):
    """
    Распределение частот периодов по градациям: среднее число периодов за
    сезон на станции и повторяемость, %. Возвращает (по станциям, по региону).
    """
    classes = classes or P.DRY_SPELL_CLASSES
    labels = [f'{a}–{b}' if b < 1000 else f'≥ {a}' for a, b in classes]
    if spells.empty:
        return pd.DataFrame(), pd.DataFrame()
    s = spells.copy()
    s['Градация'] = pd.cut(s['Дней'], [a - 0.5 for a, _ in classes] + [10_001], labels=labels)
    s = s.dropna(subset=['Градация'])
    nseas = seasons_ok[seasons_ok['Учтён']].groupby('st').size()
    cnt = s.groupby(['st', 'Градация'], observed=False).size().unstack(fill_value=0)
    per = cnt.div(nseas.reindex(cnt.index), axis=0)
    per.columns = [f'{c}, в среднем за сезон' for c in per.columns]
    reg = s.groupby(['region', 'Градация'], observed=False).size().unstack(fill_value=0)
    share = reg.div(reg.sum(axis=1), axis=0) * 100
    share.columns = [f'{c} дн., %' for c in share.columns]
    nst = seasons_ok[seasons_ok['Учтён']].merge(s[['st', 'region']].drop_duplicates(), on='st')
    reg_n = nst.groupby('region').size()
    abs_ = reg.div(reg_n.reindex(reg.index), axis=0)
    abs_.columns = [f'{c} дн., за сезон на станцию' for c in abs_.columns]
    return per.reset_index(), pd.concat([abs_, share], axis=1).reset_index()


# ─── Агроклимат ──────────────────────────────────────────────────────────
def transition_dates(t, doy, thr):
    """
    Даты устойчивого перехода через порог методом накопленных отклонений:
    весной — день после минимума накопленной суммы (T − порог) в первой
    половине года, осенью — день после максимума во второй половине.
    """
    t = np.asarray(t, float)
    doy = np.asarray(doy)
    out = [np.nan, np.nan]
    spring = doy <= 213
    autumn = doy >= 182
    if np.isnan(t[spring]).mean() < 0.1 and spring.sum() > 150:
        s = np.nancumsum(t[spring] - thr)
        i = int(np.argmin(np.r_[0, s]))
        if 0 < i < len(s):
            out[0] = doy[spring][i]
    if np.isnan(t[autumn]).mean() < 0.1 and autumn.sum() > 150:
        s = np.nancumsum(t[autumn] - thr)
        i = int(np.argmax(s))
        if i < len(s) - 1:
            out[1] = doy[autumn][i] + 1
    return out


def _complete(x, frac=0.9):
    return np.isfinite(x).mean() >= frac


def annual_indices(D, dry_mm=None):
    """
    Годовые показатели для каждой станции. Год учитывается, если в нём
    доступно ≥ 90 % суточных значений нужной переменной.
    Возвращает длинную таблицу [st, region, year, Показатель, Значение].
    """
    dry_mm = P.DRY_DAY_MM if dry_mm is None else dry_mm
    he = heat_events(D)
    ce = cold_events(D)
    pe = heavy_precip_events(D) if {'rr_n', 'rr_d'} <= set(D.columns) else pd.DataFrame()
    de = drought_events(D) if 'rr_s' in D.columns else pd.DataFrame()
    se = sukhovey_events(D)
    rr_all = D['rr_s'] if 'rr_s' in D.columns else pd.Series(dtype=float)
    p95 = D[D['rr_s'] >= 1].groupby('st')['rr_s'].quantile(0.95) if 'rr_s' in D.columns else pd.Series(dtype=float)
    rows = []

    def add(st, rg, yr, k, v):
        rows.append((st, rg, yr, k, v))

    for (st, yr), g in D.groupby(['st', 'year']):
        rg = g['region'].iloc[0]
        c = _crit(rg)
        ndays = 366 if pd.Timestamp(yr, 12, 31).dayofyear == 366 else 365
        if len(g) < ndays * 0.9:
            continue
        tx, tn, tm = g['tmax'].values, g['tmin'].values, g['tmean'].values
        if _complete(tx) and _complete(tn):
            add(st, rg, yr, 'TXx', np.nanmax(tx)); add(st, rg, yr, 'TXn', np.nanmin(tx))
            add(st, rg, yr, 'TNn', np.nanmin(tn)); add(st, rg, yr, 'TNx', np.nanmax(tn))
            add(st, rg, yr, 'Tср_год', np.nanmean(tm))
            add(st, rg, yr, 'DTR', np.nanmean(tx - tn))
            add(st, rg, yr, 'SU25', np.sum(tx > 25)); add(st, rg, yr, 'SU30', np.sum(tx >= 30))
            add(st, rg, yr, 'TR20', np.sum(tn > 20)); add(st, rg, yr, 'FD', np.sum(tn < 0))
            add(st, rg, yr, 'ID', np.sum(tx < 0))
            lo, hi = c['НЯ_жара']
            add(st, rg, yr, 'НЯ_жара_дней', np.sum((tx >= lo) & (tx <= hi)))
            for code, typ, src in [('АЖ', 'АЖ', he), ('СЖ', 'СЖ', he), ('Мороз', 'Мороз', ce), ('АХ', 'АХ', ce)]:
                ev = src[(src['st'] == st) & (src['Тип'] == typ) & (src['Начало'].dt.year == yr)] if not src.empty else src
                add(st, rg, yr, f'{code}_случаев', len(ev))
                if code == 'АЖ':
                    add(st, rg, yr, 'АЖ_дней', ev['Дней'].sum() if len(ev) else 0)
            # заморозки (воздух): последний до 15 июля, первый после
            doy = g['doy'].values
            fr = tn < 0
            sp = np.where(fr & (doy < 196))[0]
            au = np.where(fr & (doy >= 196))[0]
            ls = doy[sp[-1]] if len(sp) else np.nan
            fa = doy[au[0]] if len(au) else np.nan
            add(st, rg, yr, 'Заморозок_весна', ls); add(st, rg, yr, 'Заморозок_осень', fa)
            add(st, rg, yr, 'Безморозный', fa - ls - 1 if np.isfinite(ls) and np.isfinite(fa) else np.nan)
        if _complete(tm):
            doy = g['doy'].values
            tr = {thr: transition_dates(tm, doy, thr) for thr in (0, 5, 10)}
            add(st, rg, yr, 'Переход0_весна', tr[0][0]); add(st, rg, yr, 'Переход0_осень', tr[0][1])
            add(st, rg, yr, 'Переход5_весна', tr[5][0]); add(st, rg, yr, 'Переход5_осень', tr[5][1])
            add(st, rg, yr, 'Переход10_весна', tr[10][0]); add(st, rg, yr, 'Переход10_осень', tr[10][1])
            if np.isfinite(tr[5][0]) and np.isfinite(tr[5][1]):
                add(st, rg, yr, 'Вегетация5', tr[5][1] - tr[5][0])
            a10, b10 = tr[10]
            if np.isfinite(a10) and np.isfinite(b10):
                per = (doy >= a10) & (doy < b10)
                s10 = np.nansum(tm[per & (tm > 10)])
                add(st, rg, yr, 'Сумма10', s10)
                if 'rr_s' in g and _complete(g['rr_s'].values[per]):
                    add(st, rg, yr, 'ГТК', 10 * np.nansum(g['rr_s'].values[per]) / s10 if s10 > 0 else np.nan)
        if 'ts00' in g and _complete(g['ts00'].values, 0.85):
            ts = g['ts00'].values
            doy = g['doy'].values
            sp = np.where((ts < 0) & (doy < 196))[0]
            au = np.where((ts < 0) & (doy >= 196))[0]
            add(st, rg, yr, 'Заморозок_почва_весна', doy[sp[-1]] if len(sp) else np.nan)
            add(st, rg, yr, 'Заморозок_почва_осень', doy[au[0]] if len(au) else np.nan)
        if 'rr_s' in g and _complete(g['rr_s'].values):
            r = g['rr_s'].values
            wet = r >= 1
            add(st, rg, yr, 'PRCPTOT', np.nansum(r[wet]))
            add(st, rg, yr, 'R10mm', np.sum(r >= 10)); add(st, rg, yr, 'R20mm', np.sum(r >= 20))
            add(st, rg, yr, 'RX1day', np.nanmax(r))
            add(st, rg, yr, 'RX5day', pd.Series(r).rolling(5, min_periods=4).sum().max())
            add(st, rg, yr, 'SDII', np.nanmean(r[wet]) if wet.any() else np.nan)
            if st in p95.index:
                add(st, rg, yr, 'R95p', np.nansum(r[r > p95[st]]))
            dry = (r < dry_mm) & np.isfinite(r)
            add(st, rg, yr, 'CDD', max([b - a + 1 for a, b in runs(dry)], default=0))
            add(st, rg, yr, 'CWD', max([b - a + 1 for a, b in runs(wet)], default=0))
            if not pe.empty:
                ev = pe[(pe['st'] == st) & (pe['Начало'].dt.year == yr)]
                add(st, rg, yr, 'R35_12ч', int((ev['Тип'].str.contains('дождь') & ~ev['Тип'].str.contains('Продолж')).sum()))
                add(st, rg, yr, 'R50_12ч', int((ev['Тип'] == 'Очень сильный дождь (ОЯ)').sum()))
                add(st, rg, yr, 'R100_48ч', int((ev['Тип'] == 'Продолжительный сильный дождь (ОЯ)').sum()))
                add(st, rg, yr, 'Снег20_12ч', int((ev['Тип'] == 'Очень сильный снег (ОЯ)').sum()))
            if not de.empty:
                ev = de[(de['st'] == st) & (de['Начало'].dt.year == yr)]
            else:
                ev = de
            add(st, rg, yr, 'Атм_засуха', len(ev))
            add(st, rg, yr, 'Атм_засуха_дней', ev['Дней'].sum() if len(ev) else 0)
        if 'fm_s' in g and _complete(g['fm_s'].values):
            f = g['fm_s'].values
            add(st, rg, yr, 'Ветер10', np.sum(f > P.WIND_DAYS_THRESHOLDS[0]))
            add(st, rg, yr, 'Ветер15', np.sum(f >= P.WIND_DAYS_THRESHOLDS[1]))
            add(st, rg, yr, 'Ветер25', np.sum(f >= P.WIND_DAYS_THRESHOLDS[2]))
            add(st, rg, yr, 'FMx', np.nanmax(f))
        if 'hi_max' in g:
            jja = g['month'].isin(P.season_months(rg, 'summer')).values
            h = g['hi_max'].values[jja]
            if np.isfinite(g['rh_min'].values[jja]).mean() >= 0.85 if 'rh_min' in g else False:
                add(st, rg, yr, 'ИЖ27', np.sum(h >= 27)); add(st, rg, yr, 'ИЖ32', np.sum(h >= 32))
                add(st, rg, yr, 'ИЖ41', np.sum(h >= 41))
                add(st, rg, yr, 'ИЖ_макс', np.nanmax(h) if np.isfinite(h).any() else np.nan)
            if not se.empty:
                add(st, rg, yr, 'Суховей_случаев', int(((se['st'] == st) & (se['Начало'].dt.year == yr)).sum()))
        # таблицы Excel: снег, явления, порывы (год учитывается при полноте ≥ 90 %)
        for col, code, how in [('snow_depth', 'Снег_дней', 'days_pos'), ('snow_depth', 'Снег_макс', 'max'),
                               ('n_blizzard', 'Метель_дней', 'days'), ('n_thunder', 'Гроза_дней', 'days'),
                               ('n_fog', 'Туман_дней', 'days'), ('hail_d', 'Град_дней', 'days'),
                               ('glaze_d', 'Гололёд_дней', 'days'), ('gust', 'Порыв15_дней', 'ge15'),
                               ('gust', 'Порыв_макс', 'max')]:
            if col not in g or 'n_terms' not in g or not _complete(g['n_terms'].values):
                continue
            x = g[col].values
            if how == 'days':
                add(st, rg, yr, code, int(np.sum(np.isfinite(x))))
            elif how == 'days_pos':
                add(st, rg, yr, code, int(np.sum(x > 0)))
            elif how == 'ge15':
                add(st, rg, yr, code, int(np.sum(x >= 15)))
            elif np.isfinite(x).any():
                add(st, rg, yr, code, np.nanmax(x))
        if 'fr' in g:
            seas = g['month'].isin(P.season_months(rg, 'fire')).values
            frv = g['fr'].values[seas]
            if np.isfinite(frv).mean() >= 0.7:
                add(st, rg, yr, 'Пож4', np.sum(frv >= 4)); add(st, rg, yr, 'Пож5', np.sum(frv >= 5))
                add(st, rg, yr, 'Пож4_макс_длит', max([b - a + 1 for a, b in runs(frv >= 4)], default=0))
                doy = g['doy'].values[seas]
                hi2 = np.where(frv >= 2)[0]
                add(st, rg, yr, 'Пож_начало', doy[hi2[0]] if len(hi2) else np.nan)
                add(st, rg, yr, 'Пож_конец', doy[hi2[-1]] if len(hi2) else np.nan)
    # ГСОП: отопительный сезон июль–июнь, обозначается годом окончания
    for st, g in D.groupby('st'):
        rg = g['region'].iloc[0]
        m0 = P.seasons_for(rg)['heat_year'][0]
        key = g['year'] + (g['month'] >= m0).astype(int) if m0 > 1 else g['year']
        for yr, sub in g.groupby(key):
            if len(sub) < 330 or not _complete(sub['tmean'].values):
                continue
            t = sub['tmean'].values
            heat = t <= P.HDD_THRESHOLD
            rows.append((st, rg, yr, 'ГСОП', np.nansum(P.HDD_BASE - t[heat])))
            rows.append((st, rg, yr, 'Отоп_период', int(heat.sum())))
    return pd.DataFrame(rows, columns=['st', 'region', 'year', 'Показатель', 'Значение'])


# ─── Сводка по периодам и экстремумы ─────────────────────────────────────
def station_means(ann, years=None):
    """Средние многолетние значения показателей по станциям (для районирования и карт)."""
    a = ann if years is None else ann[ann['year'].between(*years)]
    t = a.pivot_table(index=['st', 'region'], columns='Показатель', values='Значение', aggfunc='mean')
    n = a.groupby('st')['year'].nunique().rename('Лет')
    return t.reset_index().merge(n, on='st', how='left')


def heat_summary(events, D, years, region=None):
    """
    Сводка по периодам АЖ/СЖ в формате отчёта 2016 г.: число случаев на
    станцию в год, средняя и максимальная продолжительность, повторяемость
    продолжительности и Tмакс по градациям, абсолютный максимум, число СЖ.
    """
    a, b = years
    e = events[(events['Начало'].dt.year >= a) & (events['Начало'].dt.year <= b)]
    if region:
        e = e[e['region'] == region]
        nst = D[(D['region'] == region)]['st'].nunique()
    else:
        nst = D['st'].nunique()
    nyr = b - a + 1
    aj = e[e['Тип'] == 'АЖ']
    sj = e[e['Тип'] == 'СЖ']
    out = {'Период': f'{a}–{b}', 'Станций': nst,
           'Случаев АЖ': len(aj),
           'Случаев АЖ на станцию в год': len(aj) / max(1, nst) / nyr,
           'Средняя продолжительность АЖ, дни': aj['Дней'].mean() if len(aj) else np.nan,
           'Максимальная продолжительность АЖ, дни': aj['Дней'].max() if len(aj) else np.nan}
    dcl = pd.cut(aj['Дней'], [4.5, 9.5, 14.5, 19.5, 1e4], labels=['5–9', '10–14', '15–19', '≥ 20'])
    for k, v in (dcl.value_counts(normalize=True) * 100).reindex(['5–9', '10–14', '15–19', '≥ 20']).fillna(0).items():
        out[f'Продолжительность {k} дн., %'] = v
    tcl = pd.cut(aj['Tмакс_макс'], [29.95, 32.95, 35.95, 39.95, 99], labels=['30–32', '33–35', '36–39', '≥ 40'])
    for k, v in (tcl.value_counts(normalize=True) * 100).reindex(['30–32', '33–35', '36–39', '≥ 40']).fillna(0).items():
        out[f'Tмакс {k} °C, %'] = v
    out['Абсолютный максимум в АЖ, °C'] = aj['Tмакс_макс'].max() if len(aj) else np.nan
    out['Случаев СЖ'] = len(sj)
    # совпадение СЖ с периодами АЖ
    ov = 0
    for _, r in sj.iterrows():
        m = aj[(aj['st'] == r['st']) & (aj['Начало'] <= r['Конец']) & (aj['Конец'] >= r['Начало'])]
        ov += int(len(m) > 0)
    out['СЖ внутри периодов АЖ'] = ov
    return out


def poisson_rare(counts, return_years=50):
    """Число периодов за год, которое может наблюдаться раз в N лет (распределение Пуассона)."""
    lam = np.nanmean(counts)
    if not np.isfinite(lam) or lam <= 0:
        return 0, lam
    k = 0
    while stats.poisson.sf(k - 1, lam) > 1 / return_years:
        k += 1
    return k - 1, lam


def gumbel_levels(annual_max, periods=(10, 20, 50)):
    """Уровни повторяемости раз в N лет по распределению Гумбеля (метод моментов)."""
    x = np.asarray(annual_max, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return {f'раз в {T} лет': np.nan for T in periods}
    beta = np.std(x, ddof=1) * np.sqrt(6) / np.pi
    mu = np.mean(x) - 0.5772 * beta
    return {f'раз в {T} лет': mu - beta * np.log(-np.log(1 - 1 / T)) for T in periods}


def extremes_table(D, ann):
    """По станциям: Tмакс 0,95 обеспеченности (летние месяцы региона), повторяемость годовых максимумов."""
    rows = []
    for st, g in D.groupby('st'):
        rg = g['region'].iloc[0]
        row = dict(st=st, region=rg)
        jja = g[g['month'].isin(P.season_months(rg, 'summer'))]
        row['Tмакс 0,95 обеспеченности (летние месяцы), °C'] = jja['tmax'].quantile(0.95)
        a = ann[ann['st'] == st]
        for code, nm in [('TXx', 'Tмакс'), ('RX1day', 'Суточный максимум осадков'), ('FMx', 'Максимум ветра')]:
            x = a[a['Показатель'] == code]['Значение']
            for k, v in gumbel_levels(x).items():
                row[f'{nm}, {k}'] = v
        aj = a[a['Показатель'] == 'АЖ_случаев']['Значение']
        k, lam = poisson_rare(aj)
        row['Среднее число АЖ в год'] = lam
        row['Число АЖ за год, возможное раз в 50 лет'] = k
        rows.append(row)
    return pd.DataFrame(rows)


def regional_series(ann, code, region=None):
    """Средний по станциям ряд показателя по годам (для графиков и трендов)."""
    a = ann[ann['Показатель'] == code]
    if region:
        a = a[a['region'] == region]
    return a.groupby('year')['Значение'].mean()
