"""
=============================================================================
МОДУЛЬ ПРОВЕРКИ И ВЕРИФИКАЦИИ ИСХОДНЫХ ДАННЫХ
Проверка выполняется до очистки и не меняет данные. Названия видов
проверок — по ВМО-№ 100 (2018), раздел 3.3: проверка формата, полноты,
согласованности и допустимости. Включает:
  • полноту рядов по станциям, годам и переменным;
  • пропуски по всей сети одновременно (сбои передачи);
  • согласованность форматов и раскладки (ночь + день = сутки в RR и FM,
    среднее из 8 сроков TT против Tсредней TX, масштаб TS, Td ≤ T);
  • согласованность справочника станций с файлами данных.
Результат — таблица проверок со статусом «норма / внимание / ошибка».
=============================================================================
"""

import numpy as np
import pandas as pd
import params as P

from .spatial import to_matrix
from config import cf

MAIN_COMP = {'TT': None, 'TX': 'tmean', 'RR': 'rr_s', 'FM': 'fm_s', 'TS': 'ts00',
             'RS': None, 'RH': None, 'FR': 'fr'}
MONTHS_FR = (4, 5, 6, 7, 8, 9, 10)


def _row_complete(df, var):
    comps = [c for c in df.columns if c not in ('st', 'date', 'region') and not c.endswith('_tr')
             and c != 'fr_notcalc']
    if var == 'DY':
        key = next((c for c in ('n_terms', 'rh_mean', 'p_mean') if c in df.columns), None)
        if key:
            return df[key].notna()
    if MAIN_COMP.get(var):
        return df[MAIN_COMP[var]].notna()
    return df[comps].notna().mean(axis=1) >= 0.75


def completeness(data, freq='Y'):
    """
    Полнота (% суток с данными) по станциям × годам для каждой переменной.
    Для FR знаменатель — дни апреля–октября. Возвращает длинную таблицу
    [Переменная, Регион, Станция, Год, Полнота, %].
    """
    rows = []
    for var, df in data.items():
        if df.empty:
            continue
        ok = _row_complete(df, var)
        d = df[['st', 'date', 'region']].copy()
        d['ok'] = ok.values
        d['year'] = d['date'].dt.year
        if var == 'FR':
            fr_m = d['region'].map(lambda r: tuple(P.season_months(r, 'fire')))
            d = d[[m in ms for m, ms in zip(d['date'].dt.month, fr_m)]]
        # знаменатель — все календарные дни периода наблюдений файла
        g = d.groupby(['region', 'st', 'year'])
        for (rg, st, yr), sub in g:
            if var == 'FR':
                ndays = sum(pd.Period(f'{yr}-{m:02d}').days_in_month for m in P.season_months(rg, 'fire'))
            else:
                last = min(pd.Timestamp(yr, 12, 31), df['date'].max())
                ndays = (last - pd.Timestamp(yr, 1, 1)).days + 1
            rows.append(dict(Переменная=var, Регион=rg, Станция=st, Год=yr,
                             **{'Полнота, %': 100 * sub['ok'].sum() / ndays}))
    return pd.DataFrame(rows)


def network_gaps(data, share=0.5, min_len=1):
    """
    Периоды, когда данные отсутствуют более чем на share станций региона
    одновременно (сбой передачи по всей сети). Возвращает таблицу
    [Переменная, Регион, Начало, Конец, Дней, Доля станций без данных].
    """
    rows = []
    for var, df in data.items():
        if df.empty:
            continue
        d = df[['st', 'date', 'region']].copy()
        d['miss'] = (~_row_complete(df, var)).values
        for rg, sub in d.groupby('region'):
            full_idx = pd.date_range(sub['date'].min(), sub['date'].max())
            M = sub.pivot_table(index='date', columns='st', values='miss', aggfunc='first')
            M = M.reindex(full_idx).fillna(True).astype(float)
            if var == 'FR':
                M = M[M.index.month.isin(P.season_months(rg, 'fire'))]
            frac = M.mean(axis=1)
            bad = frac > share
            if not bad.any():
                continue
            grp = (bad != bad.shift()).cumsum()
            for _, seg in frac[bad].groupby(grp[bad]):
                idx = seg.index
                # разрыв сегмента при переходе между сезонами FR
                if len(idx) >= min_len:
                    rows.append(dict(Переменная=var, Регион=rg, Начало=idx[0].date(),
                                     Конец=idx[-1].date(), Дней=len(idx),
                                     **{'Доля станций без данных, %': 100 * seg.mean()}))
    return pd.DataFrame(rows)


def _check(name, value, status, comment):
    return dict(Проверка=name, Значение=value, Статус=status, Комментарий=comment)


def consistency_checks(data, stations=None):
    """Набор проверок согласованности. Возвращает DataFrame проверок."""
    out = []
    # RR: ночь + день = сутки
    if 'RR' in data:
        w = data['RR']
        m = w[['rr_n', 'rr_d', 'rr_s']].notna().all(1) & (w['rr_s'] > 0)
        share = ((w.loc[m, 'rr_n'] + w.loc[m, 'rr_d'] - w.loc[m, 'rr_s']).abs() <= 1.05).mean() if m.any() else np.nan
        st = 'норма' if share >= 0.95 else ('внимание' if share >= 0.8 else 'ошибка')
        out.append(_check('Формат файла осадков (RR): ночь + день = сутки (± 1 мм), % суток', share * 100, st,
                          'Доля суток с осадками, где сумма полусуток совпадает с суточной. '
                          'Низкое значение указывает на иную раскладку значений в строке.'))
        n_tr = int(w[['rr_n_tr', 'rr_d_tr', 'rr_s_tr']].sum().sum())
        out.append(_check('Формат файла осадков (RR): значений «следы осадков»', n_tr, 'норма',
                          'Код «0»; учитываются как 0 мм с признаком следов.'))
    # FM: сутки = max(ночь, день)
    if 'FM' in data:
        w = data['FM']
        m = w[['fm_n', 'fm_d', 'fm_s']].notna().all(1)
        share = (w.loc[m, 'fm_s'] == w.loc[m, ['fm_n', 'fm_d']].max(1)).mean() if m.any() else np.nan
        out.append(_check('Формат файла ветра (FM): сутки = наибольшее из ночи и дня, % суток', share * 100,
                          'норма' if share >= 0.98 else 'внимание',
                          'Проверка раскладки файла ветра.'))
    # TT против TX
    if 'TT' in data and 'TX' in data:
        tt = data['TT']
        tcols = [c for c in tt.columns if c.startswith('tt')]
        t8 = tt[['st', 'date']].copy()
        t8['m8'] = tt[tcols].mean(1)
        t8['min8'] = tt[tcols].min(1)
        mm = t8.merge(data['TX'][['st', 'date', 'tmean', 'tmin', 'tmax']], on=['st', 'date'])
        mm = mm.dropna()
        if len(mm) > 100:
            r = np.corrcoef(mm['m8'], mm['tmean'])[0, 1]
            out.append(_check('Согласованность температуры по срокам (TT) и суточной температуры (TX): коэффициент корреляции среднего из 8 сроков и средней суточной', r,
                              'норма' if r > 0.97 else 'внимание',
                              'Подтверждает чтение TT и TX по датам (сроки внутри даты).'))
            share = ((mm['tmin'] - mm['min8']) > 3).mean()
            out.append(_check('Согласованность TT и TX: минимальная температура выше наименьшей из 8 сроков более чем на 3 °C, % суток', share * 100,
                              'внимание' if share > 0.05 else 'норма',
                              'Tмин и Tмакс в TX определены по полусуткам; для экстремумов '
                              'используется только TX.'))
        tx = data['TX']
        nn = int(tx[['tmin', 'tmean', 'tmax']].notna().all(1).sum())
        bad = int(((tx['tmin'] > tx['tmean'] + 1) | (tx['tmean'] > tx['tmax'] + 1)).sum())
        out.append(_check('Внутренняя согласованность TX: средняя суточная температура ниже минимальной или выше '
                          'максимальной более чем на 1 °C, суток', bad,
                          'внимание' if bad else 'норма',
                          f'{cf(100 * bad / max(nn, 1), 2)} % из {nn} суток. В пределах одних суток средняя '
                          'температура должна лежать между минимальной и максимальной. Расхождения возникают, '
                          'когда величины относятся к разным интервалам (минимум и максимум — за полусутки по '
                          'ВСВ, средняя — по восьми срокам) или при ошибке записи. На этапе контроля качества '
                          'такие сутки получают признак «требует проверки» (правило TX4) и не исключаются.'))
    # TS: масштаб
    if 'TS' in data and 'TT' in data:
        ts = data['TS'][['st', 'date', 'ts12']].merge(data['TT'][['st', 'date', 'tt12']], on=['st', 'date']).dropna()
        summer = ts[ts['date'].dt.month == 7]
        if len(summer) > 30:
            med = summer['ts12'].median()
            out.append(_check('Формат файла температуры поверхности почвы (TS): медиана в июле в 12 ВСВ, °C', med,
                              'норма' if 10 < med < 60 else 'ошибка',
                              'При масштабе ×10 медиана была бы около 1–3 °C, что физически невозможно; '
                              'значения записаны в целых градусах.'))
            r = np.corrcoef(ts['ts12'], ts['tt12'])[0, 1]
            out.append(_check('Согласованность температуры почвы (TS) и воздуха (TT): коэффициент корреляции в тот же срок', r,
                              'норма' if r > 0.9 else 'внимание', ''))
    # RS: Td > T
    if 'RS' in data and 'TT' in data:
        rs, tt = data['RS'], data['TT']
        m = rs.merge(tt, on=['st', 'date'], suffixes=('', '_t'))
        n_over, n_all = 0, 0
        for h in ['00', '03', '06', '09', '12', '15', '18', '21']:
            if f'td{h}' in m and f'tt{h}' in m:
                ok = m[[f'td{h}', f'tt{h}']].notna().all(1)
                n_over += int((m.loc[ok, f'td{h}'] > m.loc[ok, f'tt{h}'] + 0.5).sum())
                n_all += int(ok.sum())
        out.append(_check('Внутренняя согласованность RS и TT: температура точки росы выше температуры воздуха '
                          'в тот же срок более чем на 0,5 °C, сроков', n_over,
                          'внимание' if n_over else 'норма',
                          f'{cf(100 * n_over / max(n_all, 1), 2)} % из {n_all} сроков. Точка росы не может '
                          'превышать температуру воздуха (насыщение наступает при их равенстве); допуск 0,5 °C '
                          'учитывает погрешность измерений. На этапе контроля качества такие значения точки росы '
                          'получают признак «ошибочное» (правило RS1) и в расчётах не используются.'))
    if 'RH' in data:
        rh = data['RH']
        rcols = [c for c in rh.columns if c.startswith('rh')]
        n = int((rh[rcols] > 100).sum().sum())
        out.append(_check('Допустимость относительной влажности (RH): значения больше 100 %, сроков', n, 'внимание' if n else 'норма',
                          'Относительная влажность рассчитывается по T и Td (формула Магнуса); '
                          'файлы RH используются только для сравнения.'))
    if 'FR' in data:
        fr = data['FR']
        n0 = int(fr['fr_notcalc'].sum())
        out.append(_check('Формат файла класса пожарной опасности (FR): класс 0 (не рассчитывался), суток', n0, 'норма',
                          'Показатель не рассчитывался из-за низкой температуры; учитывается как пропуск.'))
    # справочник
    if stations is not None:
        no_xy = stations[stations[['lat', 'lon']].isna().any(axis=1)]
        out.append(_check('Справочник станций: станций без координат', len(no_xy), 'внимание' if len(no_xy) else 'норма',
                          ', '.join(no_xy['name'].astype(str).head(15))))
        main_vars = [v for v in data if v != 'DY']
        miss_vars = stations[stations['vars'].apply(lambda x: len([v for v in str(x).split(', ') if v in main_vars]))
                             < len(main_vars)]
        if len(miss_vars):
            out.append(_check('Справочник станций: станций, представленных не во всех файлах', len(miss_vars), 'внимание',
                              '; '.join(f"{r['name']}: {r['vars']}" for _, r in miss_vars.head(8).iterrows())))
    return pd.DataFrame(out)


def gross_error_summary(data):
    """Предварительный счёт явно невозможных значений (до очистки) — для отчёта о проверке."""
    import params as P
    rows = []
    rules = {'TX': ['tmin', 'tmax', 'tmean'], 'TT': None, 'TS': ['ts00', 'ts12'],
             'RS': None, 'RR': ['rr_n', 'rr_d', 'rr_s'], 'FM': ['fm_n', 'fm_d', 'fm_s'], 'RH': None}
    keymap = {'tmin': 'tmin', 'tmax': 'tmax', 'tmean': 'tmean'}
    for var, comps in rules.items():
        if var not in data:
            continue
        df = data[var]
        pref = {'TT': 'tt', 'RS': 'td', 'RH': 'rh'}.get(var)
        cols = comps or [c for c in df.columns if c.startswith(pref)]
        for c in cols:
            lim_key = keymap.get(c, {'TT': 'tt', 'RS': 'td', 'RH': 'rh', 'TS': 'ts',
                                     'RR': 'rr', 'FM': 'fm'}.get(var))
            lo, hi = P.LIMITS[lim_key]
            x = df[c]
            n = int(((x < lo) | (x > hi)).sum())
            if n:
                rows.append(dict(Переменная=var, Компонент=c, **{'Вне допустимых пределов': n},
                                 Мин=x.min(), Макс=x.max()))
    return pd.DataFrame(rows)
