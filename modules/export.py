"""
=============================================================================
ЭКСПОРТ РЕЗУЛЬТАТОВ
  • таблицы .xlsx с запятой как десятичным разделителем
    (имена table_NN_название.xlsx);
  • очищенные и восстановленные суточные ряды с признаками качества;
  • выгрузка помесячных рядов температуры и осадков в формат программы
    climate-zoning (листы «Температура», «Осадки», «Справочник станций») —
    для районирования по многолетним рядам в старой программе;
  • архив ZIP со всеми таблицами и рисунками.
=============================================================================
"""

import io
import zipfile
import numpy as np
import pandas as pd

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cf

MONTHS_RU = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август',
             'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']


def comma_frame(df, decimals=2):
    """Копия таблицы: дробные числа — строки с запятой; даты — ДД.ММ.ГГГГ."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[c] = s.dt.strftime('%d.%m.%Y')
        elif pd.api.types.is_float_dtype(s):
            if np.all((s.dropna() % 1) == 0) and s.notna().any() and s.abs().max() < 1e9:
                out[c] = s.map(lambda v: '' if pd.isna(v) else str(int(v)))
            else:
                out[c] = s.map(lambda v: '' if pd.isna(v) else cf(v, decimals))
    return out


def xlsx_bytes(sheets, decimals=2, comma=True):
    """sheets: dict {имя листа: DataFrame} → байты xlsx. Заголовки — обычным шрифтом."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        for name, df in sheets.items():
            d = comma_frame(df, decimals) if comma else df
            d.to_excel(w, sheet_name=str(name)[:31], index=False)
            ws = w.sheets[str(name)[:31]]
            from openpyxl.styles import Font, Alignment
            for cell in ws[1]:
                cell.font = Font(bold=False)
                cell.alignment = Alignment(wrap_text=True, vertical='top')
            for col in ws.columns:
                L = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(9, L + 2), 45)
    return buf.getvalue()


def monthly_for_zoning(data, stations, min_days=25):
    """
    Помесячные значения для climate-zoning: средняя месячная температура
    (по Tсредней TX) и месячная сумма осадков (по суточным суммам RR).
    Месяц учитывается, если в нём не менее min_days суток с данными.
    """
    out = {}
    if 'TX' in data:
        t = data['TX'][['st', 'date', 'tmean']].dropna()
        t['Год'], t['m'] = t['date'].dt.year, t['date'].dt.month
        g = t.groupby(['st', 'Год', 'm'])['tmean'].agg(['mean', 'count'])
        g = g[g['count'] >= min_days]['mean'].unstack('m')
        g.columns = [MONTHS_RU[m - 1] for m in g.columns]
        out['Температура'] = g.round(1).reset_index().rename(columns={'st': 'Станция'})
    if 'RR' in data:
        r = data['RR'][['st', 'date', 'rr_s']].dropna()
        r['Год'], r['m'] = r['date'].dt.year, r['date'].dt.month
        g = r.groupby(['st', 'Год', 'm'])['rr_s'].agg(['sum', 'count'])
        g = g[g['count'] >= min_days]['sum'].unstack('m')
        g.columns = [MONTHS_RU[m - 1] for m in g.columns]
        out['Осадки'] = g.round(1).reset_index().rename(columns={'st': 'Станция'})
    ref = stations.rename(columns={'st': 'Станция', 'name': 'Название', 'region': 'Регион',
                                   'lat': 'Широта', 'lon': 'Долгота', 'alt': 'Высота'})
    out['Справочник станций'] = ref[['Станция', 'Название', 'Регион', 'Широта', 'Долгота', 'Высота']]
    return out


def zoning_xlsx(data, stations):
    """Файл для загрузки во вкладке «Данные» climate-zoning (числа — числами, не текстом)."""
    return xlsx_bytes(monthly_for_zoning(data, stations), comma=False)


def daily_export(data, flags=None, fill_log=None):
    """Суточные ряды по переменным с признаками: R — брак удалён, C — исправлено, F — восстановлено."""
    sheets = {}
    for var, df in data.items():
        d = df.copy()
        comps = [c for c in d.columns if c not in ('st', 'date', 'region')]
        if flags and var in flags:
            f = flags[var].pivot_table(index=['st', 'date'], columns='comp', values='flag', aggfunc='first')
            f.columns = [f'{c}_признак' for c in f.columns]
            d = d.merge(f.reset_index(), on=['st', 'date'], how='left')
        if fill_log is not None and not fill_log.empty:
            fl = fill_log[fill_log['var'] == var]
            if len(fl):
                ff = fl.pivot_table(index=['st', 'date'], columns='comp', values='method', aggfunc='first')
                ff.columns = [f'{c}_восстановлено' for c in ff.columns]
                d = d.merge(ff.reset_index(), on=['st', 'date'], how='left')
        sheets[var] = d.rename(columns={'st': 'Станция', 'date': 'Дата', 'region': 'Регион'})
    return sheets


def zip_bundle(files):
    """files: dict {имя файла: байты} → байты ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, b in files.items():
            z.writestr(name, b)
    return buf.getvalue()


def fig_bytes(fig, fmt='png'):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=300, bbox_inches='tight')
    return buf.getvalue()
