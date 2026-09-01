"""
=============================================================================
МОДУЛЬ ЗАГРУЗКИ И ПРЕОБРАЗОВАНИЯ ДАННЫХ
Поддержка: Excel (.xlsx), CSV. Автоопределение широкого формата и
преобразование в длинный. Валидация.
=============================================================================
"""

import io
import pandas as pd
import numpy as np


MONTHS_RU = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
             'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']

MONTHS_RU_SHORT = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                    'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']


TYPE_KEYWORDS = {
    'Температура': ['температур', 'temp', 't_'],
    'Осадки': ['осадк', 'precip', 'rain', 'дожд'],
}

ROLE_STATION_KEYWORDS = ['справочник', 'станци', 'координат', 'stations', 'reference']

STATION_COL_KEYWORDS = {
    'Станция': ['станц', 'код', 'station', 'id', 'index'],
    'Название': ['назв', 'name', 'имя'],
    'Регион': ['регион', 'облас', 'край', 'район', 'region'],
    'Lat': ['широт', 'lat'],
    'Lon': ['долгот', 'lon', 'lng'],
}


def guess_indicator_label(sheet_or_file_name, df=None):
    """
    Пытается угадать, что за климатический показатель в таблице (Температура/
    Осадки), по названию листа/файла, а при неоднозначности — по диапазону
    значений (температура обычно в пределах -50..+45, осадки неотрицательны
    и обычно на порядок больше по модулю в тёплые месяцы).
    Возвращает угаданную подпись или None, если не удалось определить.
    """
    name_low = str(sheet_or_file_name).lower()
    for label, keywords in TYPE_KEYWORDS.items():
        if any(kw in name_low for kw in keywords):
            return label

    if df is not None:
        month_cols = [m for m in MONTHS_RU if m in df.columns]
        if month_cols:
            vals = pd.to_numeric(df[month_cols].values.ravel(), errors='coerce')
            vals = vals[~pd.isna(vals)]
            if len(vals) > 0:
                if (vals < 0).mean() > 0.05:
                    return 'Температура'
                if vals.min() >= 0 and vals.max() > 45:
                    return 'Осадки'
    return None


def guess_role(sheet_or_file_name, df):
    """
    Пытается угадать роль файла/листа: 'climate' (климатический показатель,
    широкий формат с месяцами), 'station_ref' (справочник станций) или None,
    если распознать не удалось (тогда пользователь выбирает вручную).
    """
    name_low = str(sheet_or_file_name).lower()
    if any(kw in name_low for kw in ROLE_STATION_KEYWORDS):
        return 'station_ref'

    fmt = detect_format(df)
    if fmt == 'wide_months':
        return 'climate'

    cols_low = [str(c).lower() for c in df.columns]
    has_coord_cols = any(any(kw in c for kw in STATION_COL_KEYWORDS['Lat'] + STATION_COL_KEYWORDS['Lon'])
                          for c in cols_low)
    if has_coord_cols:
        return 'station_ref'

    return None


def guess_station_column_map(df):
    """
    Пытается угадать сопоставление столбцов справочника станций
    (код/название/регион/широта/долгота) по ключевым словам в заголовках.
    Возвращает dict {роль: имя_столбца или None}.
    """
    cols = df.columns.tolist()
    result = {}
    used = set()
    for role, keywords in STATION_COL_KEYWORDS.items():
        match = None
        for c in cols:
            if c in used:
                continue
            c_low = str(c).lower()
            if any(kw in c_low for kw in keywords):
                match = c
                break
        result[role] = match
        if match:
            used.add(match)
    return result


def load_table(file, sheet_name=None):
    """
    Загружает таблицу из xlsx/csv. Возвращает DataFrame или
    словарь {sheet_name: DataFrame} для Excel с несколькими листами
    при sheet_name=None и file — путь/буфер к xlsx с >1 листом.
    """
    name = getattr(file, 'name', str(file))
    if name.lower().endswith(('.xlsx', '.xls')):
        xls = pd.ExcelFile(file)
        if sheet_name is not None:
            return pd.read_excel(xls, sheet_name=sheet_name)
        if len(xls.sheet_names) == 1:
            return pd.read_excel(xls, sheet_name=xls.sheet_names[0])
        return {sn: pd.read_excel(xls, sheet_name=sn) for sn in xls.sheet_names}
    elif name.lower().endswith('.csv'):
        # Пытаемся угадать разделитель и десятичный знак
        raw = file.read() if hasattr(file, 'read') else open(file, 'rb').read()
        if isinstance(raw, bytes):
            text = raw.decode('utf-8-sig', errors='replace')
        else:
            text = raw
        sep = ';' if text.count(';') > text.count(',') else ','
        buf = io.StringIO(text)
        df = pd.read_csv(buf, sep=sep)
        # Если числа пришли с запятой как десятичным разделителем — конвертируем
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().astype(str).head(20)
                if sample.str.match(r'^-?\d+,\d+$').mean() > 0.5:
                    df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
        return df
    else:
        raise ValueError(f'Неподдерживаемый формат файла: {name}')


def detect_format(df):
    """
    Пытается определить, является ли таблица широкой (есть столбцы-месяцы
    или несколько столбцов вида "год_показатель") или уже длинной
    (есть отдельный столбец с названием показателя/значением).

    Возвращает: 'wide_months', 'wide_generic', 'long', 'unknown'
    """
    cols = set(df.columns.astype(str))
    months_present = sum(1 for m in MONTHS_RU + MONTHS_RU_SHORT if m in cols)
    if months_present >= 6:
        return 'wide_months'

    # Признаки длинного формата: есть столбец типа "Показатель"/"Параметр"/"Variable"
    long_markers = {'показатель', 'параметр', 'variable', 'признак', 'metric'}
    value_markers = {'значение', 'value', 'val'}
    cols_lower = {c.lower() for c in cols}
    if cols_lower & long_markers and cols_lower & value_markers:
        return 'long'

    # Широкий формат общего вида: много числовых столбцов, мало id-столбцов
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 5 and len(df.columns) - len(numeric_cols) <= 3:
        return 'wide_generic'

    return 'unknown'


def wide_months_to_long(df, id_vars, value_name='Значение', month_col_name='Месяц'):
    """
    Преобразует таблицу вида [Станция, Год, Январь, Февраль, ...]
    в длинный формат [Станция, Год, Месяц, Значение].
    id_vars — список столбцов-идентификаторов (например ['Станция', 'Год']).
    """
    month_cols = [m for m in MONTHS_RU if m in df.columns]
    if not month_cols:
        month_cols = [m for m in MONTHS_RU_SHORT if m in df.columns]
    missing = [c for c in id_vars if c not in df.columns]
    if missing:
        raise ValueError(f'Не найдены id-столбцы: {missing}')

    long_df = df.melt(id_vars=id_vars, value_vars=month_cols,
                       var_name=month_col_name, value_name=value_name)
    order = {m: i + 1 for i, m in enumerate(month_cols)}
    long_df['Месяц_номер'] = long_df[month_col_name].map(order)
    long_df = long_df.sort_values(id_vars + ['Месяц_номер']).reset_index(drop=True)
    return long_df


def wide_generic_to_long(df, id_vars, var_name='Признак', value_name='Значение'):
    """Универсальный wide -> long через pd.melt для произвольных широких таблиц."""
    value_vars = [c for c in df.columns if c not in id_vars]
    return df.melt(id_vars=id_vars, value_vars=value_vars,
                    var_name=var_name, value_name=value_name)


def long_to_wide(df, index_cols, columns_col, values_col):
    """Обратное преобразование: длинный формат -> широкий (pivot)."""
    return df.pivot_table(index=index_cols, columns=columns_col,
                           values=values_col, aggfunc='first').reset_index()


def validate_dataframe(df, required_cols=None, numeric_cols=None):
    """
    Простая валидация: проверка обязательных столбцов, доли пропусков,
    типов данных. Возвращает список предупреждений (строки) — пустой,
    если всё в порядке.
    """
    warnings_list = []
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            warnings_list.append(f'Отсутствуют обязательные столбцы: {missing}')
    if numeric_cols:
        for c in numeric_cols:
            if c in df.columns:
                non_numeric = pd.to_numeric(df[c], errors='coerce').isna() & df[c].notna()
                if non_numeric.any():
                    warnings_list.append(
                        f'Столбец "{c}": {non_numeric.sum()} нечисловых значений')
    total_cells = df.shape[0] * df.shape[1]
    na_cells = df.isna().sum().sum()
    if total_cells > 0 and na_cells / total_cells > 0.3:
        warnings_list.append(
            f'Высокая доля пропусков в таблице: {na_cells / total_cells * 100:.1f}%')
    return warnings_list


def wide_months_to_long_typed(df, id_vars, tip_label, value_name='Значение'):
    """
    Как wide_months_to_long, но сразу добавляет столбец 'Тип' (например
    'Температура' / 'Осадки') — для последующего объединения нескольких
    климатических показателей в единую длинную таблицу.
    """
    long_df = wide_months_to_long(df, id_vars, value_name=value_name)
    long_df['Тип'] = tip_label
    return long_df


def merge_station_reference(long_df, stations_ref, station_col_map):
    """
    Присоединяет справочник станций (Название, Регион, Широта, Долгота)
    к длинной таблице по коду станции.

    station_col_map — словарь вида:
        {'Станция': '<столбец-код в справочнике>',
         'Название': '<столбец или None>',
         'Регион':   '<столбец или None>',
         'Lat':      '<столбец или None>',
         'Lon':      '<столбец или None>'}
    """
    ref = stations_ref.copy()
    key_col = station_col_map.get('Станция')
    if not key_col or key_col not in ref.columns:
        raise ValueError('Не указан столбец-код станции в справочнике')

    rename_map = {key_col: 'Станция'}
    keep_cols = ['Станция']
    for target in ['Название', 'Регион', 'Lat', 'Lon']:
        src = station_col_map.get(target)
        if src and src in ref.columns:
            rename_map[src] = target
            keep_cols.append(target)

    ref = ref.rename(columns=rename_map)[keep_cols].drop_duplicates(subset='Станция')
    merged = long_df.merge(ref, on='Станция', how='left')

    if 'Название' not in merged.columns:
        merged['Название'] = merged['Станция']
    else:
        merged['Название'] = merged['Название'].fillna(merged['Станция'])

    return merged


def combine_typed_datasets(datasets):
    """
    Объединяет несколько длинных таблиц с разными Тип (Температура, Осадки, ...)
    в одну общую таблицу. datasets — список DataFrame, каждый уже содержит
    столбец 'Тип'.
    """
    if not datasets:
        return pd.DataFrame()
    return pd.concat(datasets, ignore_index=True, sort=False)


def unique_values(long_df, col):
    """Список уникальных значений столбца, отсортированный, без NaN."""
    if col not in long_df.columns:
        return []
    vals = long_df[col].dropna().unique().tolist()
    try:
        return sorted(vals)
    except TypeError:
        return sorted(vals, key=str)


AGG_HINT_BY_TYPE = {
    'осадки': 'sum',
    'осадк': 'sum',
    'precip': 'sum',
}


def default_agg_for_type(tip_label):
    """
    Эвристика: для осадков по умолчанию суммируем месяцы (годовая сумма),
    для остальных показателей (температура и т.п.) — усредняем.
    """
    low = str(tip_label).lower()
    for key, agg in AGG_HINT_BY_TYPE.items():
        if key in low:
            return agg
    return 'mean'


def filter_long(long_df, tip=None, station=None, region=None, year=None, month=None):
    """
    Универсальная фильтрация объединённой длинной таблицы по типу
    показателя, станции, региону, году и месяцу. Любой параметр можно
    оставить None — тогда фильтр по нему не применяется.
    """
    d = long_df
    if tip is not None:
        d = d[d['Тип'] == tip]
    if station is not None:
        d = d[d['Станция'] == station]
    if region is not None and 'Регион' in d.columns:
        d = d[d['Регион'] == region]
    if year is not None:
        d = d[d['Год'] == year]
    if month is not None:
        d = d[d['Месяц'] == month]
    return d


def yearly_series(long_df, tip, station=None, region=None, agg=None):
    """
    Годовой ряд значений показателя (для графика динамики): агрегирует
    12 месяцев в одно значение на год (mean для температуры, sum для
    осадков по умолчанию — либо явно заданный agg).
    Если station/region не заданы — усредняет по всем станциям.
    Возвращает Series, индекс — Год.
    """
    d = filter_long(long_df, tip=tip, station=station, region=region)
    if d.empty:
        return pd.Series(dtype=float)
    agg = agg or default_agg_for_type(tip)

    # Сначала агрегируем месяцы внутри года и станции, затем — станции.
    per_station_year = d.groupby(['Станция', 'Год'])['Значение'].agg(agg)
    yearly = per_station_year.groupby('Год').mean()
    return yearly.sort_index()


def df_to_excel_bytes(df, sheet_name='Данные'):
    """Возвращает байты xlsx-файла для скачивания через st.download_button."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    buf.seek(0)
    return buf.getvalue()


def df_to_csv_bytes(df, sep=';', decimal=','):
    """Возвращает байты csv-файла (европейский стандарт: ; и запятая)."""
    return df.to_csv(index=False, sep=sep, decimal=decimal).encode('utf-8-sig')
