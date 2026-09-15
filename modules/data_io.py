"""
=============================================================================
МОДУЛЬ ЗАГРУЗКИ И ПРЕОБРАЗОВАНИЯ ДАННЫХ
Поддержка: Excel (.xlsx), CSV. Автоопределение широкого формата и
преобразование в длинный. Валидация.
=============================================================================
"""

import io
import re
import pandas as pd
import numpy as np


MONTHS_RU = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
             'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']

MONTHS_RU_SHORT = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                    'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']

# Канонические (нижний регистр) формы для гибкого распознавания —
# без учёта регистра, с поддержкой сокращений, римских цифр, цифровых
# номеров месяцев и декад ("1 декада апреля" и т.п.)
_MONTHS_CANON = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
                 'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']
_MONTHS_GENITIVE = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
                    'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']
_MONTHS_SHORT_CANON = ['янв', 'фев', 'мар', 'апр', 'май', 'июн',
                       'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']
_ROMAN_MONTHS = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6,
                  'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10, 'XI': 11, 'XII': 12}
_DECADE_RE = re.compile(r'(\d)\s*[-–]?\s*(?:я|ая|й)?\s*декад\w*\s+([а-яё]+)',
                          re.IGNORECASE | re.UNICODE)


def _match_month_word(word):
    """Сопоставляет слово с номером месяца (1-12): полное имя, сокращение
    (в любом регистре, с точкой или без)."""
    w = str(word).strip().lower().rstrip('.')
    if not w:
        return None
    if w in _MONTHS_CANON:
        return _MONTHS_CANON.index(w) + 1
    if w in _MONTHS_GENITIVE:
        return _MONTHS_GENITIVE.index(w) + 1
    if w in _MONTHS_SHORT_CANON:
        return _MONTHS_SHORT_CANON.index(w) + 1
    # Сокращение произвольной длины (от 3 букв), являющееся началом полного
    # имени (в именительном или родительном падеже)
    if len(w) >= 3:
        for i, full in enumerate(_MONTHS_CANON):
            if full.startswith(w) or _MONTHS_GENITIVE[i].startswith(w):
                return i + 1
    return None


def normalize_month_column(col):
    """
    Пытается распознать столбец как месяц или декаду месяца, независимо от
    регистра, формата написания (полное имя/сокращение/декада). Возвращает
    {'month': 1..12, 'decade': 1|2|3|None} или None, если не похоже на
    месяц/декаду. НЕ включает римские цифры и голые номера 1-12 — они
    неоднозначны (в агроклиматических таблицах римскими цифрами часто
    обозначают периоды вида "X - III", а не отдельные месяцы) и
    обрабатываются отдельно в scan_month_columns с проверкой на
    систематичность обозначения по всему файлу.
    """
    s = str(col).strip()
    if not s:
        return None

    m = _DECADE_RE.search(s)
    if m:
        decade = int(m.group(1))
        if decade in (1, 2, 3):
            month_num = _match_month_word(m.group(2))
            if month_num:
                return {'month': month_num, 'decade': decade}

    month_num = _match_month_word(s)
    if month_num:
        return {'month': month_num, 'decade': None}

    return None


def _normalize_month_column_risky(col):
    """
    Распознавание по римским цифрам или голому номеру (1-12) — само по себе
    неоднозначно (совпадает с обозначением периодов в агроклиматических
    таблицах вида "X - III"), поэтому используется только как кандидат,
    который scan_month_columns принимает лишь при систематическом
    повторении такой схемы по всему файлу.
    """
    s = str(col).strip()
    su = s.upper()
    if su in _ROMAN_MONTHS:
        return {'month': _ROMAN_MONTHS[su], 'decade': None}
    if s.isdigit() and 1 <= int(s) <= 12:
        return {'month': int(s), 'decade': None}
    return None


MIN_SYSTEMATIC_MATCHES = 6


def scan_month_columns(df):
    """
    Проходит по всем столбцам таблицы, распознавая столбцы-месяцы и
    столбцы-декады. Сначала — надёжные признаки (полное/сокращённое имя,
    декада), они принимаются всегда. Римские цифры и голые номера 1-12
    принимаются только если по всему файлу найдено не менее
    MIN_SYSTEMATIC_MATCHES таких меток (иначе это, скорее всего, случайное
    совпадение с обозначением периода, а не месяц) — и только для тех
    месяцев, что ещё не определены надёжным способом.

    Возвращает (month_map, matched_cols):
      month_map — dict {1..12: {'direct': столбец_или_None,
                                 'decades': {1: столбец, 2: ..., 3: ...}}}
      matched_cols — список всех распознанных столбцов (для исключения
                     их из id-столбцов).
    """
    month_map = {m: {'direct': None, 'decades': {}} for m in range(1, 13)}
    matched_cols = []
    risky_candidates = []

    for col in df.columns:
        info = normalize_month_column(col)
        if info is not None:
            matched_cols.append(col)
            if info['decade']:
                month_map[info['month']]['decades'][info['decade']] = col
            elif month_map[info['month']]['direct'] is None:
                month_map[info['month']]['direct'] = col
            continue
        risky_info = _normalize_month_column_risky(col)
        if risky_info is not None:
            risky_candidates.append((col, risky_info))

    if len(risky_candidates) >= MIN_SYSTEMATIC_MATCHES:
        for col, info in risky_candidates:
            if month_map[info['month']]['direct'] is None and not month_map[info['month']]['decades']:
                month_map[info['month']]['direct'] = col
                matched_cols.append(col)

    return month_map, matched_cols


def has_decade_columns(df):
    """True, если среди столбцов таблицы есть декадные (не только помесячные)."""
    month_map, _ = scan_month_columns(df)
    return any(info['decades'] for info in month_map.values())


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
        month_map, matched_cols = scan_month_columns(df)
        if matched_cols:
            vals = pd.to_numeric(df[matched_cols].values.ravel(), errors='coerce')
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
    Пытается определить, является ли таблица широкой (есть столбцы-месяцы,
    в том числе декадные, или несколько столбцов вида "год_показатель") или
    уже длинной (есть отдельный столбец с названием показателя/значением).

    Возвращает: 'wide_months', 'wide_generic', 'long', 'unknown'
    """
    month_map, matched_cols = scan_month_columns(df)
    months_covered = sum(1 for info in month_map.values()
                          if info['direct'] or info['decades'])
    if months_covered >= 6:
        return 'wide_months'

    # Признаки длинного формата: есть столбец типа "Показатель"/"Параметр"/"Variable"
    cols_lower = {str(c).lower() for c in df.columns}
    long_markers = {'показатель', 'параметр', 'variable', 'признак', 'metric'}
    value_markers = {'значение', 'value', 'val'}
    if cols_lower & long_markers and cols_lower & value_markers:
        return 'long'

    # Широкий формат общего вида: много числовых столбцов, мало id-столбцов
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 5 and len(df.columns) - len(numeric_cols) <= 3:
        return 'wide_generic'

    return 'unknown'


def wide_months_to_long(df, id_vars, value_name='Значение', month_col_name='Месяц',
                         decade_agg='mean'):
    """
    Преобразует таблицу вида [Станция, Год, <месяцы>] в длинный формат
    [Станция, Год, Месяц, Месяц_номер, Значение]. Столбцы-месяцы распознаются
    гибко (см. normalize_month_column): любой регистр, сокращения, римские
    цифры, номера 1-12. Декадные столбцы ("1 декада апреля" и т.п.)
    автоматически сворачиваются в одно месячное значение — суммой
    (decade_agg='sum', обычно для осадков) или средним (decade_agg='mean',
    обычно для температуры).
    id_vars — список столбцов-идентификаторов (например ['Станция', 'Год']).
    """
    missing = [c for c in id_vars if c not in df.columns]
    if missing:
        raise ValueError(f'Не найдены id-столбцы: {missing}')

    month_map, matched_cols = scan_month_columns(df)

    frames = []
    for month_num in range(1, 13):
        info = month_map[month_num]
        if info['direct']:
            sub = df[id_vars + [info['direct']]].copy()
            sub = sub.rename(columns={info['direct']: value_name})
        elif info['decades']:
            decade_cols = [info['decades'][d] for d in sorted(info['decades'])]
            vals = df[decade_cols].apply(pd.to_numeric, errors='coerce')
            all_nan_mask = vals.isna().all(axis=1)
            if decade_agg == 'sum':
                agg_series = vals.sum(axis=1, skipna=True)
            else:
                agg_series = vals.mean(axis=1, skipna=True)
            agg_series[all_nan_mask] = np.nan
            sub = df[id_vars].copy()
            sub[value_name] = agg_series
        else:
            continue

        sub[month_col_name] = MONTHS_RU[month_num - 1]
        sub['Месяц_номер'] = month_num
        frames.append(sub)

    if not frames:
        raise ValueError('Не удалось найти ни одного столбца-месяца в данных. '
                          'Проверьте названия столбцов (месяцы или декады месяцев).')

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.sort_values(id_vars + ['Месяц_номер']).reset_index(drop=True)
    long_df = long_df[id_vars + [month_col_name, 'Месяц_номер', value_name]]
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


def wide_months_to_long_typed(df, id_vars, tip_label, value_name='Значение', decade_agg='mean'):
    """
    Как wide_months_to_long, но сразу добавляет столбец 'Тип' (например
    'Температура' / 'Осадки') — для последующего объединения нескольких
    климатических показателей в единую длинную таблицу.
    """
    long_df = wide_months_to_long(df, id_vars, value_name=value_name, decade_agg=decade_agg)
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


def guess_id_columns(candidates):
    """
    Из списка "нецелевых" (не месяц/декада) столбцов пытается угадать,
    какие из них — настоящие идентификаторы (станция, год), а не
    дополнительные предрассчитанные показатели. По умолчанию в
    multiselect должны попадать только они, а не вообще все нецелевые
    столбцы (иначе десятки периодных сумм ошибочно попадут в id_vars).
    """
    station_kw = ['станц', 'код', 'station']
    year_kw = ['год', 'year']
    ids = [c for c in candidates
           if any(k in str(c).lower() for k in station_kw + year_kw)]
    if not ids and candidates:
        ids = list(candidates[:2])
    return ids


def build_extra_features_table(df, id_vars, extra_cols):
    """
    Формирует таблицу "как есть" из столбцов, не распознанных как месяцы/
    декады (например готовые предрассчитанные периодные суммы вида
    "X - III", "21.V - 20.VI") — без попытки интерпретировать их смысл.
    Значения приводятся к числу, где возможно.
    """
    out = df[id_vars + extra_cols].copy()
    for c in extra_cols:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out


def aggregate_extra_features_by_station(df, station_col='Станция', agg='mean', exclude_cols=None):
    """
    Сворачивает таблицу дополнительных показателей (может быть несколько
    строк на станцию — по годам) до одной строки на станцию агрегацией
    (по умолчанию — среднее по годам). exclude_cols — прочие id-столбцы
    (например 'Год'), которые не являются признаками и должны быть
    исключены из агрегации, а не усреднены как число.
    """
    exclude = set(exclude_cols or []) | {station_col}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in exclude]
    return df.groupby(station_col)[numeric_cols].agg(agg).reset_index()


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
