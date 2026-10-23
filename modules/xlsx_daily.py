"""
=============================================================================
ЧТЕНИЕ СУТОЧНЫХ ТАБЛИЦ EXCEL/CSV СО СЛОЖНЫМИ ЗАГОЛОВКАМИ
(архивы вида «Омск_1971.xlsx … Омск_2026.xlsx»: одна станция, один год,
двухстрочная шапка с объединёнными ячейками — группа и показатель)

  • шапка: первая строка с группой («Температура воздуха», «Осадки», …),
    вторая — показатель («Т ср(°C)», «ночь, мм», …); объединённые ячейки
    группы распространяются на все её столбцы;
  • столбцы сопоставляются со стандартными показателями по правилам
    (ключевые слова группы и показателя); сопоставление можно поправить
    в интерфейсе;
  • значения записаны по-разному: «+12.3», «-0.4», «12,3», целые числа,
    «<0.5» (меньше половины сантиметра), «/» (нет данных), видимость
    «20 км», «0400 м», «10 км м» — всё приводится к числам;
  • результат — переменные в том же виде, что и при чтении файлов TT/TX/RR
    (TX, RR, FM, а также DY — прочие суточные характеристики).
=============================================================================
"""

import io
import os
import re
import zipfile
import tempfile

import numpy as np
import pandas as pd

# Правила сопоставления: (группа — регулярное выражение или None, показатель, стандартное имя, описание)
RULES = [
    (r'температур', r'^т\s*ср', 'tmean', 'Средняя суточная температура, °C'),
    (r'температур', r'^т\s*мин', 'tmin', 'Минимальная температура, °C'),
    (r'температур', r'^т\s*макс', 'tmax', 'Максимальная температура, °C'),
    (r'температур', r'^adj', 'tmean_anom_adj', 'Скорректированная аномалия Tср, °C'),
    (r'температур', r'^d\s*т', 'tmean_anom', 'Аномалия Tср, °C'),
    (r'влажн', r'ср', 'rh_mean', 'Средняя относительная влажность, %'),
    (r'влажн', r'мин', 'rh_min', 'Минимальная относительная влажность, %'),
    (r'эффектив', r'^тes', 'te_s_max', 'Эффективная температура (с учётом солнца), макс., °C'),
    (r'эффектив', r'мин', 'te_min', 'Эффективная температура, мин., °C'),
    (r'эффектив', r'макс', 'te_max', 'Эффективная температура, макс., °C'),
    (r'ветер', r'порыв', 'gust', 'Максимальный порыв ветра, м/с'),
    (r'ветер', r'ср', 'wind_mean', 'Средняя скорость ветра, м/с'),
    (r'видим', None, 'vis_min', 'Минимальная видимость, м'),
    (r'давлен', r'^p\s*ср', 'p_mean', 'Давление на уровне станции, ср., гПа'),
    (r'давлен', r'^p\s*мин', 'p_min', 'Давление, мин., гПа'),
    (r'давлен', r'^p\s*макс', 'p_max', 'Давление, макс., гПа'),
    (r'давлен', r'^po\s*ср', 'p0_mean', 'Давление на уровне моря, ср., гПа'),
    (r'давлен', r'^po\s*мин', 'p0_min', 'Давление на уровне моря, мин., гПа'),
    (r'давлен', r'^po\s*макс', 'p0_max', 'Давление на уровне моря, макс., гПа'),
    (r'облачн', r'общ', 'cloud_total', 'Общая облачность, балл'),
    (r'облачн', r'нижн', 'cloud_low', 'Нижняя облачность, балл'),
    (r'осадк', r'ночь', 'rr_n', 'Осадки за ночь, мм'),
    (r'осадк', r'день', 'rr_d', 'Осадки за день, мм'),
    (r'осадк', r'сумм|сутк', 'rr_s', 'Осадки за сутки, мм'),
    (r'снеж', r'высот', 'snow_depth', 'Высота снежного покрова, см'),
    (r'снеж', r'покрыт|степен', 'snow_cover', 'Степень покрытия снегом, балл'),
    (r'число случаев', r'дожд', 'n_rain', 'Случаев дождя'),
    (r'число случаев', r'снег', 'n_snow', 'Случаев снега'),
    (r'число случаев', r'туман', 'n_fog', 'Случаев тумана'),
    (r'число случаев', r'мгла', 'n_haze', 'Случаев мглы'),
    (r'число случаев', r'метел', 'n_blizzard', 'Случаев метели'),
    (r'число случаев', r'позём', 'n_drift', 'Случаев позёмка'),
    (r'число случаев', r'гроз', 'n_thunder', 'Случаев грозы'),
    (r'число случаев', r'смерч', 'n_tornado', 'Случаев смерча'),
    (r'число случаев', r'буря', 'n_duststorm', 'Случаев пыльной бури'),
    (r'число случаев', r'п\.?\s*позём', 'n_dustdrift', 'Случаев пыльного позёмка'),
    (r'диаметр', r'град', 'hail_d', 'Град, макс. диаметр, мм'),
    (r'диаметр', r'гололед', 'glaze_d', 'Гололёд, диаметр отложения, мм'),
    (r'диаметр', r'изм', 'rime_d', 'Изморозь, диаметр отложения, мм'),
    (r'диаметр', r'сн\.?\s*отл', 'wetsnow_d', 'Налипание мокрого снега, диаметр, мм'),
    (r'число сроков', None, 'n_terms', 'Число сроков наблюдений'),
    (r'^индекс|^станц|^код', None, 'st', 'Индекс станции'),
    (r'^назв', None, 'name', 'Название станции'),
    (r'^дата', None, 'date', 'Дата'),
]
STD_LABEL = {std: lab for _, _, std, lab in RULES}
# куда раскладываются стандартные показатели
TARGET = {'tmin': ('TX', 'tmin'), 'tmax': ('TX', 'tmax'), 'tmean': ('TX', 'tmean'),
          'rr_n': ('RR', 'rr_n'), 'rr_d': ('RR', 'rr_d'), 'rr_s': ('RR', 'rr_s')}
# порыв ветра в таблицах — максимальный порыв за сутки, а FM в основных файлах — максимальная
# скорость за полусутки; величины разные, поэтому порыв хранится отдельно (DY: gust)

NUM_RE = re.compile(r'[-+]?\d+(?:[.,]\d+)?')


def parse_value(v, std=None):
    """Приведение значения ячейки к числу (NaN — нет данных)."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip().replace('\xa0', ' ').replace('−', '-')
    if s in ('', '/', '-', '—', '–', '*', 'нет', 'н/д'):
        return np.nan
    if std == 'vis_min':
        m = NUM_RE.search(s)
        if not m:
            return np.nan
        x = float(m.group().replace(',', '.'))
        unit_km = 'км' in s.split(m.group(), 1)[1][:4] or 'km' in s
        return x * 1000 if unit_km else x
    if s.startswith('<'):                      # «<0.5» — меньше половины единицы
        return 0.0
    if s.startswith('>'):
        s = s[1:]
    m = NUM_RE.search(s)
    return float(m.group().replace(',', '.')) if m else np.nan


def _read_grid(content, name, sheet=None):
    """Таблица значений листа + сведения об объединённых ячейках шапки."""
    low = name.lower()
    if low.endswith(('.xlsx', '.xlsm')):
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
        ws = wb[sheet] if sheet else wb.worksheets[0]
        grid = [list(r) for r in ws.iter_rows(values_only=True)]
        merged = [(m.min_row, m.min_col, m.max_row, m.max_col) for m in ws.merged_cells.ranges]
        return grid, merged
    if low.endswith('.xls'):
        df = pd.read_excel(io.BytesIO(content), header=None, sheet_name=sheet or 0)
        return df.where(df.notna(), None).values.tolist(), []
    text = content.decode('utf-8-sig', errors='replace')
    if text.count('�') > 10:
        text = content.decode('cp1251', errors='replace')
    sep = ';' if text.count(';') > text.count(',') else (',' if text.count(',') else '\t')
    df = pd.read_csv(io.StringIO(text), sep=sep, header=None, dtype=str, keep_default_na=False)
    return df.values.tolist(), []


def _is_date(v):
    if isinstance(v, (pd.Timestamp,)) or hasattr(v, 'year'):
        return True
    return bool(re.match(r'^\s*\d{1,4}[./-]\d{1,2}[./-]\d{1,4}\s*$', str(v or '')))


def detect_header(grid, merged):
    """
    Определяет строки шапки (всё выше первой строки с датой) и составляет
    имена столбцов «группа | показатель» с учётом объединённых ячеек.
    Возвращает (names, first_data_row, date_col).
    """
    first, dcol = None, None
    for i, row in enumerate(grid[:40]):
        for j, v in enumerate(row):
            if _is_date(v):
                first, dcol = i, j
                break
        if first is not None:
            break
    if first is None:
        raise ValueError('Не найдено ни одной строки с датой.')
    hdr = [list(r) for r in grid[:first]]
    ncol = max(len(r) for r in grid[:first + 5])
    for r in hdr:
        r += [None] * (ncol - len(r))
    # объединённые ячейки: значение левой верхней ячейки на весь диапазон
    for r0, c0, r1, c1 in merged:
        if r0 - 1 >= len(hdr):
            continue
        v = hdr[r0 - 1][c0 - 1]
        for rr in range(r0 - 1, min(r1, len(hdr))):
            for cc in range(c0 - 1, c1):
                if hdr[rr][cc] in (None, ''):
                    hdr[rr][cc] = v
    # без сведений об объединении — протягиваем группу вправо, если ниже есть показатель
    if not merged and len(hdr) >= 2:
        last = None
        for c in range(ncol):
            if hdr[0][c] not in (None, ''):
                last = hdr[0][c]
            elif hdr[1][c] not in (None, ''):
                hdr[0][c] = last
    names = []
    for c in range(ncol):
        parts = []
        for r in hdr:
            v = r[c]
            if v not in (None, '') and str(v).strip() not in parts:
                parts.append(str(v).strip())
        names.append(' | '.join(parts) if parts else f'Столбец {c + 1}')
    return names, first, dcol


def match_columns(names):
    """Сопоставление имён столбцов со стандартными показателями. Возвращает DataFrame."""
    rows = []
    used = set()
    for j, nm in enumerate(names):
        parts = [p.strip().lower() for p in nm.split('|')]
        grp, ind = parts[0], parts[-1] if len(parts) > 1 else ''
        std = None
        for g, i, s, _ in RULES:
            if s in used:
                continue
            if i is None:
                if re.search(g, grp):
                    std = s
                    break
            elif re.search(g, grp) and re.search(i, ind):
                std = s
                break
        if std:
            used.add(std)
        rows.append(dict(Столбец=j + 1, Заголовок=nm, Показатель=std or '—',
                         Описание=STD_LABEL.get(std, 'не используется')))
    return pd.DataFrame(rows)


def read_daily_table(content, name, mapping=None, sheet=None):
    """
    Читает один файл. mapping — DataFrame сопоставления (если пользователь
    его поправил). Возвращает (df [st, name, date, <показатели>], mapping, notes).
    """
    grid, merged = _read_grid(content, name, sheet)
    names, first, dcol = detect_header(grid, merged)
    mp = mapping if mapping is not None else match_columns(names)
    notes = []
    cols = {}
    for _, r in mp.iterrows():
        std = r['Показатель']
        if std in (None, '—', ''):
            continue
        j = int(r['Столбец']) - 1
        vals = [row[j] if j < len(row) else None for row in grid[first:]]
        if std == 'date':
            cols['date'] = pd.to_datetime(pd.Series([str(v).strip() if not hasattr(v, 'year') else v for v in vals]),
                                          dayfirst=True, errors='coerce')
        elif std == 'name':
            cols['name'] = [str(v).strip() if v is not None else None for v in vals]
        elif std == 'st':
            cols['st'] = [parse_value(v) for v in vals]
        else:
            cols[std] = [parse_value(v, std) for v in vals]
    if 'date' not in cols:
        vals = [row[dcol] for row in grid[first:]]
        cols['date'] = pd.to_datetime(pd.Series([str(v) for v in vals]), dayfirst=True, errors='coerce')
    df = pd.DataFrame(cols)
    bad = df['date'].isna().sum()
    df = df.dropna(subset=['date'])
    if bad:
        notes.append(f'{name}: пропущено строк без даты — {bad}')
    if 'st' in df:
        df['st'] = df['st'].ffill().astype('Int64')
    return df, mp, notes


def collect_files(uploads):
    """
    uploads: список (имя, байты). Архивы ZIP/RAR/7z распаковываются.
    Возвращает список (имя, байты) табличных файлов.
    """
    out = []
    for nm, b in uploads:
        low = nm.lower()
        if low.endswith(('.xlsx', '.xlsm', '.xls', '.csv', '.txt')):
            out.append((os.path.basename(nm), b))
        elif low.endswith('.zip'):
            z = zipfile.ZipFile(io.BytesIO(b))
            out += collect_files([(i.filename, z.read(i)) for i in z.infolist() if not i.is_dir()])
        elif low.endswith(('.rar', '.7z')):
            out += collect_files(extract_other_archive(b, nm))
    return out


def extract_other_archive(b, nm):
    """RAR/7z: через libarchive (libarchive-c) или rarfile (нужна утилита unrar)."""
    try:
        import libarchive
        res = []
        with libarchive.memory_reader(b) as arch:
            for e in arch:
                if e.isfile:
                    res.append((e.pathname, b''.join(e.get_blocks())))
        return res
    except ImportError:
        pass
    try:
        import rarfile
        rf = rarfile.RarFile(io.BytesIO(b))
        return [(i.filename, rf.read(i)) for i in rf.infolist() if not i.isdir()]
    except Exception as e:
        raise RuntimeError(f'Архив {nm} не удалось распаковать ({e}). Установите libarchive-c '
                           f'или распакуйте архив и загрузите файлы либо ZIP.')


def load_tables(files, mapping=None, progress=None):
    """
    Читает набор файлов (одна или несколько станций, по годам). Возвращает
    (daily, mapping, notes) — daily: [st, name, date, показатели…].
    """
    parts, notes, mp = [], [], mapping
    for k, (nm, b) in enumerate(sorted(files)):
        if progress:
            progress(k / max(1, len(files)), nm)
        try:
            df, mp_k, nt = read_daily_table(b, nm, mp)
            mp = mp if mp is not None else mp_k
            notes += nt
            df['_file'] = nm
            parts.append(df)
        except Exception as e:
            notes.append(f'{nm}: не прочитан — {e}')
    if not parts:
        return pd.DataFrame(), mp, notes
    d = pd.concat(parts, ignore_index=True)
    if 'st' not in d:
        # индекс по имени файла: первая группа цифр из 5 знаков, иначе условный код
        d['st'] = d['_file'].str.extract(r'(\d{5})')[0].astype(float).fillna(99999).astype(int)
    if 'name' not in d:
        d['name'] = d['_file'].str.replace(r'[_\-\s]*\d{4}\.\w+$', '', regex=True)
    dup = d.duplicated(['st', 'date']).sum()
    if dup:
        notes.append(f'Повторяющихся суток (станция и дата): {dup}; оставлены первые.')
        d = d.drop_duplicates(['st', 'date'])
    return d.drop(columns='_file').sort_values(['st', 'date']).reset_index(drop=True), mp, notes


def to_program_vars(daily, region):
    """
    Раскладывает суточную таблицу по переменным программы:
    TX (tmin, tmax, tmean), RR (rr_n, rr_d, rr_s), DY —
    остальное (влажность, снег, видимость, явления).
    """
    out = {}
    base = daily[['st', 'date']].copy()
    for var in ('TX', 'RR'):
        comps = {std: c for std, (v, c) in TARGET.items() if v == var and std in daily}
        if comps:
            d = base.copy()
            for std, c in comps.items():
                d[c] = daily[std].values
            if var == 'RR':
                for c in ('rr_n', 'rr_d', 'rr_s'):
                    if c not in d:
                        d[c] = np.nan
                    d[c + '_tr'] = False
                # сутки без полусуточных значений — сумма, если обе половины есть
                miss = d['rr_s'].isna() & d['rr_n'].notna() & d['rr_d'].notna()
                d.loc[miss, 'rr_s'] = d.loc[miss, 'rr_n'] + d.loc[miss, 'rr_d']
            if var == 'FM':
                d['fm_n'] = np.nan
                d['fm_d'] = np.nan
            d['region'] = region
            out[var] = d
    rest = [c for c in daily.columns if c not in ('st', 'name', 'date') and c not in TARGET]
    if rest:
        d = daily[['st', 'date'] + rest].copy()
        d['region'] = region
        out['DY'] = d
    return out


def merge_datasets(main, extra, prefer='main'):
    """
    Объединяет основной набор (файлы TT/TX/RR…) с данными из таблиц: для
    общих станций и дат приоритет у набора prefer, остальное дополняется.
    """
    res = dict(main)
    for var, d2 in extra.items():
        if var not in res:
            res[var] = d2
            continue
        d1 = res[var]
        a, b = (d1, d2) if prefer == 'main' else (d2, d1)
        A = a.set_index(['st', 'date'])
        B = b.set_index(['st', 'date'])
        comb = A.combine_first(B).reset_index()
        order = ['st', 'date'] + [c for c in A.columns] + [c for c in B.columns if c not in A.columns]
        res[var] = comb[order].sort_values(['st', 'date']).reset_index(drop=True)
    return res
