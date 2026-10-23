"""
=============================================================================
МОДУЛЬ ЧТЕНИЯ СУТОЧНЫХ ФАЙЛОВ МЕТЕОСТАНЦИЙ (формат ЗапСиб / Обь-Иртышского УГМС)

Поддерживаемые файлы (годовые, внутри — помесячные блоки, одна строка —
одна станция за месяц; первая строка блока — даты, вторая — сроки):
  TT  — температура воздуха по 8 срокам, °C×10
  TX  — Tмин, Tмакс, Tсредняя за сутки, °C×10
  RR  — осадки: ночь (0000), день (0012), сутки (0024), (значение−1)/10;
        «0» — следы, «1» — осадков не было
  FM  — максимальная скорость ветра за полусутки: ночь, день, сутки, м/с
  TS  — температура поверхности почвы в 00 и 12 ВСВ, целые °C
  RS  — температура точки росы по 8 срокам, °C×10; 999 — код
  RH  — относительная влажность по 8 срокам, %
  FR  — класс пожарной опасности по Нестерову (апрель–октябрь),
        блоки «Fact PO Nesterov М.ГГГГ»; 99 — пропуск; 0 — не рассчитывался
  index_* — справочник станций

Файлы могут лежать в папке или в ZIP-архиве (в т. ч. во вложенных ZIP).
Строки с «*****» или слитыми числами читаются по фиксированной ширине поля.
=============================================================================
"""

import io
import os
import re
import zipfile
import tempfile
import calendar

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as P

# Описание переменных: префикс файла → (название, компоненты, масштаб)
VAR_INFO = {
    'TT': dict(name='Температура воздуха по срокам', unit='°C'),
    'TX': dict(name='Минимальная, максимальная и средняя суточная температура воздуха', unit='°C'),
    'RR': dict(name='Количество осадков за ночь, день и сутки', unit='мм'),
    'FM': dict(name='Максимальная скорость ветра за ночь, день и сутки', unit='м/с'),
    'TS': dict(name='Температура поверхности почвы по срокам', unit='°C'),
    'RS': dict(name='Температура точки росы по срокам', unit='°C'),
    'RH': dict(name='Относительная влажность воздуха по срокам', unit='%'),
    'FR': dict(name='Класс пожарной опасности по условиям погоды (по В. Г. Нестерову)', unit='класс'),
    'DY': dict(name='Прочие суточные характеристики: давление, снежный покров, атмосферные явления, порывы ветра',
               unit=''),
}


def var_label(code):
    """Полное название переменной с обозначением в скобках: «Количество осадков … (RR)»."""
    i = VAR_INFO.get(code)
    return f'{i["name"]} ({code})' if i else str(code)

REGION_SUFFIX = {'nsk': 'НСО', 'omsk': 'Омская обл.'}

FILE_RE = re.compile(r'^(TT|TX|RR|FM|TS|RS|RH|FR)([a-zA-Z]+?)(\d{4})(?:\.\w+)?$')
DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}')
FR_HEAD_RE = re.compile(r'Nesterov\s+(\d{1,2})\.(\d{4})', re.IGNORECASE)


# ─── Файловая система и архивы ───────────────────────────────────────────
def extract_archive(zip_bytes_or_path, target_dir=None):
    """Распаковывает ZIP (включая вложенные ZIP) во временную папку. Возвращает путь."""
    target_dir = target_dir or tempfile.mkdtemp(prefix='ca_data_')
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes_or_path)
                         if isinstance(zip_bytes_or_path, (bytes, bytearray))
                         else zip_bytes_or_path)
    for info in zf.infolist():
        name = info.filename
        # Имена в архивах из Windows часто в cp866
        if not (info.flag_bits & 0x800):
            try:
                name = name.encode('cp437').decode('cp866')
            except Exception:
                pass
        dest = os.path.normpath(os.path.join(target_dir, name))
        if not dest.startswith(os.path.normpath(target_dir)):
            continue  # защита от путей вида ../
        if info.is_dir():
            os.makedirs(dest, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with zf.open(info) as src, open(dest, 'wb') as out:
            out.write(src.read())
        if dest.lower().endswith('.zip'):
            sub = os.path.splitext(dest)[0]
            os.makedirs(sub, exist_ok=True)
            extract_archive(dest, sub)
    return target_dir


def scan_folder(folder):
    """
    Находит все файлы данных и справочники станций в папке (рекурсивно).
    Возвращает (files, index_files): files — список dict {path, var, region,
    year}; при дублировании одного и того же файла (например TX в двух
    архивах) берётся последний изменённый, а совпадение отмечается.
    """
    found, index_files = {}, []
    for root, _, names in os.walk(folder):
        for nm in names:
            path = os.path.join(root, nm)
            if nm.lower().startswith('index'):
                index_files.append(path)
                continue
            m = FILE_RE.match(nm)
            if not m:
                continue
            var, suf, year = m.group(1), m.group(2).lower(), int(m.group(3))
            region = REGION_SUFFIX.get(suf, suf)
            key = (var, region, year)
            rec = dict(path=path, var=var, region=region, year=year, name=nm,
                       duplicates=[])
            if key in found:
                prev = found[key]
                rec['duplicates'] = prev['duplicates'] + [prev['path']]
                if os.path.getmtime(path) < os.path.getmtime(prev['path']):
                    prev['duplicates'].append(path)
                    continue
            found[key] = rec
    files = sorted(found.values(), key=lambda r: (r['var'], r['region'], r['year']))
    return files, sorted(set(index_files))


# ─── Разбор строк ────────────────────────────────────────────────────────
def _to_int(tok):
    try:
        return int(tok)
    except ValueError:
        return None


def split_values(rest, n_expected, width=P.FIELD_WIDTH):
    """
    Разбивает хвост строки (после кода станции) на n_expected целых.
    Сначала — по пробелам; если число элементов не совпадает или есть
    нечисловые элементы («*****», слитые числа) — по фиксированной ширине
    поля. Возвращает (массив float с NaN, метод: 'split'|'fixed').
    """
    toks = rest.split()
    vals = [_to_int(t) for t in toks]
    if n_expected is None or (len(toks) == n_expected and None not in vals):
        return np.array([np.nan if v is None else v for v in vals], float), 'split'
    raw = rest.rstrip('\r\n')
    # выравнивание: значение занимает width знаков, строка может начинаться с пробела
    if len(raw) < n_expected * width:
        raw = raw.rjust(n_expected * width)
    start = len(raw) - n_expected * width if len(raw) > n_expected * width else 0
    out = []
    for i in range(n_expected):
        chunk = raw[start + i * width: start + (i + 1) * width].strip()
        v = _to_int(chunk) if chunk else None
        out.append(np.nan if v is None else v)
    return np.array(out, float), 'fixed'


def parse_block_file(path, var):
    """
    Читает годовой файл с помесячными блоками «date:/index/строки станций».
    Возвращает (DataFrame [st, date, term, raw], stats) в «сыром» виде
    (без декодирования), stats — счётчики способов разбора строк.
    """
    rows, stats = [], {'split': 0, 'fixed': 0, 'lines': 0}
    dates, terms, nt = None, None, None
    with open(path, encoding='cp1251', errors='replace') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith('date'):
                dates = DATE_RE.findall(s)
                continue
            if s.lower().startswith('index'):
                all_terms = s.split()[1:]
                nt = max(1, len(all_terms) // max(1, len(dates or [1])))
                terms = all_terms[:nt]
                continue
            if dates is None or terms is None:
                continue
            m = re.match(r'\s*(\d{4,6})', line)
            if not m:
                continue
            st = int(m.group(1))
            rest = line[m.end():]
            toks = rest.split()
            n_tok = len(toks)
            # строки RR/RH обрезаны по длине месяца — число значений кратно nt
            n_exp = len(dates) * nt
            if n_tok != n_exp and n_tok % nt == 0 and n_tok < n_exp and \
                    all(_to_int(t) is not None for t in toks):
                n_exp = n_tok
            vals, how = split_values(rest, n_exp)
            stats[how] += 1
            stats['lines'] += 1
            nd = len(vals) // nt
            for i in range(nd):
                for j in range(nt):
                    rows.append((st, dates[i], terms[j], vals[i * nt + j], i, j, nd))
    df = pd.DataFrame(rows, columns=['st', 'date', 'term', 'raw', 'di', 'tj', 'nd'])
    return df, stats


def parse_fr_file(path):
    """Читает файл классов пожарной опасности (блоки «Fact PO Nesterov М.ГГГГ»)."""
    rows, month, year = [], None, None
    with open(path, encoding='cp1251', errors='replace') as f:
        for line in f:
            h = FR_HEAD_RE.search(line)
            if h:
                month, year = int(h.group(1)), int(h.group(2))
                continue
            if month is None:
                continue
            m = re.match(r'\s*(\d{4,6})\s', line)
            if not m:
                continue
            st = int(m.group(1))
            nd = calendar.monthrange(year, month)[1]
            toks = line[m.end():].split()
            vals, _ = split_values(line[m.end():], len(toks) if len(toks) <= 31 else nd)
            for d, v in enumerate(vals[:nd], start=1):
                rows.append((st, pd.Timestamp(year, month, d), v))
    return pd.DataFrame(rows, columns=['st', 'date', 'raw'])


# ─── Декодирование по типу переменной ───────────────────────────────────
def _nan_codes(x, extra=()):
    x = x.astype(float).copy()
    for c in tuple(P.MISSING_CODES) + tuple(extra):
        x[x == c] = np.nan
    return x


def decode_rr(raw):
    """Осадки: «0» — следы (0 мм, признак следов), «1» — 0 мм, иначе (v−1)/10."""
    raw = _nan_codes(raw)
    mm = np.where(raw <= 1, 0.0, (raw - 1) / 10.0)
    mm = np.where(np.isnan(raw), np.nan, mm)
    trace = raw == 0
    return mm, trace


def _rr_layout_score(w):
    """Доля суток, где ночь + день ≈ сутки (допуск 1,05 мм из-за отброшенных десятых)."""
    m = w[['rr_n', 'rr_d', 'rr_s']].notna().all(1) & (w['rr_s'] > 0)
    if m.sum() == 0:
        return np.nan
    return float(((w.loc[m, 'rr_n'] + w.loc[m, 'rr_d'] - w.loc[m, 'rr_s']).abs() <= 1.05).mean())


def _rr_to_wide(raw_df, layout):
    """Сборка RR в широкий вид при раскладке 'day' (тройки по датам) или 'block' (блоки по месяцу)."""
    d = raw_df.copy()
    if layout == 'block':
        # в строке: все ночные, затем все дневные, затем суточные значения
        k = d['di'] * 3 + d['tj']                  # позиция в строке
        nd = d['nd']
        comp = k // nd
        day = k % nd
        d['comp'] = comp
        d['day'] = day
        # дату берём по номеру дня из того же блока
        first = pd.to_datetime(d['date'], errors='coerce') - pd.to_timedelta(d['di'], unit='D')
        d['date'] = (first + pd.to_timedelta(day, unit='D')).dt.strftime('%Y-%m-%d')
    else:
        d['comp'] = d['tj']
    names = {0: 'rr_n', 1: 'rr_d', 2: 'rr_s'}
    d['comp'] = d['comp'].map(names)
    w = d.pivot_table(index=['st', 'date'], columns='comp', values='raw', aggfunc='first').reset_index()
    return w


def build_variable(var, raw_df, rr_layout=P.RR_LAYOUT):
    """
    Превращает «сырые» значения в таблицу суточных данных одной переменной:
    [st, date, <компоненты>], значения в физических единицах, NaN — пропуск.
    Возвращает (df, notes) — notes: список замечаний о декодировании.
    """
    notes = []
    if raw_df.empty:
        return pd.DataFrame(), notes
    if var == 'RR':
        layouts = [rr_layout] if rr_layout in ('day', 'block') else ['day', 'block']
        best, best_score, best_w = None, -1, None
        for lay in layouts:
            w = _rr_to_wide(raw_df, lay)
            for c in ['rr_n', 'rr_d', 'rr_s']:
                mm, tr = decode_rr(w[c].values)
                w[c] = mm
                w[c + '_tr'] = tr
            sc = _rr_layout_score(w)
            notes.append(f'RR, раскладка «{lay}»: ночь + день = сутки в {sc * 100:.1f} % суток с осадками'
                         .replace('.', ','))
            if sc > best_score:
                best, best_score, best_w = lay, sc, w
        notes.append(f'RR: принята раскладка «{best}»')
        df = best_w
    elif var == 'FR':
        df = raw_df.copy()
        v = df['raw'].astype(float)
        df['fr_notcalc'] = v == 0
        v[(v == 99) | (v == 0) | v.isin(list(P.MISSING_CODES))] = np.nan
        df['fr'] = v
        df = df.drop(columns='raw')
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values(['st', 'date']).reset_index(drop=True), notes
    else:
        d = raw_df.copy()
        if var == 'TX':
            names = {0: 'tmin', 1: 'tmax', 2: 'tmean'}
            d['comp'] = d['tj'].map(names)
        elif var in ('TT', 'RS', 'RH'):
            pref = {'TT': 'tt', 'RS': 'td', 'RH': 'rh'}[var]
            d['comp'] = pref + d['term'].str[-2:]
        elif var == 'FM':
            d['comp'] = d['tj'].map({0: 'fm_n', 1: 'fm_d', 2: 'fm_s'})
        elif var == 'TS':
            d['comp'] = d['tj'].map({0: 'ts00', 1: 'ts12', 2: 'ts_empty'})
        df = d.pivot_table(index=['st', 'date'], columns='comp', values='raw',
                           aggfunc='first').reset_index()
        df.columns.name = None
        comps = [c for c in df.columns if c not in ('st', 'date')]
        if 'ts_empty' in df.columns:
            df = df.drop(columns='ts_empty')
            comps.remove('ts_empty')
        for c in comps:
            x = df[c].values
            if var in ('TT', 'TX'):
                df[c] = _nan_codes(x) / P.T_SCALE
            elif var == 'RS':
                y = _nan_codes(x, extra=(P.RS_MISSING_CODE,))
                df[c] = y / P.RS_SCALE
                notes_n = int(np.sum(x == P.RS_MISSING_CODE))
                if notes_n:
                    notes.append(f'{c}: код {P.RS_MISSING_CODE} — {notes_n} значений, переведены в пропуск')
            elif var == 'TS':
                df[c] = _nan_codes(x) / P.TS_SCALE
            else:
                df[c] = _nan_codes(x)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    n_bad = int(df['date'].isna().sum())
    df = df.dropna(subset=['date'])       # 31-е числа коротких месяцев в TT/TX
    if n_bad:
        notes.append(f'Отброшено {n_bad} строк с несуществующими датами (дополнение месяца до 31 дня)')
    df.columns.name = None
    return df.sort_values(['st', 'date']).reset_index(drop=True), notes


def load_dataset(folder, rr_layout=P.RR_LAYOUT, progress=None):
    """
    Читает все файлы из папки. Возвращает dict:
      data[var]  — DataFrame [st, date, region, компоненты]
      files      — реестр файлов (DataFrame)
      notes      — замечания при чтении
      index_files — найденные справочники станций
    """
    files, index_files = scan_folder(folder)
    out, notes, reg = {}, [], []
    by_var = {}
    for k, rec in enumerate(files):
        if progress:
            progress(k / max(1, len(files)), rec['name'])
        try:
            if rec['var'] == 'FR':
                raw = parse_fr_file(rec['path'])
                stats = {'lines': raw['st'].nunique(), 'split': 0, 'fixed': 0}
            else:
                raw, stats = parse_block_file(rec['path'], rec['var'])
        except Exception as e:  # noqa
            notes.append(f'{rec["name"]}: ошибка чтения — {e}')
            continue
        raw['region'] = rec['region']
        by_var.setdefault(rec['var'], []).append(raw)
        reg.append(dict(Файл=rec['name'], Переменная=rec['var'], Регион=rec['region'],
                        Год=rec['year'], Строк=stats['lines'],
                        **{'Прочитано по ширине поля': stats['fixed']},
                        Дубликаты='; '.join(os.path.basename(p) for p in rec['duplicates'])))
    for var, parts in by_var.items():
        raw = pd.concat(parts, ignore_index=True)
        region_map = raw.groupby('st')['region'].first()
        if var == 'FR':
            df, nts = build_variable(var, raw.drop(columns='region'), rr_layout)
        else:
            df, nts = build_variable(var, raw.drop(columns='region'), rr_layout)
        df['region'] = df['st'].map(region_map)
        out[var] = df
        notes += [f'{var}: {n}' for n in nts]
    return dict(data=out, files=pd.DataFrame(reg), notes=notes, index_files=index_files)


# ─── Справочник станций ──────────────────────────────────────────────────
def parse_index_file(path):
    """
    Читает справочник станций (index_nsk, index_omsk). Ищет в строке код
    станции (5 цифр), название и, если есть, координаты/высоту.
    Возвращает DataFrame [st, name, lat, lon, alt].
    """
    rows = []
    for enc in ('cp1251', 'utf-8'):
        try:
            lines = open(path, encoding=enc).read().splitlines()
            break
        except UnicodeDecodeError:
            continue
    for line in lines:
        s = line.strip()
        if not s or s.startswith('!') or s.startswith('#'):
            continue
        toks = s.split()
        st, name, nums = None, [], []
        for t in toks:
            if st is None and re.fullmatch(r'\d{5}', t):
                st = int(t)
                continue
            if st is None:
                continue
            if re.search(r'[A-Za-zА-Яа-яЁё]', t):
                name.append(t.strip('.,;'))
            else:
                try:
                    nums.append((float(t.replace(',', '.')), ('.' in t or ',' in t)))
                except ValueError:
                    pass
        if st is None:
            continue
        dec = [v for v, d in nums if d]
        ints = [v for v, d in nums if not d]
        lat = next((v for v in dec if 40 <= v <= 75), np.nan)
        lon = next((v for v in dec if 20 <= v <= 180 and v is not lat), np.nan)
        if len(dec) >= 2 and 40 <= dec[0] <= 75:
            lat, lon = dec[0], dec[1]
        alt = next((v for v in ints if -50 <= v <= 5000), np.nan)
        nm = ' '.join(name).strip('.').replace('_', ' ')
        if nm.isupper():                       # «УСТЬ-ТАРКА» → «Усть-Тарка», «УЧЕБНАЯ ГМС» → «Учебная ГМС»
            nm = ' '.join(w if w in ('ГМС', 'ГМО', 'АМСГ', 'ОГМС', 'ГМБ') else
                          '-'.join(p.capitalize() for p in w.split('-')) for w in nm.split())
            nm = nm.replace('О.дальний', 'о. Дальний')
        rows.append(dict(st=st, name=nm or str(st), lat=lat, lon=lon, alt=alt))
    df = pd.DataFrame(rows).drop_duplicates('st')
    # целое число в конце строки index_* — код региона (одинаковый у всех станций), а не высота
    if len(df) > 3 and df['alt'].nunique(dropna=True) <= 1:
        df['alt'] = np.nan
    return df


def degmin_to_decimal(x):
    """Пересчёт координат из «градусы.минуты» (55.02 = 55°02′) в десятичные градусы."""
    x = np.asarray(x, float)
    deg = np.trunc(x)
    return deg + (x - deg) * 100 / 60


def build_station_table(index_files, data, user_table=None):
    """
    Объединяет справочники index_* с перечнем станций в данных и (при
    наличии) с пользовательской таблицей координат. Возвращает DataFrame
    [st, name, region, lat, lon, alt, vars].
    """
    parts = [parse_index_file(p) for p in index_files]
    ref = pd.concat(parts, ignore_index=True).drop_duplicates('st') if parts else \
        pd.DataFrame(columns=['st', 'name', 'lat', 'lon', 'alt'])
    present = {}
    region = {}
    for var, df in data.items():
        for st, rg in df.groupby('st')['region'].first().items():
            present.setdefault(st, []).append(var)
            region[st] = rg
    allst = sorted(set(present) | set(ref['st']))
    t = pd.DataFrame({'st': allst})
    t = t.merge(ref, on='st', how='left')
    t['region'] = t['st'].map(region)
    t['vars'] = t['st'].map(lambda s: ', '.join(sorted(present.get(s, []))))
    t['name'] = t['name'].fillna(t['st'].astype(str))
    if user_table is not None and not user_table.empty:
        u = user_table.copy()
        u = u[[c for c in ['st', 'name', 'lat', 'lon', 'alt', 'region'] if c in u.columns]]
        t = t.set_index('st')
        u = u.set_index('st')
        for c in u.columns:
            if c not in t.columns:
                t[c] = np.nan
            vals = u[c].dropna()
            t.loc[t.index.intersection(vals.index), c] = vals.loc[t.index.intersection(vals.index)]
        t = t.reset_index()
    return t[['st', 'name', 'region', 'lat', 'lon', 'alt', 'vars']]
