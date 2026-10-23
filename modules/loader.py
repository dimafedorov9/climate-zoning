"""
=============================================================================
ЕДИНАЯ ЗАГРУЗКА СУТОЧНЫХ ДАННЫХ
Принимает любые сочетания файлов и архивов (ZIP, RAR, 7z, в том числе
вложенные): файлы станционного формата (TT, TX, RR, FM, TS, RS, RH, FR,
справочники index_*) и таблицы Excel/CSV со сложной шапкой. Тип каждого
файла определяется автоматически, данные приводятся к единому виду и
объединяются; для совпадающих станций и дат выполняется сверка источников.
=============================================================================
"""

import os
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

from . import readers, xlsx_daily

ARCHIVES = ('.zip', '.rar', '.7z')
TABLES = ('.xlsx', '.xlsm', '.xls', '.csv')

# Обозначения регионов по словам в именах файлов и папок
REGION_HINTS = [(r'омск|omsk', 'Омская обл.'), (r'новосиб|nsk|нсо', 'НСО')]


def _write(path, b):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b)


def unpack(uploads, target=None):
    """
    uploads: список (имя, байты). Архивы распаковываются рекурсивно в
    target; прочие файлы копируются. Возвращает путь к папке.
    """
    target = target or tempfile.mkdtemp(prefix='ca_in_')
    stack = [(target, nm, b) for nm, b in uploads]
    while stack:
        base, nm, b = stack.pop()
        low = nm.lower()
        if low.endswith('.zip'):
            sub = os.path.join(base, os.path.splitext(os.path.basename(nm))[0])
            readers.extract_archive(b, sub)
            # вложенные RAR/7z внутри ZIP
            for root, _, names in os.walk(sub):
                for n in names:
                    if n.lower().endswith(('.rar', '.7z')):
                        p = os.path.join(root, n)
                        stack.append((root, n, open(p, 'rb').read()))
                        os.remove(p)
        elif low.endswith(('.rar', '.7z')):
            sub = os.path.join(base, os.path.splitext(os.path.basename(nm))[0])
            for inner, data in xlsx_daily.extract_other_archive(b, nm):
                if inner.lower().endswith(ARCHIVES):
                    stack.append((sub, inner, data))
                else:
                    dest = os.path.normpath(os.path.join(sub, inner))
                    if dest.startswith(os.path.normpath(sub)):
                        _write(dest, data)
        else:
            _write(os.path.join(base, os.path.basename(nm)), b)
    return target


def classify(folder):
    """Делит файлы папки на станционный формат, справочники и таблицы."""
    station, tables, other = [], [], []
    for root, _, names in os.walk(folder):
        for n in names:
            p = os.path.join(root, n)
            if readers.FILE_RE.match(n) or n.lower().startswith('index'):
                station.append(p)
            elif n.lower().endswith(TABLES) and not n.startswith('~$'):
                tables.append(p)
            else:
                other.append(p)
    return station, tables, other


def guess_region(path_or_name, default=None):
    s = str(path_or_name).lower()
    for pat, rg in REGION_HINTS:
        if re.search(pat, s):
            return rg
    return default


def compare_sources(main, extra):
    """
    Сверка значений для совпадающих станций и дат из двух источников.
    Возвращает таблицу [Переменная, Компонент, Совпадающих суток,
    Средняя абсолютная разность, Доля расхождений > допуска, %].
    """
    tol = {'tmin': 0.5, 'tmax': 0.5, 'tmean': 0.5, 'rr_n': 0.5, 'rr_d': 0.5, 'rr_s': 0.5}
    rows = []
    for var, d2 in extra.items():
        if var not in main:
            continue
        d1 = main[var]
        common = [c for c in d2.columns if c in d1.columns and c in tol]
        if not common:
            continue
        m = d1[['st', 'date'] + common].merge(d2[['st', 'date'] + common], on=['st', 'date'], suffixes=('_1', '_2'))
        for c in common:
            ok = m[[c + '_1', c + '_2']].notna().all(1)
            if not ok.any():
                continue
            diff = (m.loc[ok, c + '_1'] - m.loc[ok, c + '_2']).abs()
            rows.append({'Переменная': var, 'Компонент': c, 'Совпадающих суток': int(ok.sum()),
                         'Средняя абсолютная разность': diff.mean(),
                         'Расхождений больше допуска, %': 100 * (diff > tol[c]).mean()})
    return pd.DataFrame(rows)


def load_all(uploads=None, folder=None, table_region=None, rr_layout='auto', prefer='main', mapping=None,
             progress=None):
    """
    Загружает всё, что передано. Возвращает dict:
      data, files, notes, index_files — как readers.load_dataset;
      tables — список (имя, байты) табличных файлов; mapping — сопоставление
      заголовков; compare — сверка источников; summary — состав загрузки.
    """
    folder = folder or unpack(uploads or [])
    station, tables, other = classify(folder)
    notes, data, files, idx = [], {}, pd.DataFrame(), []
    if station:
        sdir = tempfile.mkdtemp(prefix='ca_st_')
        for p in station:
            rel = os.path.relpath(p, folder)
            _write(os.path.join(sdir, rel), open(p, 'rb').read())
        res = readers.load_dataset(sdir, rr_layout, progress=(lambda f, n: progress(0.6 * f, n)) if progress else None)
        data, files, notes, idx = res['data'], res['files'], list(res['notes']), res['index_files']
    tfiles, mp, comp = [], mapping, pd.DataFrame()
    if tables:
        tfiles = [(os.path.relpath(p, folder), open(p, 'rb').read()) for p in sorted(tables)]
        if mp is None:
            grid, merged = xlsx_daily._read_grid(tfiles[0][1], tfiles[0][0])
            names, _, _ = xlsx_daily.detect_header(grid, merged)
            mp = xlsx_daily.match_columns(names)
        daily, _, xnotes = xlsx_daily.load_tables(
            tfiles, mp, progress=(lambda f, n: progress(0.6 + 0.35 * f, n)) if progress else None)
        notes += xnotes
        if not daily.empty:
            # регион: по индексу станции в справочниках/файлах, иначе по имени файлов, иначе выбранный
            known = {}
            for d in data.values():
                known.update(d.groupby('st')['region'].first().to_dict())
            rg_auto = guess_region(' '.join(n for n, _ in tfiles), table_region)
            parts = []
            for st, g in daily.groupby('st'):
                rg = known.get(st) or rg_auto or table_region or 'Регион'
                parts.append((rg, g))
            extra = {}
            for rg, g in parts:
                for var, d in xlsx_daily.to_program_vars(g, rg).items():
                    extra.setdefault(var, []).append(d)
            extra = {v: pd.concat(ds, ignore_index=True) for v, ds in extra.items()}
            comp = compare_sources(data, extra) if data else pd.DataFrame()
            data = xlsx_daily.merge_datasets(data, extra, prefer=prefer) if data else extra
            nm = daily.groupby('st')['name'].first() if 'name' in daily else pd.Series(dtype=str)
            rgs = ', '.join(sorted({rg for rg, _ in parts}))
            reg = pd.DataFrame([dict(Файл=n, Переменная='таблица', Регион=rgs) for n, _ in tfiles])
            files = pd.concat([files, reg], ignore_index=True)
            notes.append(f'Таблицы: станций {daily["st"].nunique()}, суток {len(daily)}, период '
                         f'{daily["date"].min():%d.%m.%Y}–{daily["date"].max():%d.%m.%Y}.')
        else:
            nm = pd.Series(dtype=str)
    else:
        nm = pd.Series(dtype=str)
    if other:
        notes.append('Не распознаны и не использованы: ' + ', '.join(os.path.basename(p) for p in other[:15])
                     + (' …' if len(other) > 15 else ''))
    summary = pd.DataFrame([
        {'Источник': 'Файлы станционного формата (TT, TX, RR …)', 'Файлов': len([p for p in station
                                                                                  if readers.FILE_RE.match(os.path.basename(p))])},
        {'Источник': 'Справочники станций (index_*)', 'Файлов': len(idx)},
        {'Источник': 'Таблицы Excel/CSV', 'Файлов': len(tfiles)},
        {'Источник': 'Не распознано', 'Файлов': len(other)}])
    if progress:
        progress(1.0, 'готово')
    return dict(data=data, files=files, notes=notes, index_files=idx, tables=tfiles, mapping=mp,
                compare=comp, summary=summary, table_names=nm)
