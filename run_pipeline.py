"""
=============================================================================
ПАКЕТНЫЙ ЗАПУСК ВСЕЙ ЦЕПОЧКИ БЕЗ ИНТЕРФЕЙСА
Чтение → проверка → очистка → восстановление пропусков → показатели →
тренды → районирование → карты → таблицы и рисунки в папке результатов.
Параметры методики — в params.py; параметры запуска — ниже или в
командной строке.

Пример:
  python run_pipeline.py "C:\\Данные\\2.13" --out results --stations stations.xlsx
=============================================================================
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import params as P
from modules import (readers, verify, qc, fill, derived, indices, trends, clustering, charts, maps,
                     basemap, interp, export)

# ─── Параметры запуска (можно менять здесь) ──────────────────────────────
ZONING_FEATURES = ['SU30', 'АЖ_случаев', 'Сумма10', 'ГТК', 'PRCPTOT', 'CDD', 'R10mm', 'Ветер15', 'FD']
MAP_INDICES = ['SU30', 'Сумма10', 'ГТК', 'CDD', 'PRCPTOT']
REGION_NAMES = P.REGION_NE     # регион в данных → субъект на карте (для других регионов дополните)
TREND_CODES = ['Tср_год', 'TXx', 'SU30', 'АЖ_случаев', 'PRCPTOT', 'CDD', 'RX1day', 'Сумма10', 'ГТК', 'Ветер15']


def save_fig(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, 'figures', f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_tab(df, out, name, dec=2):
    with open(os.path.join(out, 'tables', f'{name}.xlsx'), 'wb') as f:
        f.write(export.xlsx_bytes({'Таблица': df}, dec))


def main(src, out, stations_file=None, fill_data=True, dem=False):
    os.makedirs(os.path.join(out, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(out, 'figures'), exist_ok=True)
    print('1. Чтение данных…')
    res = readers.load_dataset(src)
    data = res['data']
    user = None
    if stations_file:
        from modules import data_io
        t = data_io.load_table(stations_file)
        cm = data_io.guess_station_column_map(t)
        user = pd.DataFrame({'st': pd.to_numeric(t[cm['Станция']], errors='coerce'),
                             'lat': t[cm['Lat']] if cm.get('Lat') else np.nan,
                             'lon': t[cm['Lon']] if cm.get('Lon') else np.nan}).dropna(subset=['st'])
    stations = readers.build_station_table(res['index_files'], data, user)
    names = dict(zip(stations['st'], stations['name']))
    save_tab(res['files'], out, 'table_00_реестр_файлов', 0)
    print('   ' + '\n   '.join(res['notes']))

    print('2. Проверка…')
    save_tab(verify.consistency_checks(data, stations), out, 'table_01_проверка_согласованности')
    comp = verify.completeness(data)
    save_tab(comp, out, 'table_02_полнота_по_станциям_и_годам', 1)
    save_tab(verify.network_gaps(data), out, 'table_03_пропуски_по_сети', 1)
    for v in data:
        save_fig(charts.completeness_heatmap(comp, v, names), out, f'fig_полнота_{v}')

    print('3. Очистка…')
    log = qc.run_qc(data, stations, {'isolation_forest': True})
    clean, flags = qc.apply_qc(data, log)
    summ = qc.qc_summary(log)
    save_tab(summ, out, 'table_04_контроль_качества_сводка', 0)
    save_tab(log[log['Основное']].drop(columns='Основное'), out, 'table_05_журнал_контроля_качества')
    if not summ.empty:
        save_fig(charts.qc_summary_bar(summ), out, 'fig_контроль_качества')

    use = clean
    if fill_data:
        print('4. Восстановление пропусков…')
        filled, flog, cv = fill.fill_dataset(clean, stations, progress=lambda f, n: print(f'   {n}'))
        save_tab(fill.fill_summary(flog, clean), out, 'table_06_восстановление_сводка', 0)
        if not cv.empty:
            save_tab(cv, out, 'table_07_проверка_методов_восстановления', 3)
            for c in cv['Компонент'].unique():
                save_fig(charts.cv_bars(cv, c), out, f'fig_методы_{c}')
        use = filled
        with open(os.path.join(out, 'суточные_ряды_восстановленные.xlsx'), 'wb') as f:
            f.write(export.xlsx_bytes(export.daily_export(filled, flags, flog), 1))

    print('5. Показатели…')
    hum = None
    if 'TT' in use and 'RS' in use:
        hum = derived.daily_humidity(derived.build_terms(use['TT'], use['RS']))
    D = indices.build_daily(use, hum)
    ann = indices.annual_indices(D)
    save_tab(ann, out, 'table_08_показатели_по_годам')
    sm = stations[['st', 'name']].merge(indices.station_means(ann), on='st')
    save_tab(sm.rename(columns={c: indices.label(c) for c in indices.INDEX_INFO}), out, 'table_09_показатели_по_станциям')
    he = indices.heat_events(D)
    save_tab(he, out, 'table_10_периоды_жары', 1)
    yrs = (int(ann['year'].min()), int(ann['year'].max()))
    for rg in sorted(D['region'].dropna().unique()):
        hs = {f'{yrs[0]}–{yrs[1]} гг.': {k: v for k, v in indices.heat_summary(he, D, yrs, rg).items() if k != 'Период'}}
        save_tab(pd.DataFrame(hs).rename_axis('Характеристика').reset_index(), out, f'table_11_жара_{rg}')
    for season, kind in [('тёплый', 'dry'), ('холодный', 'dry'), ('холодный', 'nosnow')]:
        spl, ok = indices.dry_spells(D, season, kind=kind)
        _, reg = indices.spell_class_table(spl, ok)
        if not reg.empty:
            save_tab(reg, out, f'table_12_{kind}_{season}')
            save_fig(charts.spell_classes_bar(reg, f'{"Бездождные периоды" if kind == "dry" else "Периоды без снегопада"}, '
                                                   f'{season} сезон'), out, f'fig_{kind}_{season}')
    if 'rr_n' in D:
        save_tab(indices.heavy_precip_events(D), out, 'table_13_сильные_осадки', 1)
    save_tab(indices.extremes_table(D, ann), out, 'table_14_экстремумы_повторяемость', 1)
    for code in TREND_CODES:
        if code in ann['Показатель'].values:
            save_fig(charts.index_series(ann, code, indices.label(code)), out, f'fig_ряд_{code}')

    print('6. Тренды…')
    tt = trends.trend_table(ann, TREND_CODES)
    if not tt.empty:
        tt['Показатель'] = tt['Показатель'].map(indices.label)
        save_tab(tt, out, 'table_15_тренды', 3)

    print('7. Районирование и карты…')
    xy = stations.dropna(subset=['lat', 'lon'])
    if len(xy) < 4:
        print('   Нет координат станций — районирование и карты пропущены.')
        return
    for rg in sorted(xy['region'].dropna().unique()):
        rname = REGION_NAMES.get(rg)
        if rname is None:
            print(f'   {rg}: нет соответствия субъекту на карте (params.REGION_NE) — карты пропущены')
            continue
        base = xy[xy['region'] == rg][['st', 'name', 'region', 'lat', 'lon', 'alt']].merge(
            indices.station_means(ann).drop(columns='region'), on='st')
        feats = [f for f in ZONING_FEATURES if f in base.columns and base[f].notna().sum() >= 4 and base[f].std() > 0]
        if len(base) < 5 or len(feats) < 2:
            continue
        dc, Xs, _ = clustering.prepare_matrix(base, feats)
        sil = clustering.silhouette_by_k(Xs, range(2, min(6, len(dc) - 1)))
        k = clustering.best_k(sil) or 2
        lab, s_ = clustering.run_clustering(Xs, k)
        dc['Кластер'] = clustering.relabel_clusters_by_feature(dc, lab, feats[0])
        save_tab(dc, out, f'table_17_районирование_{rg}')
        g = basemap.region_geometry(rname)
        bbox = basemap.bbox_of(g, pad=0.04)
        water = dict(rivers=basemap.ne_layer('rivers', bbox), lakes=basemap.ne_layer('lakes', bbox))
        dm = None
        if dem:
            try:
                dm = basemap.fetch_dem(bbox, 8)
            except Exception as e:
                print('   Рельеф не загружен:', e)
        save_fig(maps.map_schematic(dc, g, bbox, cluster_col='Кластер', title=f'Районирование: {rname} (k = {k})',
                                    rivers=water), out, f'map_районы_схема_{rg}')
        cells = maps.thiessen(dc, g)
        save_fig(maps.map_physical(bbox, g, dem=dm[0] if dm else None, dem_extent=dm[1] if dm else None, water=water,
                                   cities=basemap.ne_layer('cities', bbox), st=dc,
                                   clusters=[{'geometry': c, 'cluster': kk} for c, kk in zip(cells, dc['Кластер'])],
                                   title=f'Районирование: {rname}'), out, f'map_районы_физ_{rg}')
        LON, LAT = interp.make_grid(bbox, 0.05)
        msk = interp.mask_by_geometry(LAT, LON, g)
        for code in MAP_INDICES:
            if code not in base.columns or base[code].notna().sum() < 5:
                continue
            b2 = base.dropna(subset=[code])
            fld, _ = interp.regression_kriging(b2['lat'], b2['lon'], b2[code], LAT, LON)
            fld = np.where(msk, fld, np.nan)
            dec = 0 if indices.INDEX_INFO[code][1] in ('дни', 'случаи', '°C') else 2
            save_fig(maps.map_schematic(b2, g, bbox, value_col=code, value_label=indices.label(code), mode='field',
                                        field=fld, grid=(LON, LAT), decimals=dec, rivers=water,
                                        title=indices.label(code).split(',')[0]), out, f'map_{code}_{rg}')
    with open(os.path.join(out, 'для_climate_zoning_помесячно.xlsx'), 'wb') as f:
        f.write(export.zoning_xlsx(use, stations))
    print('Готово:', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Подготовка климатических данных и расчёт показателей')
    ap.add_argument('src', help='папка или ZIP с файлами TT, TX, RR, …')
    ap.add_argument('--out', default='results')
    ap.add_argument('--stations', default=None, help='таблица координат станций (.xlsx/.csv)')
    ap.add_argument('--no-fill', action='store_true', help='не восстанавливать пропуски')
    ap.add_argument('--dem', action='store_true', help='загрузить рельеф (нужен интернет)')
    a = ap.parse_args()
    src = a.src
    if src.lower().endswith('.zip'):
        src = readers.extract_archive(open(src, 'rb').read())
    main(src, a.out, a.stations, not a.no_fill, a.dem)
