"""
Генератор ТЕСТОВЫХ файлов в формате исходных данных (TT, TX, RR, FM, TS,
RS, RH, FR, index). Значения синтетические и служат только для проверки
работы программы (чтение, контроль качества, заполнение, расчёты, карты).
Для расчётов по существу используются реальные файлы составителя данных.

В данные намеренно внесены ошибки тех типов, что описаны в разделе 4.3
памятки: грубые ошибки TX, суммы осадков > 100 мм в холодный сезон, ветер в
км/ч на одной станции, круглые значения 70–90 на многих станциях в одну дату,
одиночный выброс 98 м/с, код 999 и Td > T, RH > 100 % в феврале–марте,
коды −90/+98 в TS, пропуски осадков по всей сети в сентябре–декабре.

Запуск: python tests/make_test_data.py <папка> [год_начала] [год_конца]
"""

import os
import sys
import calendar
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

# Тестовые станции (координаты условные, в пределах областей)
STATIONS = {
    'nsk': [(90001, 'ТестСевер', 57.1, 79.0, 110), (90002, 'ТестЗапад', 55.4, 76.3, 115),
            (90003, 'ТестЦентр', 55.0, 82.9, 160), (90004, 'ТестЮг', 53.7, 77.7, 120),
            (90005, 'ТестВосток', 54.5, 85.4, 260), (90006, 'ТестОзёрная', 54.3, 79.9, 125),
            (90007, 'ТестРечная', 55.9, 81.0, 135), (90008, 'ТестПредгорье', 54.2, 84.2, 230),
            (90009, 'ТестСтепь', 53.9, 80.8, 140), (90010, 'ТестБолото', 56.4, 77.6, 112)],
    'omsk': [(91001, 'ТестТара', 56.9, 74.4, 75), (91002, 'ТестОмск', 54.9, 73.4, 95),
             (91003, 'ТестЧерлак', 54.2, 74.8, 100), (91004, 'ТестИсилькуль', 54.9, 71.3, 120),
             (91005, 'ТестКалачинск', 55.0, 74.6, 105), (91006, 'ТестТевриз', 57.5, 72.4, 70)],
}
TERMS8 = ['0000', '0003', '0006', '0009', '0012', '0015', '0018', '0021']


def simulate(y0, y1):
    dates = pd.date_range(f'{y0}-01-01', f'{y1}-12-31')
    allst = [s for r in STATIONS.values() for s in r]
    n, m = len(dates), len(allst)
    doy = dates.dayofyear.values
    # общая аномалия (AR1) + локальная
    common = np.zeros(n)
    for i in range(1, n):
        common[i] = 0.8 * common[i - 1] + rng.normal(0, 2.2)
    out = {}
    for k, (st, nm, lat, lon, alt) in enumerate(allst):
        loc = np.zeros(n)
        for i in range(1, n):
            loc[i] = 0.6 * loc[i - 1] + rng.normal(0, 0.9)
        clim = 1.5 - 0.6 * (lat - 55) - 0.004 * (alt - 120) - 19.5 * np.cos(2 * np.pi * (doy - 18) / 365)
        tmean = clim + common * (1 + 0.05 * (lon - 78) / 5) + loc
        amp = 5 + 3 * np.sin(2 * np.pi * (doy - 80) / 365)
        tmax = tmean + amp + rng.normal(0, 0.8, n)
        tmin = tmean - amp + rng.normal(0, 0.8, n)
        hours = np.array([0, 3, 6, 9, 12, 15, 18, 21])
        # 00 ВСВ ≈ 07 ч местного: холодно; 09 ВСВ ≈ 16 ч: тепло
        shape = -np.cos(2 * np.pi * (hours + 7 - 15) / 24)
        tt = tmean[:, None] + (amp[:, None]) * 0.9 * shape[None, :] + rng.normal(0, 0.5, (n, 8))
        dep = np.clip(4 + 4 * np.sin(2 * np.pi * (doy - 100) / 365)[:, None] * (0.5 + 0.5 * shape[None, :]) +
                      rng.gamma(2, 1.2, (n, 8)), 0.3, 30)
        td = tt - dep
        out[st] = dict(tmean=tmean, tmax=tmax, tmin=tmin, tt=tt, td=td, lat=lat)
    # осадки: общее событие + локальная вероятность
    p_common = 0.35 + 0.15 * np.sin(2 * np.pi * (doy - 120) / 365)
    ev = rng.random(n) < p_common
    for st in out:
        wet_n = ev & (rng.random(n) < 0.7) | (rng.random(n) < 0.05)
        wet_d = ev & (rng.random(n) < 0.6) | (rng.random(n) < 0.05)
        scale = 2.0 + 3.0 * np.clip(np.sin(2 * np.pi * (doy - 110) / 365), 0, None)
        rn = np.where(wet_n, rng.gamma(0.8, scale), 0)
        rd = np.where(wet_d, rng.gamma(0.8, scale), 0)
        out[st]['rr_n'], out[st]['rr_d'] = rn, rd
        out[st]['fm_n'] = np.clip(np.round(rng.gamma(4, 1.6, n)), 0, 30)
        out[st]['fm_d'] = np.clip(np.round(rng.gamma(4.5, 1.8, n)), 0, 30)
        t = out[st]['tt']
        out[st]['ts00'] = np.round(t[:, 0] - rng.uniform(0, 2, n))
        out[st]['ts12'] = np.round(t[:, 4] + np.clip(6 * np.sin(2 * np.pi * (doy - 80) / 365), 0, None))
    return dates, out


def enc_rr(x):
    """мм → код: 0 мм → 1 («не было»), иначе round(x*10)+1 (десятые до 1 мм, далее целые мм)."""
    x = np.asarray(x, float)
    code = np.where(x < 0.05, 1, np.where(x < 1, np.round(x * 10) + 1, np.round(x) * 10 + 1))
    return code.astype(int)


def write_block(path, blocks):
    with open(path, 'w', encoding='cp1251', newline='\n') as f:
        for dates, terms, rows in blocks:
            f.write('date: ' + ' '.join(dates) + '\n')
            f.write('index ' + ' '.join(terms * len(dates)) + '\n')
            for st, vals in rows:
                f.write(f'{st:5d}' + ''.join(f'{int(v):5d}' if v == v else ' 9999' for v in vals) + '\n')


def main(folder, y0=2014, y1=2016):
    dates, S = simulate(y0, y1)
    di = {d: i for i, d in enumerate(dates)}
    for suf, sts in STATIONS.items():
        R = suf.upper()
        base = os.path.join(folder, R)
        for v in ['TT', 'TX', 'RR', 'FM', 'TS', 'RS', 'RH', 'FR']:
            os.makedirs(os.path.join(base, v), exist_ok=True)
        with open(os.path.join(base, f'index_{suf}'), 'w', encoding='cp1251') as f:
            f.write('! N  обл  индекс  название  широта  долгота  высота\n')
            for k, (st, nm, lat, lon, alt) in enumerate(sts, 1):
                f.write(f'{k:3d} {suf} {st} {nm}. {lat:.2f} {lon:.2f} {alt}\n')
        for year in range(y0, y1 + 1):
            blk = {v: [] for v in ['TT', 'TX', 'RR', 'FM', 'TS', 'RS', 'RH']}
            fr_lines = []
            for mo in range(1, 13):
                nd = calendar.monthrange(year, mo)[1]
                ds31 = [f'{year}-{mo:02d}-{d:02d}' for d in range(1, 32)]
                dsn = ds31[:nd]
                rows = {v: [] for v in blk}
                for st, nm, lat, lon, alt in sts:
                    s = S[st]
                    idx = [di[pd.Timestamp(year, mo, d)] for d in range(1, nd + 1)]
                    pad = 31 - nd
                    tt = np.round(s['tt'][idx] * 10).ravel().tolist() + [9999] * 8 * pad
                    tx = np.round(np.c_[s['tmin'][idx], s['tmax'][idx], s['tmean'][idx]] * 10).ravel().tolist() + [9999] * 3 * pad
                    rn, rd = s['rr_n'][idx], s['rr_d'][idx]
                    rr = np.c_[enc_rr(rn), enc_rr(rd), enc_rr(rn + rd)].ravel().tolist()
                    fm = np.c_[s['fm_n'][idx], s['fm_d'][idx], np.maximum(s['fm_n'][idx], s['fm_d'][idx])].ravel().tolist()
                    ts = np.c_[s['ts00'][idx], s['ts12'][idx], np.full(nd, 9999)].ravel().tolist()
                    td = np.round(s['td'][idx] * 10).ravel().tolist()
                    es = 6.112 * np.exp(17.62 * s['tt'][idx] / (243.12 + s['tt'][idx]))
                    e = 6.112 * np.exp(17.62 * s['td'][idx] / (243.12 + s['td'][idx]))
                    rh = np.clip(np.round(100 * e / es + 6), 5, 100).ravel().tolist()
                    rows['TT'].append([st, tt]); rows['TX'].append([st, tx]); rows['RR'].append([st, rr])
                    rows['FM'].append([st, fm]); rows['TS'].append([st, ts]); rows['RS'].append([st, td])
                    rows['RH'].append([st, rh])
                # ─ внесённые ошибки и пропуски ─
                def setv(v, st, day, comp, ncomp, val):
                    for r in rows[v]:
                        if r[0] == st:
                            r[1][(day - 1) * ncomp + comp] = val
                s0, s1, s2 = sts[0][0], sts[1][0], sts[2][0]
                if year == y0 and mo == 9:
                    setv('TX', s1, 25, 0, 3, -625)               # Tмин = −62,5
                if year == y0 + 1 and mo == 1:
                    setv('TX', s2, 19, 1, 3, 260)                # ошибка знака Tмакс
                    setv('TX', s0, 13, 0, 3, 999)                # Tмин = +99,9
                if year == y0 + 1 and mo == 3:
                    setv('RR', s0, 20, 0, 3, 1981); setv('RR', s0, 20, 2, 3, 1991)   # 198 мм ночью в марте
                if year == y0 + 1 and mo == 7:
                    setv('TX', s2, 11, 1, 3, 389)                # Tмакс 38,9 при обычной средней
                if year >= y0 + 1 and sts[1][0] == s1 and suf == 'nsk':
                    for d in (6, 17):                           # ветер в км/ч (Барабинск-подобный случай)
                        setv('FM', s1, d, 1, 3, 54); setv('FM', s1, d, 2, 3, 54)
                if year == y0 + 1 and mo == 1:
                    for st, *_ in sts[:6]:                      # круглые значения на многих станциях
                        setv('FM', st, 5, 0, 3, int(rng.choice([70, 80, 90])))
                        setv('FM', st, 5, 2, 3, 90)
                if year == y0 and mo == 6:
                    setv('FM', s2, 15, 1, 3, 98); setv('FM', s2, 15, 2, 3, 98)        # одиночный выброс
                    setv('TS', s0, 21, 0, 3, -90)
                    setv('RS', s1, 3, 2, 8, 999)
                    setv('RS', s1, 4, 3, 8, int(S[s1]['tt'][di[pd.Timestamp(year, 6, 4)], 3] * 10 + 60))  # Td > T
                if year == y0 + 1 and mo in (2, 3):
                    for r in rows['RH']:
                        r[1] = [min(1601, int(v * 1.5)) if i % 5 == 0 else v for i, v in enumerate(r[1])]
                if year == y0 and mo >= 9:                      # пропуск осадков по всей сети
                    for r in rows['RR']:
                        r[1] = [9999] * len(r[1])
                if year == y0 + 1 and mo == 5:                  # длинный пропуск на одной станции
                    for r in rows['TX']:
                        if r[0] == s2:
                            r[1] = [9999 if i < 3 * 20 else v for i, v in enumerate(r[1])]
                # случайные одиночные пропуски
                for v in ('TT', 'TX', 'FM'):
                    for r in rows[v]:
                        if rng.random() < 0.15:
                            k = int(rng.integers(0, len(r[1]) // (8 if v == 'TT' else 3)))
                            w = 8 if v == 'TT' else 3
                            for j in range(w):
                                r[1][k * w + j] = 9999
                for v in blk:
                    terms = TERMS8 if v in ('TT', 'RS', 'RH') else (['0000', '0012', '0024'])
                    ds = ds31 if v in ('TT', 'TX') else dsn
                    blk[v].append((ds, terms, [(st, vals) for st, vals in rows[v]]))
                # FR
                if 4 <= mo <= 10:
                    fr_lines.append(f'Fact PO Nesterov {mo}.{year}')
                    for st, *_ in sts:
                        idx = [di[pd.Timestamp(year, mo, d)] for d in range(1, nd + 1)]
                        t = S[st]['tmax'][idx]; r = S[st]['rr_n'][idx] + S[st]['rr_d'][idx]
                        g, out = 0.0, []
                        for tv, rv, tdv in zip(t, r, S[st]['td'][idx, 4]):
                            g = 0 if rv > 3 else g + max(tv, 0) * max(tv - tdv, 0)
                            cls = 0 if tv < 0 else 1 + int(g > 300) + int(g > 1000) + int(g > 4000) + int(g > 10000)
                            out.append(cls)
                        fr_lines.append(f'{st:5d}' + ''.join(f'{c:3d}' for c in out))
            for v in blk:
                p = os.path.join(base, v, f'{v}{suf}{year}')
                write_block(p, blk[v])
            with open(os.path.join(base, 'FR', f'FR{suf}{year}'), 'w', encoding='cp1251') as f:
                f.write('\n'.join(fr_lines) + '\n')
    # одна строка RH с «*****» и слитыми числами
    p = os.path.join(folder, 'NSK', 'RH', f'RHnsk{y0}')
    L = open(p, encoding='cp1251').read().splitlines()
    for i, l in enumerate(L):
        if l.startswith(str(STATIONS['nsk'][3][0])):
            L[i] = l[:5] + '*****' + l[10:]
            break
    open(p, 'w', encoding='cp1251').write('\n'.join(L) + '\n')
    print('Тестовые данные записаны в', folder)


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else 'test_data'
    y0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2014
    y1 = int(sys.argv[3]) if len(sys.argv) > 3 else 2016
    main(folder, y0, y1)
