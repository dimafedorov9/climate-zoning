"""
=============================================================================
ПРОГРАММА ПОДГОТОВКИ КЛИМАТИЧЕСКИХ ДАННЫХ И ОЦЕНКИ КЛИМАТИЧЕСКИХ РИСКОВ
ДЛЯ РЕГИОНАЛЬНЫХ ПЛАНОВ АДАПТАЦИИ

Автор: Федоров Дмитрий Сергеевич, научный сотрудник отдела гидрометеорологии
и экологии ФГБУ «СибНИГМИ».

Цепочка: загрузка суточных данных → проверка → контроль качества →
восстановление пропусков → климатические показатели → тренды и сравнение
периодов → районирование → карты → экспорт. Регионы обрабатываются
раздельно. Модули оформления, кластеризации и загрузки границ общие с
программой климатического районирования (climatezoning-sibnigmi.streamlit.app).

Запуск: streamlit run app.py
=============================================================================
"""

import os
import io
import sys
import json
import pickle
import re
import tempfile
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import params as P
from config import cf
from modules import (loader, xlsx_daily, readers, verify, qc, fill, derived, indices, trends, clustering, viz,
                     charts, maps, basemap, interp, mapping, export)

APP_TITLE = 'Оценка климатических рисков для региональных планов адаптации'
st.set_page_config(page_title=APP_TITLE, layout='wide')

S = st.session_state
DEFAULTS = dict(districts_all=None, x_map=None, x_files=None, x_uploads=None, raw=None, stations=None, checks=None, comp=None, gaps=None, qc_log=None, clean=None,
                flags=None, filled=None, fill_log=None, cv=None, D=None, ann=None, events=None,
                zoning=None, districts=None, water=None, dem=None, figs={}, tables={})
for k, v in DEFAULTS.items():
    if k not in S:
        S[k] = v if not isinstance(v, dict) else dict(v)
# критерии и сезоны хранятся в сессии пользователя (у каждого пользователя свои)
import copy
if 'criteria' not in S:
    S.criteria = copy.deepcopy(P.CRITERIA)
if 'seasons' not in S:
    S.seasons = copy.deepcopy(P.SEASONS)
P.CRITERIA = S.criteria
P.SEASONS = S.seasons
MONTHS_RU = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь',
             'ноябрь', 'декабрь']
ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']


def season_txt(rg, key):
    a, b = P.seasons_for(rg)[key]
    return f'{ROMAN[a - 1]}–{ROMAN[b - 1]}'

VAR_LABEL = {k: readers.var_label(k) for k in readers.VAR_INFO}
COMP_LABEL = {'tmin': 'Минимальная температура (Tмин)', 'tmax': 'Максимальная температура (Tмакс)',
              'tmean': 'Средняя суточная температура (Tср)', 'rr_n': 'Осадки за ночь', 'rr_d': 'Осадки за день',
              'rr_s': 'Осадки за сутки', 'fm_n': 'Максимальный ветер за ночь', 'fm_d': 'Максимальный ветер за день',
              'fm_s': 'Максимальный ветер за сутки', 'ts00': 'Температура поверхности почвы, 00 ВСВ',
              'ts12': 'Температура поверхности почвы, 12 ВСВ', 'fr': 'Класс пожарной опасности'}
for h in derived.HOURS:
    COMP_LABEL[f'tt{h}'] = f'Температура воздуха (T), {h} ВСВ'
    COMP_LABEL[f'td{h}'] = f'Температура точки росы (Td), {h} ВСВ'
    COMP_LABEL[f'rh{h}'] = f'Относительная влажность (RH), {h} ВСВ'
GLOSSARY = [(k, v['name'], v['unit']) for k, v in readers.VAR_INFO.items()] + [
    ('Tмин, Tмакс, Tср', 'Минимальная, максимальная и средняя суточная температура воздуха', '°C'),
    ('Td', 'Температура точки росы', '°C'),
    ('ВСВ', 'Всемирное скоординированное время; сроки наблюдений 00, 03, …, 21 ВСВ', ''),
    ('ОЯ, НЯ', 'Опасное и неблагоприятное (комплекс неблагоприятных) гидрометеорологическое явление', ''),
    ('АЖ, СЖ', 'Аномально жаркая погода и сильная жара (критерии ОЯ)', ''),
    ('АХ', 'Аномально холодная погода (критерий ОЯ)', ''),
    ('ГТК', 'Гидротермический коэффициент Селянинова', ''),
    ('ГСОП', 'Градусо-сутки отопительного периода', '°C·сут'),
]


def pretty_codes(df):
    """Коды переменных и компонентов в таблицах заменяются полными названиями с обозначением."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    df = df.copy()
    if 'Переменная' in df and df['Переменная'].isin(list(VAR_LABEL)).any():
        df['Переменная'] = df['Переменная'].map(lambda v: VAR_LABEL.get(v, v))
    if 'Компонент' in df and df['Компонент'].isin(list(COMP_LABEL)).any():
        df['Компонент'] = df['Компонент'].map(lambda v: COMP_LABEL.get(v, v))
    return df


# ─── Вспомогательные функции интерфейса ─────────────────────────────────
ZONING_URL = 'https://climatezoning-sibnigmi.streamlit.app'
AUTHOR = ('Автор: Федоров Дмитрий Сергеевич, научный сотрудник отдела гидрометеорологии и экологии '
          'ФГБУ «СибНИГМИ»')

with st.sidebar:
    st.markdown('**Отображение**')
    FIG_W = st.slider('Ширина графиков, % окна', 25, 100, 45, 5,
                      help='Размер на экране. Файлы PNG и PDF сохраняются в исходном размере, 300 dpi.')
    MAP_W = st.slider('Ширина карт, % окна', 30, 100, 60, 5)
    st.markdown('**Обработка**')
    P.NEIGHBOR_SAME_REGION = st.checkbox('Соседние станции — только из того же региона',
                                         value=P.NEIGHBOR_SAME_REGION,
                                         help='Используется при контроле качества и восстановлении пропусков. '
                                              'Снимите флажок, чтобы учитывать и станции соседних регионов '
                                              '(полезно для станций у границы).')
    st.markdown('---')
    st.markdown(f'Программа климатического районирования: [{ZONING_URL.replace("https://", "")}]({ZONING_URL})')


def footnote(text):
    """Сноска под таблицей или графиком."""
    st.caption(f'ⓘ {text}')


def show_df(df, decimals=2, height=None):
    if df is None or len(df) == 0:
        st.caption('Нет данных.')
        return
    kw = dict(height=height) if height else {}
    df = pretty_codes(df)
    st.dataframe(export.comma_frame(df, decimals), width='stretch', hide_index=True, **kw)


def save_table(name, df, decimals=2):
    """Запоминает таблицу для итогового архива и показывает кнопку скачивания."""
    df = pretty_codes(df)
    S.tables[name] = (df, decimals)
    st.download_button(f'Скачать {name}.xlsx', export.xlsx_bytes({'Таблица': df}, decimals),
                       file_name=f'{name}.xlsx', key=f'dl_{name}')


def show_fig(fig, name, wide=False, kind_='chart'):
    """График на экране (ширина — из боковой панели) + кнопки PNG/PDF."""
    import matplotlib.pyplot as plt
    S.figs[name] = fig
    w = (MAP_W if kind_ == 'map' else min(100, FIG_W * (1.6 if wide else 1))) / 100
    c = st.columns([w, 1 - w])[0] if w < 0.999 else st.container()
    with c:
        st.pyplot(fig, width='stretch')
        a, b = st.columns(2)
        a.download_button('PNG, 300 dpi', export.fig_bytes(fig, 'png'), file_name=f'{name}.png', key=f'png_{name}')
        b.download_button('PDF', export.fig_bytes(fig, 'pdf'), file_name=f'{name}.pdf', key=f'pdf_{name}')
    plt.close(fig)


def station_names():
    if S.stations is None:
        return {}
    return dict(zip(S.stations['st'], S.stations['name']))


_NAMES = {}


def refresh_names():
    global _NAMES
    _NAMES = station_names()


def st_label(s):
    return f'{_NAMES.get(s, s)} ({s})'


def regions():
    """Регионы, представленные в загруженных данных (в порядке справочника станций)."""
    if S.stations is None:
        return []
    return [r for r in pd.unique(S.stations['region'].dropna().astype(str))]


def region_tabs():
    """Вкладки по регионам: каждый регион анализируется отдельно."""
    rs = regions()
    if len(rs) <= 1:
        return [(r, st.container()) for r in rs]
    return list(zip(rs, st.tabs(rs)))


def region_data(data, rg):
    return {v: d[d['region'] == rg] for v, d in data.items()}


def region_stations(rg):
    return S.stations[S.stations['region'] == rg]


def numeric_stations(t):
    t = t.copy()
    for c in ('lat', 'lon', 'alt'):
        if c in t:
            t[c] = pd.to_numeric(t[c].astype(str).str.replace(',', '.'), errors='coerce')
    t['st'] = pd.to_numeric(t['st'], errors='coerce').astype(int)
    return t


def apply_regions():
    """Регион станции из справочника переносится во все переменные данных."""
    if not S.raw or S.stations is None:
        return
    rmap = dict(zip(S.stations['st'], S.stations['region']))
    for d in S.raw['data'].values():
        d['region'] = d['st'].map(rmap).fillna(d['region'])


def current_data(prefer='filled'):
    if prefer == 'filled' and S.filled is not None:
        return S.filled
    if S.clean is not None:
        return S.clean
    return S.raw['data'] if S.raw else None


def stations_xy():
    t = S.stations
    if t is None:
        return None
    t = numeric_stations(t)
    return t.dropna(subset=['lat', 'lon'])


refresh_names()
st.title(APP_TITLE)
st.markdown(f'<div style="color:#555;margin-top:-0.6rem">{AUTHOR}</div>', unsafe_allow_html=True)
st.caption('Подготовка суточных данных метеостанций (проверка, контроль качества, восстановление пропусков), '
           'расчёт показателей климатических рисков, районирование и карты')

tabs = st.tabs(['1. Данные', '2. Проверка', '3. Контроль качества', '4. Пропуски', '5. Показатели',
                '6. Тренды и периоды', '7. Районирование', '8. Карты', '9. Экспорт'])

# ═══════════════════════════ 1. ДАННЫЕ ═══════════════════════════════════
with tabs[0]:
    st.header('Загрузка суточных данных')
    st.markdown('Загрузите файлы или архивы в любом сочетании — программа сама определит их тип, прочитает, '
                'сверит и объединит:')
    st.markdown('- файлы станционного формата: ' + '; '.join(f'{VAR_LABEL[k]}' for k in readers.VAR_INFO
                                                            if k != 'DY') + '; справочники станций `index_*`;\n'
                '- суточные таблицы Excel/CSV по станциям с многострочной шапкой (значения вида «+12.3», «12,3», '
                '«<0.5», «/», «20 км» приводятся к числам);\n'
                '- архивы ZIP, RAR, 7z, в том числе вложенные.')
    with st.expander('Обозначения переменных и сокращения', expanded=False):
        st.dataframe(pd.DataFrame(GLOSSARY, columns=['Обозначение', 'Расшифровка', 'Единицы']),
                     width='stretch', hide_index=True)
    ups = st.file_uploader('Файлы и архивы с данными', accept_multiple_files=True,
                           type=['zip', 'rar', '7z', 'xlsx', 'xls', 'csv', 'txt', 'dat'])
    known = list(dict.fromkeys(regions() + list(P.CRITERIA)))
    with st.expander('Параметры чтения', expanded=False):
        c1, c2 = st.columns(2)
        x_reg_sel = c1.selectbox('Регион станций из таблиц, если его нельзя определить автоматически',
                                 ['Определить автоматически'] + known + ['Другой…'], key='x_region_sel',
                                 help='Регион таблицы определяется по индексу станции (если она есть в '
                                      'справочниках станционных файлов) или по имени файла.')
        x_region = (c1.text_input('Название региона', key='x_region_new') if x_reg_sel == 'Другой…'
                    else (None if x_reg_sel == 'Определить автоматически' else x_reg_sel))
        x_mode = c2.radio('При совпадении станции и даты в обоих источниках', ['main', 'tables'], key='x_mode',
                          format_func={'main': 'Приоритет у файлов станционного формата',
                                       'tables': 'Приоритет у таблиц Excel/CSV'}.get)
        rr_layout = c2.selectbox('Раскладка значений в файлах осадков (RR)', ['auto', 'day', 'block'],
                                 format_func={'auto': 'Определить автоматически',
                                              'day': 'Ночь, день, сутки по датам',
                                              'block': 'Блоками по месяцу'}.get)

    def _read(mapping=None, uploads=None):
        bar = st.progress(0.0, text='Чтение файлов…')
        test_dir = os.environ.get('CA_TEST_DIR')     # для автоматических тестов программы
        if not uploads and test_dir:
            uploads = [(n, open(os.path.join(test_dir, n), 'rb').read()) for n in sorted(os.listdir(test_dir))
                       if os.path.isfile(os.path.join(test_dir, n))]
        res = loader.load_all(uploads=uploads,
                              table_region=x_region, rr_layout=rr_layout, prefer=x_mode, mapping=mapping,
                              progress=lambda f, n: bar.progress(min(1.0, f), text=f'Чтение: {n}'))
        bar.empty()
        if not res['data']:
            st.error('Данные не найдены: среди загруженных файлов нет файлов станционного формата или таблиц.')
            return
        S.raw = res
        S.x_map = res['mapping']
        stt = readers.build_station_table(res['index_files'], res['data'])
        nm = res['table_names']
        if len(nm):
            stt['name'] = np.where(stt['name'].astype(str) == stt['st'].astype(str),
                                   stt['st'].map(nm).fillna(stt['name']), stt['name'])
        S.stations = numeric_stations(stt)
        refresh_names()
        for k in ['checks', 'qc_log', 'clean', 'flags', 'filled', 'fill_log', 'cv', 'D', 'ann', 'events', 'zoning']:
            S[k] = None

    if st.button('Прочитать данные', type='primary'):
        if not ups and not os.environ.get('CA_TEST_DIR'):
            st.error('Загрузите файлы или архивы.')
        else:
            try:
                _read(uploads=[(u.name, u.getvalue()) for u in ups] if ups else None)
                S.x_uploads = [(u.name, u.getvalue()) for u in ups] if ups else None
            except Exception as e:
                st.error(f'Ошибка чтения: {e}')
    if S.raw:
        res = S.raw
        st.subheader('Состав загруженных данных')
        if 'summary' in res:
            show_df(res['summary'], 0)
        nv = pd.DataFrame([{'Переменная': v, 'Станций': d['st'].nunique(), 'Суток': len(d),
                            'Начало': d['date'].min().strftime('%d.%m.%Y'), 'Конец': d['date'].max().strftime('%d.%m.%Y'),
                            'Регионы': ', '.join(sorted(d['region'].dropna().astype(str).unique()))}
                           for v, d in res['data'].items()])
        show_df(nv, 0)
        if res.get('compare') is not None and len(res['compare']):
            st.markdown('**Сверка источников** (совпадающие станции и даты в файлах станционного формата и в таблицах)')
            show_df(res['compare'], 2)
            footnote('Средняя абсолютная разность и доля суток, где значения источников расходятся больше чем на '
                     '0,5 °C (температура) или 0,5 мм (осадки). Заметные расхождения Tмин и Tмакс обычно связаны с '
                     'разными интервалами определения экстремумов (полусутки или сутки). В объединённый ряд '
                     'берётся значение источника, выбранного в параметрах чтения; второй дополняет пропуски.')
        if res.get('tables'):
            with st.expander('Сопоставление заголовков таблиц с показателями', expanded=False):
                st.caption('Сопоставление построено автоматически по первой таблице. Если показатель определён '
                           'неверно, исправьте его и прочитайте данные заново.')
                opts = ['—'] + [r[2] for r in xlsx_daily.RULES]
                mp = st.data_editor(S.x_map, hide_index=True, width='stretch', key='x_map_ed',
                                    column_config={'Показатель': st.column_config.SelectboxColumn(options=opts),
                                                   'Столбец': st.column_config.NumberColumn(disabled=True),
                                                   'Заголовок': st.column_config.TextColumn(disabled=True),
                                                   'Описание': st.column_config.TextColumn(disabled=True)})
                if st.button('Прочитать заново с этим сопоставлением'):
                    try:
                        mp = mp.assign(Описание=mp['Показатель'].map(xlsx_daily.STD_LABEL).fillna('не используется'))
                        _read(mapping=mp, uploads=S.get('x_uploads'))
                        st.rerun()
                    except Exception as e:
                        st.error(f'Ошибка чтения: {e}')
        st.subheader('Реестр файлов')
        show_df(res['files'], 0, height=240)
        if res['notes']:
            with st.expander('Замечания при чтении', expanded=False):
                for n in res['notes']:
                    st.markdown(f'- {n}')
        st.subheader('Справочник станций')
        st.markdown('Координаты нужны для пространственной проверки, интерполяции и карт. Их можно '
                    'ввести в таблице или загрузить файлом (столбцы: индекс станции, широта, долгота, '
                    'высота).')
        coord = st.file_uploader('Файл с координатами (.xlsx, .csv)', type=['xlsx', 'csv'], key='coordfile')
        if coord is not None:
            try:
                from modules import data_io
                tab = data_io.load_table(coord)
                if isinstance(tab, dict):
                    tab = list(tab.values())[0]
                cmap_ = data_io.guess_station_column_map(tab)
                u = pd.DataFrame({'st': pd.to_numeric(tab[cmap_['Станция']], errors='coerce')})
                for k_from, k_to in [('Lat', 'lat'), ('Lon', 'lon'), ('Название', 'name')]:
                    if cmap_.get(k_from):
                        u[k_to] = tab[cmap_[k_from]].values
                alt_col = next((c for c in tab.columns if 'выс' in str(c).lower() or 'alt' in str(c).lower()), None)
                if alt_col:
                    u['alt'] = pd.to_numeric(tab[alt_col], errors='coerce').values
                S.stations = numeric_stations(readers.build_station_table(res['index_files'], res['data'],
                                                                          u.dropna(subset=['st'])))
                st.success('Координаты добавлены.')
            except Exception as e:
                st.error(f'Не удалось прочитать файл координат: {e}')
        degmin = st.checkbox('Координаты записаны в формате «градусы.минуты» — пересчитать в десятичные')
        st.caption('Регионы в данных: ' + ', '.join(f'{r} ({int((S.stations["region"] == r).sum())} ст.)'
                                                      for r in regions()) +
                   '. Регион определяется по имени файла (…nsk… — НСО, …omsk… — Омская обл., иначе — '
                   'буквенная часть имени), для таблиц Excel/CSV — по индексу станции или по параметрам чтения; '
                   'его можно изменить в таблице.')
        ed = st.data_editor(numeric_stations(S.stations), width='stretch', hide_index=True, key='st_editor',
                            column_config={'st': st.column_config.NumberColumn('Индекс', disabled=True, format='%d'),
                                           'name': 'Название', 'region': 'Регион',
                                           'lat': st.column_config.NumberColumn('Широта', format='%.2f'),
                                           'lon': st.column_config.NumberColumn('Долгота', format='%.2f'),
                                           'alt': st.column_config.NumberColumn('Высота, м', format='%.0f'),
                                           'vars': st.column_config.TextColumn('Переменные', disabled=True)})
        if st.button('Сохранить справочник'):
            ed = ed.copy()
            if degmin:
                ed['lat'] = readers.degmin_to_decimal(ed['lat'])
                ed['lon'] = readers.degmin_to_decimal(ed['lon'])
            S.stations = numeric_stations(ed)
            apply_regions()
            refresh_names()
            for k in ['checks', 'qc_log', 'clean', 'flags', 'filled', 'fill_log', 'cv', 'D', 'ann', 'events', 'zoning']:
                S[k] = None
            st.success('Справочник сохранён. Результаты следующих этапов сброшены — выполните их заново.')

        st.subheader('Границы сезонов по регионам')
        st.markdown('Сроки сезонов зависят от климата региона и используются при контроле качества, расчёте '
                    'полноты и показателей. Значения по умолчанию заданы для юга Западной Сибири; для другого '
                    'региона проверьте и при необходимости измените их.')
        for rg in regions():
            P.seasons_for(rg)
        rows = []
        for key, lab in P.SEASON_LABELS.items():
            r = {'Код': key, 'Сезон': lab}
            for rg in regions():
                a_, b_ = P.SEASONS[rg][key]
                r[f'{rg}: начало'] = MONTHS_RU[a_ - 1]
                r[f'{rg}: окончание'] = MONTHS_RU[b_ - 1]
            rows.append(r)
        cc = {'Код': None, 'Сезон': st.column_config.TextColumn(disabled=True, width='large')}
        for rg in regions():
            for part in ('начало', 'окончание'):
                cc[f'{rg}: {part}'] = st.column_config.SelectboxColumn(options=MONTHS_RU, required=True)
        sez = st.data_editor(pd.DataFrame(rows), hide_index=True, width='stretch', key='season_ed',
                             column_config=cc)
        changed = False
        for _, r in sez.iterrows():
            for rg in regions():
                v = (MONTHS_RU.index(r[f'{rg}: начало']) + 1, MONTHS_RU.index(r[f'{rg}: окончание']) + 1)
                if P.SEASONS[rg][r['Код']] != v:
                    P.SEASONS[rg][r['Код']] = v
                    changed = True
        if changed:
            for k in ['checks', 'qc_log', 'clean', 'flags', 'filled', 'fill_log', 'cv', 'D', 'ann', 'events', 'zoning']:
                S[k] = None
            st.info('Границы сезонов изменены. Результаты следующих этапов сброшены — выполните их заново.')
        footnote('Сезон, переходящий через новый год (например, ноябрь–март), относится к году окончания: '
                 'зима 2016/17 г. — 2017 г. Отопительный год используется только для расчёта ГСОП.')

# ═══════════════════════════ 2. ПРОВЕРКА ═════════════════════════════════
with tabs[1]:
    st.header('Проверка исходных данных')
    st.caption('Проверка не меняет данные. Виды проверок — по Руководству по климатологической практике '
               '(ВМО-№ 100, 2018, раздел 3.3): проверка формата, полноты, согласованности и допустимости.')
    if not S.raw:
        st.info('Сначала загрузите данные.')
    else:
        data = S.raw['data']
        if st.button('Выполнить проверку', type='primary'):
            with st.spinner('Проверка…'):
                S.checks = {rg: verify.consistency_checks(region_data(data, rg), region_stations(rg))
                            for rg in regions()}
                S.comp = verify.completeness(data)
                S.gaps = verify.network_gaps(data)
                S.gross = {rg: verify.gross_error_summary(region_data(data, rg)) for rg in regions()}
        if S.checks is not None:
            var = st.selectbox('Переменная для карты полноты', sorted(v for v in data if v != 'DY'),
                               format_func=VAR_LABEL.get, key='comp_var')
            for rg, tab in region_tabs():
                with tab:
                    st.subheader('Проверка формата и согласованности')
                    chk = S.checks[rg].copy()
                    chk['Значение'] = chk['Значение'].map(lambda v: (cf(v, 0) if float(v).is_integer() else cf(v, 2))
                                                          if isinstance(v, (int, float, np.number)) and np.isfinite(v) else v)
                    st.dataframe(chk.style.apply(
                        lambda r: ['background-color: #FDECEA' if r['Статус'] == 'ошибка' else
                                   ('background-color: #FFF4E0' if r['Статус'] == 'внимание' else '') for _ in r],
                        axis=1), width='stretch', hide_index=True)
                    save_table(f'table_01_проверка_согласованности_{rg}', S.checks[rg])
                    footnote('Статус: «норма» — расхождений нет или они в допустимых пределах; «внимание» — есть '
                             'значения, которые обрабатываются на этапе контроля качества; «ошибка» — вероятно '
                             'неверное чтение формата (например, иная раскладка значений в строке).')
                    st.subheader('Проверка полноты')
                    cr = S.comp[S.comp['Регион'] == rg]
                    if not cr[cr['Переменная'] == var].empty:
                        show_fig(charts.completeness_heatmap(cr, var, station_names()), f'fig_полнота_{var}_{rg}')
                    summ = cr.groupby('Переменная')['Полнота, %'].agg(['mean', 'min']).reset_index()
                    summ.columns = ['Переменная', 'Средняя полнота, %', 'Наименьшая полнота, %']
                    show_df(summ, 1)
                    save_table(f'table_02_полнота_{rg}', cr, 1)
                    footnote(f'Полнота — доля суток с данными от числа календарных суток года (для класса пожарной опасности FR — '
                             f'пожароопасный сезон региона, {season_txt(rg, "fire")}). '
                             'На карте подписаны значения меньше 100 %.')
                    st.subheader('Пропуски по всей сети региона')
                    g = S.gaps[S.gaps['Регион'] == rg] if not S.gaps.empty else S.gaps
                    show_df(g, 1)
                    save_table(f'table_03_пропуски_по_сети_{rg}', g, 1)
                    footnote('Периоды, когда данных нет более чем у половины станций региона одновременно. Такие '
                             'пропуски нельзя восстановить по соседним станциям.')
                    st.subheader('Проверка допустимости (до очистки)')
                    show_df(S.gross[rg], 1)
                    footnote(qc.RULE_HELP['L'])

# ═══════════════════════════ 3. ОЧИСТКА ══════════════════════════════════
with tabs[2]:
    st.header('Контроль качества')
    if not S.raw:
        st.info('Сначала загрузите данные.')
    else:
        data = S.raw['data']
        with st.expander('Правила и пороги', expanded=False):
            st.markdown('Пороги по умолчанию — в файле params.py; здесь их можно изменить для текущего расчёта.')
            c1, c2, c3 = st.columns(3)
            P.RR_MAX_COLD = c1.number_input('Осадки вне лета: ошибочное, мм за сутки >', value=float(P.RR_MAX_COLD),
                                            help=qc.RULE_HELP['RR1'])
            P.RR_MAX_WARM = c1.number_input('Осадки летом: ошибочное, мм за сутки ≥', value=float(P.RR_MAX_WARM))
            P.NETWORK_MEDIAN_DEV_T = c2.number_input('Отклонение от медианы по сети, °C',
                                                     value=float(P.NETWORK_MEDIAN_DEV_T), help=qc.RULE_HELP['N'])
            P.SPATIAL_Z_SUSPECT = c2.number_input('Соседние станции: |z| для «требует проверки»',
                                                  value=float(P.SPATIAL_Z_SUSPECT), help=qc.RULE_HELP['S'])
            P.SPATIAL_Z_REJECT = c2.number_input('Соседние станции: |z| для «ошибочное»', value=float(P.SPATIAL_Z_REJECT))
            P.SPATIAL_MIN_ABS = c2.number_input('Соседние станции: наименьшее расхождение для «ошибочное», °C',
                                                value=float(P.SPATIAL_MIN_ABS))
            P.NEIGHBOR_MAX_KM = c3.number_input('Радиус поиска соседних станций, км', value=float(P.NEIGHBOR_MAX_KM))
            P.FM_ROUND_MIN_STATIONS = int(c3.number_input('Ветер: круглые значения одновременно не менее чем на … станциях',
                                                          value=int(P.FM_ROUND_MIN_STATIONS), step=1,
                                                          help=qc.RULE_HELP['FM1']))
            sp = st.text_input('Периоды повышенной строгости (начало–конец, через «;»)',
                               value='; '.join(f'{a}–{b}' for a, b in P.STRICT_PERIODS),
                               help='В эти периоды пороги проверок по соседним станциям и медиане сети умножаются на 0,6 '
                                    '(например, при известном сбое исходных телеграмм).')
            try:
                P.STRICT_PERIODS = [tuple(x.strip().split('–')) for x in sp.split(';') if '–' in x]
            except Exception:
                pass
            sts = S.stations
            fm_st = st.multiselect('Ветер: станции, где завышенные значения делятся на 10',
                                   [int(x) for x in sts['st']],
                                   default=[s for s in P.FM_DIVIDE_BY_10 if s in sts['st'].values],
                                   format_func=st_label, help=qc.RULE_HELP['FM2'])
            fm_year = st.number_input('… начиная с года', value=2018, step=1)
            P.FM_DIVIDE_BY_10 = {s: int(fm_year) for s in fm_st}
        c1, c2, c3, c4 = st.columns(4)
        opts = dict(isolation_forest=c1.checkbox('Многомерная проверка TX (Isolation Forest)', value=True,
                                                 help=qc.RULE_HELP['IF']),
                    persistence=c2.checkbox('Поиск неизменных значений', value=True, help=qc.RULE_HELP['P']),
                    correct_sign=c3.checkbox('Исправлять вероятную ошибку знака Tмакс', value=False,
                                             help=qc.RULE_HELP['TX3']),
                    fm_kmh_convert=c4.checkbox('Ветер в км/ч пересчитывать в м/с', value=False,
                                               help=qc.RULE_HELP['FM3']))
        if st.button('Выполнить контроль качества', type='primary'):
            bar = st.progress(0.0)
            S.qc_log = qc.run_qc(data, S.stations, opts, progress=lambda f, n: bar.progress(f, text=f'Проверка: {n}'))
            bar.empty()
            S.clean, S.flags = qc.apply_qc(data, S.qc_log)
            S.filled = None
        if S.qc_log is not None:
            log = S.qc_log
            if log.empty:
                st.success('Значений, не прошедших проверки, нет.')
            else:
                with st.expander('Правила контроля качества и категории значений', expanded=False):
                    st.markdown('**Категории.** «Ошибочное» — значение исключается из ряда. «Требует проверки» — '
                                'значение остаётся; по выбранным правилам такие значения можно исключить. '
                                '«Исправленное» — ошибка известного происхождения устранена (единицы измерения, '
                                'лишняя цифра, суточный максимум). «Восстановленное» — пропуск заполнен на этапе 4.')
                    show_df(pd.DataFrame([{'Код': k, 'Правило': qc.RULES[k], 'Пояснение': qc.RULE_HELP.get(k, '')}
                                          for k in qc.RULES]), 0)
                sus_rules = sorted(log[log['Действие'] == qc.SUSPECT]['Правило'].unique())
                rej = st.multiselect('Исключать значения «требует проверки» по правилам', sus_rules,
                                     format_func=lambda r: f'{r} — {qc.RULES.get(r, r)}')
                if st.button('Применить'):
                    S.clean, S.flags = qc.apply_qc(data, log, reject_suspect_rules=rej)
                    S.filled = None
                    st.success('Очищенные данные обновлены.')
                for rg, tab in region_tabs():
                    with tab:
                        lr = log[log['Регион'] == rg]
                        summ = qc.qc_summary(lr)
                        if summ.empty:
                            st.success('Значений, не прошедших проверки, нет.')
                            continue
                        st.subheader('Сводка')
                        show_fig(charts.qc_summary_bar(summ), f'fig_контроль_качества_{rg}')
                        show_df(summ.drop(columns='Регион'), 0)
                        save_table(f'table_04_контроль_качества_{rg}', summ, 0)
                        st.subheader('Журнал')
                        c1, c2, c3 = st.columns(3)
                        fv = c1.multiselect('Переменная', sorted(lr['Переменная'].unique()), key=f'jv_{rg}')
                        fa = c2.multiselect('Категория', sorted(lr['Действие'].unique()), key=f'ja_{rg}')
                        fs = c3.multiselect('Станция', sorted(int(x) for x in lr['Станция'].unique()),
                                            format_func=st_label, key=f'js_{rg}')
                        L = lr[lr['Основное']]
                        if fv:
                            L = L[L['Переменная'].isin(fv)]
                        if fa:
                            L = L[L['Действие'].isin(fa)]
                        if fs:
                            L = L[L['Станция'].isin(fs)]
                        L = L.assign(Название=L['Станция'].map(station_names()))
                        show_df(L[['Переменная', 'Компонент', 'Станция', 'Название', 'Дата', 'Значение',
                                   'Новое значение', 'Действие', 'Правило', 'Описание правила', 'Комментарий']]
                                .rename(columns={'Действие': 'Категория'}), 1, height=320)
                        save_table(f'table_05_журнал_контроля_качества_{rg}', lr[lr['Основное']].drop(columns='Основное'), 2)
                        footnote('Если к одному значению применимо несколько правил, в журнале показано одно: '
                                 'исправленное → ошибочное → требует проверки. Коды правил — в разделе «Правила '
                                 'контроля качества» выше.')
                        st.subheader('Ряд станции до и после контроля')
                        c1, c2, c3 = st.columns(3)
                        vv = [x for x in ['TX', 'TT', 'RR', 'FM', 'TS', 'RS'] if x in data]
                        v = c1.selectbox('Переменная', vv, key=f'qc_v_{rg}')
                        comps = [c for c in qc.VAR_COMPS[v] if c in data[v].columns]
                        cpt = c2.selectbox('Компонент', comps, format_func=lambda c: COMP_LABEL.get(c, c), key=f'qc_c_{rg}')
                        cand = [int(x) for x in L[L['Переменная'] == v]['Станция'].value_counts().index]
                        cand += [int(x) for x in region_stations(rg)['st'] if int(x) not in cand]
                        sst = c3.selectbox('Станция', cand, format_func=st_label, key=f'qc_s_{rg}')
                        fig = charts.qc_series(data[v], cpt, sst, log, S.clean[v],
                                               title=f'{COMP_LABEL.get(cpt, cpt)}: {st_label(sst)}',
                                               ylabel=readers.VAR_INFO[v]['unit'])
                        show_fig(fig, f'fig_контроль_{v}_{cpt}_{sst}', wide=True)

# ═══════════════════════════ 4. ПРОПУСКИ ═════════════════════════════════
with tabs[3]:
    st.header('Восстановление пропусков')
    if S.clean is None:
        st.info('Сначала выполните контроль качества (вкладка 3).')
    else:
        st.markdown('Для каждой переменной выберите метод или «по проверке точности». Во втором случае для каждого '
                    f'региона отдельно из имеющихся значений скрывается {cf(P.FILL_CV_SHARE * 100, 0)} %, они '
                    'оцениваются каждым методом, и выбирается метод с наименьшей RMSE. Пропуски длиннее '
                    f'{P.FILL_MAX_GAP_DAYS} суток и сутки без данных на всех соседних станциях не заполняются.')
        footnote('Методы — по разделу 5.9 «Оценивание данных» ВМО-№ 100: временная интерполяция (короткие разрывы), '
                 'методы пространственной оценки (регрессия на соседние станции, обратно-взвешенное расстояние, '
                 'отношение к соседним станциям для осадков и ветра), а также методы машинного обучения '
                 '(k ближайших соседей, Random Forest), где признаки — значения соседних станций и день года.')
        opts_m = {'auto': 'По проверке точности', 'none': 'Не восстанавливать', **fill.METHODS}
        cols = st.columns(6)
        meth = {}
        for i, v in enumerate(['TX', 'TT', 'RS', 'TS', 'RR', 'FM']):
            if v in S.clean:
                choices = ['auto', 'none'] + (['ratio', 'regress', 'knn', 'rf', 'clim'] if v in ('RR', 'FM')
                                               else ['interp', 'regress', 'idw', 'knn', 'rf', 'clim'])
                meth[v] = cols[i].selectbox(v, choices, index=choices.index(fill.DEFAULT_METHODS.get(v, 'auto')),
                                            format_func=opts_m.get, key=f'fm_{v}')
        P.FILL_MAX_GAP_DAYS = int(st.number_input('Наибольшая длина восстанавливаемого пропуска, сутки',
                                                  value=int(P.FILL_MAX_GAP_DAYS), step=5))
        if st.button('Восстановить пропуски', type='primary'):
            bar = st.progress(0.0)
            S.filled, S.fill_log, S.cv = fill.fill_dataset(S.clean, S.stations, meth,
                                                           progress=lambda f, n: bar.progress(f, text=n))
            bar.empty()
        if S.filled is not None:
            for rg, tab in region_tabs():
                with tab:
                    st.subheader('Результат')
                    fl_r = S.fill_log[S.fill_log['region'] == rg] if 'region' in S.fill_log else S.fill_log
                    fs = fill.fill_summary(fl_r, S.clean)
                    show_df(fs.drop(columns='Регион', errors='ignore'), 0)
                    save_table(f'table_06_восстановление_{rg}', fs, 0)
                    cv = S.cv[S.cv['Регион'] == rg] if S.cv is not None and 'Регион' in S.cv else S.cv
                    if cv is not None and not cv.empty:
                        st.subheader('Проверка точности методов')
                        show_df(cv.drop(columns=['Ошибка', 'Регион'], errors='ignore'), 2)
                        save_table(f'table_07_проверка_методов_{rg}', cv, 3)
                        footnote('MAE — средняя абсолютная ошибка, RMSE — средняя квадратическая ошибка, смещение — '
                                 'средняя разность «оценка − факт», R² — доля объяснённой дисперсии; все — на '
                                 'искусственно скрытых значениях. «Восстановлено, %» — доля скрытых значений, '
                                 'которые метод смог оценить.')
                        cc = st.selectbox('Компонент', cv['Компонент'].unique(), format_func=lambda c: COMP_LABEL.get(c, c),
                                          key=f'cvc_{rg}')
                        show_fig(charts.cv_bars(cv, cc), f'fig_методы_{cc}_{rg}')
                    st.subheader('Восстановленные значения на графике')
                    c1, c2, c3 = st.columns(3)
                    v = c1.selectbox('Переменная', [x for x in ['TX', 'TT', 'RR', 'FM', 'TS', 'RS'] if x in S.filled],
                                     key=f'f_v_{rg}')
                    cpt = c2.selectbox('Компонент', [c for c in fill.FILL_COMPS[v] if c in S.filled[v].columns],
                                       format_func=lambda c: COMP_LABEL.get(c, c), key=f'f_c_{rg}')
                    fl = fl_r[(fl_r['var'] == v) & (fl_r['comp'] == cpt)]
                    cand = [int(x) for x in fl['st'].value_counts().index] or [int(x) for x in region_stations(rg)['st']]
                    sst = c3.selectbox('Станция', cand, format_func=st_label, key=f'f_s_{rg}')
                    fig = charts.qc_series(S.clean[v], cpt, sst, None, None, fl,
                                           title=f'{COMP_LABEL.get(cpt, cpt)}: {st_label(sst)}',
                                           ylabel=readers.VAR_INFO[v]['unit'])
                    show_fig(fig, f'fig_восстановление_{v}_{cpt}_{sst}', wide=True)

# ═══════════════════════════ 5. ПОКАЗАТЕЛИ ═══════════════════════════════
with tabs[4]:
    st.header('Климатические показатели для планов адаптации')
    data = current_data()
    if data is None:
        st.info('Сначала загрузите данные.')
    else:
        src = st.radio('Данные для расчёта', ['Восстановленные', 'Очищенные (без восстановления)', 'Исходные'],
                       horizontal=True, index=0 if S.filled is not None else (1 if S.clean is not None else 2))
        use = {'Восстановленные': S.filled, 'Очищенные (без восстановления)': S.clean,
               'Исходные': S.raw['data']}[src]
        P.DRY_DAY_MM = st.number_input('Порог сухого дня, мм (сутки с осадками меньше порога)',
                                       value=float(P.DRY_DAY_MM), step=0.1,
                                       help='1,0 мм — порог индексов ETCCDI (CDD, CWD). Для бездождных периодов по '
                                            'другим определениям можно задать 0,1 мм.')
        st.subheader('Критерии опасных и неблагоприятных явлений')
        new_regions = [rg for rg in regions() if rg not in P.CRITERIA_CHECKED]
        if new_regions:
            st.warning('Для регионов ' + ', '.join(new_regions) + ' критерии ОЯ и НЯ заполнены значениями по умолчанию. '
                       'Введите критерии УГМС этих регионов в таблице ниже.')
        for rg in regions():
            P.criteria_for(rg)
        crit = pd.DataFrame({rg: {k: (f'({cf(v[0], 1)}; {cf(v[1], 1)})' if isinstance(v, tuple) else cf(v, 1))
                                  for k, v in P.CRITERIA[rg].items()} for rg in regions()})
        crit.insert(0, 'Критерий', [P.CRITERIA_LABELS.get(k, k) for k in crit.index])
        crit = crit.reset_index().rename(columns={'index': 'Код'})
        crit_ed = st.data_editor(crit.astype(str), hide_index=True, width='stretch', key='crit_ed',
                                 disabled=['Код', 'Критерий'])
        try:
            for rg in regions():
                for _, r in crit_ed.iterrows():
                    raw = str(r[rg]).strip()
                    if raw.startswith('('):
                        nums = re.findall(r'-?\d+(?:[.,]\d+)?', raw.replace(', ', ' '))
                        P.CRITERIA[rg][r['Код']] = tuple(float(x.replace(',', '.')) for x in nums)
                    else:
                        P.CRITERIA[rg][r['Код']] = float(raw.replace(',', '.'))
        except Exception:
            st.warning('Критерии не приняты: проверьте формат чисел (диапазон — в скобках, например «(30,0; 34,9)»).')
        if use is None:
            st.warning('Выбранный набор данных ещё не сформирован.')
        elif st.button('Рассчитать показатели', type='primary'):
            with st.spinner('Расчёт…'):
                hum = None
                if 'TT' in use and 'RS' in use:
                    hum = derived.daily_humidity(derived.build_terms(use['TT'], use['RS']))
                D = indices.build_daily(use, hum)
                S.D = D
                S.ann = indices.annual_indices(D)
                S.events = dict(heat=indices.heat_events(D), cold=indices.cold_events(D),
                                precip=indices.heavy_precip_events(D) if 'rr_n' in D else pd.DataFrame(),
                                drought=indices.drought_events(D) if 'rr_s' in D else pd.DataFrame(),
                                sukh=indices.sukhovey_events(D))
        if S.ann is not None:
            ann, D = S.ann, S.D
            with st.expander('Определения показателей и источники'):
                show_df(pd.DataFrame([{'Показатель': indices.label(c), 'Определение': h}
                                      for c, h in indices.INDEX_HELP.items()]), 0)
                for r in indices.REFERENCES:
                    st.markdown(f'- {r}')
            groups = sorted({v[2] for v in indices.INDEX_INFO.values()})
            grp = st.multiselect('Группы показателей в таблицах', groups,
                                 default=[g for g in ['Жара', 'Осадки', 'Засуха', 'Агроклимат'] if g in groups])
            code = st.selectbox('Показатель для графика межгодового хода', sorted(ann['Показатель'].unique()),
                                format_func=indices.label)
            for rg, tab in region_tabs():
                with tab:
                    a = ann[ann['region'] == rg]
                    Dr = D[D['region'] == rg]
                    if a.empty:
                        st.info('Нет лет с достаточной полнотой данных.')
                        continue
                    yrs = (int(a['year'].min()), int(a['year'].max()))
                    st.caption(f'Годы с достаточной полнотой: {yrs[0]}–{yrs[1]}. Год учитывается, если доступно '
                               '≥ 90 % суточных значений нужной переменной.')
                    per = st.slider('Период осреднения', yrs[0], yrs[1], yrs, key=f'ind_period_{rg}') \
                        if yrs[0] < yrs[1] else yrs
                    sm = indices.station_means(a, per)
                    sm = S.stations[['st', 'name']].merge(sm, on='st')
                    codes = [c for c, v in indices.INDEX_INFO.items() if v[2] in grp and c in sm.columns]
                    tab_ = sm[['st', 'name', 'Лет'] + codes].rename(columns={**{c: indices.label(c) for c in codes},
                                                                           'st': 'Станция', 'name': 'Название'})
                    st.subheader('Средние многолетние значения по станциям')
                    show_df(tab_, 1, height=330)
                    save_table(f'table_08_показатели_по_станциям_{rg}', tab_, 2)
                    rm = a[a['year'].between(*per)].groupby('Показатель')['Значение'].agg(['mean', 'min', 'max'])
                    rm = rm.reindex([c for c in indices.INDEX_INFO if c in rm.index])
                    rm.index = [indices.label(c) for c in rm.index]
                    rm.columns = ['Среднее по станциям', 'Наименьшее', 'Наибольшее']
                    st.subheader('Сводка по региону')
                    show_df(rm.reset_index().rename(columns={'index': 'Показатель'}), 2, height=300)
                    save_table(f'table_09_показатели_регион_{rg}', rm.reset_index().rename(columns={'index': 'Показатель'}), 2)
                    footnote('Среднее — по годовым значениям всех станций региона за выбранный период; наименьшее и '
                             'наибольшее — по отдельным годам и станциям.')
                    st.subheader('Межгодовой ход')
                    if code in a['Показатель'].values:
                        show_fig(charts.index_series(a, code, indices.label(code)), f'fig_ряд_{code}_{rg}')
                        footnote('Линия — среднее по станциям региона; пунктир — наклон Сена; * — тренд значим по тесту '
                                 'Манна — Кендалла (p < 0,05). Для рядов короче 30 лет оценка предварительная (ВМО).')
                    st.subheader('Аномально жаркая погода и сильная жара')
                    ev = S.events['heat']
                    ev = ev[ev['region'] == rg] if not ev.empty else ev
                    c1, c2 = st.columns(2)
                    pa = c1.slider('Период 1', yrs[0], yrs[1], yrs, key=f'hp1_{rg}') if yrs[0] < yrs[1] else yrs
                    pb = c2.slider('Период 2 (необязательно)', yrs[0], yrs[1], yrs, key=f'hp2_{rg}') \
                        if yrs[0] < yrs[1] else yrs
                    recs = {}
                    for pp in ([pa] if pa == pb else [pa, pb]):
                        r = indices.heat_summary(ev, Dr, pp, rg)
                        recs[f'{r.pop("Период")} гг.'] = r
                    hs = pd.DataFrame(recs)
                    hs.index.name = 'Характеристика'
                    hs = hs.reset_index()
                    with st.expander('Базовые значения для сравнения (например, по ранее выполненным расчётам)'):
                        base = st.data_editor(pd.DataFrame({'Характеристика': hs['Характеристика'],
                                                            'Базовый период': [None] * len(hs)}),
                                              hide_index=True, width='stretch', key=f'base_heat_{rg}')
                        if base['Базовый период'].notna().any():
                            hs = hs.merge(base, on='Характеристика')
                    show_df(hs, 2)
                    save_table(f'table_10_жара_{rg}', hs, 2)
                    crit_r = P.criteria_for(rg)
                    footnote(f'АЖ — Tмакс ≥ {cf(crit_r["АЖ_порог"], 0)} °C не менее {crit_r["АЖ_дней"]} суток подряд; '
                             f'СЖ — Tмакс ≥ {cf(crit_r["СЖ_порог"], 0)} °C не менее {crit_r["СЖ_дней"]} суток подряд '
                             '(критерии региона, таблица выше). Пропуск данных прерывает период.')
                    with st.expander('Перечень периодов АЖ и СЖ'):
                        show_df(ev.assign(Название=ev['st'].map(station_names())), 1, height=260)
                        save_table(f'table_11_периоды_жары_{rg}', ev, 1)
                    st.subheader('Бездождные периоды и периоды без снегопада')
                    ps = st.slider('Период', yrs[0], yrs[1], yrs, key=f'dp_{rg}') if yrs[0] < yrs[1] else yrs
                    Dp = Dr[Dr['date'].dt.year.between(ps[0] - 1, ps[1])]
                    for season, kind, ttl in [('тёплый', 'dry', f'Бездождные периоды, тёплый сезон ({season_txt(rg, "warm")})'),
                                              ('холодный', 'dry', f'Бездождные периоды, холодный сезон ({season_txt(rg, "cold")})'),
                                              ('холодный', 'nosnow', f'Периоды без снегопада, холодный сезон ({season_txt(rg, "cold")})')]:
                        spl, ok = indices.dry_spells(Dp, season, kind=kind)
                        if ok.empty:
                            continue
                        ok = ok[ok['Сезон'].between(*ps)]
                        spl = spl[spl['Сезон'].between(*ps)] if not spl.empty else spl
                        by_st, by_reg = indices.spell_class_table(spl, ok)
                        st.markdown(f'**{ttl}.** Учтено сезонов: {int(ok["Учтён"].sum())} из {len(ok)}.')
                        if not by_reg.empty:
                            show_df(by_reg.drop(columns='region'), 2)
                            show_fig(charts.spell_classes_bar(by_reg, ttl), f'fig_{kind}_{season}_{rg}')
                            save_table(f'table_12_{kind}_{season}_{rg}', by_reg, 2)
                            with st.expander('По станциям'):
                                show_df(by_st.assign(Название=by_st['st'].map(station_names())), 2)
                    footnote(f'Сухой день — осадки < {cf(P.DRY_DAY_MM, 1)} мм. Период относится к сезону, в котором '
                             'начался; холодный сезон обозначен годом окончания (зима 2016/17 г. — 2017 г.). Сезон с '
                             f'пропусками > {cf(P.SEASON_MAX_MISSING * 100, 0)} % исключается. День со снегопадом — '
                             f'осадки ≥ порога сухого дня при Tмакс ≤ {cf(P.SNOW_TMAX, 0)} °C.')
                    st.subheader('Опасные явления: перечни')
                    for k, ttl in [('precip', 'Сильные осадки и снег'),
                                   ('cold', 'Сильный мороз, аномально холодная погода'),
                                   ('drought', 'Атмосферная засуха'), ('sukh', 'Суховеи')]:
                        e = S.events.get(k)
                        e = e[e['region'] == rg] if e is not None and len(e) else e
                        with st.expander(f'{ttl}: {0 if e is None else len(e)} случаев'):
                            if e is not None and len(e):
                                show_df(e.assign(Название=e['st'].map(station_names())), 1, height=240)
                                save_table(f'table_13_события_{k}_{rg}', e, 1)
                    st.subheader('Экстремумы и повторяемость')
                    ext = indices.extremes_table(Dr[Dr['year'].between(*per)], a[a['year'].between(*per)])
                    ext = S.stations[['st', 'name']].merge(ext, on='st').drop(columns='region')
                    show_df(ext, 1, height=260)
                    save_table(f'table_14_экстремумы_{rg}', ext, 1)
                    footnote(f'Tмакс 0,95 обеспеченности — 95-й процентиль суточной Tмакс за летние месяцы ({season_txt(rg, "summer")}). Уровни «раз в N '
                             'лет» — по распределению Гумбеля для годовых максимумов (метод моментов); число случаев АЖ, '
                             'возможное раз в 50 лет, — по распределению Пуассона. При длине ряда < 30 лет оценки '
                             'приближённые.')

# ═══════════════════════════ 6. ТРЕНДЫ ═══════════════════════════════════
with tabs[5]:
    st.header('Тренды и сравнение периодов')
    if S.ann is None:
        st.info('Сначала рассчитайте показатели (вкладка 5).')
    else:
        ann = S.ann
        codes = st.multiselect('Показатели', sorted(ann['Показатель'].unique()), format_func=indices.label,
                               default=[c for c in ['Tср_год', 'TXx', 'SU30', 'АЖ_случаев', 'PRCPTOT', 'CDD', 'RX1day',
                                                    'Сумма10', 'ГТК'] if c in ann['Показатель'].unique()])
        by = st.radio('Уровень', ['region', 'st'], format_func={'region': 'Средние по региону', 'st': 'По станциям'}.get,
                      horizontal=True)
        footnote('Наклон Сена — изменение за 10 лет; тест Манна — Кендалла с поправкой Хамеда — Рао на автокорреляцию '
                 '(p < 0,05 — тренд значим); тест Петтитта — год наиболее вероятной смены среднего уровня ряда. По '
                 'рекомендации ВМО выводы о трендах делаются по рядам не короче 30 лет.')
        for rg, tab in region_tabs():
            with tab:
                a = ann[ann['region'] == rg]
                tt = trends.trend_table(a, codes, by=by)
                if not tt.empty:
                    tt['Показатель'] = tt['Показатель'].map(indices.label)
                    if by == 'st':
                        tt.insert(1, 'Название', tt['Станция'].map(station_names()))
                    show_df(tt.drop(columns='Регион', errors='ignore'), 3)
                    save_table(f'table_15_тренды_{rg}', tt, 3)
                else:
                    st.info('Ряды короче 5 лет — тренды не оцениваются.')
                yrs = (int(a['year'].min()), int(a['year'].max())) if not a.empty else (0, 0)
                if yrs[1] - yrs[0] < 5:
                    continue
                st.subheader('Сравнение периодов')
                c1, c2 = st.columns(2)
                p1 = c1.slider('Период 1', yrs[0], yrs[1], (yrs[0], (yrs[0] + yrs[1]) // 2), key=f'cp1_{rg}')
                p2 = c2.slider('Период 2', yrs[0], yrs[1], ((yrs[0] + yrs[1]) // 2 + 1, yrs[1]), key=f'cp2_{rg}')
                cmp = trends.compare_periods(a, p1, p2, codes, by='region')
                if not cmp.empty:
                    code0 = st.selectbox('Показатель для графика', codes, format_func=indices.label, key=f'cmpc_{rg}')
                    fig = charts.period_compare_bar(cmp, code0, indices.label(code0))
                    cmp['Показатель'] = cmp['Показатель'].map(indices.label)
                    show_df(cmp.drop(columns='Регион', errors='ignore'), 3)
                    save_table(f'table_16_сравнение_периодов_{rg}', cmp, 3)
                    footnote('Разность — среднее периода 2 минус среднее периода 1; 95 % доверительный интервал — '
                             'бутстреп (2000 повторов); p — критерий Манна — Уитни.')
                    if fig:
                        show_fig(fig, f'fig_периоды_{code0}_{rg}')

# ═══════════════════════════ 7. РАЙОНИРОВАНИЕ ════════════════════════════
with tabs[6]:
    st.header('Районирование по климатическим показателям')
    if S.ann is None:
        st.info('Сначала рассчитайте показатели (вкладка 5).')
    else:
        ann = S.ann
        rg = st.selectbox('Регион', regions(), key='z_region')
        a = ann[ann['region'] == rg]
        yrs = (int(a['year'].min()), int(a['year'].max()))
        per = st.slider('Период осреднения', yrs[0], yrs[1], yrs, key='z_per') if yrs[0] < yrs[1] else yrs
        sm = indices.station_means(a, per)
        base = S.stations[['st', 'name', 'region', 'lat', 'lon', 'alt']].merge(sm.drop(columns='region'), on='st')
        base = base[base['region'] == rg].reset_index(drop=True)
        feats_all = [c for c in indices.INDEX_INFO if c in base.columns and base[c].notna().sum() >= 3
                     and base[c].std() > 0]
        default = [c for c in ['SU30', 'АЖ_случаев', 'Сумма10', 'ГТК', 'PRCPTOT', 'CDD', 'R10mm', 'Ветер15', 'Пож4',
                               'FD'] if c in feats_all]
        feats = st.multiselect('Показатели для районирования', feats_all, default=default, format_func=indices.label)
        if len(base) < 4:
            st.warning('Для районирования нужно не менее 4 станций с данными.')
        elif len(feats) < 2:
            st.warning('Выберите минимум 2 показателя.')
        else:
            corr = base[feats].corr()
            hi = [(x, y, corr.loc[x, y]) for i, x in enumerate(feats) for y in feats[i + 1:] if abs(corr.loc[x, y]) > 0.9]
            if hi:
                footnote('Сильно связанные показатели (|r| > 0,9), один из пары можно исключить: ' +
                         '; '.join(f'{indices.label(x)} — {indices.label(y)} ({cf(r, 2)})' for x, y, r in hi))
            dc, Xs, _ = clustering.prepare_matrix(base, feats)
            st.caption(f'Станций с полными значениями: {len(dc)} из {len(base)}')
            kmax = max(3, min(7, len(dc) - 1))
            sil = clustering.silhouette_by_k(Xs, range(2, kmax))
            ari = clustering.ari_by_k(Xs, dc['lat'].values, range(2, kmax)) if dc['lat'].notna().all() else None
            kbest = clustering.best_k(clustering.combined_scores(sil, ari) if ari else sil)
            c1, c2, c3 = st.columns(3)
            k = c1.number_input('Число районов (k)', 2, kmax - 1 if kmax > 2 else 2, int(kbest or 2))
            alg = c2.selectbox('Алгоритм', ['kmeans', 'hierarchical', 'gmm'],
                               format_func={'kmeans': 'k-средних', 'hierarchical': 'Иерархический (Уорд)',
                                            'gmm': 'Смесь гауссовых распределений'}.get)
            order = c3.selectbox('Нумерация районов по убыванию', feats, format_func=indices.label)
            show_fig(viz.k_selection_plot(sil, ari, chosen_k=k, title='Выбор числа районов'), f'fig_выбор_k_{rg}')
            footnote('Силуэт — согласованность станций внутри районов (от −1 до 1, больше — лучше); ARI — согласие с '
                     'делением по широте. Алгоритмы и критерии — как в программе климатического районирования '
                     f'([{ZONING_URL}]({ZONING_URL})).')
            labels, s_ = clustering.run_clustering(Xs, int(k), algorithm=alg)
            dc = dc.copy()
            dc['Кластер'] = clustering.relabel_clusters_by_feature(dc, labels, order)
            S.zoning = dict(df=dc, feats=feats, k=int(k), sil=s_, period=per, region=rg)
            st.success(f'Районирование выполнено: k = {int(k)}, силуэт = {cf(s_, 3)}.')
            out = dc[['st', 'name', 'Кластер'] + feats].rename(
                columns={**{c: indices.label(c) for c in feats}, 'st': 'Станция', 'name': 'Название', 'Кластер': 'Район'})
            show_df(out, 2)
            save_table(f'table_17_районирование_{rg}', out, 2)
            summ = clustering.cluster_summary_table(dc, feats, 'Кластер')
            summ.columns = [c.replace(c.split(' (')[0], indices.label(c.split(' (')[0])) if ' (' in c else c
                            for c in summ.columns]
            show_df(summ, 2)
            save_table(f'table_18_характеристика_районов_{rg}', summ, 2)
            Xp, pm, ev = clustering.pca_transform(Xs)
            show_fig(viz.pca_biplot(Xp, pm, dc['Кластер'].values, [indices.label(f).split(',')[0] for f in feats], ev,
                                    title='Главные компоненты: станции и показатели'), f'fig_pca_{rg}')
            Z = clustering.hierarchical_linkage(Xs)
            show_fig(viz.dendrogram_plot(Z, dc['name'].astype(str).str[:12].tolist(), title='Дендрограмма (Уорд)'),
                     f'fig_дендрограмма_{rg}', wide=True)

# ═══════════════════════════ 8. КАРТЫ ════════════════════════════════════
@st.fragment
def maps_tab():
    """Вкладка карт выполняется отдельно: изменение её настроек не пересчитывает остальные вкладки."""
    st.header('Карты')
    stx = stations_xy()
    if stx is None or stx.empty:
        st.info('Нужны координаты станций (вкладка 1).')
    else:
        c1, c2, c3 = st.columns(3)
        kind = c1.selectbox('Вид карты', ['Схематическая', 'По административным районам', 'Физическая',
                                          'На растровой подложке', 'Интерактивная'],
                            help='Схематическая — граница и станции с полигонами Тиссена или изолиниями; по районам — '
                                 'заливка административных районов; физическая — рельеф, реки и озёра; на подложке — '
                                 'растровая топографическая или физическая карта; интерактивная — для просмотра.')
        rg = c2.selectbox('Регион', [r for r in regions() if r in set(stx['region'])], key='map_region')
        regs_ne = basemap.region_names()
        guess = P.REGION_NE.get(rg) or next((n for n in regs_ne if str(rg).lower()[:4] in n.lower()), None)
        terr = c3.multiselect('Граница субъекта на карте', regs_ne, default=[guess] if guess in regs_ne else [],
                              help='Для региона, которого нет в списке соответствий, выберите субъект вручную.')
        rgeom = basemap.region_geometry(terr) if terr else None
        if S.get('districts_all') and S.get('districts_region') != rg:
            S.districts_all, S.districts = None, None       # районы другого региона не используются
        if S.get('districts_all'):
            if st.checkbox('Граница субъекта — по загруженным районам OpenStreetMap (точнее встроенной)',
                           value=True, key='osm_border'):
                from shapely.ops import unary_union
                rgeom = unary_union([p['geometry'] for p in S.districts_all]).buffer(0)
        sdf = stx[stx['region'] == rg].copy()
        bbox = basemap.bbox_of(rgeom, pad=0.04) if rgeom is not None else basemap.bbox_of(lat=sdf['lat'], lon=sdf['lon'])
        what_opts = ['Только станции']
        if S.zoning is not None and S.zoning.get('region') == rg:
            what_opts.append('Районы-кластеры')
        if S.ann is not None:
            what_opts.append('Показатель')
        what = st.radio('Что показать', what_opts, horizontal=True, index=len(what_opts) - 1 if len(what_opts) > 1 else 0)
        value_col, cluster_col, vlabel, dec, per = None, None, '', 1, None
        if what == 'Районы-кластеры':
            sdf = sdf.merge(S.zoning['df'][['st', 'Кластер']], on='st')
            cluster_col = 'Кластер'
        elif what == 'Показатель':
            a = S.ann[S.ann['region'] == rg]
            yrs = (int(a['year'].min()), int(a['year'].max()))
            cc1, cc2 = st.columns(2)
            code = cc1.selectbox('Показатель', sorted(a['Показатель'].unique()), format_func=indices.label, key='map_code')
            per = cc2.slider('Период осреднения', yrs[0], yrs[1], yrs, key='map_per') if yrs[0] < yrs[1] else yrs
            sm = indices.station_means(a, per)[['st', code]]
            sdf = sdf.merge(sm, on='st').dropna(subset=[code])
            value_col, vlabel = code, indices.label(code)
            dec = 0 if indices.INDEX_INFO.get(code, ('', ''))[1] in ('дни', 'случаи', 'день года', '°C·сут') else 1
            dec = st.number_input('Знаков после запятой', 0, 3, dec)
        labels_on = st.checkbox('Подписи станций', value=True)
        title = st.text_input('Заголовок', value=(vlabel.split(',')[0] + (f', {per[0]}–{per[1]} гг.' if per else '')
                                                  if value_col else ('Климатическое районирование' if cluster_col
                                                                     else 'Метеостанции')))
        field, grid = None, None
        if value_col and len(sdf) >= 4:
            im = st.selectbox('Интерполяция между станциями', ['rk', 'idw', 'nearest'],
                              format_func={'rk': 'Регрессионный кригинг (широта, долгота, высота)', 'idw': 'IDW',
                                           'nearest': 'Ближайшая станция (Тиссен)'}.get,
                              help='Регрессионный кригинг: линейная регрессия показателя на широту, долготу и высоту '
                                   '(если загружен рельеф), затем обычный кригинг остатков. IDW — обратно-взвешенное '
                                   'расстояние.')
            step = st.select_slider('Шаг сетки, °', [0.02, 0.05, 0.1], value=0.05)
            LON, LAT = interp.make_grid(bbox, step)
            galt, alt = None, None
            if S.dem is not None and im == 'rk' and sdf['alt'].notna().all():
                galt = basemap.sample_dem(S.dem[0], S.dem[1], LAT.ravel(), LON.ravel()).reshape(LAT.shape)
                alt = sdf['alt'].values
            if im == 'rk':
                field, info = interp.regression_kriging(sdf['lat'], sdf['lon'], sdf[value_col], LAT, LON, alt=alt, galt=galt)
                footnote(f'Доля дисперсии, объяснённая регрессией: R² = {cf(info["r2_trend"], 2)}'
                         f'{" (с высотой по ЦМР)" if info["use_alt"] else " (без высоты: для её учёта загрузите рельеф)"}.')
            elif im == 'idw':
                field = interp.idw(sdf['lat'], sdf['lon'], sdf[value_col], LAT, LON)
            else:
                field = interp.nearest(sdf['lat'], sdf['lon'], sdf[value_col], LAT, LON)
            # поле считается по всему охвату карты; на картах заливка обрезается точно по границе субъекта
            grid = (LON, LAT)
            with st.expander('Проверка интерполяции (поочерёдное исключение станций)'):
                cvt = interp.loo_cv(sdf['lat'], sdf['lon'], sdf[value_col],
                                    alt=sdf['alt'].values if sdf['alt'].notna().all() else None)
                show_df(cvt, 2)
                footnote('Каждая станция по очереди исключается, её значение оценивается по остальным; MAE, RMSE, R² '
                         '— по этим оценкам.')
        # ── субъект в OpenStreetMap: выбирается из списка, запросы — по идентификатору ──
        osm_rel = None
        need_osm = kind in ('Схематическая', 'По административным районам', 'Физическая')
        with st.expander('Субъект и административные районы по данным OpenStreetMap',
                         expanded=kind == 'По административным районам'):
            st.caption('Список субъектов запрашивается в OpenStreetMap по охвату станций региона, поэтому название '
                       'выбирается, а не вводится вручную. Границы районов загружаются по идентификатору субъекта.')
            ok_key = f'osm_subj_{rg}'
            if st.button('Получить список субъектов из OpenStreetMap', key='btn_osm_subj'):
                try:
                    with st.spinner('Запрос к Overpass API…'):
                        S[ok_key] = mapping.list_osm_subjects(basemap.bbox_of(lat=sdf['lat'], lon=sdf['lon'], pad=0.05))
                except basemap.ServiceError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f'Список субъектов не получен: {e}')
            subj = S.get(ok_key)
            if subj is not None and len(subj):
                names = list(subj['name'])
                g0 = (terr[0] if terr else str(rg)).lower()
                stem = g0.split()[0][:5]
                default = next((i for i, n in enumerate(names) if n.lower() == g0), None)
                if default is None:
                    default = next((i for i, n in enumerate(names) if stem and stem in n.lower()), 0)
                osm_name = st.selectbox('Субъект в OpenStreetMap', names, index=default, key=f'osm_name_{rg}')
                osm_rel = int(subj.loc[subj['name'] == osm_name, 'id'].iloc[0])
                cc1, cc2 = st.columns(2)
                al = cc1.selectbox('Уровень административного деления', [6, 8], key='d_al',
                                   format_func={6: 'Муниципальные районы и округа (admin_level 6)',
                                                8: 'Поселения (admin_level 8)'}.get)
                if cc2.button('Загрузить районы субъекта'):
                    try:
                        with st.spinner('Запрос к Overpass API (при первой загрузке до 2 минут)…'):
                            S.districts_all = mapping.fetch_osm_districts(osm_rel, admin_level=int(al))
                        S.districts = list(S.districts_all)
                        S.districts_region = rg
                        st.success(f'Загружено районов: {len(S.districts_all)}.')
                    except basemap.ServiceError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(str(e))
            elif subj is not None:
                st.warning('В охвате станций не найдено субъектов OpenStreetMap.')
            gj = st.file_uploader('или свой файл границ районов (GeoJSON)', type=['geojson', 'json'])
            if gj is not None:
                S.districts_all = mapping.read_geojson_file(gj.getvalue())
                S.districts = list(S.districts_all)
                S.districts_region = rg
            if S.get('districts_all'):
                all_names = [p['name'] for p in S.districts_all]
                sel = st.multiselect('Районы на карте', all_names, default=all_names, key='d_sel',
                                     help='По умолчанию — все районы субъекта; лишние можно убрать из списка.')
                S.districts = [p for p in S.districts_all if p['name'] in set(sel)]
                if st.button('Очистить загруженные районы'):
                    S.districts_all, S.districts = None, None
            show_d = st.checkbox('Показывать границы районов', value=bool(S.districts))
        water = None
        if need_osm:
            wsrc = st.radio('Реки и озёра', ['Крупные (встроенные, Natural Earth)', 'Подробные (OpenStreetMap)',
                                             'Не показывать'], horizontal=True)
            if wsrc.startswith('Крупные'):
                water = dict(rivers=basemap.ne_layer('rivers', bbox), lakes=basemap.ne_layer('lakes', bbox))
            elif wsrc.startswith('Подробные'):
                if osm_rel is None:
                    st.caption('Выберите субъект в разделе «Субъект и административные районы» выше; без него '
                               'реки запрашиваются по прямоугольному охвату карты.')
                if st.button('Загрузить реки и озёра (Overpass API)'):
                    try:
                        with st.spinner('Запрос к Overpass API (до 3 минут при первой загрузке)…'):
                            S.water = basemap.fetch_osm_water(bbox=bbox, rel_id=osm_rel)
                        st.success(f'Рек: {len(S.water["rivers"])}, водоёмов: {len(S.water["lakes"])}.')
                    except basemap.ServiceError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f'Реки и озёра не загружены: {e}')
                water = S.water or dict(rivers=basemap.ne_layer('rivers', bbox), lakes=basemap.ne_layer('lakes', bbox))
        districts = None
        if show_d and S.districts:
            districts = S.districts
        dem = None
        show_cities = True
        if kind == 'Физическая':
            with st.expander('Рельеф', expanded=True):
                rsrc = st.radio('Источник рельефа', ['Тайлы высот Terrarium (AWS, SRTM)', 'Свой файл ЦМР (.tif, .asc)',
                                                     'Без рельефа'], horizontal=True)
                if rsrc.startswith('Тайлы'):
                    zoom = st.select_slider('Детальность (уровень тайлов)', [6, 7, 8, 9], value=8)
                    if st.button('Загрузить рельеф'):
                        try:
                            bar = st.progress(0.0, text='Загрузка тайлов высот…')
                            S.dem = basemap.fetch_dem(bbox, zoom=zoom, progress=lambda f: bar.progress(f))
                            bar.empty()
                        except basemap.ServiceError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f'Не удалось загрузить рельеф: {e}')
                elif rsrc.startswith('Свой'):
                    df_ = st.file_uploader('ЦМР в географических координатах', type=['tif', 'tiff', 'asc'])
                    if df_ is not None:
                        try:
                            S.dem = basemap.load_dem_file(df_.getvalue(), df_.name)
                        except Exception as e:
                            st.error(f'Ошибка чтения ЦМР: {e}')
                if not rsrc.startswith('Без') and S.dem is not None:
                    dem = S.dem
            show_cities = st.checkbox('Города', value=True)
        polys_assigned = None
        if districts and (cluster_col or value_col):
            if cluster_col:
                polys_assigned = mapping.assign_clusters_to_polygons(
                    districts, sdf.rename(columns={'lat': 'Lat', 'lon': 'Lon', 'name': 'Название'}), cluster_col='Кластер')
            elif field is not None:
                vals = interp.zonal_mean(field, grid[1], grid[0], districts)
                polys_assigned = [{**p, 'value': v} for p, v in zip(districts, vals)]
        try:
            if sdf.empty:
                st.info('Нет станций с координатами в выбранном регионе.')
            elif kind == 'Схематическая':
                mode = st.radio('Способ', ['thiessen', 'field', 'points'], horizontal=True,
                                format_func={'thiessen': 'Полигоны Тиссена', 'field': 'Изолинии (интерполяция)',
                                             'points': 'Только точки'}.get,
                                index=1 if value_col else 0,
                                help='Полигоны Тиссена — каждой точке территории присваивается значение ближайшей станции.')
                if rgeom is None and mode == 'thiessen':
                    st.info('Для полигонов Тиссена выберите границу субъекта.')
                else:
                    fig = maps.map_schematic(sdf, rgeom, bbox, cluster_col=cluster_col, value_col=value_col,
                                             value_label=vlabel, title=title, mode=mode, field=field, grid=grid,
                                             decimals=dec, labels=labels_on, rivers=water, districts=districts)
                    show_fig(fig, f'map_схематическая_{rg}', kind_='map')
            elif kind == 'По административным районам':
                if not polys_assigned:
                    st.info('Загрузите границы районов и выберите районы-кластеры или показатель.')
                else:
                    show_lab = st.checkbox('Подписывать районы', value=False)
                    fig = maps.map_districts(polys_assigned, sdf, bbox, key='cluster' if cluster_col else 'value',
                                             value_label=vlabel, title=title, decimals=dec, show_labels=show_lab,
                                             region_geom=rgeom, rivers=water, station_labels=labels_on)
                    show_fig(fig, f'map_районы_{rg}', kind_='map')
                    tbl = pd.DataFrame([{'Административный район': p['name'],
                                         ('Район-кластер' if cluster_col else 'Значение'):
                                             p.get('cluster' if cluster_col else 'value'),
                                         'Станций внутри': p.get('n_stations_inside', '')} for p in polys_assigned])
                    show_df(tbl, 2)
                    save_table(f'table_19_районы_карта_{rg}', tbl, 2)
                    footnote('Для кластеров: район окрашен по преобладающему кластеру станций внутри него, без станций — '
                             'по ближайшей станции. Для показателя: среднее значение интерполированного поля в границах '
                             'района.')
            elif kind == 'Физическая':
                clusters_poly = None
                if cluster_col:
                    if polys_assigned:
                        clusters_poly = polys_assigned
                    elif rgeom is not None:
                        cells = maps.thiessen(sdf, rgeom)
                        clusters_poly = [{'geometry': c, 'cluster': k} for c, k in zip(cells, sdf['Кластер'])
                                         if c is not None]
                fig = maps.map_physical(bbox, rgeom, dem=dem[0] if dem else None, dem_extent=dem[1] if dem else None,
                                        water=water, cities=basemap.ne_layer('cities', bbox) if show_cities else None,
                                        st=sdf, st_labels=labels_on, field=field if value_col else None, grid=grid,
                                        value_label=vlabel, decimals=dec, districts=districts, title=title,
                                        clusters=clusters_poly)
                show_fig(fig, f'map_физическая_{rg}', kind_='map')
                if dem is None:
                    footnote('Рельеф не загружен: нажмите «Загрузить рельеф» (нужен интернет) или загрузите файл ЦМР.')
            elif kind == 'На растровой подложке':
                prov = st.selectbox('Подложка', list(basemap.TILE_PROVIDERS))
                zoom = st.select_slider('Детальность', [5, 6, 7, 8, 9], value=7)
                alpha = st.slider('Непрозрачность заливки', 0.0, 1.0, 0.45)
                if st.button('Построить'):
                    try:
                        bar = st.progress(0.0, text='Загрузка тайлов…')
                        img, ext = basemap.fetch_mosaic(bbox, basemap.TILE_PROVIDERS[prov], zoom,
                                                        progress=lambda f: bar.progress(f))
                        bar.empty()
                        polys = polys_assigned
                        if polys is None and cluster_col and rgeom is not None:
                            cells = maps.thiessen(sdf, rgeom)
                            polys = [{'geometry': c, 'cluster': k, 'name': ''} for c, k in zip(cells, sdf['Кластер'])
                                     if c is not None]
                        fig = maps.map_tiles(bbox, img, ext, basemap.TILE_ATTRIB[prov], rgeom, polys,
                                             key='cluster' if cluster_col else 'value', alpha=alpha, st=sdf,
                                             st_labels=labels_on, title=title, value_label=vlabel, decimals=dec)
                        show_fig(fig, f'map_подложка_{rg}', kind_='map')
                    except basemap.ServiceError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f'Не удалось загрузить подложку: {e}')
            else:
                import streamlit.components.v1 as components
                cap = f'<b>{title}</b><br>'
                if value_col:
                    cap += (f'Цвет станций и заливка — {vlabel}' + (', поле интерполировано между станциями' if field is
                                                                    not None else '') + '. Наведите на станцию для значения.')
                elif cluster_col:
                    cap += 'Цвет станций и районов — номер климатического района.'
                else:
                    cap += 'Метеостанции и граница субъекта. Выберите «Показатель» или «Районы-кластеры», чтобы раскрасить карту.'
                m = maps.folium_map(sdf, polys_assigned, key='cluster' if cluster_col else 'value', region_geom=rgeom,
                                    value_col=value_col, value_label=vlabel, cluster_col=cluster_col,
                                    field=field, grid=grid, decimals=dec, caption=cap)
                html = m.get_root().render()
                # карта встраивается как готовая веб-страница: перемещение, масштаб и смена подложки
                # выполняются в браузере и не запускают пересчёт программы
                h_px = int(560 * MAP_W / 60)
                if hasattr(st, 'iframe'):
                    st.iframe(html, height=h_px)
                else:
                    components.html(html, height=h_px, scrolling=False)
                st.download_button('Скачать интерактивную карту (HTML)', html.encode('utf-8'),
                                   file_name=f'map_интерактивная_{rg}.html', mime='text/html', key='dl_folium')
                footnote('Слои (подложки, поле, районы, станции) включаются и отключаются в правом верхнем углу карты. '
                         'Скачанный файл HTML открывается в любом браузере и работает без программы (для подложек '
                         'нужен интернет).')
        except Exception as e:
            st.exception(e)


with tabs[7]:
    maps_tab()

# ═══════════════════════════ 9. ЭКСПОРТ ══════════════════════════════════
with tabs[8]:
    st.header('Экспорт результатов')
    if not S.raw:
        st.info('Нет данных.')
    else:
        st.subheader('Передача в программу климатического районирования')
        st.markdown(f'Программа климатического районирования территорий: [{ZONING_URL}]({ZONING_URL}). Файл ниже '
                    'содержит помесячные ряды температуры (по Tсредней) и осадков (суточные суммы) по прошедшим '
                    'контролю данным и лист «Справочник станций» — в формате, который её вкладка «Данные» распознаёт '
                    'автоматически. Месяц учитывается при наличии не менее 25 суток с данными.')
        src = current_data()
        st.download_button('Скачать помесячные ряды для районирования (.xlsx)', export.zoning_xlsx(src, S.stations),
                           file_name='помесячные_ряды_для_районирования.xlsx')
        st.subheader('Суточные ряды')
        which = st.radio('Набор', ['Восстановленные', 'Очищенные', 'Исходные'], horizontal=True)
        dd = {'Восстановленные': S.filled, 'Очищенные': S.clean, 'Исходные': S.raw['data']}[which]
        if dd is not None and st.button('Сформировать файл суточных рядов'):
            sheets = export.daily_export(dd, S.flags if which != 'Исходные' else None,
                                         S.fill_log if which == 'Восстановленные' else None)
            st.download_button('Скачать суточные ряды (.xlsx)', export.xlsx_bytes(sheets, 1),
                               file_name=f'суточные_ряды_{which.lower()}.xlsx')
            footnote('Признаки качества: R — ошибочное значение исключено, C — исправленное, S — требует проверки; '
                     'в столбцах «…_восстановлено» — метод восстановления.')
        st.subheader('Все таблицы и рисунки')
        st.caption(f'Сохранено таблиц: {len(S.tables)}, рисунков: {len(S.figs)} (то, что открывалось на вкладках).')
        if st.button('Собрать архив'):
            files = {}
            for n, (df, d) in S.tables.items():
                files[f'tables/{n}.xlsx'] = export.xlsx_bytes({'Таблица': df}, d)
            for n, f in S.figs.items():
                files[f'figures/{n}.png'] = export.fig_bytes(f, 'png')
                files[f'figures/{n}.pdf'] = export.fig_bytes(f, 'pdf')
            if S.qc_log is not None:
                files['журнал_контроля_качества.xlsx'] = export.xlsx_bytes({'Журнал': S.qc_log}, 2)
            st.download_button('Скачать архив (.zip)', export.zip_bundle(files), file_name='результаты.zip')
        st.subheader('Параметры обработки')
        prm = {k: getattr(P, k) for k in dir(P) if k.isupper()}
        st.download_button('Скачать параметры (.json)', json.dumps(prm, ensure_ascii=False, indent=1, default=str),
                           file_name='параметры_обработки.json')
