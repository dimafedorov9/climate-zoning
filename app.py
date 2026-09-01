"""
=============================================================================
ПРОГРАММА КЛИМАТИЧЕСКОГО РАЙОНИРОВАНИЯ ТЕРРИТОРИЙ
MVP — инкремент 2: многотиповая модель данных (температура/осадки/
справочник станций), контекстная описательная статистика, настраиваемые
графики, автопроверка мультиколлинеарности.
Запуск: streamlit run app.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from modules import data_io, descriptive, correlation, viz, indicators, tests_stat, clustering, mapping
from config import PALETTE

st.set_page_config(page_title='Климатическое районирование', layout='wide')

# ─── Состояние сессии ────────────────────────────────────────────────────
if 'typed_datasets' not in st.session_state:
    st.session_state.typed_datasets = {}       # {'Температура': long_df, 'Осадки': long_df, ...}
if 'stations_ref' not in st.session_state:
    st.session_state.stations_ref = None        # DataFrame справочника станций (сырой)
if 'stations_ref_map' not in st.session_state:
    st.session_state.stations_ref_map = None     # словарь сопоставления столбцов
if 'long_combined' not in st.session_state:
    st.session_state.long_combined = None        # итоговая объединённая таблица
if 'indicators_table' not in st.session_state:
    st.session_state.indicators_table = None     # рассчитанные климатические индексы
if 'subsamples' not in st.session_state:
    st.session_state.subsamples = {}             # {'Весь период': (y_min, y_max), 'Ранний период': (...), ...}
if 'feature_sets' not in st.session_state:
    st.session_state.feature_sets = {}           # {'название набора': [список признаков]}

st.title('🌍 Климатическое районирование территорий')
st.caption('Кластерный анализ и статистика климатических характеристик')

tab_data, tab_sub, tab_desc, tab_ind, tab_corr, tab_cluster = st.tabs([
    '1. Данные', '2. Подвыборки и периоды', '3. Описательная статистика',
    '4. Климатические индексы', '5. Корреляция и мультиколлинеарность',
    '6. Районирование (кластеризация)'
])

CHART_SIZE_PRESETS = {
    'Компактный (8×5)': (8, 5),
    'Матрица (10×8)': (10, 8),
    'Широкий (12×8)': (12, 8),
    'Крупный (14×12)': (14, 12),
}


def size_and_color_controls(key_prefix, default_size='Компактный (8×5)', with_color=True):
    """Универсальная панель настройки графика: размер, цвет, легенда."""
    with st.expander('⚙️ Настройки графика (размер, цвет, легенда)'):
        c1, c2, c3 = st.columns(3)
        with c1:
            size_label = st.selectbox('Размер', list(CHART_SIZE_PRESETS.keys()),
                                       index=list(CHART_SIZE_PRESETS.keys()).index(default_size),
                                       key=f'{key_prefix}_size')
            custom = st.checkbox('Свой размер', key=f'{key_prefix}_custom_size')
            if custom:
                w = st.slider('Ширина, дюймы', 4, 20, CHART_SIZE_PRESETS[size_label][0],
                               key=f'{key_prefix}_w')
                h = st.slider('Высота, дюймы', 3, 16, CHART_SIZE_PRESETS[size_label][1],
                               key=f'{key_prefix}_h')
                figsize = (w, h)
            else:
                figsize = CHART_SIZE_PRESETS[size_label]
        color = None
        with c2:
            if with_color:
                use_custom_color = st.checkbox('Свой цвет', key=f'{key_prefix}_custom_color')
                if use_custom_color:
                    color = st.color_picker('Цвет', PALETTE[0], key=f'{key_prefix}_color')
                else:
                    color_name = st.selectbox('Цвет из палитры',
                                               [f'Цвет {i+1}' for i in range(len(PALETTE))],
                                               key=f'{key_prefix}_palette_color')
                    color = PALETTE[int(color_name.split()[-1]) - 1]
        with c3:
            legend_loc = st.selectbox('Позиция легенды', list(viz.LEGEND_LOC_OPTIONS.keys()),
                                       key=f'{key_prefix}_legend')
    return figsize, color, legend_loc


def show_caption(chart_key):
    title, desc = viz.CHART_DESCRIPTIONS[chart_key]
    st.markdown(f'**{title}.** {desc}')


def show_plot(fig, width_frac=0.62):
    """
    Показывает график в колонке ограниченной ширины, а не на весь экран —
    удобнее для восприятия при широком layout='wide'.
    """
    col, _ = st.columns([width_frac, 1 - width_frac])
    with col:
        st.pyplot(fig, use_container_width=True)


def subsample_selector(ld, key):
    """
    Выпадающий список подвыборок (периодов), сформированных во вкладке
    "Подвыборки и периоды". Возвращает отфильтрованный по годам DataFrame.
    Если подвыборок ещё нет — возвращает исходный ld без изменений.
    """
    if not st.session_state.subsamples:
        return ld, 'Весь период (по умолчанию)'
    names = list(st.session_state.subsamples.keys())
    name = st.selectbox('Подвыборка (период)', names, key=key)
    y1, y2 = st.session_state.subsamples[name]
    return ld[(ld['Год'] >= y1) & (ld['Год'] <= y2)], name


def download_row(fig, base_name, key_prefix):
    c1, c2 = st.columns(2)
    with c1:
        st.download_button('📥 Скачать PNG', viz.fig_to_bytes(fig, 'png'),
                            file_name=f'{base_name}.png', key=f'{key_prefix}_png')
    with c2:
        st.download_button('📥 Скачать PDF', viz.fig_to_bytes(fig, 'pdf'),
                            file_name=f'{base_name}.pdf', key=f'{key_prefix}_pdf')


def bbox_filter_controls(polygons, d_clean, key_prefix):
    """
    UI-блок обрезки полигонов по прямоугольнику координат. По умолчанию
    предлагает границы с запасом вокруг станций текущего набора данных.
    Возвращает обрезанный список полигонов.
    """
    with st.expander('✂️ Обрезать по прямоугольнику координат (если полигонов слишком много)'):
        lat_min_d, lat_max_d = float(d_clean['Lat'].min()), float(d_clean['Lat'].max())
        lon_min_d, lon_max_d = float(d_clean['Lon'].min()), float(d_clean['Lon'].max())
        pad_lat = (lat_max_d - lat_min_d) * 0.3 + 0.5
        pad_lon = (lon_max_d - lon_min_d) * 0.3 + 0.5
        use_bbox = st.checkbox('Включить обрезку', value=False, key=f'{key_prefix}_use_bbox')
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            lon_min = st.number_input('Долгота от', value=round(lon_min_d - pad_lon, 2),
                                       key=f'{key_prefix}_lon_min')
        with c2:
            lon_max = st.number_input('Долгота до', value=round(lon_max_d + pad_lon, 2),
                                       key=f'{key_prefix}_lon_max')
        with c3:
            lat_min = st.number_input('Широта от', value=round(lat_min_d - pad_lat, 2),
                                       key=f'{key_prefix}_lat_min')
        with c4:
            lat_max = st.number_input('Широта до', value=round(lat_max_d + pad_lat, 2),
                                       key=f'{key_prefix}_lat_max')
        if use_bbox:
            cropped = mapping.filter_polygons_by_bbox(polygons, lon_min, lon_max, lat_min, lat_max)
            st.info(f'После обрезки осталось полигонов: {len(cropped)} из {len(polygons)}')
            return cropped
    return polygons


def build_district_assignment_table(assigned):
    """
    Таблица район→кластер: сверху районы, где кластер определён по станциям
    внутри границ (отсортированы по числу станций, потом по имени), затем —
    районы без станций, где кластер присвоен по ближайшей станции. Каждая
    строка поясняет, каким способом определён кластер.
    """
    rows = []
    for a in assigned:
        has_stations = a['n_stations_inside'] > 0
        method = ('По станциям внутри района' if has_stations
                  else 'По ближайшей станции (в границах района станций нет)')
        rows.append({
            'Район': a['name'], 'Кластер': a['cluster'],
            'Станций внутри': a['n_stations_inside'],
            'Способ определения кластера': method,
            '_has_stations': has_stations,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(['_has_stations', 'Станций внутри', 'Район'],
                         ascending=[False, False, True]).drop(columns='_has_stations')
    return df.reset_index(drop=True)


# =============================================================================
# ВКЛАДКА 1: ДАННЫЕ
# =============================================================================
with tab_data:
    st.header('Загрузка данных')
    st.markdown(
        'Загрузите один или несколько файлов: обычно нужны отдельные наборы '
        '**температуры** и **осадков**, и по желанию — **справочник станций** '
        '(код станции, название, регион, широта/долгота — для карты).'
    )

    uploaded_files = st.file_uploader(
        'Файлы (.xlsx, .csv) — можно выбрать несколько сразу',
        type=['xlsx', 'xls', 'csv'], accept_multiple_files=True)

    if uploaded_files:
        for f in uploaded_files:
            with st.expander(f'📄 {f.name}', expanded=True):
                try:
                    loaded = data_io.load_table(f)
                except Exception as e:
                    st.error(f'Ошибка загрузки: {e}')
                    continue

                if isinstance(loaded, dict):
                    sheet = st.selectbox('Лист Excel', list(loaded.keys()),
                                          key=f'sheet_{f.name}')
                    df_raw = loaded[sheet]
                else:
                    df_raw = loaded
                    sheet = None

                st.caption(f'{df_raw.shape[0]} строк × {df_raw.shape[1]} столбцов')
                st.dataframe(df_raw.head(8), use_container_width=True)

                name_for_guess = sheet if sheet else f.name.rsplit('.', 1)[0]
                guessed_role = data_io.guess_role(name_for_guess, df_raw)
                role_options = ['— выбрать —', 'Климатический показатель (температура, осадки и т.п.)',
                                 'Справочник станций (координаты, регион)', 'Пропустить']
                role_default_idx = {'climate': 1, 'station_ref': 2}.get(guessed_role, 0)
                if guessed_role:
                    st.caption(f'🔍 Похоже на: {"климатический показатель" if guessed_role == "climate" else "справочник станций"} '
                               f'(можно поменять ниже, если ошибся)')

                role = st.selectbox('Роль этого файла/листа', role_options,
                                     index=role_default_idx, key=f'role_{f.name}_{sheet}')

                if role == 'Климатический показатель (температура, осадки и т.п.)':
                    fmt = data_io.detect_format(df_raw)
                    if fmt != 'wide_months':
                        st.warning('Не похоже на широкий формат "Станция/Год + 12 месяцев". '
                                   'Убедитесь, что столбцы месяцев называются как в списке '
                                   '(Январь, Февраль, ...).')
                    guessed_label = data_io.guess_indicator_label(name_for_guess, df_raw)
                    default_label = guessed_label or name_for_guess
                    tip_label = st.text_input('Название показателя (например "Температура", '
                                               '"Осадки")', value=default_label,
                                               key=f'label_{f.name}_{sheet}')
                    id_candidates = [c for c in df_raw.columns
                                      if c not in data_io.MONTHS_RU + data_io.MONTHS_RU_SHORT]
                    id_vars = st.multiselect('Столбцы-идентификаторы (Станция, Год)',
                                              id_candidates, default=id_candidates,
                                              key=f'idvars_{f.name}_{sheet}')
                    if st.button(f'➕ Добавить как показатель «{tip_label}»',
                                 key=f'add_{f.name}_{sheet}'):
                        try:
                            long_typed = data_io.wide_months_to_long_typed(
                                df_raw, id_vars, tip_label)
                            st.session_state.typed_datasets[tip_label] = long_typed
                            st.success(f'Показатель «{tip_label}» добавлен '
                                       f'({len(long_typed)} строк).')
                        except Exception as e:
                            st.error(f'Ошибка преобразования: {e}')

                elif role == 'Справочник станций (координаты, регион)':
                    cols = df_raw.columns.tolist()
                    guessed_map = data_io.guess_station_column_map(df_raw)

                    def _idx(options, guessed):
                        return options.index(guessed) if guessed in options else 0

                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        col_station = st.selectbox('Код станции', cols,
                                                    index=_idx(cols, guessed_map.get('Станция')),
                                                    key=f'ref_st_{f.name}')
                    with c2:
                        opts_name = ['—'] + cols
                        col_name = st.selectbox('Название (опц.)', opts_name,
                                                 index=_idx(opts_name, guessed_map.get('Название')),
                                                 key=f'ref_name_{f.name}')
                    with c3:
                        opts_region = ['—'] + cols
                        col_region = st.selectbox('Регион (опц.)', opts_region,
                                                   index=_idx(opts_region, guessed_map.get('Регион')),
                                                   key=f'ref_region_{f.name}')
                    with c4:
                        opts_lat = ['—'] + cols
                        col_lat = st.selectbox('Широта (опц.)', opts_lat,
                                                index=_idx(opts_lat, guessed_map.get('Lat')),
                                                key=f'ref_lat_{f.name}')
                    with c5:
                        opts_lon = ['—'] + cols
                        col_lon = st.selectbox('Долгота (опц.)', opts_lon,
                                                index=_idx(opts_lon, guessed_map.get('Lon')),
                                                key=f'ref_lon_{f.name}')
                    if st.button('➕ Использовать как справочник станций',
                                 key=f'add_ref_{f.name}'):
                        st.session_state.stations_ref = df_raw
                        st.session_state.stations_ref_map = {
                            'Станция': col_station,
                            'Название': None if col_name == '—' else col_name,
                            'Регион': None if col_region == '—' else col_region,
                            'Lat': None if col_lat == '—' else col_lat,
                            'Lon': None if col_lon == '—' else col_lon,
                        }
                        st.success('Справочник станций сохранён.')

        st.divider()
        st.subheader('Текущее состояние')
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('**Загруженные климатические показатели:**')
            if st.session_state.typed_datasets:
                for tip, d in st.session_state.typed_datasets.items():
                    st.write(f'• {tip}: {len(d)} строк, '
                             f'{d["Станция"].nunique()} станций, '
                             f'{d["Год"].min()}–{d["Год"].max()} гг.')
            else:
                st.write('— пока ничего не добавлено')
        with c2:
            st.markdown('**Справочник станций:**')
            if st.session_state.stations_ref is not None:
                st.write(f'✅ загружен, {len(st.session_state.stations_ref)} записей')
            else:
                st.write('— не загружен (карта и региональный анализ будут недоступны)')

        st.divider()
        if st.session_state.typed_datasets:
            if st.button('🔗 Собрать общий набор данных', type='primary'):
                with st.status('⏳ Идёт расчёт: объединение данных...', expanded=True) as status:
                    combined = data_io.combine_typed_datasets(
                        list(st.session_state.typed_datasets.values()))
                    if st.session_state.stations_ref is not None:
                        try:
                            combined = data_io.merge_station_reference(
                                combined, st.session_state.stations_ref,
                                st.session_state.stations_ref_map)
                        except Exception as e:
                            st.warning(f'Не удалось присоединить справочник станций: {e}')
                    status.update(label='✅ Данные объединены', state='complete')
                st.session_state.long_combined = combined
                y_min, y_max = int(combined['Год'].min()), int(combined['Год'].max())
                st.session_state.subsamples['Весь период'] = (y_min, y_max)
                st.success(f'Общий набор данных собран: {len(combined)} строк.')

        if st.session_state.long_combined is not None:
            st.subheader('Предпросмотр общего набора данных')
            st.dataframe(st.session_state.long_combined.head(20), use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.download_button('Скачать (.xlsx)',
                                    data_io.df_to_excel_bytes(st.session_state.long_combined),
                                    file_name='combined_long.xlsx')
            with c2:
                st.download_button('Скачать (.csv)',
                                    data_io.df_to_csv_bytes(st.session_state.long_combined),
                                    file_name='combined_long.csv')
    else:
        st.info('Загрузите хотя бы один файл, чтобы начать.')


# =============================================================================
# ВКЛАДКА 2: ПОДВЫБОРКИ И ПЕРИОДЫ
# =============================================================================
with tab_sub:
    st.header('Проверка однородности ряда и формирование подвыборок')
    st.markdown(
        'Тест Петтитта ищет год наиболее вероятного изменения климатического '
        'режима, ITA (Şen, 2012) проверяет наличие значимого тренда. На основе '
        'этих тестов формируется рекомендация — стоит ли делить весь период '
        'на подвыборки (например «ранний» и «современный»), и предлагается '
        'год разбиения.'
    )

    ld = st.session_state.long_combined
    if ld is None or ld.empty:
        st.info('Сначала соберите общий набор данных во вкладке "Данные".')
    else:
        tips = data_io.unique_values(ld, 'Тип')
        c1, c2 = st.columns(2)
        with c1:
            test_tip = st.selectbox('Показатель для проверки', tips, key='test_tip')
        with c2:
            level_opt = ['Все станции (среднее)']
            if 'Регион' in ld.columns and ld['Регион'].notna().any():
                level_opt.append('Конкретный регион')
            level_opt.append('Конкретная станция')
            test_level = st.selectbox('Уровень', level_opt, key='test_level')

        test_station, test_region = None, None
        if test_level == 'Конкретная станция':
            names = data_io.unique_values(ld, 'Название') or data_io.unique_values(ld, 'Станция')
            lbl = st.selectbox('Станция', names, key='test_station_lbl')
            test_station = (ld.loc[ld['Название'] == lbl, 'Станция'].iloc[0]
                             if 'Название' in ld.columns else lbl)
        elif test_level == 'Конкретный регион':
            test_region = st.selectbox('Регион', data_io.unique_values(ld, 'Регион'),
                                        key='test_region_sel')

        agg = data_io.default_agg_for_type(test_tip)
        ys = data_io.yearly_series(ld, test_tip, station=test_station, region=test_region, agg=agg)

        if len(ys.dropna()) < 8:
            st.warning('Недостаточно точек ряда для тестов (нужно минимум 8 лет).')
        else:
            rec = tests_stat.recommend_periods(ys)
            st.markdown(rec['text'])

            figsize, color, _ = size_and_color_controls('pettitt', default_size='Широкий (12×8)')
            show_caption('pettitt')
            fig = viz.pettitt_chart(ys, rec['pettitt'],
                                     title=f'{test_tip}: {test_level}'
                                           + (f' ({test_station or test_region})'
                                              if (test_station or test_region) else ''),
                                     figsize=figsize, color=color)
            show_plot(fig)
            download_row(fig, f'graph_pettitt_{test_tip}', 'pettitt')

            st.divider()
            st.subheader('Создать именованную подвыборку')
            y_min_all, y_max_all = int(ld['Год'].min()), int(ld['Год'].max())
            with st.form('add_subsample_form'):
                c1, c2, c3 = st.columns(3)
                with c1:
                    sub_name = st.text_input('Название подвыборки',
                                              placeholder='например "Современный период"')
                with c2:
                    sub_y1 = st.number_input('Год с', min_value=y_min_all, max_value=y_max_all,
                                              value=y_min_all)
                with c3:
                    sub_y2 = st.number_input('Год по', min_value=y_min_all, max_value=y_max_all,
                                              value=y_max_all)
                submitted = st.form_submit_button('➕ Добавить подвыборку')
                if submitted:
                    if not sub_name.strip():
                        st.error('Укажите название подвыборки.')
                    elif sub_y1 > sub_y2:
                        st.error('"Год с" не может быть больше "Год по".')
                    else:
                        st.session_state.subsamples[sub_name.strip()] = (int(sub_y1), int(sub_y2))
                        st.success(f'Подвыборка «{sub_name}» ({sub_y1}–{sub_y2}) добавлена.')

            if rec['pettitt']['change_year'] is not None:
                cy = rec['pettitt']['change_year']
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(f'⚡ Быстро создать «Ранний период» ({y_min_all}–{cy})'):
                        st.session_state.subsamples['Ранний период'] = (y_min_all, cy)
                        st.success('Добавлено.')
                with c2:
                    if st.button(f'⚡ Быстро создать «Современный период» ({cy+1}–{y_max_all})'):
                        st.session_state.subsamples['Современный период'] = (cy + 1, y_max_all)
                        st.success('Добавлено.')

        st.divider()
        st.subheader('Текущие подвыборки')
        if st.session_state.subsamples:
            for name, (y1, y2) in list(st.session_state.subsamples.items()):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.write(f'• **{name}**: {y1}–{y2} гг.')
                with c2:
                    if name != 'Весь период' and st.button('🗑️ Удалить', key=f'del_sub_{name}'):
                        del st.session_state.subsamples[name]
                        st.rerun()
        else:
            st.info('Подвыборок пока нет — соберите данные во вкладке "Данные", '
                    '«Весь период» появится автоматически.')

        st.caption('Выбор подвыборки (периода) доступен на вкладках "Описательная '
                   'статистика", "Климатические индексы" и "Корреляция" — везде, где '
                   'нужно ограничить анализ конкретным диапазоном лет.')


# =============================================================================
# ВКЛАДКА 2: ОПИСАТЕЛЬНАЯ СТАТИСТИКА
# =============================================================================
with tab_desc:
    st.header('Описательная статистика')

    ld = st.session_state.long_combined
    if ld is None or ld.empty:
        st.info('Сначала соберите общий набор данных во вкладке "Данные".')
    else:
        ld, active_sub = subsample_selector(ld, 'sub_desc')
        st.subheader('Контекст анализа')
        c1, c2, c3 = st.columns(3)
        with c1:
            tips = data_io.unique_values(ld, 'Тип')
            tip = st.selectbox('Показатель', tips)
        with c2:
            level_options = ['Все станции']
            if 'Регион' in ld.columns and ld['Регион'].notna().any():
                level_options.append('Конкретный регион')
            level_options.append('Конкретная станция')
            level = st.selectbox('Уровень агрегации', level_options)
        with c3:
            period_mode = st.selectbox('Период', ['Весь период', 'Конкретный год',
                                                    'Конкретный месяц (по всем годам)'])

        station, region, year, month = None, None, None, None
        if level == 'Конкретная станция':
            station_names = data_io.unique_values(ld, 'Название') or data_io.unique_values(ld, 'Станция')
            station_label = st.selectbox('Станция', station_names)
            if 'Название' in ld.columns:
                station = ld.loc[ld['Название'] == station_label, 'Станция'].iloc[0]
            else:
                station = station_label
        elif level == 'Конкретный регион':
            region = st.selectbox('Регион', data_io.unique_values(ld, 'Регион'))

        if period_mode == 'Конкретный год':
            year = st.selectbox('Год', data_io.unique_values(ld, 'Год'))
        elif period_mode == 'Конкретный месяц (по всем годам)':
            month = st.selectbox('Месяц', data_io.unique_values(ld, 'Месяц'))

        filtered = data_io.filter_long(ld, tip=tip, station=station, region=region,
                                        year=year, month=month)

        context_desc = f'{active_sub} · {tip} · {level}'
        if station:
            context_desc += f' ({station})'
        if region:
            context_desc += f' ({region})'
        context_desc += f' · {period_mode}'
        if year:
            context_desc += f' ({year})'
        if month:
            context_desc += f' ({month})'
        st.info(f'**Текущий контекст:** {context_desc} — {len(filtered)} наблюдений')

        if filtered.empty:
            st.warning('Нет данных для выбранного сочетания фильтров.')
        else:
            st.subheader(f'Описательная статистика: {context_desc}')
            desc_target = descriptive.descriptive_target(filtered['Значение'], tip)
            st.dataframe(desc_target, use_container_width=True)
            if not desc_target.empty:
                st.markdown(f'**Вывод о нормальности:** '
                            f'{descriptive.normality_conclusion(desc_target.iloc[0])}')
            safe_name = context_desc.replace(' ', '_').replace('·', '-')
            st.download_button('Скачать таблицу (.xlsx)',
                                data_io.df_to_excel_bytes(desc_target, 'Описательная стат.'),
                                file_name=f'table_01_descriptive_{safe_name}.xlsx')

            st.divider()
            st.subheader('Графики')

            chart_type = st.radio('Тип графика',
                                   ['Гистограмма с KDE', 'Ящик с усами (по группам)',
                                    'Динамика по годам', 'Совмещённая динамика 2 показателей'],
                                   horizontal=True)

            if chart_type == 'Гистограмма с KDE':
                show_caption('hist_kde')
                figsize, color, legend_loc = size_and_color_controls('hist')
                fig = viz.hist_kde(filtered['Значение'], title=f'Распределение: {context_desc}',
                                    xlabel=tip, figsize=figsize, color=color, legend_loc=legend_loc)
                show_plot(fig)
                download_row(fig, f'graph_hist_{safe_name}', 'hist')

            elif chart_type == 'Ящик с усами (по группам)':
                show_caption('boxplot')
                group_by = st.selectbox('Группировать по', ['Станция', 'Год', 'Регион']
                                         if 'Регион' in filtered.columns else ['Станция', 'Год'])
                annotate_on_chart = st.checkbox('Показывать Мин/Макс подписи на самом графике',
                                                 value=False,
                                                 help='При многих группах подписи накладываются '
                                                      'друг на друга — по умолчанию сводка '
                                                      'выводится отдельной таблицей ниже.')
                figsize, color, legend_loc = size_and_color_controls('box', with_color=False)
                group_col = 'Название' if group_by == 'Станция' and 'Название' in filtered.columns else group_by
                fig = viz.boxplot_groups(filtered, 'Значение', group_col,
                                          title=f'{tip} по {group_by.lower()}ам', ylabel=tip,
                                          figsize=figsize, legend_loc=legend_loc,
                                          annotate_minmax=annotate_on_chart)
                show_plot(fig)
                download_row(fig, f'graph_box_{safe_name}', 'box')

                st.markdown('**Мин/Макс/Среднее/Медиана по группам:**')
                summary_df = viz.boxplot_minmax_summary(filtered, 'Значение', group_col)
                st.dataframe(summary_df, use_container_width=True)

            elif chart_type == 'Динамика по годам':
                show_caption('timeseries')
                agg_mode = st.radio('Агрегация месяцев в год', ['Среднее', 'Сумма'],
                                     horizontal=True,
                                     index=0 if data_io.default_agg_for_type(tip) == 'mean' else 1)
                agg = 'mean' if agg_mode == 'Среднее' else 'sum'
                show_trend = st.checkbox('Показать линию тренда', value=True)
                show_smooth = st.checkbox('Показать скользящее среднее (11 лет)', value=False)
                figsize, color, legend_loc = size_and_color_controls('ts')
                ys = data_io.yearly_series(ld, tip, station=station, region=region, agg=agg)
                fig = viz.time_series_line(ys, title=f'Динамика: {context_desc}',
                                            ylabel=tip, figsize=figsize, color=color,
                                            show_trend=show_trend, show_smooth=show_smooth,
                                            legend_loc=legend_loc)
                show_plot(fig)
                download_row(fig, f'graph_dynamics_{safe_name}', 'ts')

            elif chart_type == 'Совмещённая динамика 2 показателей':
                show_caption('timeseries_combo')
                if len(tips) < 2:
                    st.warning('Нужно как минимум два показателя (например Температура и Осадки).')
                else:
                    other_tip = st.selectbox('Второй показатель', [t for t in tips if t != tip])
                    agg1 = data_io.default_agg_for_type(tip)
                    agg2 = data_io.default_agg_for_type(other_tip)
                    figsize, _, legend_loc = size_and_color_controls('combo', with_color=False)
                    ys1 = data_io.yearly_series(ld, tip, station=station, region=region, agg=agg1)
                    ys2 = data_io.yearly_series(ld, other_tip, station=station, region=region, agg=agg2)
                    fig = viz.time_series_combo(ys1, ys2, tip, other_tip,
                                                 title=f'{tip} и {other_tip}: {level}'
                                                       + (f' ({station or region})' if (station or region) else ''),
                                                 figsize=figsize, legend_loc=legend_loc)
                    show_plot(fig)
                    download_row(fig, f'graph_combo_{safe_name}', 'combo')


# =============================================================================
# ВКЛАДКА 3: КЛИМАТИЧЕСКИЕ ИНДЕКСЫ
# =============================================================================
with tab_ind:
    st.header('Расчёт климатических показателей')
    st.markdown(
        'На основе температуры и осадков рассчитываются производные '
        'характеристики по каждой станции: сезонные средние/суммы, индекс '
        'континентальности Конрада, индексы аридности и увлажнения, ГТК, '
        'линейные тренды. Эта таблица — основа для проверки '
        'мультиколлинеарности и дальнейшего отбора признаков.'
    )

    ld = st.session_state.long_combined
    if ld is None or ld.empty:
        st.info('Сначала соберите общий набор данных во вкладке "Данные".')
    else:
        ld, active_sub_ind = subsample_selector(ld, 'sub_ind')
        st.caption(f'Активная подвыборка: **{active_sub_ind}**')
        tips = data_io.unique_values(ld, 'Тип')
        if len(tips) < 2:
            st.warning('Нужны минимум два показателя (температура и осадки), '
                       'чтобы рассчитать производные индексы.')
        else:
            c1, c2 = st.columns(2)
            with c1:
                temp_tip = st.selectbox('Какой показатель — температура?', tips,
                                         index=next((i for i, t in enumerate(tips)
                                                      if 'температ' in t.lower()), 0))
            with c2:
                precip_candidates = [t for t in tips if t != temp_tip]
                precip_tip = st.selectbox('Какой показатель — осадки?', precip_candidates,
                                           index=next((i for i, t in enumerate(precip_candidates)
                                                        if 'осад' in t.lower()), 0))

            has_lat = 'Lat' in ld.columns and ld['Lat'].notna().any()
            if not has_lat:
                st.info('Справочник станций с широтой не загружен — индекс '
                        'континентальности Конрада не будет рассчитан (нужна широта). '
                        'Остальные показатели считаются без неё.')

            if st.button('🧮 Рассчитать индексы', type='primary'):
                lat_map = None
                if has_lat:
                    lat_map = (ld[['Станция', 'Lat']].drop_duplicates('Станция')
                               .set_index('Станция')['Lat'].to_dict())
                with st.status('⏳ Идёт расчёт: климатические показатели по станциям...',
                                expanded=True) as status:
                    ind_table = indicators.compute_indicators(ld, temp_tip, precip_tip, lat_map=lat_map)
                    status.update(label='✅ Индексы рассчитаны', state='complete')
                st.session_state.indicators_table = ind_table
                st.success(f'Рассчитано для {len(ind_table)} станций.')

            if st.session_state.indicators_table is not None:
                ind_table = st.session_state.indicators_table
                st.subheader('Таблица индексов по станциям')
                st.dataframe(ind_table, use_container_width=True)
                st.download_button('Скачать таблицу (.xlsx)',
                                    data_io.df_to_excel_bytes(ind_table, 'Индексы'),
                                    file_name='table_indicators.xlsx')

                with st.expander('📖 Справка: как считается каждый показатель'):
                    for name, formula in indicators.INDICATOR_FORMULAS.items():
                        st.markdown(f'**{name}** — {formula}')

                st.divider()
                st.subheader('Описательная статистика по индексам')
                feat_cols = indicators.indicator_feature_columns(ind_table)
                desc_all = descriptive.descriptive_all_features(ind_table, feat_cols)
                st.dataframe(desc_all, use_container_width=True)
                st.download_button('Скачать (.xlsx)',
                                    data_io.df_to_excel_bytes(desc_all, 'Стат. по индексам'),
                                    file_name='table_02_descriptive_indicators.xlsx')


# =============================================================================
# ВКЛАДКА 4: КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# =============================================================================
with tab_corr:
    st.header('Корреляционный анализ и мультиколлинеарность')

    ld = st.session_state.long_combined
    if ld is None or ld.empty:
        st.info('Сначала соберите общий набор данных во вкладке "Данные".')
    else:
        ld, active_sub_corr = subsample_selector(ld, 'sub_corr')
        st.caption(f'Активная подвыборка (для режимов «месяцы» и «сопоставление»): '
                   f'**{active_sub_corr}**. Режим по индексам использует подвыборку, '
                   f'выбранную при расчёте во вкладке 4.')
        st.markdown(
            'Здесь анализируется связь **между месяцами** одного показателя '
            '(например, насколько похожи январь и февраль между собой — это '
            'полезно для отбора неизбыточных признаков), связь между '
            '**двумя показателями** (например температура и осадки), либо '
            'мультиколлинеарность среди **рассчитанных климатических индексов** '
            '(вкладка 3) — рекомендуется именно этот режим для финального '
            'отбора признаков.'
        )

        mode_options = ['Мультиколлинеарность месяцев одного показателя',
                         'Сопоставление двух показателей (T vs P и т.п.)']
        if st.session_state.indicators_table is not None:
            mode_options.insert(0, 'Мультиколлинеарность климатических индексов (рекоменд.)')
        mode = st.radio('Режим анализа', mode_options, horizontal=False)

        if mode == 'Мультиколлинеарность климатических индексов (рекоменд.)':
            ind_table = st.session_state.indicators_table
            feat_cols = indicators.indicator_feature_columns(ind_table)
            st.subheader('Матрица корреляций между индексами')
            figsize, _, _ = size_and_color_controls('heat_ind', with_color=False,
                                                      default_size='Матрица (10×8)')
            corr_df, pval_df = correlation.correlation_matrix(ind_table, feat_cols, method='pearson')
            show_caption('heatmap')
            fig = viz.correlation_heatmap(corr_df, title='Корреляция климатических индексов',
                                           figsize=figsize)
            show_plot(fig)
            download_row(fig, 'graph_heatmap_indicators', 'heat_ind')

            st.divider()
            st.subheader('Автоматическая проверка мультиколлинеарности')
            threshold = st.slider('Порог |r| для признания признаков избыточными',
                                   0.5, 0.99, 0.8, 0.01, key='thr_ind')
            pairs_df, keep_list, drop_list = correlation.auto_flag_multicollinearity(
                ind_table, feat_cols, threshold=threshold)
            if pairs_df.empty:
                st.success('Пар с корреляцией выше порога не найдено.')
                keep_list = feat_cols
            else:
                st.dataframe(pairs_df, use_container_width=True)
                st.markdown(f'**Рекомендуется оставить:** {", ".join(keep_list)}')
                st.markdown(f'**Рекомендуется исключить (избыточны):** '
                            f'{", ".join(drop_list) if drop_list else "—"}')
                st.download_button('Скачать список пар (.xlsx)',
                                    data_io.df_to_excel_bytes(pairs_df, 'Мультиколлинеарность'),
                                    file_name='table_multicollinearity_indicators.xlsx')

            st.divider()
            st.subheader('💾 Сохранить набор признаков')
            st.caption('Отредактируйте список при необходимости и сохраните под именем — '
                       'этот набор станет доступен во вкладке "Важность признаков" и при '
                       'дальнейшем отборе/кластеризации.')
            chosen_features = st.multiselect('Признаки в наборе', feat_cols,
                                              default=keep_list, key='fs_multiselect')
            fs_name = st.text_input('Название набора', value='без мультиколлинеарности',
                                     key='fs_name_input')
            if st.button('💾 Сохранить набор признаков', type='primary'):
                if not fs_name.strip():
                    st.error('Укажите название набора.')
                elif not chosen_features:
                    st.error('Выберите хотя бы один признак.')
                else:
                    st.session_state.feature_sets[fs_name.strip()] = chosen_features
                    st.success(f'Набор «{fs_name}» сохранён ({len(chosen_features)} признаков).')

            if st.session_state.feature_sets:
                st.markdown('**Сохранённые наборы признаков:**')
                for name, feats in list(st.session_state.feature_sets.items()):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.write(f'• **{name}**: {", ".join(feats)}')
                    with c2:
                        if st.button('🗑️', key=f'del_fs_{name}'):
                            del st.session_state.feature_sets[name]
                            st.rerun()

        elif mode == 'Мультиколлинеарность месяцев одного показателя':
            tip = st.selectbox('Показатель', data_io.unique_values(ld, 'Тип'), key='corr_tip')
            d = data_io.filter_long(ld, tip=tip)
            wide = data_io.long_to_wide(d, ['Станция', 'Год'], 'Месяц', 'Значение')
            month_cols = [m for m in data_io.MONTHS_RU if m in wide.columns]

            if len(month_cols) < 2:
                st.warning('Недостаточно месяцев для анализа.')
            else:
                st.subheader('Матрица корреляций между месяцами')
                figsize, _, _ = size_and_color_controls('heat', with_color=False,
                                                          default_size='Матрица (10×8)')
                corr_df, pval_df = correlation.correlation_matrix(wide, month_cols, method='pearson')
                show_caption('heatmap')
                fig = viz.correlation_heatmap(corr_df, title=f'{tip}: корреляция месяцев',
                                               figsize=figsize)
                show_plot(fig)
                download_row(fig, f'graph_heatmap_{tip}', 'heat')

                st.divider()
                st.subheader('Автоматическая проверка мультиколлинеарности')
                threshold = st.slider('Порог |r| для признания признаков избыточными',
                                       0.5, 0.99, 0.8, 0.01)
                pairs_df, keep_list, drop_list = correlation.auto_flag_multicollinearity(
                    wide, month_cols, threshold=threshold)
                if pairs_df.empty:
                    st.success('Пар с корреляцией выше порога не найдено — '
                               'мультиколлинеарность не выявлена.')
                else:
                    st.dataframe(pairs_df, use_container_width=True)
                    st.markdown(f'**Рекомендуется оставить:** {", ".join(keep_list)}')
                    st.markdown(f'**Рекомендуется исключить (избыточны):** '
                                f'{", ".join(drop_list) if drop_list else "—"}')
                    st.download_button('Скачать список пар (.xlsx)',
                                        data_io.df_to_excel_bytes(pairs_df, 'Мультиколлинеарность'),
                                        file_name=f'table_multicollinearity_{tip}.xlsx')

        else:
            tip1 = st.selectbox('Показатель 1', data_io.unique_values(ld, 'Тип'), key='corr_t1')
            remaining = [t for t in data_io.unique_values(ld, 'Тип') if t != tip1]
            if not remaining:
                st.warning('Нужно как минимум два разных показателя (например Температура и Осадки).')
            else:
                tip2 = st.selectbox('Показатель 2', remaining, key='corr_t2')
                agg1 = data_io.default_agg_for_type(tip1)
                agg2 = data_io.default_agg_for_type(tip2)
                st.caption(f'Агрегация в годовое значение: {tip1} → '
                           f'{"среднее" if agg1=="mean" else "сумма"}, '
                           f'{tip2} → {"среднее" if agg2=="mean" else "сумма"}')

                d1 = data_io.filter_long(ld, tip=tip1)
                d2 = data_io.filter_long(ld, tip=tip2)
                y1 = d1.groupby(['Станция', 'Год'])['Значение'].agg(agg1).reset_index()
                y2 = d2.groupby(['Станция', 'Год'])['Значение'].agg(agg2).reset_index()
                merged = y1.merge(y2, on=['Станция', 'Год'], suffixes=(f'_{tip1}', f'_{tip2}'))
                col1, col2 = f'Значение_{tip1}', f'Значение_{tip2}'
                merged = merged.rename(columns={f'Значение_{tip1}': col1, f'Значение_{tip2}': col2})

                if merged.empty:
                    st.warning('Нет совпадающих станций/лет между показателями.')
                else:
                    st.subheader('Диаграмма рассеяния')
                    show_caption('scatter')
                    figsize, color, legend_loc = size_and_color_controls('scatter2')
                    fig = viz.scatter_with_trend(
                        merged, col1, col2,
                        title=f'{tip1} vs {tip2} (по станциям и годам)',
                        xlabel=tip1, ylabel=tip2, figsize=figsize, color=color,
                        legend_loc=legend_loc)
                    show_plot(fig)
                    download_row(fig, f'graph_scatter_{tip1}_{tip2}', 'scatter2')

                    st.divider()
                    st.subheader('Ранжированная корреляция')
                    st.caption('Здесь считается прямая связь между годовыми значениями '
                               'двух показателей по всем станциям — не «целевая переменная», '
                               'а просто сопоставление рядов.')
                    r_p, p_p = correlation.stats.pearsonr(merged[col1], merged[col2])
                    r_s, p_s = correlation.stats.spearmanr(merged[col1], merged[col2])
                    st.write(f'Пирсон: r = {r_p:.3f}, p = {p_p:.4g} '
                             f'{correlation.sig_stars(p_p)}')
                    st.write(f'Спирмен: ρ = {r_s:.3f}, p = {p_s:.4g} '
                             f'{correlation.sig_stars(p_s)}')



# =============================================================================
# ВКЛАДКА 6: РАЙОНИРОВАНИЕ (КЛАСТЕРНЫЙ АНАЛИЗ)
# =============================================================================
with tab_cluster:
    st.header('Районирование: кластерный анализ')
    st.markdown(
        'K-means по стандартизированным климатическим индексам с подбором '
        'оптимального числа кластеров по силуэтному коэффициенту. Результат — '
        'разбиение станций на однородные климатические районы, PCA-биплот для '
        'визуализации структуры и дендрограмма для верификации.'
    )

    ind_table = st.session_state.indicators_table
    if ind_table is None or ind_table.empty:
        st.info('Сначала рассчитайте климатические индексы во вкладке '
                '"Климатические индексы" — кластеризация проводится по ним.')
    else:
        all_feat_cols = indicators.indicator_feature_columns(ind_table)

        st.subheader('Выбор признаков для кластеризации')
        fs_options = ['Все индексы'] + list(st.session_state.feature_sets.keys())
        fs_choice = st.selectbox('Набор признаков', fs_options, key='cluster_fs')
        if fs_choice == 'Все индексы':
            default_features = all_feat_cols
        else:
            default_features = [c for c in st.session_state.feature_sets[fs_choice]
                                 if c in all_feat_cols]
        chosen_features = st.multiselect('Признаки', all_feat_cols,
                                          default=default_features, key='cluster_features')

        nan_cols = [c for c in chosen_features if ind_table[c].isna().all()]
        if nan_cols:
            st.warning(f'Признаки полностью без данных исключены автоматически: '
                       f'{", ".join(nan_cols)} (например K_Конрад — нужна широта станций).')
            chosen_features = [c for c in chosen_features if c not in nan_cols]

        if len(chosen_features) < 2:
            st.error('Выберите минимум 2 признака.')
        else:
            d_clean, Xs, scaler = clustering.prepare_matrix(ind_table, chosen_features)
            st.caption(f'Станций с полными данными по выбранным признакам: {len(d_clean)}')

            st.subheader('Подбор числа кластеров')
            k_max = min(7, len(d_clean) - 1)
            k_range = range(2, max(3, k_max))
            has_lat = 'Lat' in d_clean.columns and d_clean['Lat'].notna().any()
            with st.status('⏳ Идёт расчёт: подбор количества кластеров...',
                            expanded=False) as status:
                sil_scores = clustering.silhouette_by_k(Xs, k_range=k_range)
                ari_scores = clustering.ari_by_k(Xs, d_clean['Lat'].values, k_range=k_range) \
                    if has_lat else None
                status.update(label='✅ Подбор количества кластеров завершён', state='complete')

            k_method_options = ['По силуэту']
            if has_lat:
                k_method_options += ['По ARI (широтные зоны)', 'Комбинированно (0,6·силуэт + 0,4·ARI)']
            k_method = st.radio('Метод определения количества кластеров',
                                 k_method_options, horizontal=False)

            if k_method == 'По силуэту':
                score_for_k = sil_scores
            elif k_method == 'По ARI (широтные зоны)':
                score_for_k = ari_scores
            else:
                score_for_k = clustering.combined_scores(sil_scores, ari_scores)

            auto_k = clustering.best_k(score_for_k)

            c1, c2 = st.columns(2)
            with c1:
                use_auto = st.checkbox('Подобрать k автоматически', value=True)
            with c2:
                if use_auto:
                    k = auto_k
                    st.metric('Автоматически выбрано k', k)
                else:
                    k = st.slider('Число кластеров (k)', 2, max(2, k_max - 1),
                                   auto_k or 2)

            figsize_s, _, _ = size_and_color_controls('sil', with_color=False)
            fig_sil = viz.k_selection_plot(
                sil_scores, ari_scores=ari_scores if k_method != 'По силуэту' else None,
                chosen_k=k, title=f'Подбор k: {k_method}', figsize=figsize_s)
            show_plot(fig_sil)
            download_row(fig_sil, 'graph_k_selection', 'sil')
            if not has_lat:
                st.caption('ARI-критерий недоступен: нет широты станций (загрузите '
                           'справочник станций с координатами во вкладке "Данные").')

            st.divider()
            st.subheader('Результат кластеризации')
            c1, c2 = st.columns(2)
            with c1:
                algorithm_label = st.selectbox(
                    'Алгоритм кластеризации',
                    ['K-means', 'Иерархическая (Ward)', 'Gaussian Mixture'])
            with c2:
                order_feature = st.selectbox('Нумеровать кластеры по убыванию признака',
                                              chosen_features,
                                              index=chosen_features.index('P_год')
                                              if 'P_год' in chosen_features else 0)
            algorithm_map = {'K-means': 'kmeans', 'Иерархическая (Ward)': 'hierarchical',
                              'Gaussian Mixture': 'gmm'}
            with st.status(f'⏳ Идёт расчёт: кластеризация ({algorithm_label})...',
                            expanded=False) as status:
                labels, sil = clustering.run_clustering(Xs, k, algorithm=algorithm_map[algorithm_label])
                relabeled = clustering.relabel_clusters_by_feature(d_clean, labels, order_feature)
                status.update(label='✅ Кластеризация завершена', state='complete')
            d_clean = d_clean.copy()
            d_clean['Кластер'] = relabeled
            st.success(f'Кластеризация выполнена: {algorithm_label}, k={k}, силуэт={sil:.3f}')

            result_cols = ['Станция'] + [c for c in ['Название', 'Регион', 'Lat', 'Lon']
                                          if c in d_clean.columns] + ['Кластер'] + chosen_features
            st.dataframe(d_clean[result_cols], use_container_width=True)
            st.download_button('📥 Скачать результат (станция → кластер, .xlsx)',
                                data_io.df_to_excel_bytes(d_clean[result_cols], 'Районирование'),
                                file_name='table_clustering_result.xlsx')

            st.divider()
            st.subheader('PCA-биплот')
            st.markdown('**PCA-биплот.** Точки — станции, окрашенные по кластеру; '
                        'стрелки — вклад каждого признака в первые две главные '
                        'компоненты. Направление и длина стрелки показывают, какой '
                        'признак сильнее всего определяет положение станции.')
            figsize_p, _, _ = size_and_color_controls('pca', with_color=False,
                                                       default_size='Широкий (12×8)')
            with st.status('⏳ Идёт расчёт: построение PCA...', expanded=False) as status:
                Xpca, pca_model, ev = clustering.pca_transform(Xs)
                fig_pca = viz.pca_biplot(Xpca, pca_model, relabeled, chosen_features, ev,
                                          title=f'PCA: k={k}', figsize=figsize_p)
                status.update(label='✅ PCA построен', state='complete')
            show_plot(fig_pca)
            download_row(fig_pca, 'graph_pca_biplot', 'pca')

            st.divider()
            st.subheader('Дендрограмма (верификация)')
            figsize_d, _, _ = size_and_color_controls('dendro', with_color=False,
                                                       default_size='Широкий (12×8)')
            with st.status('⏳ Идёт расчёт: построение дендрограммы...', expanded=False) as status:
                Z = clustering.hierarchical_linkage(Xs)
                leaf_labels = (d_clean['Название'] if 'Название' in d_clean.columns
                               else d_clean['Станция']).astype(str).str[:10].tolist()
                thr = (Z[-(k - 1), 2] + Z[-k, 2]) / 2 if k > 1 and len(Z) >= k else None
                fig_dendro = viz.dendrogram_plot(Z, leaf_labels, threshold=thr,
                                                  title='Иерархическая кластеризация (Уорд)',
                                                  figsize=figsize_d)
                status.update(label='✅ Дендрограмма построена', state='complete')
            show_plot(fig_dendro, width_frac=0.85)
            download_row(fig_dendro, 'graph_dendrogram', 'dendro')

            st.divider()
            st.subheader('Характеристика кластеров')
            summary_cols = st.multiselect('Признаки для сводной таблицы', chosen_features,
                                           default=chosen_features[:5], key='cluster_summary_feats')
            if summary_cols:
                summ = clustering.cluster_summary_table(d_clean, summary_cols, 'Кластер')
                st.dataframe(summ, use_container_width=True)
                st.download_button('Скачать сводку по кластерам (.xlsx)',
                                    data_io.df_to_excel_bytes(summ, 'Сводка кластеров'),
                                    file_name='table_cluster_summary.xlsx')

                st.divider()
                st.subheader('Боксплот по кластерам')
                box_feat = st.selectbox('Признак для боксплота', summary_cols, key='cluster_box_feat')
                annotate_on_chart_c = st.checkbox('Показывать Мин/Макс на графике', value=False,
                                                   key='cluster_box_annotate')
                figsize_b, color_b, legend_b = size_and_color_controls('cbox', with_color=False)
                fig_box = viz.boxplot_groups(d_clean, box_feat, 'Кластер',
                                              title=f'{box_feat} по кластерам', ylabel=box_feat,
                                              figsize=figsize_b, legend_loc=legend_b,
                                              annotate_minmax=annotate_on_chart_c)
                show_plot(fig_box)
                download_row(fig_box, f'graph_cluster_box_{box_feat}', 'cbox')
                st.markdown('**Мин/Макс/Среднее/Медиана по кластерам:**')
                st.dataframe(viz.boxplot_minmax_summary(d_clean, box_feat, 'Кластер'),
                             use_container_width=True)

            st.divider()
            st.subheader('Карта')
            has_coords = 'Lat' in d_clean.columns and 'Lon' in d_clean.columns \
                and d_clean['Lat'].notna().any() and d_clean['Lon'].notna().any()
            if not has_coords:
                st.info('Координаты станций не загружены — карта недоступна. '
                        'Загрузите справочник станций с широтой/долготой во вкладке '
                        '"Данные" (роль "Справочник станций"), затем пересоберите '
                        'общий набор данных и рассчитайте индексы заново.')
            else:
                map_mode = st.radio('Тип карты',
                                     ['Точки станций (простая)',
                                      'Раскраска районов (Overpass API — автозагрузка)',
                                      'Раскраска районов (свой файл GeoJSON/Shapefile)'])

                if map_mode == 'Точки станций (простая)':
                    show_labels = st.checkbox('Показывать подписи станций на карте', value=True)
                    figsize_m, _, _ = size_and_color_controls('map', with_color=False,
                                                                default_size='Широкий (12×8)')
                    label_col = 'Название' if 'Название' in d_clean.columns else 'Станция'
                    fig_map = viz.map_scatter(d_clean, cluster_col='Кластер', label_col=label_col,
                                               title=f'Пространственное распределение кластеров (k={k})',
                                               figsize=figsize_m, show_labels=show_labels)
                    show_plot(fig_map, width_frac=0.75)
                    download_row(fig_map, 'graph_cluster_map', 'map')

                elif map_mode == 'Раскраска районов (Overpass API — автозагрузка)':
                    st.markdown(
                        'Границы районов скачиваются напрямую с OpenStreetMap через '
                        'Overpass API — это работает с вашего компьютера (обычный '
                        'доступ в интернет нужен именно у вас). Первая '
                        'загрузка может занять 10–60 секунд.'
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        region_name = st.text_input('Название региона (как в OSM)',
                                                     value='Новосибирская область')
                    with c2:
                        admin_level = st.number_input('Admin_level районов', min_value=2,
                                                        max_value=10, value=6,
                                                        help='Обычно 6 — районы области; '
                                                             '4 — сама область/край; '
                                                             '8 — сельсоветы/города')
                    cache_key = f'{region_name}__{admin_level}'
                    if st.button('🌐 Скачать границы районов'):
                        try:
                            with st.status('⏳ Идёт расчёт: запрос к Overpass API '
                                           '(может занять до минуты)...', expanded=True) as status:
                                polys = mapping.fetch_boundaries_overpass(
                                    region_name, admin_level=int(admin_level))
                                status.update(label='✅ Границы районов загружены', state='complete')
                            st.session_state['overpass_cache'] = {cache_key: polys}
                            st.success(f'Загружено {len(polys)} районов.')
                        except Exception as e:
                            st.error(f'Не удалось загрузить границы: {e}')

                    cached = st.session_state.get('overpass_cache', {}).get(cache_key)
                    if cached:
                        polys_ov = bbox_filter_controls(cached, d_clean, 'ov_bbox')
                        with st.status('⏳ Идёт расчёт: определение кластеров по районам...',
                                       expanded=False) as status:
                            assigned = mapping.assign_clusters_to_polygons(polys_ov, d_clean)
                            status.update(label='✅ Готово', state='complete')
                        show_district_labels = st.checkbox('Подписывать районы', value=False)
                        show_station_pts = st.checkbox('Показывать точки станций', value=True,
                                                        key='overpass_show_pts')
                        figsize_ch, _, _ = size_and_color_controls('choropleth_ov', with_color=False,
                                                                    default_size='Широкий (12×8)')
                        with st.status('⏳ Идёт расчёт: отрисовка карты...', expanded=False) as status:
                            fig_ch = viz.choropleth_map(
                                assigned, stations_df=d_clean if show_station_pts else None,
                                title=f'Районирование по кластерам (k={k})', figsize=figsize_ch,
                                show_station_points=show_station_pts,
                                show_district_labels=show_district_labels)
                            status.update(label='✅ Карта готова', state='complete')
                        show_plot(fig_ch, width_frac=0.8)
                        download_row(fig_ch, 'graph_choropleth_overpass', 'choropleth_ov')

                        assign_table = build_district_assignment_table(assigned)
                        st.markdown('Сначала — районы, где кластер определён по станциям '
                                    'внутри границ; затем — районы без станций (кластер '
                                    'присвоен по ближайшей станции).')
                        st.dataframe(assign_table, use_container_width=True)
                        st.download_button('Скачать таблицу район→кластер (.xlsx)',
                                            data_io.df_to_excel_bytes(assign_table, 'Районы'),
                                            file_name='table_district_clusters.xlsx')

                else:  # свой файл
                    st.markdown('Загрузите **GeoJSON** (один файл) или **Shapefile** '
                               '(три файла: .shp, .dbf, .shx одновременно).')
                    upload_kind = st.radio('Формат файла', ['GeoJSON', 'Shapefile (.shp+.dbf+.shx)'],
                                            horizontal=True)
                    polys_uploaded = None
                    if upload_kind == 'GeoJSON':
                        gj_file = st.file_uploader('Файл .geojson', type=['geojson', 'json'])
                        if gj_file is not None:
                            try:
                                polys_uploaded = mapping.read_geojson_file(gj_file.read())
                                st.success(f'Загружено {len(polys_uploaded)} полигонов.')
                            except Exception as e:
                                st.error(f'Ошибка чтения GeoJSON: {e}')
                    else:
                        shp_files = st.file_uploader('Файлы .shp, .dbf, .shx (выберите все три)',
                                                      type=['shp', 'dbf', 'shx'],
                                                      accept_multiple_files=True)
                        if shp_files and len(shp_files) >= 3:
                            by_ext = {f.name.rsplit('.', 1)[-1].lower(): f for f in shp_files}
                            if all(ext in by_ext for ext in ['shp', 'dbf', 'shx']):
                                try:
                                    polys_uploaded, fields = mapping.read_shapefile(
                                        by_ext['shp'].read(), by_ext['dbf'].read(),
                                        by_ext['shx'].read())
                                    st.success(f'Загружено {len(polys_uploaded)} полигонов. '
                                               f'Поля атрибутов: {", ".join(fields)}')
                                except Exception as e:
                                    st.error(f'Ошибка чтения Shapefile: {e}')
                            else:
                                st.warning('Нужны все три файла: .shp, .dbf, .shx')

                    if polys_uploaded:
                        st.caption(f'Всего загружено полигонов: {len(polys_uploaded)}')
                        available_fields = sorted({k for p in polys_uploaded
                                                    for k in p.get('properties', {}).keys()})
                        st.markdown('**Фильтр по региону** (нужно, если файл содержит всю '
                                    'страну — например GADM с районами всей России)')
                        c1, c2 = st.columns(2)
                        with c1:
                            filter_field = st.selectbox('Поле атрибутов для фильтра',
                                                          ['— без фильтра —'] + available_fields,
                                                          key='filter_field')
                        with c2:
                            filter_value = ''
                            if filter_field != '— без фильтра —':
                                field_values = mapping.list_property_values(polys_uploaded, filter_field)
                                filter_value = st.selectbox('Значение (например область/край)',
                                                             ['— все —'] + field_values,
                                                             key='filter_value')

                        if filter_field != '— без фильтра —' and filter_value not in ('', '— все —'):
                            polys_uploaded = mapping.filter_polygons_by_attribute(
                                polys_uploaded, filter_field, filter_value)
                            st.info(f'После фильтра осталось полигонов: {len(polys_uploaded)}')

                        polys_uploaded = bbox_filter_controls(polys_uploaded, d_clean, 'upload_bbox')

                        with st.status('⏳ Идёт расчёт: определение кластеров по районам...',
                                       expanded=False) as status:
                            assigned = mapping.assign_clusters_to_polygons(polys_uploaded, d_clean)
                            status.update(label='✅ Готово', state='complete')
                        show_district_labels_u = st.checkbox('Подписывать районы', value=False,
                                                              key='upload_show_labels')
                        show_station_pts_u = st.checkbox('Показывать точки станций', value=True,
                                                          key='upload_show_pts')
                        figsize_chu, _, _ = size_and_color_controls('choropleth_up', with_color=False,
                                                                     default_size='Широкий (12×8)')
                        with st.status('⏳ Идёт расчёт: отрисовка карты...', expanded=False) as status:
                            fig_chu = viz.choropleth_map(
                                assigned, stations_df=d_clean if show_station_pts_u else None,
                                title=f'Районирование по кластерам (k={k})', figsize=figsize_chu,
                                show_station_points=show_station_pts_u,
                                show_district_labels=show_district_labels_u)
                            status.update(label='✅ Карта готова', state='complete')
                        show_plot(fig_chu, width_frac=0.8)
                        download_row(fig_chu, 'graph_choropleth_upload', 'choropleth_up')

                        assign_table_u = build_district_assignment_table(assigned)
                        st.markdown('Сначала — районы, где кластер определён по станциям '
                                    'внутри границ; затем — районы без станций (кластер '
                                    'присвоен по ближайшей станции).')
                        st.dataframe(assign_table_u, use_container_width=True)
                        st.download_button('Скачать таблицу район→кластер (.xlsx)',
                                            data_io.df_to_excel_bytes(assign_table_u, 'Районы'),
                                            file_name='table_district_clusters.xlsx')
