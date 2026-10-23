"""
=============================================================================
МОДУЛЬ КАРТ: ГРАНИЦЫ РАЙОНОВ + РАСКРАСКА ПО КЛАСТЕРАМ
Два источника границ:
  1. Overpass API (OpenStreetMap) — автоматическая загрузка по названию
     региона и admin_level. Работает из окружения пользователя, поэтому
     требует обычного доступа в интернет.
  2. Загруженный пользователем файл — GeoJSON или shapefile (.shp/.dbf/.shx).

Полигоны сопоставляются со станциями (какой кластер преобладает внутри
района) через point-in-polygon на shapely, без зависимости от geopandas/GDAL
(легче ставится на Windows).
=============================================================================
"""

import json
import numpy as np
import pandas as pd
from shapely.geometry import shape, Point, Polygon, MultiPolygon

OVERPASS_ENDPOINTS = [
    'https://overpass-api.de/api/interpreter',
    'https://maps.mail.ru/osm/tools/overpass/api/interpreter',
]


def overpass(query, timeout=60):
    """
    Выполняет запрос Overpass QL на одном из серверов. При сбое всех
    серверов — ServiceError с понятным описанием причины.
    """
    import requests
    from .basemap import ServiceError, service_message, UA
    last = None
    for url in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(url, data={'data': query}, timeout=timeout + 20, headers=UA)
            r.raise_for_status()
            data = r.json()
            rem = str(data.get('remark', ''))
            if 'runtime error' in rem.lower() or 'timed out' in rem.lower():
                raise requests.exceptions.Timeout(rem)
            return data
        except Exception as e:  # noqa
            last = e
    raise ServiceError(service_message('Overpass API (OpenStreetMap)', last))


def _cache(key):
    import hashlib
    import os
    from .basemap import CACHE
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, 'osm_' + hashlib.md5(key.encode()).hexdigest() + '.pkl')


def list_osm_subjects(bbox, timeout=60):
    """
    Субъекты (admin_level = 4) OpenStreetMap, пересекающие охват станций.
    Возвращает DataFrame [name, id, iso] — названия точно как в OSM, поэтому
    дальнейшие запросы выполняются по идентификатору, без ошибок в написании.
    """
    import os
    import pickle
    q = (f'[out:json][timeout:{timeout}];relation["boundary"="administrative"]["admin_level"="4"]'
         f'({bbox[2]:.3f},{bbox[0]:.3f},{bbox[3]:.3f},{bbox[1]:.3f});out tags;')
    cp = _cache(q)
    if os.path.exists(cp):
        return pickle.load(open(cp, 'rb'))
    data = overpass(q, timeout)
    rows = [dict(name=e['tags'].get('name:ru') or e['tags'].get('name'), id=e['id'],
                 iso=e['tags'].get('ISO3166-2', '')) for e in data.get('elements', []) if e.get('tags')]
    df = pd.DataFrame(rows, columns=['name', 'id', 'iso']).dropna(subset=['name']).sort_values('name')
    df = df.reset_index(drop=True)
    pickle.dump(df, open(cp, 'wb'))
    return df


def fetch_osm_districts(rel_id, admin_level=6, timeout=120):
    """
    Границы административных единиц уровня admin_level внутри субъекта с
    идентификатором отношения rel_id. Возвращает список dict(name, geometry).
    """
    import os
    import pickle
    q = (f'[out:json][timeout:{timeout}];area({3600000000 + int(rel_id)})->.a;'
         f'relation["boundary"="administrative"]["admin_level"="{admin_level}"](area.a);out geom;')
    cp = _cache(q)
    if os.path.exists(cp):
        return pickle.load(open(cp, 'rb'))
    data = overpass(q, timeout)
    import osm2geojson
    geo = osm2geojson.json2geojson(data)
    res = []
    for feat in geo.get('features', []):
        props = feat.get('properties', {})
        if props.get('type') != 'relation':
            continue
        tags = props.get('tags', {})
        name = tags.get('name:ru') or tags.get('name')
        try:
            g = shape(feat['geometry'])
        except Exception:
            continue
        if name and g.geom_type in ('Polygon', 'MultiPolygon'):
            res.append({'name': name, 'geometry': g if g.is_valid else g.buffer(0)})
    if not res:
        raise RuntimeError(f'В OpenStreetMap для выбранного субъекта нет границ уровня {admin_level}. '
                           'Попробуйте другой уровень (6 — муниципальные районы и округа, 8 — поселения).')
    res.sort(key=lambda p: p['name'])
    pickle.dump(res, open(cp, 'wb'))
    return res


def build_overpass_query(region_name, admin_level=6, region_admin_level=4, timeout=60):
    """
    Формирует Overpass QL запрос: находит регион по имени (area на заданном
    admin_level, обычно 4 — область/край), затем внутри него все отношения
    boundary=administrative на заданном admin_level (обычно 6 — районы).
    """
    return f"""
[out:json][timeout:{timeout}];
area["name"="{region_name}"]["admin_level"="{region_admin_level}"]->.a;
relation["admin_level"="{admin_level}"]["boundary"="administrative"](area.a);
out geom;
""".strip()


def fetch_boundaries_overpass(region_name, admin_level=6, region_admin_level=4,
                               timeout=60, endpoint=None):
    """
    Скачивает границы районов через Overpass API. Возвращает список dict:
    {'name': str, 'geometry': shapely geometry}. Бросает исключение с
    понятным текстом при сетевой ошибке или пустом результате.
    """
    import requests

    query = build_overpass_query(region_name, admin_level, region_admin_level, timeout)
    endpoints = [endpoint] if endpoint else OVERPASS_ENDPOINTS
    last_error = None

    for url in endpoints:
        try:
            resp = requests.post(url, data={'data': query}, timeout=timeout + 10)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            last_error = e
            data = None
    if data is None:
        raise RuntimeError(f'Не удалось получить ответ от Overpass API: {last_error}')

    import osm2geojson
    geo = osm2geojson.json2geojson(data)

    results = []
    for feat in geo.get('features', []):
        props = feat.get('properties', {})
        if props.get('type') != 'relation':
            continue
        tags = props.get('tags', {})
        name = tags.get('name')
        geom = feat.get('geometry')
        if not name or not geom:
            continue
        try:
            geom = shape(geom)
        except Exception:
            continue
        if geom.geom_type not in ('Polygon', 'MultiPolygon'):
            continue
        results.append({'name': name, 'geometry': geom})

    if not results:
        raise RuntimeError(
            f'Overpass вернул пустой результат для региона "{region_name}" '
            f'(admin_level={admin_level}). Проверьте точное название региона '
            f'в OSM (например "Новосибирская область") и уровень (обычно '
            f'4 — область/край, 6 — районы, 8 — сельсоветы/города).'
        )
    return results


def filter_polygons_by_attribute(polygons, field, value_substring):
    """
    Оставляет только полигоны, у которых значение поля properties[field]
    содержит value_substring (без учёта регистра). Используется, например,
    чтобы из общероссийского файла GADM оставить только районы одной области
    по полю вида NAME_1.
    """
    if not value_substring:
        return polygons
    needle = value_substring.strip().lower()
    return [p for p in polygons
            if needle in str(p.get('properties', {}).get(field, '')).lower()]


def filter_polygons_by_bbox(polygons, lon_min, lon_max, lat_min, lat_max):
    """
    Оставляет только полигоны, пересекающиеся с прямоугольником координат
    (долгота/широта) — быстрый способ обрезать общероссийский файл до нужной
    территории, если атрибутов для фильтрации по региону нет или неудобны.
    """
    from shapely.geometry import box
    bbox = box(lon_min, lat_min, lon_max, lat_max)
    return [p for p in polygons if p['geometry'].intersects(bbox)]


def list_property_values(polygons, field):
    """Уникальные значения поля properties[field] среди всех полигонов (для UI-выбора)."""
    vals = {str(p.get('properties', {}).get(field, '')) for p in polygons}
    return sorted(v for v in vals if v)


def read_geojson_file(file_bytes_or_path):
    """Читает GeoJSON (FeatureCollection) — из байтов или пути. Возвращает список полигонов."""
    if isinstance(file_bytes_or_path, (bytes, bytearray)):
        data = json.loads(file_bytes_or_path.decode('utf-8'))
    else:
        with open(file_bytes_or_path, encoding='utf-8') as f:
            data = json.load(f)

    results = []
    for feat in data.get('features', []):
        props = feat.get('properties', {})
        name = (props.get('name') or props.get('NAME') or props.get('Name') or
                props.get('district') or props.get('NAME_2') or props.get('NAME_3') or
                next((v for k, v in props.items() if 'name' in k.lower() and v), None) or
                f'Полигон {len(results)+1}')
        geom = feat.get('geometry')
        if not geom:
            continue
        try:
            g = shape(geom)
        except Exception:
            continue
        if g.geom_type not in ('Polygon', 'MultiPolygon'):
            continue
        results.append({'name': str(name), 'geometry': g, 'properties': props})
    return results


def read_shapefile(shp_bytes, dbf_bytes, shx_bytes, name_field=None):
    """
    Читает shapefile из байтов трёх обязательных компонентов (.shp/.dbf/.shx)
    через pyshp (без GDAL/geopandas). Возвращает (список полигонов, список
    имён полей атрибутов). Каждый полигон содержит 'properties' — полный
    словарь атрибутов записи, для последующей фильтрации по региону.
    name_field — имя поля атрибутов для названия района; если None —
    берётся первое текстовое поле или используется "Полигон N".
    """
    import shapefile
    import io

    reader = shapefile.Reader(shp=io.BytesIO(shp_bytes), dbf=io.BytesIO(dbf_bytes),
                               shx=io.BytesIO(shx_bytes))
    field_names = [f[0] for f in reader.fields[1:]]  # первое поле — DeletionFlag

    if name_field is None:
        # Предпочитаем более специфичный уровень (NAME_2 район > NAME_1 область)
        priority = ['NAME_2', 'name_2', 'NAME_3', 'district', 'район']
        name_field = next((f for f in priority if f in field_names), None)
        if name_field is None:
            name_field = next((f for f in field_names if 'name' in f.lower()
                                or 'назв' in f.lower() or 'район' in f.lower()), None)

    results = []
    for i, sr in enumerate(reader.shapeRecords()):
        geom_dict = sr.shape.__geo_interface__
        try:
            geom = shape(geom_dict)
        except Exception:
            continue
        if geom.geom_type not in ('Polygon', 'MultiPolygon'):
            continue
        rec = sr.record.as_dict()
        name = str(rec.get(name_field, f'Полигон {i+1}')) if name_field else f'Полигон {i+1}'
        results.append({'name': name, 'geometry': geom, 'properties': rec})
    return results, field_names


def assign_clusters_to_polygons(polygons, stations_df, lat_col='Lat', lon_col='Lon',
                                 cluster_col='Кластер'):
    """
    Для каждого полигона определяет преобладающий кластер: станции, чьи
    координаты попадают внутрь полигона (большинство голосов); если ни одна
    станция не попала внутрь — берётся кластер ближайшей станции к центроиду.
    Возвращает список полигонов с добавленным ключом 'cluster' (может быть None).
    """
    pts = []
    for _, row in stations_df.iterrows():
        lat, lon = row.get(lat_col), row.get(lon_col)
        if pd.notna(lat) and pd.notna(lon):
            pts.append((Point(lon, lat), row[cluster_col], row.get('Название', row.get('Станция'))))

    out = []
    for poly in polygons:
        geom = poly['geometry']
        inside = [(c, name) for pt, c, name in pts if geom.contains(pt)]
        if inside:
            clusters = [c for c, _ in inside]
            # преобладающий кластер (мода)
            vals, counts = np.unique(clusters, return_counts=True)
            cluster = vals[np.argmax(counts)]
            n_stations = len(inside)
        elif pts:
            centroid = geom.centroid
            dists = [(centroid.distance(pt), c) for pt, c, _ in pts]
            dists.sort(key=lambda x: x[0])
            cluster = dists[0][1]
            n_stations = 0
        else:
            cluster = None
            n_stations = 0
        out.append({**poly, 'cluster': cluster, 'n_stations_inside': n_stations})
    return out
