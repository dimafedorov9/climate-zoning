"""
=============================================================================
КАРТОГРАФИЧЕСКАЯ ОСНОВА
  • Встроенные слои Natural Earth (общественное достояние): границы
    субъектов РФ и Казахстана, крупные реки, озёра, города — работают без
    интернета.
  • Детальные реки, озёра и водохранилища OpenStreetMap через Overpass API
    (как загрузка границ районов в climate-zoning) — нужен интернет.
  • Рельеф: тайлы высот Terrarium (AWS Open Data, SRTM/GMTED), отмывка
    рельефа и послойная окраска; либо свой файл ЦМР (GeoTIFF, ESRI ASCII).
  • Растровые подложки (OpenTopoMap, OpenStreetMap, Esri World Physical /
    Topo) — переводятся из Web Mercator в географические координаты без
    GDAL.
Все скачанные данные кешируются в папке ~/.climate_adapt_cache.
=============================================================================
"""

import io
import os
import json
import math
import hashlib

import numpy as np
from shapely.geometry import shape, box, LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NE_DIR = os.path.join(HERE, 'data', 'basemap')
CACHE = os.path.join(os.path.expanduser('~'), '.climate_adapt_cache')
os.makedirs(CACHE, exist_ok=True)
UA = {'User-Agent': 'climate-adapt/1.0 (SibNIGMI research software)'}

TILE_PROVIDERS = {
    'OpenTopoMap (топографическая)': 'https://a.tile.opentopomap.org/{z}/{x}/{y}.png',
    'Esri World Physical (физическая)': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Physical_Map/MapServer/tile/{z}/{y}/{x}',
    'Esri World Topo': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
    'Esri World Shaded Relief (отмывка)': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}',
    'OpenStreetMap': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
}
TILE_ATTRIB = {
    'OpenTopoMap (топографическая)': '© OpenStreetMap, SRTM | стиль © OpenTopoMap (CC-BY-SA)',
    'Esri World Physical (физическая)': 'Источник: Esri, US National Park Service',
    'Esri World Topo': 'Источник: Esri и др.',
    'Esri World Shaded Relief (отмывка)': 'Источник: Esri',
    'OpenStreetMap': '© участники OpenStreetMap',
}
TERRARIUM = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'


# ─── Встроенные слои ─────────────────────────────────────────────────────
def _load_geojson(name):
    with open(os.path.join(NE_DIR, name), encoding='utf-8') as f:
        return json.load(f)


def ne_layer(name, bbox=None):
    """Слой Natural Earth: 'regions', 'rivers', 'lakes', 'cities' → список dict(name, geometry, props)."""
    d = _load_geojson(f'{name}.geojson')
    B = box(bbox[0], bbox[2], bbox[1], bbox[3]) if bbox else None
    out = []
    for f in d['features']:
        g = shape(f['geometry'])
        if B is not None and not g.intersects(B):
            continue
        p = f['properties']
        out.append(dict(name=p.get('name_ru') or p.get('name'), geometry=g, props=p))
    return out


def region_names():
    return sorted({r['name'] for r in ne_layer('regions')})


def region_geometry(names):
    """Объединённая граница выбранных субъектов (по русским названиям Natural Earth)."""
    if isinstance(names, str):
        names = [names]
    geoms = [r['geometry'] for r in ne_layer('regions') if r['name'] in names]
    return unary_union(geoms) if geoms else None


def bbox_of(geom=None, lat=None, lon=None, pad=0.15):
    if geom is not None:
        x0, y0, x1, y1 = geom.bounds
    else:
        x0, x1, y0, y1 = np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return (x0 - dx, x1 + dx, y0 - dy, y1 + dy)


# ─── Overpass: детальная гидросеть ───────────────────────────────────────
def _cache_path(key, ext):
    return os.path.join(CACHE, hashlib.md5(key.encode()).hexdigest() + ext)


def fetch_osm_water(region_name=None, bbox=None, timeout=180, min_lake_km2=1.0, rel_id=None):
    """
    Реки (waterway=river) и водоёмы (natural=water) из OpenStreetMap через
    Overpass API — по названию субъекта (как в OSM) или по bbox.
    Возвращает dict(rivers=[...], lakes=[...]); результат кешируется.
    """
    import requests
    from .mapping import OVERPASS_ENDPOINTS
    if rel_id:
        area = f'area({3600000000 + int(rel_id)})->.a;'
        sel = '(area.a)'
    elif region_name:
        area = f'area["name"="{region_name}"]["admin_level"="4"]->.a;'
        sel = '(area.a)'
    else:
        area = ''
        sel = f'({bbox[2]},{bbox[0]},{bbox[3]},{bbox[1]})'
    q = f"""[out:json][timeout:{timeout}];{area}
(way["waterway"="river"]{sel};
 way["natural"="water"]["water"~"lake|reservoir|oxbow"]{sel};
 relation["natural"="water"]["water"~"lake|reservoir"]{sel};);
out geom;"""
    cp = _cache_path(q, '.json')
    if os.path.exists(cp):
        data = json.load(open(cp, encoding='utf-8'))
    else:
        from .mapping import overpass
        data = overpass(q, timeout)
        json.dump(data, open(cp, 'w', encoding='utf-8'))
    rivers, lakes = [], []
    for el in data.get('elements', []):
        tags = el.get('tags', {})
        if el['type'] == 'way' and 'geometry' in el:
            pts = [(p['lon'], p['lat']) for p in el['geometry']]
            if tags.get('waterway') == 'river' and len(pts) >= 2:
                rivers.append(dict(name=tags.get('name', ''), geometry=LineString(pts), props=tags))
            elif tags.get('natural') == 'water' and len(pts) >= 4:
                try:
                    pg = Polygon(pts)
                    if pg.is_valid and _area_km2(pg) >= min_lake_km2:
                        lakes.append(dict(name=tags.get('name', ''), geometry=pg, props=tags))
                except Exception:
                    pass
        elif el['type'] == 'relation':
            outers = [LineString([(p['lon'], p['lat']) for p in m['geometry']])
                      for m in el.get('members', []) if m.get('role') == 'outer' and 'geometry' in m]
            if outers:
                from shapely.ops import polygonize, linemerge
                try:
                    polys = list(polygonize(linemerge(outers)))
                    if polys:
                        mp = unary_union(polys)
                        if _area_km2(mp) >= min_lake_km2:
                            lakes.append(dict(name=tags.get('name', ''), geometry=mp, props=tags))
                except Exception:
                    pass
    return dict(rivers=rivers, lakes=lakes)


def _area_km2(g):
    lat = g.centroid.y
    return g.area * 111.32 ** 2 * math.cos(math.radians(lat))


# ─── Тайлы: общие функции ───────────────────────────────────────────────
def _deg2num(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def _num2lat(y, z):
    n = 2 ** z
    return np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))


class ServiceError(RuntimeError):
    """Ошибка внешнего картографического сервиса (не программы)."""


def service_message(service, e):
    """Понятное сообщение о сбое внешнего сервиса с указанием причины."""
    import requests
    code = getattr(getattr(e, 'response', None), 'status_code', None)
    if isinstance(e, requests.exceptions.Timeout) or 'timed out' in str(e).lower():
        why = 'сервер не ответил за отведённое время (перегружен)'
    elif isinstance(e, requests.exceptions.ConnectionError):
        why = 'нет соединения с сервером (сервер недоступен или нет доступа в интернет)'
    elif code == 429:
        why = 'сервер ограничил число запросов (HTTP 429)'
    elif code in (502, 503, 504):
        why = f'сервер перегружен или на обслуживании (HTTP {code})'
    elif code:
        why = f'сервер вернул ошибку HTTP {code}'
    else:
        why = str(e)[:200]
    return (f'Ошибка на стороне внешнего сервиса «{service}»: {why}. Это не ошибка программы и не ошибка '
            f'загруженных данных. Повторите попытку через несколько минут; уже полученные данные сохраняются '
            f'и повторно не запрашиваются.')


def _get_tile(url_tpl, z, x, y, timeout=30):
    import requests
    from urllib.parse import urlparse
    url = url_tpl.format(z=z, x=x, y=y)
    cp = _cache_path(url, '.png')
    if os.path.exists(cp):
        return open(cp, 'rb').read()
    try:
        r = requests.get(url, timeout=timeout, headers=UA)
        r.raise_for_status()
    except Exception as e:
        raise ServiceError(service_message(f'сервер тайлов {urlparse(url).netloc}', e)) from None
    open(cp, 'wb').write(r.content)
    return r.content


def auto_zoom(bbox, max_tiles=64):
    for z in range(12, 2, -1):
        x0, y1 = _deg2num(bbox[2], bbox[0], z)
        x1, y0 = _deg2num(bbox[3], bbox[1], z)
        if (int(x1) - int(x0) + 1) * (int(y1) - int(y0) + 1) <= max_tiles:
            return z
    return 3


def fetch_mosaic(bbox, url_tpl, zoom=None, mode='RGB', progress=None):
    """
    Собирает мозаику тайлов и пересчитывает её на регулярную сетку широта ×
    долгота (строки — равномерно по широте). Возвращает (array, extent).
    """
    from PIL import Image
    z = zoom or auto_zoom(bbox)
    x0f, y1f = _deg2num(bbox[2], bbox[0], z)
    x1f, y0f = _deg2num(bbox[3], bbox[1], z)
    x0, x1, y0, y1 = int(x0f), int(x1f), int(y0f), int(y1f)
    W, H = (x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256
    canvas = Image.new(mode, (W, H))
    total = (x1 - x0 + 1) * (y1 - y0 + 1)
    k = 0
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            img = Image.open(io.BytesIO(_get_tile(url_tpl, z, tx, ty))).convert(mode)
            canvas.paste(img, ((tx - x0) * 256, (ty - y0) * 256))
            k += 1
            if progress:
                progress(k / total)
    A = np.asarray(canvas)
    # координаты пикселей мозаики
    px_lon = (x0 + (np.arange(W) + 0.5) / 256) / 2 ** z * 360 - 180
    py_lat = _num2lat(y0 + (np.arange(H) + 0.5) / 256, z)
    # вырезка по bbox и пересчёт строк на равномерную сетку широт
    cols = (px_lon >= bbox[0]) & (px_lon <= bbox[1])
    out_h = int(np.sum((py_lat >= bbox[2]) & (py_lat <= bbox[3])))
    lat_t = np.linspace(bbox[3], bbox[2], max(out_h, 2))
    rows = np.clip(np.searchsorted(-py_lat, -lat_t), 0, H - 1)
    A = A[rows][:, cols]
    extent = (px_lon[cols][0], px_lon[cols][-1], bbox[2], bbox[3])
    return A, extent


def fetch_dem(bbox, zoom=None, progress=None):
    """Цифровая модель рельефа из тайлов Terrarium: высота = R·256 + G + B/256 − 32768, м."""
    A, ext = fetch_mosaic(bbox, TERRARIUM, zoom=zoom or min(auto_zoom(bbox, 36), 9), progress=progress)
    A = A.astype(float)
    dem = A[..., 0] * 256 + A[..., 1] + A[..., 2] / 256 - 32768
    return dem, ext


def load_dem_file(file_bytes, name):
    """
    Чтение своей ЦМР без GDAL: ESRI ASCII Grid (.asc) или GeoTIFF в
    географических координатах (EPSG:4326, через tifffile).
    Возвращает (dem, extent).
    """
    if name.lower().endswith('.asc'):
        txt = file_bytes.decode('ascii', errors='replace').splitlines()
        hdr = {}
        i = 0
        while i < len(txt) and txt[i].split()[0].lower() in ('ncols', 'nrows', 'xllcorner', 'yllcorner',
                                                              'xllcenter', 'yllcenter', 'cellsize', 'nodata_value'):
            k, v = txt[i].split()[:2]
            hdr[k.lower()] = float(v)
            i += 1
        dem = np.loadtxt(txt[i:])
        nod = hdr.get('nodata_value')
        if nod is not None:
            dem[dem == nod] = np.nan
        cs = hdr['cellsize']
        x0 = hdr.get('xllcorner', hdr.get('xllcenter', 0) - cs / 2)
        y0 = hdr.get('yllcorner', hdr.get('yllcenter', 0) - cs / 2)
        return dem, (x0, x0 + cs * dem.shape[1], y0, y0 + cs * dem.shape[0])
    import tifffile
    with tifffile.TiffFile(io.BytesIO(file_bytes)) as tf:
        page = tf.pages[0]
        dem = page.asarray().astype(float)
        tags = page.tags
        scale = tags['ModelPixelScaleTag'].value
        tie = tags['ModelTiepointTag'].value
        x0, y1 = tie[3], tie[4]
        ext = (x0, x0 + scale[0] * dem.shape[1], y1 - scale[1] * dem.shape[0], y1)
        nod = tags.get('GDAL_NODATA')
        if nod is not None:
            dem[dem == float(nod.value)] = np.nan
    return dem, ext


def sample_dem(dem, extent, lat, lon):
    """Высота ЦМР в точках (ближайший узел)."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    h, w = dem.shape
    j = np.clip(((lon - extent[0]) / (extent[1] - extent[0]) * w).astype(int), 0, w - 1)
    i = np.clip(((extent[3] - lat) / (extent[3] - extent[2]) * h).astype(int), 0, h - 1)
    return dem[i, j]


def hillshade(dem, extent, azimuth=315, altitude=45, z_factor=1.0):
    """Отмывка рельефа (0…1) с учётом размера ячейки в метрах на данной широте."""
    from matplotlib.colors import LightSource
    h, w = dem.shape
    lat0 = (extent[2] + extent[3]) / 2
    dx = (extent[1] - extent[0]) / w * 111320 * math.cos(math.radians(lat0))
    dy = (extent[3] - extent[2]) / h * 110540
    z = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    return ls.hillshade(z, vert_exag=z_factor, dx=dx, dy=dy)
