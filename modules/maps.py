"""
=============================================================================
КАРТЫ: ПЯТЬ ВИДОВ
  1. Схематическая — граница территории, станции, полигоны Тиссена
     (или изолинии интерполированного поля), окраска по районам-кластерам
     или значению показателя.
  2. По административным районам — границы районов (Overpass API / свой
     файл / встроенные субъекты); район окрашивается по преобладающему
     кластеру станций или по среднему значению интерполированного поля.
  3. Физическая — отмывка рельефа и послойная окраска высот, реки, озёра,
     водохранилища, города, границы; поверх — станции и (при выборе)
     полупрозрачное поле показателя с изолиниями.
  4. На растровой подложке — OpenTopoMap / Esri World Physical и др.,
     поверх — районы или станции.
  5. Интерактивная (в приложении, folium/Leaflet) — для просмотра.
Оформление: запятая в подписях координат и шкал, стрелка «С», масштабная
линейка, без лишних элементов (правило Тафти).
=============================================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from shapely.geometry import MultiPoint, Point
from shapely.ops import voronoi_diagram, unary_union

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PALETTE, cf

HALO = [pe.withStroke(linewidth=2.2, foreground='white')]
WATER = '#5B9BD5'
WATER_FILL = '#BFD9F2'
CLUSTER_COLORS = PALETTE + ['#3D405B', '#81B29A', '#F2CC8F', '#E07A5F', '#6D6875']

# послойная окраска высот (низменность → возвышенности), мягкие тона
HYPSO = LinearSegmentedColormap.from_list('hypso', [
    (0.00, '#8DB580'), (0.12, '#B5CF8E'), (0.25, '#DDE3A3'), (0.40, '#EED9A0'),
    (0.55, '#DDB77E'), (0.70, '#C79A6B'), (0.85, '#A98466'), (1.00, '#F2EFEA')])


# ─── Общие элементы ──────────────────────────────────────────────────────
def setup_ax(ax, bbox, grid=True):
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect(1 / math.cos(math.radians((bbox[2] + bbox[3]) / 2)))
    fmt = FuncFormatter(lambda v, p: f'{cf(v, 0 if abs(v - round(v)) < 1e-6 else 1)}°')
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(labelsize=8, length=2, color='#777')
    for s in ax.spines.values():
        s.set_color('#555')
        s.set_linewidth(0.8)
    if grid:
        ax.grid(color='#999', lw=0.3, alpha=0.5, zorder=1)


def north_arrow(ax, x=0.05, y=0.90, size=0.06):
    ax.annotate('', xy=(x, y), xytext=(x, y - size), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='#222', lw=1.6, mutation_scale=14), zorder=20)
    ax.text(x, y + 0.01, 'С', transform=ax.transAxes, ha='center', va='bottom', fontsize=10,
            fontweight='bold', path_effects=HALO, zorder=20)


def scale_bar(ax, bbox, loc=(0.05, 0.05)):
    lat0 = (bbox[2] + bbox[3]) / 2
    km_deg = 111.32 * math.cos(math.radians(lat0))
    width_km = (bbox[1] - bbox[0]) * km_deg
    bar = min([10, 20, 25, 50, 100, 200, 250, 500], key=lambda c: abs(c - width_km / 5))
    x0 = bbox[0] + (bbox[1] - bbox[0]) * loc[0]
    y0 = bbox[2] + (bbox[3] - bbox[2]) * loc[1]
    seg = bar / km_deg / 2
    hgt = (bbox[3] - bbox[2]) * 0.008
    for i, col in enumerate(['#222', 'white']):
        ax.add_patch(plt.Rectangle((x0 + i * seg, y0), seg, hgt, fc=col, ec='#222', lw=0.6, zorder=20))
    for xx, t in [(x0, '0'), (x0 + seg, f'{bar // 2}'), (x0 + 2 * seg, f'{bar} км')]:
        ax.text(xx, y0 + hgt * 2.2, t, ha='center', va='bottom', fontsize=7, path_effects=HALO, zorder=20)


def _iter_polys(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == 'Polygon':
        return [g]
    if g.geom_type in ('MultiPolygon', 'GeometryCollection'):
        return [p for p in g.geoms if p.geom_type == 'Polygon']
    return []


def _poly_path(p):
    from matplotlib.path import Path
    verts, codes = [], []
    for ring in [p.exterior] + list(p.interiors):
        xy = np.asarray(ring.coords)
        if len(xy) < 3:
            continue
        verts.append(xy)
        codes += [Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY]
    return Path(np.concatenate(verts), codes)


def draw_poly(ax, g, fc='none', ec='#333', lw=0.8, alpha=1.0, zorder=3, hatch=None):
    """Полигон с внутренними «дырами» (озёра, анклавы) через составной путь."""
    from matplotlib.patches import PathPatch
    for p in _iter_polys(g):
        ax.add_patch(PathPatch(_poly_path(p), fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zorder,
                               hatch=hatch))


def clip_to(ax, artists, geom):
    """Обрезка заливки и изолиний точно по границе территории (вместо маски по узлам сетки)."""
    if geom is None:
        return
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    path = Path.make_compound_path(*[_poly_path(p) for p in _iter_polys(geom)])
    patch = PathPatch(path, transform=ax.transData, fc='none', ec='none')
    ax.add_patch(patch)
    for a in artists:
        if a is None:
            continue
        if hasattr(a, 'set_clip_path'):
            a.set_clip_path(patch)
        for c in getattr(a, 'collections', []):
            c.set_clip_path(patch)
    return patch


def label_contours(ax, cs, geom, decimals):
    """Подписи изолиний внутри территории: по одной на каждый уровень, в середине участка линии внутри границы."""
    fmt = lambda v: cf(v, decimals)
    if geom is None:
        return ax.clabel(cs, fmt=fmt, fontsize=7)
    import shapely
    pts = []
    try:
        segs = cs.allsegs
    except AttributeError:
        return ax.clabel(cs, fmt=fmt, fontsize=7)
    inner = geom.buffer(-0.08)
    for lev_segs in segs:
        best = None
        for seg in lev_segs:
            if len(seg) < 2:
                continue
            ins = shapely.contains_xy(inner if not inner.is_empty else geom, seg[:, 0], seg[:, 1])
            if ins.sum() > (0 if best is None else len(best)):
                best = seg[ins]
        if best is not None and len(best):
            pts.append(tuple(best[len(best) // 2]))
    if not pts:
        return []
    return ax.clabel(cs, fmt=fmt, fontsize=7, manual=pts, inline=True)


def inside_values(field, grid, geom):
    """Значения поля внутри территории — для шкалы цветов."""
    if geom is None or field is None:
        return np.ravel(field)
    import shapely
    LON, LAT = grid
    m = shapely.contains_xy(geom.buffer(0.05), LON, LAT)
    v = np.asarray(field)[m]
    return v if np.isfinite(v).any() else np.ravel(field)


def draw_lines(ax, g, color=WATER, lw=0.8, zorder=4, alpha=1.0):
    if g is None:
        return
    geoms = g.geoms if g.geom_type.startswith('Multi') or g.geom_type == 'GeometryCollection' else [g]
    for l in geoms:
        if l.geom_type == 'LineString':
            x, y = l.xy
            ax.plot(x, y, color=color, lw=lw, zorder=zorder, alpha=alpha, solid_capstyle='round')
        elif l.geom_type in ('Polygon', 'MultiPolygon'):
            for p in _iter_polys(l):
                x, y = p.exterior.xy
                ax.plot(x, y, color=color, lw=lw, zorder=zorder, alpha=alpha)


def draw_water(ax, water, clip=None, river_lw=(1.6, 0.6)):
    """Реки и озёра: water — dict(rivers=[...], lakes=[...]) в формате basemap."""
    for lk in water.get('lakes', []):
        g = lk['geometry'] if clip is None else lk['geometry'].intersection(clip)
        draw_poly(ax, g, fc=WATER_FILL, ec=WATER, lw=0.4, zorder=4)
    for rv in water.get('rivers', []):
        g = rv['geometry'] if clip is None else rv['geometry'].intersection(clip)
        sr = rv.get('props', {}).get('scalerank')
        lw = river_lw[1] if sr is None else max(river_lw[1], river_lw[0] - 0.12 * (float(sr) - 2))
        draw_lines(ax, g, color=WATER, lw=lw, zorder=4)


def draw_cities(ax, cities, bbox, max_rank=8, fontsize=7, st=None):
    near = []
    if st is not None and len(st):
        near = list(zip(st['lon'].values, st['lat'].values, st['name'].astype(str).str.lower().values))
    for c in cities:
        x, y = c['geometry'].x, c['geometry'].y
        if not (bbox[0] <= x <= bbox[1] and bbox[2] <= y <= bbox[3]):
            continue
        rank = c['props'].get('rank', 10)
        if rank is not None and rank > max_rank:
            continue
        big = rank is not None and rank <= 4
        # город с одноимённой метеостанцией рядом не подписывается дважды
        cn = str(c['name']).lower()[:5]
        if not big and any(abs(x - sx) < 0.2 and abs(y - sy) < 0.2 and sn[:5] == cn for sx, sy, sn in near):
            continue
        ax.plot(x, y, marker='s', ms=4 if big else 2.5, color='#333', zorder=9)
        ax.text(x + 0.05, y + 0.04, c['name'], fontsize=fontsize + (1 if big else 0), color='#222',
                fontweight='bold' if big else 'normal', path_effects=HALO, zorder=9)


def draw_stations(ax, st, color='#222', size=28, labels=True, label_col='name', fontsize=6.5, zorder=12,
                  marker='o', edge='white'):
    ax.scatter(st['lon'], st['lat'], s=size, c=color, edgecolors=edge, linewidths=0.8, zorder=zorder, marker=marker)
    if labels:
        for _, r in st.iterrows():
            ax.text(r['lon'] + 0.04, r['lat'] - 0.04, str(r[label_col])[:14], fontsize=fontsize, va='top',
                    color='#111', path_effects=HALO, zorder=zorder + 1)


def _value_norm(values, cmap, n_levels=8, vmin=None, vmax=None):
    v = np.asarray(values, float)
    lo = np.nanmin(v) if vmin is None else vmin
    hi = np.nanmax(v) if vmax is None else vmax
    if lo == hi:
        hi = lo + 1
    levels = np.linspace(lo, hi, n_levels + 1)
    cm = plt.get_cmap(cmap, n_levels)
    return cm, BoundaryNorm(levels, cm.N), levels


def _colorbar(fig, ax, cm, norm, label, decimals=1, orientation='vertical'):
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    cb = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02, orientation=orientation, fraction=0.04)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: cf(v, decimals)))
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: cf(v, decimals)))
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.5)
    return cb


def legend_top(ax, handles, ncol=None):
    """Легенда над полем карты (между заголовком и рамкой) — не перекрывает объекты карты."""
    if not handles:
        return 0
    ncol = ncol or min(len(handles), 4)
    ax.legend(handles=handles, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=ncol,
              frameon=False, handlelength=1.6, columnspacing=1.4, borderaxespad=0.2)
    return int(np.ceil(len(handles) / ncol))


def set_title(ax, title, legend_rows=0):
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10 + 17 * legend_rows)


def _cluster_legend(ax, clusters, colors, title='Район', loc=None, extra=None):
    h = [Patch(fc=colors[c], ec='#333', lw=0.5, label=f'{title} {c}') for c in clusters]
    if extra:
        h += extra
    return legend_top(ax, h)


def thiessen(st, clip_geom):
    """Полигоны Тиссена (Вороного) для станций, обрезанные границей территории."""
    pts = MultiPoint([Point(x, y) for x, y in zip(st['lon'], st['lat'])])
    env = clip_geom.envelope.buffer(2) if clip_geom is not None else pts.envelope.buffer(2)
    vor = voronoi_diagram(pts, envelope=env)
    cells = []
    for _, r in st.iterrows():
        p = Point(r['lon'], r['lat'])
        cell = next((c for c in vor.geoms if c.contains(p)), None)
        if cell is not None and clip_geom is not None:
            cell = cell.intersection(clip_geom)
        cells.append(cell)
    return cells


# ─── 1. Схематическая ────────────────────────────────────────────────────
def map_schematic(st, region_geom, bbox, cluster_col=None, value_col=None, value_label='',
                  title='', mode='thiessen', field=None, grid=None, cmap='viridis', decimals=1,
                  n_levels=8, labels=True, rivers=None, figsize=(10, 8), districts=None):
    """
    Схематическая карта. mode: 'thiessen' — полигоны Тиссена; 'field' —
    интерполированное поле (field на сетке grid=(LON, LAT)) с изолиниями;
    'points' — только станции. Окраска — по cluster_col или value_col.
    """
    fig, ax = plt.subplots(figsize=figsize)
    setup_ax(ax, bbox, grid=False)
    ax.set_facecolor('white')
    colors = None
    if cluster_col:
        cl = sorted(st[cluster_col].dropna().unique())
        colors = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(cl)}
    elif value_col:
        cm, norm, levels = _value_norm(st[value_col] if field is None else
                                       np.r_[st[value_col].values, inside_values(field, grid, region_geom)],
                                       cmap, n_levels)
    if mode == 'thiessen' and (cluster_col or value_col):
        cells = thiessen(st, region_geom)
        for cell, (_, r) in zip(cells, st.iterrows()):
            if cell is None:
                continue
            fc = colors[r[cluster_col]] if cluster_col else cm(norm(r[value_col]))
            draw_poly(ax, cell, fc=fc, ec='white', lw=0.6, alpha=0.9, zorder=2)
    elif mode == 'field' and field is not None:
        LON, LAT = grid
        F = np.ma.masked_invalid(field)
        if cluster_col:
            cl_levels = np.arange(min(colors) - 0.5, max(colors) + 1.5)
            cmc = ListedColormap([colors[c] for c in sorted(colors)])
            cf_ = ax.contourf(LON, LAT, F, levels=cl_levels, cmap=cmc, alpha=0.9, zorder=2)
            clip_to(ax, [cf_], region_geom)
        else:
            cf_ = ax.contourf(LON, LAT, F, levels=levels, cmap=cm, norm=norm, zorder=2, extend='both')
            cs = ax.contour(LON, LAT, F, levels=levels, colors='#333', linewidths=0.4, zorder=3)
            clip_to(ax, [cf_, cs], region_geom)
            label_contours(ax, cs, region_geom, decimals)
    if districts:
        for d in districts:
            draw_poly(ax, d['geometry'], ec='#777', lw=0.4, zorder=5)
    if rivers:
        draw_water(ax, rivers, clip=region_geom.buffer(0.02) if region_geom is not None else None)
    if region_geom is not None:
        draw_poly(ax, region_geom, ec='#222', lw=1.2, zorder=6)
    pt_color = '#222' if mode != 'points' or not (cluster_col or value_col) else \
        ([colors[c] for c in st[cluster_col]] if cluster_col else [cm(norm(v)) for v in st[value_col]])
    draw_stations(ax, st, color=pt_color, size=30 if mode != 'points' else 70, labels=labels,
                  edge='white' if mode != 'points' else '#222')
    rows = 0
    if cluster_col:
        rows = _cluster_legend(ax, sorted(colors), colors)
    elif value_col:
        _colorbar(fig, ax, cm, norm, value_label, decimals)
    north_arrow(ax)
    scale_bar(ax, bbox)
    set_title(ax, title, rows)
    fig.tight_layout()
    return fig


# ─── 2. Районы ───────────────────────────────────────────────────────────
def map_districts(polys, st, bbox, key='cluster', value_label='', title='', cmap='viridis',
                  decimals=1, n_levels=8, show_labels=False, show_stations=True, region_geom=None,
                  rivers=None, figsize=(10, 8), station_labels=False):
    """
    Районы, окрашенные по кластеру (key='cluster') или по значению
    (key='value', например среднее интерполированного поля).
    polys — список dict(name, geometry, cluster|value).
    """
    fig, ax = plt.subplots(figsize=figsize)
    setup_ax(ax, bbox, grid=False)
    if key == 'cluster':
        cl = sorted({p['cluster'] for p in polys if p.get('cluster') is not None})
        colors = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(cl)}
    else:
        cm, norm, _ = _value_norm([p.get('value', np.nan) for p in polys], cmap, n_levels)
    for p in polys:
        v = p.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            fc = '#E6E6E6'
        else:
            fc = colors[v] if key == 'cluster' else cm(norm(v))
        draw_poly(ax, p['geometry'], fc=fc, ec='white', lw=0.7, zorder=2)
        draw_poly(ax, p['geometry'], ec='#666', lw=0.35, zorder=3)
        if show_labels:
            c = p['geometry'].representative_point()
            nm = str(p['name']).replace(' район', '').replace('муниципальный ', '')
            ax.text(c.x, c.y, nm[:16], fontsize=6, ha='center', va='center', path_effects=HALO, zorder=8)
    if rivers:
        draw_water(ax, rivers, clip=region_geom.buffer(0.02) if region_geom is not None else None,
                   river_lw=(1.2, 0.4))
    if region_geom is not None:
        draw_poly(ax, region_geom, ec='#222', lw=1.2, zorder=6)
    if show_stations and st is not None:
        draw_stations(ax, st, color='#222', size=20, labels=station_labels)
    if key == 'cluster':
        extra = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#222', markersize=6,
                        label='Метеостанция')] if show_stations else None
        rows = _cluster_legend(ax, cl, colors, extra=extra)
    else:
        rows = 0
        _colorbar(fig, ax, cm, norm, value_label, decimals)
    north_arrow(ax)
    scale_bar(ax, bbox)
    set_title(ax, title, rows)
    fig.tight_layout()
    return fig


# ─── 3. Физическая ───────────────────────────────────────────────────────
def map_physical(bbox, region_geom=None, dem=None, dem_extent=None, water=None, cities=None,
                 st=None, st_color=None, st_labels=True, field=None, grid=None, value_label='',
                 cmap='coolwarm', decimals=1, n_levels=8, field_alpha=0.45, districts=None,
                 neighbors_geoms=None, title='', figsize=(10, 8), clusters=None, cluster_alpha=0.45,
                 relief_legend=True):
    """
    Физическая карта: отмывка и послойная окраска рельефа (при наличии
    ЦМР), реки и озёра, города, границы; поверх — станции, поле показателя
    (полупрозрачно, с изолиниями) или районы-кластеры (clusters — список
    dict(geometry, cluster)).
    """
    from . import basemap as B
    fig, ax = plt.subplots(figsize=figsize)
    setup_ax(ax, bbox, grid=True)
    ax.set_facecolor('#F4F1EA')
    if dem is not None:
        hs = B.hillshade(dem, dem_extent, z_factor=6)
        lo, hi = np.nanpercentile(dem, 1), np.nanpercentile(dem, 99.5)
        ax.imshow(np.ma.masked_invalid(dem), extent=dem_extent, origin='upper', cmap=HYPSO,
                  vmin=lo, vmax=max(hi, lo + 50), zorder=0, interpolation='bilinear', aspect='auto')
        ax.imshow(hs, extent=dem_extent, origin='upper', cmap='gray', alpha=0.35, zorder=0.5,
                  interpolation='bilinear', aspect='auto')
        ax.set_aspect(1 / math.cos(math.radians((bbox[2] + bbox[3]) / 2)))
    # затемнение за пределами территории
    if region_geom is not None:
        from shapely.geometry import box as sbox
        outside = sbox(bbox[0] - 1, bbox[2] - 1, bbox[1] + 1, bbox[3] + 1).difference(region_geom)
        draw_poly(ax, outside, fc='white', ec='none', alpha=0.45, zorder=1)
    if neighbors_geoms:
        for g in neighbors_geoms:
            draw_poly(ax, g, ec='#999', lw=0.5, zorder=5)
    if field is not None:
        LON, LAT = grid
        F = np.ma.masked_invalid(field)
        vin = inside_values(field, grid, region_geom)
        cm, norm, levels = _value_norm(vin[np.isfinite(vin)], cmap, n_levels)
        cf_ = ax.contourf(LON, LAT, F, levels=levels, cmap=cm, norm=norm, alpha=field_alpha, zorder=2, extend='both')
        cs = ax.contour(LON, LAT, F, levels=levels, colors='#333', linewidths=0.45, zorder=3)
        clip_to(ax, [cf_, cs], region_geom)
        label_contours(ax, cs, region_geom, decimals)
    if clusters:
        cl = sorted({c['cluster'] for c in clusters if c.get('cluster') is not None})
        colors = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(cl)}
        for c in clusters:
            if c.get('cluster') is not None:
                draw_poly(ax, c['geometry'], fc=colors[c['cluster']], ec='white', lw=0.5,
                          alpha=cluster_alpha, zorder=2)
    if districts:
        for d in districts:
            draw_poly(ax, d['geometry'], ec='#6E6E6E', lw=0.35, zorder=5)
    if water:
        draw_water(ax, water, clip=None)
    if region_geom is not None:
        draw_poly(ax, region_geom, ec='#B03A2E', lw=1.4, zorder=6)
    if cities:
        draw_cities(ax, cities, bbox, st=st)
    if st is not None:
        draw_stations(ax, st, color=st_color or '#222', size=26, labels=st_labels, marker='^', edge='white')
    handles = []
    if st is not None:
        handles.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='#222', markersize=7,
                              label='Метеостанция'))
    if water:
        handles += [Line2D([0], [0], color=WATER, lw=1.4, label='Река'),
                    Patch(fc=WATER_FILL, ec=WATER, lw=0.5, label='Озеро, водохранилище')]
    if region_geom is not None:
        handles.append(Line2D([0], [0], color='#B03A2E', lw=1.4, label='Граница субъекта'))
    if clusters:
        handles += [Patch(fc=colors[c], ec='#333', lw=0.4, alpha=0.8, label=f'Район {c}') for c in cl]
    rows = legend_top(ax, handles)
    if field is not None:
        _colorbar(fig, ax, cm, norm, value_label, decimals)
    elif dem is not None and relief_legend:
        from matplotlib.colors import Normalize
        _colorbar(fig, ax, HYPSO, Normalize(lo, max(hi, lo + 50)), 'Высота, м', 0)
    north_arrow(ax)
    scale_bar(ax, bbox)
    set_title(ax, title, rows)
    fig.tight_layout()
    return fig


# ─── 4. Растровая подложка ───────────────────────────────────────────────
def map_tiles(bbox, img, img_extent, attribution='', region_geom=None, polys=None, key='cluster',
              alpha=0.5, st=None, st_labels=True, title='', value_label='', cmap='viridis', decimals=1,
              figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    setup_ax(ax, bbox, grid=False)
    ax.imshow(img, extent=img_extent, origin='upper', zorder=0, interpolation='bilinear', aspect='auto')
    ax.set_aspect(1 / math.cos(math.radians((bbox[2] + bbox[3]) / 2)))
    if polys:
        if key == 'cluster':
            cl = sorted({p['cluster'] for p in polys if p.get('cluster') is not None})
            colors = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(cl)}
            for p in polys:
                if p.get('cluster') is not None:
                    draw_poly(ax, p['geometry'], fc=colors[p['cluster']], ec='white', lw=0.6, alpha=alpha, zorder=2)
            rows = _cluster_legend(ax, cl, colors)
        else:
            rows = 0
            cm, norm, _ = _value_norm([p.get('value', np.nan) for p in polys], cmap, 8)
            for p in polys:
                v = p.get('value', np.nan)
                if np.isfinite(v):
                    draw_poly(ax, p['geometry'], fc=cm(norm(v)), ec='white', lw=0.6, alpha=alpha, zorder=2)
            _colorbar(fig, ax, cm, norm, value_label, decimals)
        for p in polys:
            draw_poly(ax, p['geometry'], ec='#444', lw=0.4, zorder=3)
    if region_geom is not None:
        draw_poly(ax, region_geom, ec='#B03A2E', lw=1.4, zorder=6)
    if st is not None:
        draw_stations(ax, st, color='#222', size=24, labels=st_labels)
    ax.text(0.995, 0.005, attribution, transform=ax.transAxes, ha='right', va='bottom', fontsize=6,
            color='#333', path_effects=HALO, zorder=20)
    north_arrow(ax)
    scale_bar(ax, bbox)
    set_title(ax, title, rows if polys else 0)
    fig.tight_layout()
    return fig


# ─── 5. Интерактивная ────────────────────────────────────────────────────
def folium_map(st, polys=None, key='cluster', region_geom=None, value_col=None, value_label='',
               cluster_col=None, field=None, grid=None, decimals=1, caption=''):
    """
    Интерактивная карта: подложки (топографическая, физическая, OSM), граница
    территории, районы (по кластеру или значению), поле показателя
    (полупрозрачный растр), станции, окрашенные по кластеру или значению,
    с всплывающими подписями; легенда и пояснение «что показано».
    """
    import io
    import base64
    import folium
    import branca.colormap as bcm
    from shapely.geometry import mapping
    center = [float(st['lat'].mean()), float(st['lon'].mean())]
    m = folium.Map(location=center, zoom_start=6, tiles=None, control_scale=True)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Physical_Map/MapServer/tile/{z}/{y}/{x}',
                     name='Физическая (Esri)', attr='Esri').add_to(m)
    folium.TileLayer('OpenTopoMap', name='Топографическая (OpenTopoMap)', attr='© OpenTopoMap, © OSM').add_to(m)
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    colors, cmap_v = {}, None
    vals_all = []
    if cluster_col and cluster_col in st:
        cl = sorted(st[cluster_col].dropna().unique())
        colors = {c: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, c in enumerate(cl)}
    if value_col and value_col in st:
        vals_all = list(st[value_col].dropna().values)
        if field is not None:
            vin = inside_values(field, grid, region_geom)
            vals_all += list(vin[np.isfinite(vin)])
        lo, hi = float(np.nanmin(vals_all)), float(np.nanmax(vals_all))
        if hi == lo:
            hi = lo + 1
        vir = plt.get_cmap('viridis')
        cmap_v = bcm.LinearColormap([matplotlib_hex(vir(x)) for x in np.linspace(0, 1, 9)], vmin=lo, vmax=hi,
                                    caption=value_label)
    # поле показателя
    if field is not None and cmap_v is not None:
        LON, LAT = grid
        F = np.asarray(field, float)
        # растр уточняется в 6 раз, строки располагаются равномерно в проекции Меркатора (как у подложки),
        # затем обрезается точно по границе — заливка доходит до границы без «ступенек» и смещений
        from scipy.interpolate import RegularGridInterpolator
        lon0 = LON[0, :]
        lat0 = LAT[:, 0]
        if lat0[0] > lat0[-1]:
            lat0, F = lat0[::-1], F[::-1]
        Ff = np.where(np.isfinite(F), F, np.nanmean(F))
        itp = RegularGridInterpolator((lat0, lon0), Ff, bounds_error=False, fill_value=None)
        merc = lambda la: np.log(np.tan(np.pi / 4 + np.radians(la) / 2))
        imerc = lambda y: np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)
        ny, nx = len(lat0) * 6, len(lon0) * 6
        lat_f = imerc(np.linspace(merc(lat0.max()), merc(lat0.min()), ny))     # сверху вниз
        lon_f = np.linspace(lon0.min(), lon0.max(), nx)
        LON, LAT = np.meshgrid(lon_f, lat_f)
        F = itp(np.c_[LAT.ravel(), LON.ravel()]).reshape(LAT.shape)
        if region_geom is not None:
            import shapely
            F = np.where(shapely.contains_xy(region_geom, LON, LAT), F, np.nan)
        norm = (F - cmap_v.vmin) / (cmap_v.vmax - cmap_v.vmin)
        rgba = plt.get_cmap('viridis')(np.clip(norm, 0, 1))
        rgba[..., 3] = np.where(np.isfinite(F), 0.75, 0)
        buf = io.BytesIO()
        plt.imsave(buf, rgba, format='png')
        url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        folium.raster_layers.ImageOverlay(url, bounds=[[float(LAT.min()), float(LON.min())],
                                                       [float(LAT.max()), float(LON.max())]],
                                          name=f'Поле: {value_label}', opacity=1.0, interactive=False).add_to(m)
    # районы
    if polys:
        fg = folium.FeatureGroup(name='Административные районы')
        if key == 'cluster':
            pcl = sorted({p['cluster'] for p in polys if p.get('cluster') is not None})
            for c in pcl:
                colors.setdefault(c, CLUSTER_COLORS[len(colors) % len(CLUSTER_COLORS)])
        for p in polys:
            v = p.get(key)
            if key == 'cluster':
                col = colors.get(v, '#999')
                tip = f"{p['name']}: район {v}"
            else:
                ok = v is not None and np.isfinite(v)
                col = cmap_v(v) if (cmap_v is not None and ok) else '#bbb'
                tip = f"{p['name']}: {cf(v, decimals) if ok else 'нет данных'}"
            folium.GeoJson(mapping(p['geometry']), style_function=lambda f, c=col: dict(
                fillColor=c, color='#444', weight=0.7, fillOpacity=0.5), tooltip=tip).add_to(fg)
        fg.add_to(m)
    if region_geom is not None:
        folium.GeoJson(mapping(region_geom), name='Граница субъекта', style_function=lambda f: dict(
            fill=False, color='#B03A2E', weight=2.2)).add_to(m)
    fs = folium.FeatureGroup(name='Метеостанции')
    for _, r in st.iterrows():
        txt = f"<b>{r.get('name', r['st'])}</b> ({r['st']})"
        fill = '#F18F01'
        if cluster_col and cluster_col in r and r[cluster_col] in colors:
            fill = colors[r[cluster_col]]
            txt += f"<br>Район {r[cluster_col]}"
        if value_col and value_col in r and np.isfinite(r[value_col]):
            fill = cmap_v(r[value_col]) if not cluster_col else fill
            txt += f"<br>{value_label}: {cf(r[value_col], decimals)}"
        folium.CircleMarker([r['lat'], r['lon']], radius=6, color='#222', weight=1, fill=True, fill_color=fill,
                            fill_opacity=0.95, tooltip=folium.Tooltip(txt)).add_to(fs)
    fs.add_to(m)
    if cmap_v is not None and not cluster_col:
        cmap_v.add_to(m)
    if colors:
        items = ''.join(f'<div><span style="display:inline-block;width:14px;height:14px;background:{c};'
                        f'border:1px solid #333;margin-right:6px;vertical-align:middle"></span>Район {k}</div>'
                        for k, c in colors.items())
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;bottom:30px;left:12px;z-index:9999;background:white;padding:6px 10px;'
            f'border:1px solid #999;border-radius:4px;font:13px sans-serif">{items}</div>'))
    if caption:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:12px;left:60px;z-index:9999;background:rgba(255,255,255,.92);'
            f'padding:6px 10px;border:1px solid #999;border-radius:4px;font:13px sans-serif;max-width:55%">'
            f'{caption}</div>'))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def matplotlib_hex(rgba):
    from matplotlib.colors import to_hex
    return to_hex(rgba)
