"""Stream annual product ZIPs into observation-pooled distributions and SHPs.

No inversion, source modification, calendar reweighting, or missing-value fill.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time
import uuid

import fiona
from affine import Affine
from numcodecs import Blosc
from numcodecs.blosc import set_nthreads
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
import zarr

from statistics_core import TAGS, UNITS, add_observations, lake_scalar, selected_months, temporal_map

BASE = Path(__file__).resolve().parent
VERSION = '1.0.0-pooled-temporal-mean'
OUTPUT_NAMES = ('climatology_month', 'annual', 'climatology_annual')
METHOD = {
    'order': 'finite-observation arithmetic temporal mean per pixel, then equal-pixel spatial mean',
    'time_weighting': 'Each selected acquisition record has unit weight; no equal-year or equal-month averaging; no deduplication.',
    'validity': 'Only finite product values contribute to sum/count; NaN and infinity are invalid; no fill with zero.',
    'global_coverage': 'pixel finite observation count over ALL selected years/seasonal months / ALL selected acquisition records >= 0.05',
    'period_coverage': 'pixel finite observation count within current calendar year or pooled calendar month / ALL selected records of that group >= 0.05',
    'coverage_combination': 'global condition AND current-period condition AND original lake polygon; positive count required; zero-record groups are invalid',
    'empty_scenes': 'Included in every applicable denominator irrespective of scene_status and valid_pixel_count.',
    'climatology_month': 'Pool all observations of the same calendar month across selected years.',
    'annual': 'Pool all selected observations within each calendar year.',
    'climatology_annual': 'Pool all selected observations across the entire requested period.',
    'season': 'EPL and YGPL: January-December; IMXL, TPL and NPML: May-October before ANY denominators or sums.',
    'precision': 'float64 sums, uint32 raw observation counts, float32 saved maps; lake means computed in float64 from reopened saved float32 maps.',
    'spatial_weighting': 'Each retained grid pixel has equal weight, not geodesic-area weighting; valid fraction denominator is original polygon pixel-center rasterization.',
    'nobs': 'Raw finite counts retained even where either coverage condition rejects the mean.',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':')).encode('utf-8')


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    temporary.write_bytes(encoded(value))
    os.replace(temporary, path)


def fingerprint(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = hash_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Input changed during hashing: {path}')
    return {'path': str(path), 'size_bytes': after.st_size, 'mtime_ns': after.st_mtime_ns, 'sha256': digest}


def assert_unchanged(info):
    stat = Path(info['path']).stat()
    if stat.st_size != info['size_bytes'] or stat.st_mtime_ns != info['mtime_ns']:
        raise RuntimeError(f'Input changed during processing: {info["path"]}')


def init_worker():
    set_nthreads(1)
    zarr.config.set({'async.concurrency': 1, 'threading.max_workers': 1})


def resolve_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (BASE / path).resolve()


def validate_lake_ids(lake_ids):
    """Validate an explicit subset; IDs are canonical, not codes or aliases."""
    if (not isinstance(lake_ids, list) or not lake_ids
            or any(not isinstance(value, str) or not value or value != value.strip() for value in lake_ids)):
        raise ValueError('lake_ids must be a nonempty list of canonical lake_id strings')
    if len(set(lake_ids)) != len(lake_ids):
        raise ValueError('lake_ids must not contain duplicates')


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    for key in ('source_root', 'output_root', 'boundaries', 'work_root'):
        config[key] = str(resolve_path(config[key]))
    years = config['years']
    if not years or any(isinstance(y, bool) or not isinstance(y, int) for y in years) or sorted(set(years)) != years:
        raise ValueError('years must be a strictly increasing list of calendar years')
    if float(config['coverage_fraction']) != .05:
        raise ValueError('This approved method requires coverage_fraction=0.05')
    if config.get('read_scenes') != 16:
        raise ValueError('read_scenes must be 16 to match the source chunk layout')
    for key in ('workers', 'spatial_chunk'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if 'lake_ids' in config:
        validate_lake_ids(config['lake_ids'])
    return config


def read_boundaries(path, lake_filter=None, lake_ids=None):
    """Select in memory while preserving the full original boundary file/hash.

    Optional canonical lake_ids restrict the configured scope. --lake may then
    narrow that scope to one ID/code/name; it never expands the configured set.
    Output ordering follows the boundary file, independent of lake_ids order.
    """
    collection = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    all_features = collection['features']
    ids = [item['properties']['lake_id'] for item in all_features]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate lake_id in boundaries')
    features = all_features
    if lake_ids is not None:
        validate_lake_ids(lake_ids)
        unknown = sorted(set(lake_ids) - set(ids))
        if unknown:
            raise ValueError(f'Unknown configured lake_ids: {unknown}')
        configured = set(lake_ids)
        features = [f for f in all_features if f['properties']['lake_id'] in configured]
    if lake_filter:
        selected = [f for f in all_features if lake_filter in (f['properties']['lake_id'], f['properties']['code'], f['properties']['name'])]
        if len(selected) != 1:
            raise ValueError(f'--lake must select exactly one lake: {lake_filter}')
        if lake_ids is not None and selected[0]['properties']['lake_id'] not in set(lake_ids):
            raise ValueError(f'--lake {lake_filter!r} is outside configured lake_ids')
        features = selected
    for item in features:
        properties = item['properties']
        selected_months(properties['region'])
        if item['geometry']['type'] not in ('Polygon', 'MultiPolygon'):
            raise ValueError('Expected original polygon or multipolygon lake geometry')
    return features


def make_plan(config, features):
    missing = []
    for feature in features:
        lake = feature['properties']['lake_id']
        for year in config['years']:
            path = Path(config['source_root']) / lake / f'{year}.zarr.zip'
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError('All explicitly requested years are required; no implicit partial climatology. Missing: ' + '\n'.join(missing[:20]) + (f'\n... {len(missing)} missing in total' if len(missing) > 20 else ''))
    boundary = fingerprint(config['boundaries'])
    code_hashes = {name: hash_file(BASE / name) for name in ('pipeline.py', 'statistics_core.py')}
    tasks = []
    for feature in features:
        lake = feature['properties']['lake_id']
        sources = []
        for year in config['years']:
            item = fingerprint(Path(config['source_root']) / lake / f'{year}.zarr.zip')
            item['year'] = year
            sources.append(item)
        identity = {'version': VERSION, 'feature': feature['properties'], 'source_years': config['years'],
                    'sources': sources, 'boundary_sha256': boundary['sha256'], 'code_sha256': code_hashes,
                    'coverage_fraction': .05, 'method': METHOD, 'spatial_chunk': config['spatial_chunk']}
        lake_id = hashlib.sha256(encoded(identity)).hexdigest()
        tasks.append({'feature': feature, 'sources': sources, 'boundary': boundary, 'identity': identity,
                      'lake_fingerprint': lake_id, 'work_dir': str(Path(config['work_root']) / 'lakes' / lake / lake_id),
                      'spatial_chunk': config['spatial_chunk']})
    run_id = hashlib.sha256(encoded({'lake_fingerprints': [t['lake_fingerprint'] for t in tasks], 'version': VERSION})).hexdigest()
    return tasks, run_id


def grid_from_group(group, year, lake, boundary_sha256):
    attrs = dict(group.attrs)
    if attrs.get('lake_id') != lake or attrs.get('year') != year:
        raise ValueError(f'Archive lake/year metadata mismatch: {lake}/{year}')
    if attrs.get('boundary_sha256') != boundary_sha256:
        raise ValueError(f'Archive and statistics boundary hashes differ: {lake}/{year}')
    metadata = json.loads(group['metadata_json'][:].tobytes())
    if metadata.get('complete') is not True:
        raise ValueError(f'Annual product is not complete: {lake}/{year}')
    shape = tuple(group['Chla'].shape)
    if len(shape) != 3:
        raise ValueError('Expected products(time,y,x)')
    for tag in TAGS:
        if tuple(group[tag].shape) != shape or group[tag].dtype != np.dtype('float32'):
            raise ValueError(f'Wrong product schema: {lake}/{year}/{tag}')
        if group[tag].attrs.get('units') != UNITS[tag]:
            raise ValueError(f'Unexpected units: {lake}/{year}/{tag}')
    times = np.asarray(group['acquisition_time'][:], dtype=np.int64)
    if times.shape != (shape[0],):
        raise ValueError('Time coordinate length does not match products')
    months = np.empty(len(times), dtype=np.uint8)
    for i, value in enumerate(times):
        date = datetime.strptime(str(int(value)), '%Y%m%d%H%M%S')
        if date.year != year:
            raise ValueError(f'Observation belongs to a different year: {value}')
        months[i] = date.month
    transform = tuple(float(v) for v in attrs['transform'][:6])
    if len(transform) != 6 or not np.isfinite(transform).all():
        raise ValueError('Invalid rasterio Affine transform')
    affine = Affine(*transform)
    if affine.b != 0 or affine.d != 0:
        raise ValueError('Rotated grids require two-dimensional coordinates')
    crs = rasterio.crs.CRS.from_wkt(attrs['crs_wkt'])
    x, y = np.asarray(group['x'][:]), np.asarray(group['y'][:])
    if x.shape != (shape[2],) or y.shape != (shape[1],):
        raise ValueError('Grid coordinate lengths do not match products')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite grid coordinates')
    np.testing.assert_array_equal(x, affine.c + (np.arange(shape[2]) + .5) * affine.a)
    np.testing.assert_array_equal(y, affine.f + (np.arange(shape[1]) + .5) * affine.e)
    return {'shape': shape[1:], 'transform': transform, 'crs_wkt': attrs['crs_wkt'], 'crs': crs, 'x': x, 'y': y}, months


def require_same_grid(reference, current):
    if current['shape'] != reference['shape'] or current['transform'] != reference['transform'] or current['crs'] != reference['crs']:
        raise ValueError('Cross-year grid/transform/CRS mismatch; resampling is not permitted')
    np.testing.assert_array_equal(current['x'], reference['x'])
    np.testing.assert_array_equal(current['y'], reference['y'])


def polygon_mask(feature, grid):
    geometry = feature['geometry']
    if grid['crs'] != rasterio.crs.CRS.from_epsg(4326):
        geometry = transform_geom('EPSG:4326', grid['crs'], geometry)
    mask = rasterize([(geometry, 1)], out_shape=grid['shape'], transform=Affine(*grid['transform']),
                     fill=0, all_touched=False, dtype='uint8').astype(bool)
    if not mask.any():
        raise ValueError('Original lake polygon has no pixel-center intersections with this grid')
    return mask


def create_array(group, name, shape, dtype, chunks, dims, fill=0, attrs=None):
    return group.create_array(name, shape=shape, dtype=dtype, chunks=chunks, fill_value=fill,
                              compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE),
                              attributes={'_ARRAY_DIMENSIONS': dims, **(attrs or {})})


def write_coordinate(group, name, values, dim, attrs=None):
    values = np.asarray(values)
    array = create_array(group, name, values.shape, values.dtype, tuple(max(1, n) for n in values.shape), [dim], attrs=attrs)
    array[:] = values


def output_groups(paths, grid, task, month_counts, year_counts):
    """Open final ZIP writers; write every product chunk once to avoid ZIP duplicates."""
    stores, groups = {}, {}
    years = task['identity']['source_years']
    used = np.array([m in selected_months(task['feature']['properties']['region']) for m in range(1, 13)], dtype=np.uint8)
    common = {**task['identity'], 'lake_fingerprint': task['lake_fingerprint'], 'lake_id': task['feature']['properties']['lake_id'],
              'lake_name': task['feature']['properties']['name'], 'region': task['feature']['properties']['region'],
              'created_utc': now(), 'crs_wkt': grid['crs_wkt'], 'transform': list(grid['transform']),
              'transform_convention': 'rasterio.Affine a,b,c,d,e,f (NOT GDAL order)',
              'sourceyears': years, 'input_fingerprints': task['sources'], 'boundary_source': task['boundary'],
              'global_n_scenes': int(sum(year_counts)), 'month_used_definition': '1 included by regional season; 0 excluded before sums and denominators',
              'complete': True}
    try:
        for name in OUTPUT_NAMES:
            stores[name] = zarr.storage.ZipStore(str(paths[name]), mode='w')
            groups[name] = zarr.group(store=stores[name], zarr_format=2, attributes={**common, 'statistic': name})
            group = groups[name]
            write_coordinate(group, 'x', grid['x'], 'x')
            write_coordinate(group, 'y', grid['y'], 'y')
            write_coordinate(group, 'month', np.arange(1, 13, dtype=np.uint8), 'month')
            write_coordinate(group, 'month_used', used, 'month')
            if name == 'climatology_month':
                write_coordinate(group, 'n_scenes', np.asarray(month_counts, dtype=np.uint32), 'month')
                leading, dims = (12,), ['month', 'y', 'x']
            elif name == 'annual':
                write_coordinate(group, 'year', np.asarray(years, dtype=np.int16), 'year')
                write_coordinate(group, 'n_scenes', np.asarray(year_counts, dtype=np.uint32), 'year')
                leading, dims = (len(years),), ['year', 'y', 'x']
            else:
                array = create_array(group, 'n_scenes', (), 'uint32', (), [])
                array[()] = np.uint32(sum(year_counts))
                leading, dims = (), ['y', 'x']
            shape = leading + grid['shape']
            spatial = tuple(min(task['spatial_chunk'], size) for size in grid['shape'])
            chunks = ((1,) if leading else ()) + spatial
            for tag in TAGS:
                create_array(group, tag, shape, 'float32', chunks, dims, np.nan, {'units': UNITS[tag], 'cell_methods': 'time: mean', 'ancillary_variables': tag + '_nobs'})
                create_array(group, tag + '_nobs', shape, 'uint32', chunks, dims, attrs={'long_name': 'raw finite observation count before coverage filtering'})
        return stores, groups
    except BaseException:
        for store in stores.values():
            store.close()
        raise


def summarize_archive(path, name, feature, lake_mask, years):
    rows = []
    with zarr.storage.ZipStore(str(path), mode='r') as store:
        group = zarr.open_group(store=store, mode='r')
        if group.attrs.get('complete') is not True:
            raise ValueError('Statistics archive missing completion marker')
        periods = list(range(12)) if name == 'climatology_month' else list(range(len(years))) if name == 'annual' else [None]
        for index in periods:
            if name == 'climatology_month' and not int(group['month_used'][index]):
                continue
            record = {'lake_id': feature['properties']['lake_id'], 'name_cn': feature['properties']['name'],
                      'region': feature['properties']['region'], 'year': years[index] if name == 'annual' else None,
                      'month': index + 1 if name == 'climatology_month' else None,
                      'yr_start': min(years), 'yr_end': max(years), 'n_lakepix': int(lake_mask.sum()),
                      'nsce': int(group['n_scenes'][index] if index is not None else group['n_scenes'][()])}
            for tag in TAGS:
                saved = group[tag][index] if index is not None else group[tag][:]
                scalar = lake_scalar(saved, lake_mask)
                record[tag] = scalar['mean']
                record[tag + '_npix'] = scalar['n_valid']
                record[tag + '_frac'] = scalar['valid_fraction']
            rows.append(record)
    return rows


def process_lake(task):
    init_worker()
    started = time.perf_counter()
    lake = task['feature']['properties']['lake_id']
    work = Path(task['work_dir'])
    result_dir, scratch = work / 'result', work / 'scratch'
    completion = result_dir / 'lake_result.json'
    if completion.is_file():
        result = json.loads(completion.read_text(encoding='utf-8'))
        if result.get('lake_fingerprint') != task['lake_fingerprint']:
            raise ValueError('Local statistics belong to a different input/method fingerprint')
        for name, info in result['outputs'].items():
            if hash_file(result_dir / info['filename']) != info['sha256']:
                raise ValueError(f'Completed local statistics checksum mismatch: {lake}/{name}')
        for item in task['sources']:
            assert_unchanged(item)
        return result
    result_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    allowed = selected_months(task['feature']['properties']['region'])
    grid, scene_months, month_counts, year_counts = None, [], np.zeros(12, dtype=np.uint64), []
    for source in task['sources']:
        assert_unchanged(source)
        with zarr.storage.ZipStore(source['path'], mode='r') as store:
            group = zarr.open_group(store=store, mode='r')
            current, months = grid_from_group(group, source['year'], lake, task['boundary']['sha256'])
        if grid is None:
            grid = current
        else:
            require_same_grid(grid, current)
        scene_months.append(months)
        include = np.isin(months, allowed)
        year_counts.append(int(include.sum()))
        month_counts += np.bincount(months[include], minlength=13)[1:13].astype(np.uint64)
    if sum(year_counts) > np.iinfo(np.uint32).max:
        raise OverflowError('Selected acquisition count exceeds uint32')
    mask = polygon_mask(task['feature'], grid)
    paths = {name: result_dir / (name + '.zarr.zip') for name in OUTPUT_NAMES}
    temporary = {name: path.with_name(path.name + '.partial-' + uuid.uuid4().hex) for name, path in paths.items()}
    stores, groups = output_groups(temporary, grid, task, month_counts, year_counts)
    try:
        for tag in TAGS:
            monthly_sum = np.zeros((12,) + grid['shape'], dtype=np.float64)
            monthly_count = np.zeros((12,) + grid['shape'], dtype=np.uint32)
            total_sum = np.zeros(grid['shape'], dtype=np.float64)
            total_count = np.zeros(grid['shape'], dtype=np.uint32)
            for yi, (source, months) in enumerate(zip(task['sources'], scene_months)):
                annual_sum = np.zeros(grid['shape'], dtype=np.float64)
                annual_count = np.zeros(grid['shape'], dtype=np.uint32)
                with zarr.storage.ZipStore(source['path'], mode='r') as store:
                    group = zarr.open_group(store=store, mode='r')
                    for start in range(0, len(months), 16):
                        end = min(start + 16, len(months))
                        block_months = months[start:end]
                        choose = np.isin(block_months, allowed)
                        if not choose.any():
                            continue
                        block = np.asarray(group[tag][start:end])[choose]
                        if np.any(np.isfinite(block) & ~mask[None, :, :]):
                            raise ValueError(f'Finite source values outside original lake geometry: {lake}/{source["year"]}/{tag}')
                        add_observations(annual_sum, annual_count, block)
                        kept_months = block_months[choose]
                        for month in np.unique(kept_months):
                            add_observations(monthly_sum[month - 1], monthly_count[month - 1], block[kept_months == month])
                total_sum += annual_sum
                total_count += annual_count
                np.save(scratch / f'{tag}_{source["year"]}_sum.npy', annual_sum)
                np.save(scratch / f'{tag}_{source["year"]}_count.npy', annual_count)
                write_json(work / 'progress.json', {'lake_id': lake, 'tag': tag, 'year': source['year'], 'phase': 'accumulated', 'updated_utc': now()})
                assert_unchanged(source)
            if not np.array_equal(total_count, monthly_count.sum(axis=0, dtype=np.uint64)):
                raise AssertionError('Monthly and full-period observation counts differ')
            for mi in range(12):
                groups['climatology_month'][tag][mi] = temporal_map(monthly_sum[mi], monthly_count[mi], int(month_counts[mi]), total_count, sum(year_counts), mask)
                groups['climatology_month'][tag + '_nobs'][mi] = monthly_count[mi]
            for yi, source in enumerate(task['sources']):
                annual_sum = np.load(scratch / f'{tag}_{source["year"]}_sum.npy', mmap_mode='r')
                annual_count = np.load(scratch / f'{tag}_{source["year"]}_count.npy', mmap_mode='r')
                groups['annual'][tag][yi] = temporal_map(annual_sum, annual_count, year_counts[yi], total_count, sum(year_counts), mask)
                groups['annual'][tag + '_nobs'][yi] = annual_count
            groups['climatology_annual'][tag][:] = temporal_map(total_sum, total_count, sum(year_counts), total_count, sum(year_counts), mask)
            groups['climatology_annual'][tag + '_nobs'][:] = total_count
    finally:
        for store in stores.values():
            store.close()
    for name in OUTPUT_NAMES:
        os.replace(temporary[name], paths[name])
    rows = {name: summarize_archive(paths[name], name, task['feature'], mask, task['identity']['source_years']) for name in OUTPUT_NAMES}
    for item in task['sources']:
        assert_unchanged(item)
    result = {'complete': True, 'lake_id': lake, 'lake_fingerprint': task['lake_fingerprint'],
              'result_dir': str(result_dir), 'n_lakepix': int(mask.sum()), 'rows': rows,
              'outputs': {name: {'filename': path.name, 'sha256': hash_file(path), 'size_bytes': path.stat().st_size} for name, path in paths.items()},
              'sourceyears': task['identity']['source_years'], 'input_fingerprints': task['sources'],
              'elapsed_seconds': time.perf_counter() - started, 'completed_utc': now()}
    write_json(completion, result)
    return result


SHP_FIELDS = {'lake_id': 'str:60', 'name_cn': 'str:80', 'region': 'str:8', 'year': 'int:6', 'month': 'int:3',
              'yr_start': 'int:6', 'yr_end': 'int:6', 'nsce': 'int:18', 'n_lakepix': 'int:18'}
for _tag in TAGS:
    SHP_FIELDS[_tag] = 'float:24.12'
    SHP_FIELDS[_tag + '_npix'] = 'int:18'
    SHP_FIELDS[_tag + '_frac'] = 'float:20.12'


def write_shapefiles(destination, results, features):
    geometry = {f['properties']['lake_id']: f['geometry'] for f in features}
    for name in OUTPUT_NAMES:
        path = destination / (name + '.shp')
        schema = {'geometry': 'Polygon', 'properties': SHP_FIELDS}
        with fiona.open(str(path), mode='w', driver='ESRI Shapefile', schema=schema, crs='EPSG:4326', encoding='UTF-8') as writer:
            for result in results:
                for row in result['rows'][name]:
                    writer.write({'geometry': geometry[result['lake_id']], 'properties': row})
        expected = sum(len(r['rows'][name]) for r in results)
        with fiona.open(str(path), encoding='UTF-8') as reader:
            if len(reader) != expected or rasterio.crs.CRS.from_user_input(reader.crs) != rasterio.crs.CRS.from_epsg(4326):
                raise ValueError('Shapefile row count or CRS verification failed')
            if any(len(key) > 10 for key in reader.schema['properties']):
                raise ValueError('Shapefile field name exceeds 10 characters')


def verify_delivery(directory, run_id):
    path = Path(directory) / 'run_manifest.json'
    if not path.is_file():
        raise FileExistsError(f'Existing destination has no complete statistics manifest: {directory}')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('run_fingerprint') != run_id or manifest.get('complete') is not True:
        raise FileExistsError(f'Existing destination contains different statistics; refusing to overwrite: {directory}')
    for relative, expected in manifest['files'].items():
        actual = Path(directory) / relative
        if not actual.is_file() or hash_file(actual) != expected['sha256']:
            raise FileExistsError(f'Existing destination checksum differs; refusing to overwrite: {actual}')
    return manifest


def assemble_delivery(config, tasks, results, run_id, features):
    destination = Path(config['work_root']) / 'deliveries' / run_id
    if destination.exists():
        verify_delivery(destination, run_id)
        return destination
    staging = destination.with_name(run_id + '.partial-' + uuid.uuid4().hex)
    staging.mkdir(parents=True, exist_ok=False)
    for result in results:
        target = staging / result['lake_id']
        target.mkdir()
        for info in result['outputs'].values():
            shutil.copy2(Path(result['result_dir']) / info['filename'], target / info['filename'])
        shutil.copy2(Path(result['result_dir']) / 'lake_result.json', target / 'lake_result.json')
    write_shapefiles(staging, results, features)
    shutil.copy2(BASE / 'README.txt', staging / 'README.txt')
    write_json(staging / 'config_used.json', config)
    files = {str(path.relative_to(staging)).replace('\\', '/'): {'sha256': hash_file(path), 'size_bytes': path.stat().st_size}
             for path in sorted(staging.rglob('*')) if path.is_file()}
    write_json(staging / 'run_manifest.json', {'complete': True, 'version': VERSION, 'created_utc': now(),
               'run_fingerprint': run_id, 'sourceyears': config['years'], 'method': METHOD,
               'lake_fingerprints': {t['feature']['properties']['lake_id']: t['lake_fingerprint'] for t in tasks},
               'input_fingerprints': {t['feature']['properties']['lake_id']: t['sources'] for t in tasks},
               'boundary_source': tasks[0]['boundary'], 'files': files})
    os.rename(staging, destination)
    verify_delivery(destination, run_id)
    return destination


def publish_delivery(local, output, run_id):
    output = Path(output)
    if output.exists():
        verify_delivery(output, run_id)
        return 'already_verified'
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + '.partial-' + uuid.uuid4().hex)
    shutil.copytree(local, staging)
    verify_delivery(staging, run_id)
    # Same-directory rename publishes a fully copied/verified new directory.
    # Windows rename refuses an existing destination; no existing result is removed.
    os.rename(staging, output)
    verify_delivery(output, run_id)
    return 'published'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(BASE / 'config' / 'run.json'))
    parser.add_argument('--lake', help='Exact lake_id, code, or Chinese name; must be inside configured lake_ids when provided')
    parser.add_argument('--local-only', action='store_true', help='Build and validate local distributions/SHPs without publishing to NAS')
    parser.add_argument('--workers', type=int, help='Override the configured worker process count')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.workers is not None:
        if args.workers < 1:
            parser.error('--workers must be positive')
        config['workers'] = args.workers
    features = read_boundaries(config['boundaries'], args.lake, lake_ids=config.get('lake_ids'))
    init_worker()
    tasks, run_id = make_plan(config, features)
    if Path(config['output_root']).exists() and not args.local_only:
        verify_delivery(config['output_root'], run_id)
        print(json.dumps({'status': 'already_verified', 'output': config['output_root'], 'run_fingerprint': run_id}, ensure_ascii=False), flush=True)
        return 0
    print(json.dumps({'status': 'planned', 'lakes': len(tasks), 'years': config['years'], 'workers': min(config['workers'], len(tasks)), 'run_fingerprint': run_id}, ensure_ascii=False), flush=True)
    results = []
    if len(tasks) == 1 or config['workers'] == 1:
        for task in tasks:
            result = process_lake(task)
            results.append(result)
            print(json.dumps({'status': 'lake_complete', 'lake': result['lake_id'], 'elapsed_seconds': result['elapsed_seconds']}, ensure_ascii=False), flush=True)
    else:
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=min(config['workers'], len(tasks)), mp_context=context, initializer=init_worker) as pool:
            futures = {pool.submit(process_lake, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps({'status': 'lake_complete', 'lake': result['lake_id'], 'elapsed_seconds': result['elapsed_seconds']}, ensure_ascii=False), flush=True)
    order = {f['properties']['lake_id']: i for i, f in enumerate(features)}
    results.sort(key=lambda result: order[result['lake_id']])
    local = assemble_delivery(config, tasks, results, run_id, features)
    status = 'local_only' if args.local_only else publish_delivery(local, config['output_root'], run_id)
    print(json.dumps({'status': status, 'local_result': str(local), 'output': None if args.local_only else config['output_root'], 'run_fingerprint': run_id}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
