"""Per-year monthly products using frozen daily-first temporal arithmetic.

Task source ``read_path`` values must refer to caches already SHA256-verified by
the caller. This engine never opens or stats original NAS product paths. It
checks cache size/stability, validates product provenance, and verifies the
small prior full-period statistics archive against its trusted checksum.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

from affine import Affine
import numpy as np
import rasterio
import zarr

import collection_pipeline as collection
import daily_collection_pipeline as daily
import daily_statistics_core as core
import pipeline as storage
from daily_safe_runner import safe_write_json

VERSION = '1.0.0-daily-first-year-month'
AGGREGATION = core.AGGREGATION
TAGS, UNITS = core.TAGS, core.UNITS
METHOD = {key: value for key, value in core.METHOD.items()
          if key not in ('climatology_month', 'annual', 'climatology_annual')}
METHOD.update(
    period_coverage='pixel finite-day count in the current calendar year and month / number of distinct selected source acquisition dates in that year-month >= 0.05',
    year_month='Each archive represents one available source year and contains 12 calendar months; each month averages its finite daily means with equal day weights.',
    global_reference='The all-available-year finite-day counts and source-day denominator come from a SHA256-pinned daily-first climatology_annual archive using exactly the same sources, grid, lake boundary and regional season.',
    excluded_months='Months outside the regional season contain NaN means and zero counts, with month_used=0; no lake-level SHP row is produced for them.')
SHP_FIELDS = {**daily.SHP_FIELDS, 'ym': 'int:6'}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _normal(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def _source_evidence(source):
    evidence = {key: source[key] for key in ('year', 'path', 'size_bytes', 'mtime_ns', 'sha256')}
    if source.get('run_fingerprint'):
        evidence['run_fingerprint'] = source['run_fingerprint']
    return evidence


def _local_stamp(path):
    info = Path(path).stat()
    return info.st_size, info.st_mtime_ns


def _check_local_stamps(snapshots):
    for path, stamp in snapshots.items():
        if _local_stamp(path) != stamp:
            raise RuntimeError('Local input changed while producing monthly maps: ' + str(path))


def _validate_task(task):
    lake = task['feature']['properties']['lake_id']
    years = [source['year'] for source in task['sources']]
    if not years or years != sorted(set(years)):
        raise ValueError('Monthly sources must have unique increasing years')
    if task['identity'].get('source_years') != years:
        raise ValueError('Monthly identity differs from available source years')
    for key in ('lake_fingerprint', 'daily_lake_fingerprint', 'global_archive_sha256'):
        if not isinstance(task.get(key), str) or re.fullmatch('[0-9a-f]{64}', task[key]) is None:
            raise ValueError('Missing/invalid frozen monthly evidence: ' + key)
    if hashlib.sha256(storage.encoded(task['identity'])).hexdigest() != task['lake_fingerprint']:
        raise ValueError('Monthly scientific fingerprint cannot be reconstructed')
    cache_root = Path(task['cache_root'])
    if not cache_root.is_absolute() or str(cache_root.drive).startswith('\\\\'):
        raise ValueError('Monthly computation requires an absolute local cache root')
    snapshots = {}
    for source in task['sources']:
        if re.fullmatch('[0-9a-f]{64}', source['sha256']) is None:
            raise ValueError('Invalid source SHA256')
        read_path = Path(source['read_path'])
        expected = cache_root / lake / f'{source["year"]}_{source["sha256"]}.zarr.zip'
        if not read_path.is_absolute() or _normal(read_path) != _normal(expected):
            raise ValueError('Monthly source read_path differs from its frozen local cache')
        stamp = _local_stamp(read_path)
        if stamp[0] != source['size_bytes']:
            raise ValueError('Monthly local cache size differs from frozen source')
        snapshots[str(read_path)] = stamp
    return years, snapshots


def read_global_reference(task):
    """Load proven global valid-day counts; never reconstruct them from scenes."""
    path = Path(task['global_archive'])
    if not path.is_absolute() or str(path.drive).startswith('\\\\'):
        raise ValueError('The full-period reference must be a local archive')
    before = _local_stamp(path)
    if storage.hash_file(path) != task['global_archive_sha256']:
        raise ValueError('Full-period reference SHA256 differs')
    sources = [_source_evidence(source) for source in task['sources']]
    years = [source['year'] for source in task['sources']]
    feature = task['feature']['properties']
    with zarr.storage.ZipStore(str(path), mode='r') as store:
        group = zarr.open_group(store=store, mode='r')
        attrs = dict(group.attrs)
        if (attrs.get('complete') is not True or attrs.get('statistic') != 'climatology_annual'
                or attrs.get('aggregation') != AGGREGATION or attrs.get('coverage_unit') != 'source_acquisition_day'
                or attrs.get('lake_fingerprint') != task['daily_lake_fingerprint']
                or attrs.get('lake_id') != feature['lake_id'] or attrs.get('feature') != feature
                or attrs.get('region') != feature['region']
                or attrs.get('boundary_sha256') != task['boundary']['sha256']
                or attrs.get('coverage_fraction') != .05 or attrs.get('method') != core.METHOD):
            raise ValueError('Full-period reference scientific identity differs')
        if (attrs.get('source_years') != years or attrs.get('sourceyears') != years
                or [_source_evidence(s) for s in attrs.get('input_fingerprints', [])] != sources
                or [_source_evidence(s) for s in attrs.get('sources', [])] != sources):
            raise ValueError('Full-period reference source set differs')
        used = np.asarray([month in core.selected_months(feature['region']) for month in range(1, 13)], dtype=np.uint8)
        np.testing.assert_array_equal(group['month'][:], np.arange(1, 13))
        np.testing.assert_array_equal(group['month_used'][:], used)
        counts = {}
        shape = tuple(task['shape_hw'])
        total_days, total_scenes = attrs['global_n_days'], attrs['global_n_scenes']
        for name, value in (('global_n_days', total_days), ('global_n_scenes', total_scenes)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= np.iinfo(np.uint32).max:
                raise ValueError('Invalid full-period count: ' + name)
        if total_days > total_scenes or int(group['n_days'][()]) != total_days or int(group['n_scenes'][()]) != total_scenes:
            raise ValueError('Inconsistent full-period date/scene denominators')
        for tag in TAGS:
            array = group[tag + '_ndays']
            if tuple(array.shape) != shape or array.dtype != np.dtype('uint32'):
                raise ValueError('Invalid full-period per-pixel day count schema')
            counts[tag] = np.asarray(array[:])
            if np.any(counts[tag] > total_days):
                raise ValueError('Full-period valid-day counts exceed source dates')
        transform = tuple(float(value) for value in attrs['transform'][:6])
        if len(transform) != 6 or not np.isfinite(transform).all():
            raise ValueError('Invalid full-period transform')
        affine = Affine(*transform)
        if affine.b != 0 or affine.d != 0:
            raise ValueError('Rotated full-period grid is unsupported')
        x, y = np.asarray(group['x'][:]), np.asarray(group['y'][:])
        if x.shape != (shape[1],) or y.shape != (shape[0],):
            raise ValueError('Full-period grid coordinate shape differs')
        np.testing.assert_array_equal(x, affine.c + (np.arange(shape[1]) + .5) * affine.a)
        np.testing.assert_array_equal(y, affine.f + (np.arange(shape[0]) + .5) * affine.e)
        grid = {'shape': shape, 'transform': transform, 'crs_wkt': attrs['crs_wkt'],
                'crs': rasterio.crs.CRS.from_wkt(attrs['crs_wkt']), 'x': x, 'y': y}
    if _local_stamp(path) != before:
        raise RuntimeError('Full-period archive changed while loading reference counts')
    return {'grid': grid, 'ndays': counts, 'n_days': total_days, 'n_scenes': total_scenes,
            'path': str(path), 'sha256': task['global_archive_sha256'], 'stamp': before,
            'daily_lake_fingerprint': task['daily_lake_fingerprint']}


def _source_groups(task, reference):
    allowed = core.selected_months(task['feature']['properties']['region'])
    plans = []
    for source in task['sources']:
        with zarr.storage.ZipStore(source['read_path'], mode='r') as store:
            group = zarr.open_group(store=store, mode='r')
            grid, _, source_fp = collection.validate_product(group, source, task)
            storage.require_same_grid(reference['grid'], grid)
            dates = core.day_groups(group['acquisition_time'][:], source['year'], allowed)
        n_days = np.zeros(12, dtype=np.uint32)
        n_scenes = np.zeros(12, dtype=np.uint32)
        for _, month, indices in dates:
            n_days[month - 1] += 1
            n_scenes[month - 1] += len(indices)
        plans.append({'source': source, 'dates': dates, 'n_days': n_days,
                      'n_scenes': n_scenes, 'source_fingerprint': source_fp})
    if (sum(int(p['n_days'].sum(dtype=np.uint64)) for p in plans) != reference['n_days']
            or sum(int(p['n_scenes'].sum(dtype=np.uint64)) for p in plans) != reference['n_scenes']):
        raise ValueError('Source dates/scenes differ from the full-period reference')
    return plans


def _new_archive(path, task, reference, plan):
    grid, year = reference['grid'], plan['source']['year']
    feature = task['feature']['properties']
    used = np.asarray([month in core.selected_months(feature['region']) for month in range(1, 13)], dtype=np.uint8)
    attrs = {**task['identity'], 'complete': True, 'version': VERSION, 'statistic': 'year_month',
             'aggregation': AGGREGATION, 'coverage_unit': 'source_acquisition_day', 'method': METHOD,
             'lake_fingerprint': task['lake_fingerprint'], 'lake_id': feature['lake_id'],
             'lake_name': feature['name'], 'region': feature['region'], 'year': year,
             'created_utc': storage.now(), 'crs_wkt': grid['crs_wkt'], 'transform': list(grid['transform']),
             'transform_convention': 'rasterio.Affine a,b,c,d,e,f (NOT GDAL order)',
             'sourceyears': [source['year'] for source in task['sources']],
             'input_fingerprints': [_source_evidence(source) for source in task['sources']],
             'annual_input_fingerprint': _source_evidence(plan['source']),
             'boundary_source': task['boundary'], 'global_n_days': reference['n_days'],
             'global_n_scenes': reference['n_scenes'],
             'global_coverage_reference': {key: reference[key] for key in ('path', 'sha256', 'daily_lake_fingerprint')},
             'month_used_definition': '1 included by regional season; 0 excluded before sums and denominators'}
    store = zarr.storage.ZipStore(str(path), mode='w')
    try:
        group = zarr.group(store=store, zarr_format=2, attributes=attrs)
        storage.write_coordinate(group, 'x', grid['x'], 'x')
        storage.write_coordinate(group, 'y', grid['y'], 'y')
        storage.write_coordinate(group, 'month', np.arange(1, 13, dtype=np.uint8), 'month')
        storage.write_coordinate(group, 'month_used', used, 'month')
        coordinate = storage.create_array(group, 'year', (), 'int16', (), [])
        coordinate[()] = np.int16(year)
        for name in ('n_days', 'n_scenes'):
            storage.write_coordinate(group, name, plan[name], 'month', {'long_name': METHOD[name]})
        shape = (12,) + grid['shape']
        chunks = (1,) + tuple(min(task['spatial_chunk'], size) for size in grid['shape'])
        for tag in TAGS:
            storage.create_array(group, tag, shape, 'float32', chunks, ['month', 'y', 'x'], np.nan,
                {'units': UNITS[tag], 'cell_methods': 'time: mean within days time: mean over days',
                 'aggregation': AGGREGATION, 'ancillary_variables': tag + '_ndays ' + tag + '_nobs'})
            for suffix in ('ndays', 'nobs'):
                storage.create_array(group, tag + '_' + suffix, shape, 'uint32', chunks,
                                     ['month', 'y', 'x'], attrs={'long_name': METHOD[suffix]})
        return store, group
    except BaseException:
        store.close()
        raise


def summarize_year(path, task, lake_mask, year):
    feature = task['feature']['properties']
    years = [source['year'] for source in task['sources']]
    requested = task['identity']['requested_years']
    missing = sorted(set(requested) - set(years))
    rows = []
    with zarr.storage.ZipStore(str(path), mode='r') as store:
        group = zarr.open_group(store=store, mode='r')
        if group.attrs.get('complete') is not True or int(group['year'][()]) != year:
            raise ValueError('Incomplete/wrong-year monthly archive')
        for month in core.selected_months(feature['region']):
            index = month - 1
            row = {'lake_id': feature['lake_id'], 'name_cn': feature['name'], 'region': feature['region'],
                   'year': year, 'month': month, 'ym': year * 100 + month,
                   'yr_start': min(years), 'yr_end': max(years), 'n_lakepix': int(lake_mask.sum()),
                   'nsce': int(group['n_scenes'][index]), 'ndays': int(group['n_days'][index]),
                   'n_years': len(years), 'yrs_used': ','.join(map(str, years)),
                   'missing_yr': ','.join(map(str, missing)), 'yr_full': int(not missing),
                   'req_start': min(requested), 'req_end': max(requested)}
            for tag in TAGS:
                scalar = core.lake_scalar(group[tag][index], lake_mask)
                row[tag] = scalar['mean']
                row[tag + '_npix'] = scalar['n_valid']
                row[tag + '_frac'] = scalar['valid_fraction']
            rows.append(row)
    return rows


def process_lake(task):
    """Return flat SHP rows and year-keyed ZIP outputs; caller owns scheduling."""
    started = time.perf_counter()
    years, snapshots = _validate_task(task)
    reference = read_global_reference(task)
    snapshots[reference['path']] = reference['stamp']
    work = Path(task['work_dir'])
    result_dir = work / 'result'
    completion = result_dir / 'lake_result.json'
    expected_names = {str(year): f'{year}_monthly.zarr.zip' for year in years}
    if completion.is_file():
        result = _read_json(completion)
        if (result.get('complete') is not True or result.get('lake_fingerprint') != task['lake_fingerprint']
                or result.get('aggregation') != AGGREGATION or result.get('statistic') != 'year_month'
                or result.get('sourceyears') != years or set(result.get('outputs', {})) != set(expected_names)
                or result.get('global_archive_sha256') != reference['sha256']
                or result.get('input_fingerprints') != [_source_evidence(s) for s in task['sources']]):
            raise ValueError('Existing monthly result belongs to a different frozen task')
        for year, filename in expected_names.items():
            info = result['outputs'][year]
            if info['filename'] != filename or storage.hash_file(result_dir / filename) != info['sha256']:
                raise ValueError('Existing monthly ZIP checksum differs')
        _check_local_stamps(snapshots)
        return result
    plans = _source_groups(task, reference)
    grid = reference['grid']
    lake_mask = storage.polygon_mask(task['feature'], grid)
    if any(np.any(count[~lake_mask] != 0) for count in reference['ndays'].values()):
        raise ValueError('Full-period valid-day counts fall outside the original lake polygon')
    result_dir.mkdir(parents=True, exist_ok=True)
    outputs, rows = {}, []
    observed_total = {tag: np.zeros(grid['shape'], dtype=np.uint32) for tag in TAGS}
    for plan in plans:
        year = plan['source']['year']
        output = result_dir / expected_names[str(year)]
        temporary = output.with_name(output.name + '.partial-' + uuid.uuid4().hex)
        writer, target = _new_archive(temporary, task, reference, plan)
        try:
            with zarr.storage.ZipStore(plan['source']['read_path'], mode='r') as reader:
                source = zarr.open_group(store=reader, mode='r')
                for tag in TAGS:
                    sums = np.zeros((12,) + grid['shape'], dtype=np.float64)
                    ndays = np.zeros((12,) + grid['shape'], dtype=np.uint32)
                    nobs = np.zeros((12,) + grid['shape'], dtype=np.uint32)
                    for _, month, daily_mean, raw_count in core.iter_daily_means(source[tag], plan['dates'], lake_mask, 16):
                        index = month - 1
                        core.add_day(sums[index], ndays[index], nobs[index], daily_mean, raw_count)
                    total = ndays.sum(axis=0, dtype=np.uint64)
                    if np.any(total > np.iinfo(np.uint32).max - observed_total[tag].astype(np.uint64)):
                        raise OverflowError('Global pixel-day count exceeds uint32')
                    observed_total[tag] += total.astype(np.uint32)
                    for index in range(12):
                        target[tag][index] = core.daily_temporal_map(sums[index], ndays[index], int(plan['n_days'][index]),
                            reference['ndays'][tag], reference['n_days'], lake_mask)
                        target[tag + '_ndays'][index] = ndays[index]
                        target[tag + '_nobs'][index] = nobs[index]
                    del sums, ndays, nobs
        finally:
            writer.close()
        _check_local_stamps(snapshots)
        os.replace(temporary, output)
        rows.extend(summarize_year(output, task, lake_mask, year))
        outputs[str(year)] = {'filename': output.name, 'sha256': storage.hash_file(output), 'size_bytes': output.stat().st_size}
        safe_write_json(work / 'progress.json', {'lake_id': task['feature']['properties']['lake_id'],
            'phase': 'year_month_completed', 'year': year, 'completed_years': len(outputs),
            'total_years': len(years), 'updated_utc': storage.now()})
    for tag in TAGS:
        if not np.array_equal(observed_total[tag], reference['ndays'][tag]):
            raise ValueError('Recomputed monthly valid-day counts differ from the full-period reference: ' + tag)
    _check_local_stamps(snapshots)
    result = {'complete': True, 'version': VERSION, 'statistic': 'year_month', 'aggregation': AGGREGATION,
              'lake_id': task['feature']['properties']['lake_id'], 'lake_fingerprint': task['lake_fingerprint'],
              'daily_lake_fingerprint': task['daily_lake_fingerprint'], 'result_dir': str(result_dir),
              'sourceyears': years, 'outputs': outputs, 'rows': rows, 'n_lakepix': int(lake_mask.sum()),
              'global_n_days': reference['n_days'], 'global_n_scenes': reference['n_scenes'],
              'global_archive_sha256': reference['sha256'], 'global_archive': reference['path'],
              'input_fingerprints': [_source_evidence(source) for source in task['sources']],
              'elapsed_seconds': time.perf_counter() - started, 'completed_utc': storage.now()}
    safe_write_json(completion, result)
    return result
