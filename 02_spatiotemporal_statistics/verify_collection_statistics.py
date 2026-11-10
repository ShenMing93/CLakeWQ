"""Independent QA of available-year collection statistics.
No production or statistics-core module is imported. All saved maps are checked;
deterministic source pixels are independently recalculated in representative lakes.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import fiona
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from affine import Affine
from shapely.geometry import shape
from shapely import normalize
import zarr
from verify_statistics import Verifier as OriginalVerifier, read_json, sha256, same_float, TAGS, KINDS, BASE

PRODUCT_BASE = BASE.parent / 'OLCI_typical_products'
SAMPLE_LAKES = ['F36A001_Poyang', 'F43A001_Dongting', 'H53A302_Xingyun',
                'A22S304_Shunshanbao', 'NMB01_Huhenuoer', 'R63D205_Xitaijinaier']
UNITS = {'Chla': 'ug L-1', 'SDD': 'm', 'SPM': 'mg L-1'}

def resolve(value, base=BASE):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (base / p).resolve()

def lexical_path(value):
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(value))))

def path_under_root(value, root_aliases, canonical_root):
    key = lexical_path(value)
    for alias in sorted(root_aliases, key=len, reverse=True):
        if key == alias or key.startswith(alias + os.sep):
            return canonical_root + key[len(alias):]
    return key

def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':')).encode('utf-8')

def exact_support(count, n, global_count, global_n, polygon):
    """Independent integer arithmetic for the approved inclusive 1/20 threshold."""
    return (np.asarray(polygon, bool) & (np.asarray(count) > 0) & (n > 0) & (global_n > 0)
            & (np.asarray(count, np.uint64) * 20 >= int(n))
            & (np.asarray(global_count, np.uint64) * 20 >= int(global_n)))

def product_fingerprint(attrs, feature):
    value = {'lake': attrs['lake_id'], 'year': attrs['year'], 'source': attrs['source'],
             'feature': feature, 'exported_parameters': attrs['exported_mask_parameters'],
             'boundary_sha256': attrs['boundary_sha256'], 'boundary_provenance': attrs['boundary_provenance'],
             'model_provenance': attrs['models'], 'software': attrs['software'], 'code_hashes': attrs['code_hashes']}
    return hashlib.sha256(encoded(value)).hexdigest()

def collection_rows(manifest, wanted, requested):
    rows = {}
    for row in manifest['completed']:
        key = (row['lake'], int(row['year']))
        if key in rows:
            raise ValueError('Duplicate completed product lake/year: ' + str(key))
        rows[key] = row
    missing = {(r['lake'], int(r['year'])) for r in manifest.get('missing_sources', [])}
    result = {}
    for lake in wanted:
        available = sorted(y for l, y in rows if l == lake and y in requested)
        absent = sorted(set(requested) - set(available))
        if not available:
            raise ValueError('No usable product years: ' + lake)
        if any((lake, y) not in missing for y in absent):
            raise ValueError('Missing product is not a declared missing source: ' + lake)
        if any((lake, y) in missing for y in available):
            raise ValueError('Completed product also declared missing: ' + lake)
        result[lake] = {'available_years': available, 'missing_years': absent}
    return rows, result

class Verifier(OriginalVerifier):
    def __init__(self, args):
        self.args = args
        self.cfg = read_json(args.config)
        self.output = resolve(args.output_root) if args.output_root else resolve(self.cfg['output_root'])
        self.source = resolve(self.cfg['source_root'])
        self.source_key = lexical_path(self.source)
        self.source_aliases = {self.source_key, lexical_path(self.cfg['source_root'])}
        self.source_stamps = {}
        self.work = resolve(self.cfg['work_root'])
        self.requested_years = self.cfg['years']
        self.years = self.requested_years
        self.coverage = float(self.cfg['coverage_fraction'])
        self.report_path = resolve(args.report)
        self.report = {'status': 'running', 'started_local': time.strftime('%Y-%m-%dT%H:%M:%S'),
                       'config': str(resolve(args.config)), 'output_root': str(self.output),
                       'requested_years': self.requested_years, 'climatology_policy': 'available_years',
                       'coverage_fraction': self.coverage, 'checks_passed': 0, 'geometry': [], 'maps': [],
                       'source_samples': [], 'failures': [], 'source_sampling_enabled': not args.skip_source_samples}
        self.original_stamps = {}
        self.expected_rows = {}
        self.sample_data = {}
        self.source_meta = {}
        self.hashes = {}
        self.route_data = {}
        self.source_read_paths = {}
        self.check(self.cfg.get('climatology_policy') == 'available_years', 'Expected explicitly authorized available_years policy')
        self.check(self.coverage == .05, 'Coverage must be exactly 5%')
        self.check(self.requested_years == list(range(min(self.requested_years), max(self.requested_years) + 1)),
                   'Requested calendar period must be consecutive')
        self.frozen_paths = {}
        for key in ('product_config', 'frozen_manifest', 'input_audit'):
            path = resolve(self.cfg[key])
            expected = self.cfg[key + '_sha256']
            self.check(self.digest(path) == expected, 'Frozen dependency hash differs: ' + key)
            self.original_stamp(path)
            self.frozen_paths[key] = path
        self.product_cfg = read_json(self.frozen_paths['product_config'])
        self.audit = read_json(self.frozen_paths['input_audit'])
        self.check(self.audit.get('status') in ('PASS', 'pass', 'complete', 'validated'), 'Input audit is not passed')
        for key in ('product_config', 'frozen_manifest'):
            self.check(self.audit[key]['sha256'] == self.cfg[key + '_sha256'], 'Audit pins disagree with configuration')
        self.audit_lakes = {r['lake_id']: r for r in self.audit['lakes']}
        self.frozen = read_json(self.frozen_paths['frozen_manifest'])
        self.check(self.frozen.get('available_scope_finished') is True, 'Product available scope is not finished')
        self.check(not self.frozen.get('failed'), 'Frozen product manifest has failures')
        self.check(Path(self.product_cfg['output_root']).resolve() == self.source, 'Statistics source root differs from product configuration')
        self.boundary_path = resolve(self.product_cfg['boundaries'], PRODUCT_BASE)
        self.check(self.digest(self.boundary_path) == self.product_cfg['boundary_snapshot_sha256'], 'Scope boundary snapshot hash differs')
        all_features = read_json(self.boundary_path)['features']
        all_ids = [f['properties']['lake_id'] for f in all_features]
        self.check(len(set(all_ids)) == len(all_ids), 'Duplicate boundary lake ID')
        configured = self.cfg.get('lake_ids', self.product_cfg['lake_ids'])
        self.check(isinstance(configured, list) and configured and len(set(configured)) == len(configured), 'Invalid configured lake IDs')
        self.check(set(configured) <= set(self.product_cfg['lake_ids']) & set(all_ids), 'Configured lakes outside frozen product scope')
        self.configured_lake_ids = set(configured)
        self.features = [f for f in all_features if f['properties']['lake_id'] in self.configured_lake_ids]
        if args.lake:
            self.check(len(set(args.lake)) == len(args.lake) and set(args.lake) <= self.configured_lake_ids,
                       '--lake requires unique canonical IDs within config')
            self.features = [f for f in self.features if f['properties']['lake_id'] in set(args.lake)]
        self.selected_lake_ids = {f['properties']['lake_id'] for f in self.features}
        self.full_config_selected = self.selected_lake_ids == self.configured_lake_ids
        self.rows, self.scopes = collection_rows(self.frozen, self.configured_lake_ids, self.requested_years)
        self.sample_lakes = list(args.sample_lakes or [x for x in SAMPLE_LAKES if x in self.selected_lake_ids])
        self.check(len(set(self.sample_lakes)) == len(self.sample_lakes) and set(self.sample_lakes) <= self.selected_lake_ids,
                   'Sample IDs must be unique selected lake IDs')
        if not args.skip_source_samples:
            self.check(bool(self.sample_lakes), 'At least one source sample lake is required')
            if set(SAMPLE_LAKES) <= self.selected_lake_ids:
                self.check(set(SAMPLE_LAKES) <= set(self.sample_lakes), 'Full collection QA must sample all six representative lakes')
        self.models = read_json(resolve(self.product_cfg['model_manifest'], PRODUCT_BASE))
        self.check(self.digest(resolve(self.product_cfg['model_manifest'], PRODUCT_BASE)) == self.product_cfg['model_manifest_sha256'],
                   'Frozen model manifest differs')
        self.report.update(selected_lakes=sorted(self.selected_lake_ids), configured_lakes=sorted(self.configured_lake_ids),
                           full_config_selected=self.full_config_selected, source_sample_lakes=self.sample_lakes,
                           expected_annual_rows=sum(len(self.scopes[l]['available_years']) for l in self.selected_lake_ids),
                           missing_lake_years=sum(len(self.scopes[l]['missing_years']) for l in self.selected_lake_ids),
                           frozen_dependencies={k: {'path': str(v), 'sha256': self.digest(v)} for k, v in self.frozen_paths.items()})
        self.manifest = read_json(self.output / 'run_manifest.json')
        self.report.update(config_sha256=self.digest(resolve(args.config)), run_id=self.manifest.get('run_fingerprint'), manifest_sha256=self.digest(self.output / 'run_manifest.json'),
                           lake_ids=sorted(self.selected_lake_ids), lake_count=len(self.selected_lake_ids),
                           expected_counts={'lakes':len(self.selected_lake_ids), 'zarr_archives':len(self.selected_lake_ids)*3,
                             'annual':sum(len(self.scopes[l]['available_years']) for l in self.selected_lake_ids),
                             'climatology_annual':len(self.selected_lake_ids),
                             'climatology_month':sum(12 if f['properties']['region'] in ('EPL','YGPL') else 6 for f in self.features)})
        self.original_stamp(resolve(args.config))

    def progress(self, phase, lake=None, year=None):
        payload = {'phase':phase, 'updated_local':time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'lakes_verified':len(self.report['maps']) // 3, 'total_lakes':len(self.features),
                   'current_lake':lake, 'current_source_year':year,
                   'sample_lakes_verified':len(self.report['source_samples']), 'sample_lakes_total':len(self.sample_lakes),
                   'config_sha256':self.report.get('config_sha256'), 'run_id':self.report.get('run_id'),
                   'output_root':str(self.output), 'report':str(self.report_path)}
        self.report['progress'] = payload
        path = self.report_path.parent / 'verification_progress.json'
        # This sidecar is advisory. Final report persistence remains mandatory.
        if time.monotonic() < getattr(self, '_progress_retry_after', 0):
            self._progress_warning['skipped_updates_during_cooldown'] += 1
            return False
        temporary = path.with_suffix('.json.tmp')
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        deadline = time.monotonic() + 5.0
        delay = .05
        while True:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(serialized, encoding='utf-8')
                os.replace(temporary, path)
                if hasattr(self, '_progress_warning'):
                    self._progress_warning['last_recovered_local'] = payload['updated_local']
                return True
            except PermissionError as exc:
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    self.report['progress_sidecar_retries'] = self.report.get('progress_sidecar_retries', 0) + 1
                    time.sleep(min(delay, remaining))
                    delay = min(.5, delay * 2)
                    continue
                self._warn_progress_failure(path, exc, payload)
                return False
            except OSError as exc:
                # Non-permission IO errors are not retried; scientific QA continues.
                self._warn_progress_failure(path, exc, payload)
                return False

    def _warn_progress_failure(self, path, error, payload):
        warning = getattr(self, '_progress_warning', None)
        if warning is None:
            warning = {'type':'progress_sidecar_write_failed', 'path':str(path),
                       'first_seen_local':payload['updated_local'], 'failed_write_cycles':0,
                       'skipped_updates_during_cooldown':0,
                       'impact':'Advisory progress only; all scientific checks and final QA report remain mandatory.'}
            self.report.setdefault('warnings', []).append(warning)
            self._progress_warning = warning
        warning.update(last_seen_local=payload['updated_local'], last_phase=payload['phase'],
                       error_type=type(error).__name__, error=str(error),
                       failed_write_cycles=warning['failed_write_cycles'] + 1)
        # A permanently locked sidecar must not add five seconds to every source.
        self._progress_retry_after = time.monotonic() + 60.0

    def digest(self, path):
        p = Path(path)
        stat = p.stat()
        key = (lexical_path(p), stat.st_size, stat.st_mtime_ns)
        if key not in self.hashes:
            self.hashes[key] = sha256(p)
        return self.hashes[key]

    def source_path_key(self, value):
        return path_under_root(value, self.source_aliases, self.source_key)

    def route(self, lake):
        legacy = resolve(self.product_cfg['legacy_boundaries'], PRODUCT_BASE)
        if not hasattr(self, 'legacy_ids'):
            self.legacy_ids = {f['properties']['lake_id'] for f in read_json(legacy)['features']}
        if lake in self.product_cfg.get('boundary_overrides', {}):
            cfg = self.product_cfg['boundary_overrides'][lake]
            values = cfg['boundaries'], cfg['boundary_provenance'], cfg['boundary_sha256'], cfg['boundary_provenance_sha256']
        elif lake in self.legacy_ids:
            cfg = self.product_cfg
            values = cfg['legacy_boundaries'], cfg['legacy_boundary_provenance'], cfg['legacy_boundary_sha256'], cfg['legacy_boundary_provenance_sha256']
        else:
            cfg = self.product_cfg
            values = cfg['default_boundaries'], cfg['default_boundary_provenance'], cfg['default_boundary_sha256'], cfg['default_boundary_provenance_sha256']
        bp, pp = resolve(values[0], PRODUCT_BASE), resolve(values[1], PRODUCT_BASE)
        self.check(self.digest(bp) == values[2] and self.digest(pp) == values[3], 'Boundary route hash mismatch ' + lake)
        if str(bp) not in self.route_data:
            self.route_data[str(bp)] = ({f['properties']['lake_id']: f for f in read_json(bp)['features']}, read_json(pp))
            self.original_stamp(bp); self.original_stamp(pp)
        features, proof = self.route_data[str(bp)]
        self.check(lake in features, 'Routed snapshot omits lake ' + lake)
        return bp, pp, features[lake], proof

    def verify_boundary_snapshot(self):
        self.original_geometry = {}
        self.boundary_routes = {}
        for feature in self.features:
            lake = feature['properties']['lake_id']
            bp, pp, routed, proof = self.route(lake)
            self.check(encoded(feature) == encoded(routed), 'Merged scope and frozen production feature differ ' + lake)
            geometry = shape(routed['geometry'])
            self.original_geometry[lake] = geometry
            self.boundary_routes[lake] = (bp, pp, routed, proof)
            record = next((r for r in proof.get('lakes', []) if r['lake_id'] == lake), None)
            if record:
                self.check(hashlib.sha256(encoded(routed['geometry'])).hexdigest() == record['geometry_sha256'],
                           'Frozen geometry provenance hash differs ' + lake)
            self.report['geometry'].append({'lake': lake, 'route_snapshot': str(bp), 'snapshot_sha256': self.digest(bp),
                                             'provenance': str(pp), 'provenance_sha256': self.digest(pp),
                                             'geometry_exact_to_frozen_product_boundary': True,
                                             'geometry_type': geometry.geom_type,
                                             'source_association_method': record.get('match_method') if record else 'legacy14_exact_source_shp',
                                             'datum_caveat': record.get('datum_caveat') if record else None})

    def read_source_metadata(self, feature):
        lake = feature['properties']['lake_id']
        used = self.month_selection(feature['properties']['region'])
        metadata, expected = {}, None
        bp, pp, routed, proof = self.boundary_routes[lake]
        mask_path = resolve(self.product_cfg['default_mask_parameters'], PRODUCT_BASE)
        mask = read_json(mask_path)
        for year in self.scopes[lake]['available_years']:
            self.progress('source_metadata_and_hash', lake, year)
            row = self.rows[lake, year]
            audited = next(r for r in self.audit_lakes[lake]['annual'] if r['year'] == year)
            self.check(audited['sha256'] == row['sha256'] and audited['size_bytes'] == row['size_bytes'], 'Audit/source manifest evidence differs')
            expected_origin = self.source / lake / f'{year}.zarr.zip'
            origin = Path(row['destination'])
            self.check(self.source_path_key(origin) == self.source_path_key(expected_origin), 'Frozen source destination differs')
            self.check(self.source_path_key(audited['path']) == self.source_path_key(expected_origin), 'Audited source path differs')
            stamp = origin.stat()
            self.source_stamps[lake, year] = (stamp.st_size, stamp.st_mtime_ns)
            self.original_stamps[str(origin)] = (stamp.st_size, stamp.st_mtime_ns)
            self.check((stamp.st_size, stamp.st_mtime_ns) == (row['size_bytes'], audited['mtime_ns']), 'Original source stamp changed')
            cached = self.work / 'source_cache' / lake / f'{year}_{row["sha256"]}.zarr.zip'
            read_path = cached if cached.is_file() else origin
            self.original_stamp(read_path)
            self.check(read_path.stat().st_size == row['size_bytes'] and self.digest(read_path) == row['sha256'],
                       'Source/cache SHA differs from frozen manifest ' + lake + '/' + str(year))
            self.source_read_paths[lake, year] = str(read_path)
            with zarr.storage.ZipStore(str(read_path), mode='r') as store:
                g = zarr.open_group(store=store, mode='r')
                attrs = dict(g.attrs)
                complete = json.loads(g['metadata_json'][:].tobytes())
                self.check(complete.get('complete') is True, 'Incomplete source product')
                self.check(attrs['lake_id'] == lake and attrs['year'] == year, 'Source lake/year metadata mismatch')
                self.check(attrs['boundary_sha256'] == self.digest(bp), 'Source boundary route hash differs')
                self.check(attrs['boundary_provenance'] == proof, 'Source boundary provenance differs')
                self.check(attrs['exported_mask_parameters'] == mask, 'Source exported threshold parameters differ')
                self.check(attrs['models'] == self.models, 'Source RF models differ')
                self.check(product_fingerprint(attrs, routed) == attrs['run_fingerprint'], 'Source run fingerprint reconstruction differs')
                times = np.asarray(g['acquisition_time'][:], np.int64)
                dates = [datetime.strptime(str(int(v)), '%Y%m%d%H%M%S') for v in times]
                self.check(all(d.year == year for d in dates), 'Source acquisition time outside year')
                months = np.asarray([d.month for d in dates], np.int8)
                grid = (tuple(g['Chla'].shape[-2:]), tuple(attrs['transform'][:6]), rasterio.crs.CRS.from_wkt(attrs['crs_wkt']))
                affine = Affine(*grid[1])
                self.check(affine.b == 0 and affine.d == 0, 'Rotated source grids not supported')
                if expected is None:
                    expected = grid
                self.check(grid == expected, 'Source grids differ across available years ' + lake)
                self.check(tuple(g['Chla'].shape) == tuple(row['shape']), 'Frozen source shape differs')
                self.check(g['Chla'].shape[0] == len(times), 'Source time length differs')
                np.testing.assert_array_equal(g['x'][:], affine.c + (np.arange(grid[0][1]) + .5) * affine.a)
                np.testing.assert_array_equal(g['y'][:], affine.f + (np.arange(grid[0][0]) + .5) * affine.e)
                for tag in TAGS:
                    self.check(g[tag].shape == g['Chla'].shape and g[tag].dtype == np.dtype('float32'), 'Source array schema differs')
                    self.check(g[tag].attrs.get('units') == UNITS[tag], 'Source product units differ')
                metadata[year] = {'path': str(origin), 'read_path': str(read_path), 'times': times, 'months': months,
                                  'eligible': used[months - 1], 'attrs': attrs, 'shape': g['Chla'].shape,
                                  'run_fingerprint': attrs['run_fingerprint'], 'sha256': row['sha256']}
        self.source_meta[lake] = metadata
        return metadata, expected, used

    def verify_scope(self, value, lake, context):
        actual = self.scopes[lake]['available_years']
        missing = self.scopes[lake]['missing_years']
        self.check(value.get('requested_years') == self.requested_years, context + ' requested years differ')
        self.check(value.get('available_years', value.get('source_years')) == actual, context + ' actual years differ')
        self.check(value.get('missing_years') == missing, context + ' missing years differ')
        self.check(value.get('n_years') == len(actual), context + ' number of years differs')
        self.check(value.get('complete_requested_period') is (not missing), context + ' complete-period flag differs')

    def verify_records(self, records, lake):
        years = self.scopes[lake]['available_years']
        self.check(len(records) == len(years) and {int(r['year']) for r in records} == set(years), 'Source fingerprint year set differs')
        for record in records:
            year = int(record['year'])
            meta = self.source_meta[lake][year]
            origin = Path(meta['path'])
            size_bytes, mtime_ns = self.source_stamps[lake, year]
            self.check(self.source_path_key(record['path']) == self.source_path_key(origin), 'Source fingerprint origin path differs')
            self.check(record['sha256'] == self.rows[lake, year]['sha256'], 'Input SHA differs from frozen source')
            self.check(record['size_bytes'] == size_bytes and record['mtime_ns'] == mtime_ns, 'Source fingerprint stamp differs')
            if 'run_fingerprint' in record:
                self.check(record['run_fingerprint'] == meta['run_fingerprint'], 'Recorded source run fingerprint differs')

    def verify_lake_maps(self, feature):
        p = feature['properties']; lake = p['lake_id']
        years = self.scopes[lake]['available_years']
        sources, grid, used = self.read_source_metadata(feature)
        hw, transform, crs = grid
        affine = Affine(*transform)
        geom = feature['geometry'] if crs == rasterio.crs.CRS.from_epsg(4326) else transform_geom('EPSG:4326', crs, feature['geometry'])
        polygon = rasterize([(geom, 1)], out_shape=hw, transform=affine, fill=0, all_touched=False, dtype='uint8').astype(bool)
        lakepix = int(polygon.sum())
        self.check(lakepix > 0, 'Empty polygon rasterization')
        annual_n = [int(sources[y]['eligible'].sum()) for y in years]
        monthly_n = [sum(int(np.count_nonzero(s['eligible'] & (s['months'] == m))) for s in sources.values()) for m in range(1, 13)]
        global_n = sum(annual_n)
        self.check(sum(monthly_n) == global_n, 'Seasonal source denominator sums disagree')
        with zarr.storage.ZipStore(str(self.output / lake / 'climatology_annual.zarr.zip'), mode='r') as store:
            g = zarr.open_group(store=store, mode='r')
            global_counts = {t: g[t + '_nobs'][:] for t in TAGS}
        sample = None
        if lake in self.sample_lakes and not self.args.skip_source_samples:
            coords = self.coordinates(global_counts['Chla'], polygon)
            sample = {'lake': lake, 'region': p['region'], 'coords': coords, 'polygon': polygon[coords], 'periods': {}}
            self.sample_data[lake] = sample
        pooled_counts = {k: {t: np.zeros(hw, np.uint64) for t in TAGS} for k in ('annual', 'climatology_month')}
        bp = self.boundary_routes[lake][0]
        for kind in KINDS:
            path = self.output / lake / (kind + '.zarr.zip')
            self.validate_zip(path)
            with zarr.storage.ZipStore(str(path), mode='r') as store:
                g = zarr.open_group(store=store, mode='r'); a = dict(g.attrs)
                self.check(a.get('complete') is True, 'Statistics archive incomplete')
                self.check(a.get('sourceyears') == years and a.get('source_years') == years, 'Statistics actual source years differ')
                self.verify_scope(a, lake, kind)
                self.check(a.get('climatology_policy') == 'available_years', 'Statistics policy differs')
                self.check(a.get('coverage_fraction') == .05 and a.get('global_n_scenes') == global_n, 'Coverage/denominator differs')
                self.check(a['lake_id'] == lake and a['region'] == p['region'], 'Statistics lake/region differs')
                self.check(a['boundary_sha256'] == self.digest(bp), 'Statistics boundary route differs')
                self.check(tuple(a['transform'][:6]) == transform and rasterio.crs.CRS.from_wkt(a['crs_wkt']) == crs, 'Statistics grid differs')
                self.check(a['lake_fingerprint'] == self.manifest['lake_fingerprints'][lake], 'Lake fingerprint differs between ZIP and manifest')
                identity_keys = ('version','requested_years','source_years','available_years','missing_years','n_years','complete_requested_period','climatology_policy','feature','sources','boundary_sha256','boundary_provenance_sha256','model_manifest_sha256','code_sha256','coverage_fraction','method','spatial_chunk')
                identity = {k:a[k] for k in identity_keys}
                self.check(hashlib.sha256(encoded(identity)).hexdigest() == a['lake_fingerprint'], 'Statistics lake identity hash differs')
                self.check(a['feature'] == p and a['sources'] == a['input_fingerprints'], 'Statistics identity feature/sources differ')
                self.check(a['model_manifest_sha256'] == self.product_cfg['model_manifest_sha256'], 'Statistics model manifest differs')
                for name, digest in a['code_sha256'].items():
                    self.check(name in ('collection_pipeline.py','pipeline.py','statistics_core.py') and self.digest(BASE / name) == digest, 'Statistics scientific code hash differs')
                self.check(a['boundary_provenance_sha256'] == self.digest(self.boundary_routes[lake][1]), 'Statistics provenance SHA differs')
                self.verify_records(a['input_fingerprints'], lake)
                np.testing.assert_array_equal(g['x'][:], affine.c + (np.arange(hw[1]) + .5) * affine.a)
                np.testing.assert_array_equal(g['y'][:], affine.f + (np.arange(hw[0]) + .5) * affine.e)
                self.check(g['x'].dtype == np.dtype('float64') and g['y'].dtype == np.dtype('float64'), 'Statistics coordinate dtype differs')
                np.testing.assert_array_equal(g['month'][:], np.arange(1, 13))
                np.testing.assert_array_equal(g['month_used'][:].astype(bool), used)
                periods = list(range(1, 13)) if kind == 'climatology_month' else years if kind == 'annual' else [0]
                denominators = monthly_n if kind == 'climatology_month' else annual_n if kind == 'annual' else [global_n]
                expected_shape = hw if kind == 'climatology_annual' else (len(periods), *hw)
                if kind == 'annual':
                    np.testing.assert_array_equal(g['year'][:], years)
                self.check(g['n_scenes'].shape == (() if kind == 'climatology_annual' else (len(periods),)), 'n_scenes dimensions differ')
                np.testing.assert_array_equal(np.atleast_1d(g['n_scenes'][...]), denominators)
                self.check(g['n_scenes'].dtype == np.dtype('uint32'), 'n_scenes type differs')
                records = []
                for i, period in enumerate(periods):
                    n = denominators[i]
                    record = {'lake_id': lake, 'name_cn': p['name'], 'region': p['region'],
                              'year': period if kind == 'annual' else None, 'month': period if kind == 'climatology_month' else None,
                              'yr_start': min(years), 'yr_end': max(years), 'nsce': n, 'n_lakepix': lakepix}
                    for tag in TAGS:
                        self.check(g[tag].shape == expected_shape and g[tag + '_nobs'].shape == expected_shape, 'Map/count shape differs')
                        self.check(g[tag].dtype == np.dtype('float32') and g[tag + '_nobs'].dtype == np.dtype('uint32'), 'Map/count type differs')
                        self.check(g[tag].attrs.get('units') == UNITS[tag], 'Statistics product units differ')
                        mean = g[tag][:] if kind == 'climatology_annual' else g[tag][i]
                        count = g[tag + '_nobs'][:] if kind == 'climatology_annual' else g[tag + '_nobs'][i]
                        if kind in pooled_counts:
                            pooled_counts[kind][tag] += count
                        support = exact_support(count, n, global_counts[tag], global_n, polygon)
                        np.testing.assert_array_equal(np.isfinite(mean), support)
                        self.check(not np.isinf(mean).any() and np.all(count <= n), 'Invalid mean/count range')
                        self.check(not count[~polygon].any(), 'Observations outside lake boundary')
                        finite = mean[np.isfinite(mean)]
                        # This RF collection was delivered with positive physical outputs; this is not a generic mean rule.
                        self.check(np.all(finite > 0), 'Unexpected nonpositive statistic')
                        if kind == 'climatology_month' and not used[period - 1]:
                            self.check(n == 0 and not count.any() and not finite.size, 'Excluded seasonal month contains data')
                        record[tag] = float(finite.mean(dtype=np.float64)) if finite.size else None
                        record[tag + '_npix'] = int(finite.size)
                        record[tag + '_frac'] = int(finite.size) / lakepix
                        if sample is not None:
                            entry = sample['periods'].setdefault((kind, period), {'n_scenes': n, 'products': {}})
                            entry['products'][tag] = {'mean': mean[sample['coords']].copy(), 'count': count[sample['coords']].copy()}
                    if kind != 'climatology_month' or used[period - 1]:
                        self.expected_rows[kind, lake, record['year'], record['month']] = record
                    records.append(record)
                self.report['maps'].append({'lake': lake, 'kind': kind, 'shape': list(expected_shape), 'sha256': self.digest(path),
                                            'available_years': years, 'missing_years': self.scopes[lake]['missing_years'],
                                            'global_n_scenes': global_n, 'n_scenes': denominators, 'periods': records})
        for kind in pooled_counts:
            for tag in TAGS:
                np.testing.assert_array_equal(pooled_counts[kind][tag], global_counts[tag])
        self.check(True, 'Full map coverage and count pooling passed')
        self.progress('lake_maps_complete', lake)
        self.save()
        print(json.dumps({'verified_lake_maps': lake, 'actual_years': years, 'missing_years': self.scopes[lake]['missing_years'],
                          'global_n_scenes': global_n, 'sample_pixels': len(sample['coords'][0]) if sample else 0}), flush=True)

    def verify_shapefiles(self):
        super().verify_shapefiles()
        for kind in KINDS:
            with fiona.open(self.output / (kind + '.shp')) as records:
                for feature in records:
                    p = dict(feature['properties']); lake = p['lake_id']
                    if lake not in self.selected_lake_ids:
                        continue
                    scope = self.scopes[lake]; years = scope['available_years']; missing = scope['missing_years']
                    expected = {'n_years': len(years), 'yrs_used': ','.join(map(str, years)),
                                'missing_yr': ','.join(map(str, missing)), 'yr_full': int(not missing),
                                'req_start': min(self.requested_years), 'req_end': max(self.requested_years)}
                    for key, value in expected.items():
                        actual = p.get(key)
                        self.check(actual == value or (value == '' and actual is None), 'SHP available-year field differs ' + lake + ' ' + key)

    def verify_manifest(self):
        m = self.manifest
        self.check(m.get('complete') is True, 'Statistics manifest incomplete')
        self.verify_manifest_lake_scope(m)
        self.check(m.get('requested_years', m.get('sourceyears')) == self.requested_years, 'Manifest requested years differ')
        self.check(set(m['lake_scopes']) == self.configured_lake_ids, 'Manifest lake scopes differ')
        self.check(m['lake_count'] == len(self.configured_lake_ids), 'Manifest lake count differs')
        self.check(m['annual_input_count'] == sum(len(self.scopes[x]['available_years']) for x in self.configured_lake_ids), 'Manifest annual count differs')
        run_identity = {'version':m['version'], 'lake_fingerprints':sorted(m['lake_fingerprints'].values()), 'frozen_manifest_sha256':self.cfg['frozen_manifest_sha256']}
        self.check(hashlib.sha256(encoded(run_identity)).hexdigest() == m['run_fingerprint'], 'Statistics run identity hash differs')
        for pin in m['input_pins']:
            self.check(self.digest(pin['path']) == pin['sha256'], 'Manifest input pin differs')
        expected_outputs = {f'{lake}/{kind}.zarr.zip' for lake in self.configured_lake_ids for kind in KINDS}
        expected_outputs |= {f'{lake}/lake_result.json' for lake in self.configured_lake_ids}
        expected_outputs |= {kind + suffix for kind in KINDS for suffix in ('.shp','.shx','.dbf','.prj','.cpg')}
        self.check(expected_outputs <= set(m['files']), 'Manifest omits required statistics outputs')
        checked = 0
        for name, record in m['files'].items():
            path = self.output / name
            self.check(path.resolve().is_relative_to(self.output.resolve()), 'Unsafe manifest relative path')
            self.check(path.is_file() and path.stat().st_size == record['size_bytes'], 'Manifest output file size differs ' + name)
            self.check(self.digest(path) == record['sha256'], 'Manifest output SHA differs ' + name)
            self.original_stamp(path); checked += 1
        for f in self.features:
            lake = f['properties']['lake_id']
            self.verify_scope(m['lake_scopes'][lake], lake, 'manifest')
            self.verify_records(m['input_fingerprints'][lake], lake)
            result = read_json(self.output / lake / 'lake_result.json')
            self.check(result.get('complete') is True and result['lake_fingerprint'] == m['lake_fingerprints'][lake], 'Lake result fingerprint differs')
            for kind in KINDS:
                record = result['outputs'][kind]
                path = self.output / lake / record['filename']
                self.check(path.stat().st_size == record['size_bytes'] and self.digest(path) == record['sha256'], 'Lake result output SHA differs')
        self.report['manifest'] = {'path': str(self.output / 'run_manifest.json'), 'sha256': self.digest(self.output / 'run_manifest.json'),
                                   'complete': True, 'files_checked': checked, 'run_fingerprint': m.get('run_fingerprint')}
        self.original_stamp(self.output / 'run_manifest.json')

    def verify_source_samples(self):
        for lake, sample in self.sample_data.items():
            self.progress('source_samples', lake)
            ys, xs = sample['coords']; blocks = {t: [] for t in TAGS}; years = []; months = []
            started = time.perf_counter(); excluded = 0
            for year in self.scopes[lake]['available_years']:
                self.progress('source_samples', lake, year)
                item = self.source_meta[lake][year]; keep = item['eligible']; excluded += int((~keep).sum())
                with zarr.storage.ZipStore(item['read_path'], mode='r') as store:
                    group = zarr.open_group(store=store, mode='r')
                    for start in range(0, len(keep), 16):
                        end = min(start + 16, len(keep)); include = keep[start:end]
                        if not include.any():
                            continue
                        for tag in TAGS:
                            block = group[tag][start:end]
                            blocks[tag].append(block[include][:, ys, xs])
                        years.extend([year] * int(include.sum()))
                        months.extend(item['months'][start:end][include].tolist())
                        if self.args.read_delay > 0:
                            time.sleep(self.args.read_delay)
            years = np.asarray(years); months = np.asarray(months); total_n = len(years)
            values = {t: np.concatenate(v, axis=0) if v else np.empty((0, len(ys)), np.float32) for t, v in blocks.items()}
            global_count = {t: np.isfinite(v).sum(axis=0, dtype=np.uint32) for t, v in values.items()}
            comparisons = 0
            for (kind, period), stored in sample['periods'].items():
                include = months == period if kind == 'climatology_month' else years == period if kind == 'annual' else np.ones(total_n, bool)
                n = int(include.sum())
                self.check(n == stored['n_scenes'], 'Independent sampled denominator differs')
                for tag in TAGS:
                    data = values[tag][include]; finite = np.isfinite(data)
                    count = finite.sum(axis=0, dtype=np.uint32)
                    sums = np.where(finite, data, 0).sum(axis=0, dtype=np.float64)
                    allowed = exact_support(count, n, global_count[tag], total_n, sample['polygon'])
                    expected = np.full(len(ys), np.nan, np.float64)
                    np.divide(sums, count, out=expected, where=allowed)
                    expected = expected.astype(np.float32)
                    target = stored['products'][tag]
                    np.testing.assert_array_equal(target['count'], count)
                    np.testing.assert_array_equal(target['mean'], expected)
                    np.testing.assert_array_equal(target['mean'][allowed].view(np.uint32), expected[allowed].view(np.uint32))
                    comparisons += len(ys)
            self.report['source_samples'].append({'lake': lake, 'region': sample['region'],
                'available_years': self.scopes[lake]['available_years'], 'sample_pixel_count': len(ys),
                'rows': ys.tolist(), 'columns': xs.tolist(), 'selected_records_including_empty': total_n,
                'excluded_month_records': excluded, 'periods_checked': len(sample['periods']),
                'product_pixel_comparisons': comparisons, 'all_counts_exact': True, 'all_float32_means_bitwise_equal': True,
                'source_hash_evidence': [{'year': y, 'origin': v['path'], 'read_path': v['read_path'], 'sha256': v['sha256']}
                                         for y, v in self.source_meta[lake].items()],
                'seconds': round(time.perf_counter() - started, 3)})
            self.save()
            print(json.dumps({'source_recalculation_passed': lake, 'pixels': len(ys), 'comparisons': comparisons}), flush=True)

    def run(self):
        self.progress('boundary_provenance')
        self.verify_boundary_snapshot()
        for feature in self.features:
            self.verify_lake_maps(feature)
        self.progress('shapefiles')
        self.verify_shapefiles()
        self.progress('manifest')
        self.verify_manifest()
        if not self.args.skip_source_samples:
            self.verify_source_samples()
        for name, before in self.original_stamps.items():
            s = Path(name).stat()
            self.check((s.st_size, s.st_mtime_ns) == before, 'File changed during verification ' + name)
        self.report.update(status='passed', finished_local=time.strftime('%Y-%m-%dT%H:%M:%S'),
                           lakes_verified=len(self.features), zarr_archives_verified=len(self.report['maps']),
                           source_and_output_size_mtime_unchanged=True,
                           source_hashes_checked=sum(len(self.source_meta[x]) for x in self.source_meta),
                           actual_year_count_distribution=dict(Counter(len(self.scopes[x]['available_years']) for x in self.selected_lake_ids)))
        self.report['limitations'] = [
            'Independent source numerical recalculation is limited to deterministic pixels in the reported sample lakes; every saved map is fully checked for support, coverage, counts and SHP spatial means.',
            'Boundary QA checks exact frozen production snapshots and their pinned provenance; historical source-association and stated datum-accuracy caveats are inherited, not newly resolved.',
            'When a cache is present, its complete SHA256 is matched to the frozen source manifest; original NAS source paths are checked for size and modification-time stability against recorded input fingerprints.']
        if self.args.skip_source_samples:
            self.report['limitations'].insert(0, 'Source numerical recalculation explicitly skipped; structural QA only.')
        self.progress('complete')
        self.save()
        print(json.dumps({'status': 'passed', 'lakes': len(self.features), 'zarr_archives': len(self.report['maps']),
                          'source_sample_lakes': len(self.report['source_samples']), 'report': str(self.report_path)}), flush=True)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-root')
    parser.add_argument('--report', required=True)
    parser.add_argument('--lake', nargs='+', help='Optional canonical IDs within config, for explicitly partial QA')
    parser.add_argument('--sample-lakes', nargs='+')
    parser.add_argument('--skip-source-samples', action='store_true', help='Structural QA only; not final production verification')
    parser.add_argument('--read-delay', type=float, default=0.0)
    args = parser.parse_args(argv)
    qa = None
    try:
        qa = Verifier(args)
        qa.run()
        return 0
    except Exception as exc:
        report = qa.report if qa else {'checks_passed': 0, 'failures': []}
        report.update(status='failed', finished_local=time.strftime('%Y-%m-%dT%H:%M:%S'))
        report['failures'].append({'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        path = resolve(args.report); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
