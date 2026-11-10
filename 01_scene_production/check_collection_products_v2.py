"""Read-only independent verification of the v2 append-compatible OLCI collection.

Default: --sample 3 (recent newly completed archives first). --all explicitly
hashes every completed archive in the manifest snapshot. No producer imports,
raw Rrs reads, RF prediction, archive writes or production-state changes.
Exit 0 means the selected checks passed, NOT that the collection is complete.
Exit 1 means verification failed; --require-complete returns 2 for unfinished scope.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
import zipfile
import numpy as np
import zarr

BASE = Path(__file__).resolve().parent
TAGS = ('Chla', 'SDD', 'SPM')
SUMMARY_FIELDS = ('lake','year','scenes','processed_scenes','empty_scenes',
                  'valid_product_pixels','shp_pixels','shape')
EXPECTED_ARRAYS = set(TAGS) | {'x','y','acquisition_time','valid_pixel_count','scene_status','metadata_json'}
zarr.config.set({'async.concurrency': 1, 'threading.max_workers': 1})

def require(value, message):
    if not bool(value):
        raise ValueError(message)

def now():
    return datetime.now(timezone.utc).isoformat()

def locate(value):
    p = Path(value)
    return p if p.is_absolute() else BASE / p

def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')

def load(path):
    raw = Path(path).read_bytes()
    return json.loads(raw.decode('utf-8-sig')), hashlib.sha256(raw).hexdigest()

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def key(row):
    require(isinstance(row['year'], int) and not isinstance(row['year'], bool), 'Year must be an integer')
    return row.get('lake', row.get('lake_id')), row['year']

def samepath(left, right):
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))

def index(rows, label):
    result = {}
    for row in rows:
        k = key(row)
        require(k not in result, 'Duplicate ' + label + ': ' + str(k))
        result[k] = row
    return result

def proof(value):
    require(value.get('all_chunks_read') is True, 'Missing all-chunk verification evidence')
    require(value.get('zip_crc') == 'passed', 'Missing successful ZIP CRC evidence')
    require(value.get('outside_shp_finite_pixels') == 0, 'Finite pixels outside exact SHP')
    require(value.get('same_support_all_products') is True, 'Product supports were not verified equal')
    require(set(value.get('ranges', {})) == set(TAGS), 'Incomplete product-range evidence')
    for tag in TAGS:
        lo, hi = value['ranges'][tag]
        require((lo is None and hi is None) or
                (lo is not None and hi is not None and np.isfinite([lo, hi]).all() and 0 < lo <= hi),
                'Invalid positive product range: ' + tag)

class CollectionCheck:
    def __init__(self, filename):
        self.config_path = locate(filename)
        self.cfg, self.config_sha = load(self.config_path)
        c = self.cfg
        require(c.get('format') == 'olci-incremental-collection' and c.get('version') == 2,
                'Unsupported collection configuration')
        self.output, self.work = locate(c['output_root']), locate(c['work_root'])
        self.source = locate(c['source_root'])
        self.years = c['years']
        require(self.years == list(range(2016, 2026)), 'Unexpected requested year range')
        self.eligible = set(c['lake_ids'])
        require(len(self.eligible) == len(c['lake_ids']) and self.eligible, 'Duplicate or empty eligible lake IDs')
        for lake in self.eligible:
            require(Path(lake).name == lake and lake not in ('.', '..'), 'Unsafe lake directory ID')
        blocked = c.get('blocked_lakes', [])
        self.blocked = {r['lake_id']: r for r in blocked}
        require(len(self.blocked) == len(blocked) and not self.eligible.intersection(self.blocked),
                'Blocked and eligible lake sets overlap or contain duplicates')
        self.requested = self.eligible | set(self.blocked)
        self.inventory, self.inventory_sha = load(locate(c['source_inventory']))
        require(not self.inventory.get('errors'), 'Source inventory contains errors')
        ilakes = {r['lake_id']: r for r in self.inventory['lakes']}
        require(len(ilakes) == len(self.inventory['lakes']) and set(ilakes) == self.requested,
                'Source inventory and requested lakes differ')
        self.sources = {}
        for lake, row in ilakes.items():
            for annual in row['annual']:
                if annual['year'] not in self.years:
                    continue
                k = (lake, annual['year'])
                require(k not in self.sources, 'Duplicate inventory annual source')
                require(samepath(annual['path'], self.source / lake / (str(annual['year']) + '.zarr.zip')),
                        'Unexpected inventory source path: ' + str(k))
                self.sources[k] = annual
        self.expected = {k for k in self.sources if k[0] in self.eligible}
        requested_grid = {(lake, year) for lake in self.requested for year in self.years}
        self.missing_all = requested_grid - set(self.sources)
        self.missing_eligible = {k for k in self.missing_all if k[0] in self.eligible}
        for lake, row in self.blocked.items():
            require(set(row['years']).intersection(self.years) ==
                    {y for l, y in self.sources if l == lake}, 'Blocked source years differ: ' + lake)

        self.default, default_sha = load(locate(c['default_mask_parameters']))
        require(default_sha == c['default_mask_sha256'], 'Exported default mask hash differs')
        require(self.default['params']['water']['threshold'] ==
                self.default['params']['ndwi']['threshold'] == 0.2, 'Default water thresholds changed')
        self.parameters, self.parameter_hashes = {}, {}
        documents = {}
        for lake in sorted(self.eligible):
            p = locate(c['default_mask_parameters'])
            if str(p) not in documents:
                documents[str(p)] = load(p)
            params, h = documents[str(p)]
            expected = copy.deepcopy(self.default)
            require(params == expected, 'Entire exported mask JSON differs: ' + lake)
            self.parameters[lake], self.parameter_hashes[lake] = params, h


        # Independently select the scientific snapshot for each lake. The merged
        # top-level boundary document is checked as a scope view only.
        self.full, self.full_sha = load(locate(c['boundaries']))
        self.full_prov, self.full_prov_sha = load(locate(c['boundary_provenance']))
        self.default_boundary, self.default_boundary_sha = load(locate(c['default_boundaries']))
        self.default_boundary_prov, self.default_boundary_prov_sha = load(locate(c['default_boundary_provenance']))
        self.legacy, self.legacy_sha = load(locate(c['legacy_boundaries']))
        self.legacy_prov, self.legacy_prov_sha = load(locate(c['legacy_boundary_provenance']))

        def features(document, label):
            rows = document.get('features', [])
            result = {}
            for f in rows:
                name = f.get('properties', {}).get('lake_id')
                require(isinstance(name, str) and name, label + ': feature lacks lake ID')
                require(name not in result, label + ': duplicate lake ID ' + name)
                geom = f.get('geometry') or {}
                require(geom.get('type') in ('Polygon','MultiPolygon') and geom.get('coordinates'),
                        label + ': missing polygon ' + name)
                result[name] = f
            require(result, label + ': empty feature collection')
            return result

        def bound_proof(actual, provenance, expected, proof_sha, expected_proof_sha, label):
            declared = provenance.get('snapshot', {}).get('sha256') or provenance.get('output_geojson_sha256')
            require(expected is not None or declared is not None, label + ': snapshot has no frozen checksum')
            if expected is not None:
                require(actual == expected, label + ': snapshot SHA differs')
            if declared is not None:
                require(actual == declared, label + ': provenance snapshot SHA differs')
            if expected_proof_sha is not None:
                require(proof_sha == expected_proof_sha, label + ': provenance SHA differs')

        bound_proof(self.full_sha, self.full_prov, c.get('boundary_snapshot_sha256'),
                    self.full_prov_sha, c.get('boundary_provenance_sha256'), 'Scope view')
        bound_proof(self.default_boundary_sha, self.default_boundary_prov, c.get('default_boundary_sha256'),
                    self.default_boundary_prov_sha, c.get('default_boundary_provenance_sha256'), 'Frozen default')
        bound_proof(self.legacy_sha, self.legacy_prov, c.get('legacy_boundary_sha256'),
                    self.legacy_prov_sha, c.get('legacy_boundary_provenance_sha256'), 'Legacy')
        scope = features(self.full, 'Scope view')
        original = features(self.default_boundary, 'Frozen default')
        old = features(self.legacy, 'Legacy')
        require(set(old) <= set(original), 'Legacy lakes are absent from the frozen default snapshot')
        for lake in old:
            require(old[lake] == original[lake], 'Legacy original geometry/properties differ: ' + lake)
        require(self.eligible <= set(scope), 'Scope view lacks eligible lakes')
        overrides = c.get('boundary_overrides', {})
        require(isinstance(overrides, dict), 'Boundary overrides must be an object')
        require(set(overrides) <= self.eligible, 'Boundary override outside eligible scope')
        require(not set(overrides).intersection(old), 'Legacy boundary cannot be overridden')
        require(not set(overrides).intersection(original), 'Frozen default boundary cannot be overridden')
        require(self.eligible - set(original) == set(overrides), 'New-lake boundary override coverage differs')
        self.legacy_ids = set(old)
        self.features = {}
        self.selected_boundaries = {}
        self.boundary_assignments = {}
        cache = {}
        for lake in sorted(self.eligible):
            if lake in old:
                feature, h, provenance, ph = old[lake], self.legacy_sha, self.legacy_prov, self.legacy_prov_sha
                bpath, ppath, role = c['legacy_boundaries'], c['legacy_boundary_provenance'], 'legacy'
            elif lake in original:
                feature, h, provenance, ph = (original[lake], self.default_boundary_sha,
                                              self.default_boundary_prov, self.default_boundary_prov_sha)
                bpath, ppath, role = c['default_boundaries'], c['default_boundary_provenance'], 'frozen_default'
            else:
                spec = overrides[lake]
                required = {'boundaries','boundary_provenance','boundary_sha256'}
                require(isinstance(spec, dict) and required <= set(spec) and
                        set(spec) <= required | {'boundary_provenance_sha256'}, 'Invalid boundary override: ' + lake)
                bpath, ppath, role = spec['boundaries'], spec['boundary_provenance'], 'supplement'
                pair = (str(locate(bpath)), str(locate(ppath)))
                if pair not in cache:
                    document, dh = load(pair[0]); pv, pvh = load(pair[1])
                    cache[pair] = features(document, 'Supplement'), dh, pv, pvh
                fs, h, provenance, ph = cache[pair]
                bound_proof(h, provenance, spec['boundary_sha256'], ph,
                            spec.get('boundary_provenance_sha256'), 'Supplement ' + lake)
                require(lake in fs, 'Supplement snapshot missing lake: ' + lake)
                feature = fs[lake]
            require(scope[lake] == feature, 'Scope view differs from scientific snapshot: ' + lake)
            require(feature['properties']['code'] == lake.split('_',1)[0], 'Exact lake code differs: ' + lake)
            # Where the snapshot records per-lake geometry hashes, also bind the
            # exact preserved rings/coordinates to that evidence.
            records = provenance.get('lakes')
            if isinstance(records, list):
                entries = [r for r in records if r.get('lake_id') == lake]
                require(len(entries) == 1, 'Per-lake provenance identity is missing/duplicate: ' + lake)
                if 'geometry_sha256' in entries[0]:
                    require(hashlib.sha256(encoded(feature['geometry'])).hexdigest() ==
                            entries[0]['geometry_sha256'], 'Preserved geometry SHA differs: ' + lake)
            self.features[lake] = feature
            self.selected_boundaries[lake] = (h, provenance)
            self.boundary_assignments[lake] = dict(role=role, boundaries=str(locate(bpath)),
                boundary_provenance=str(locate(ppath)), boundary_sha256=h, provenance_sha256=ph)

        self.models, self.models_sha = load(locate(c['model_manifest']))
        require(self.models_sha == c['model_manifest_sha256'], 'RF manifest hash differs')
        require(set(self.models) == set(TAGS), 'RF targets differ')
        for tag, bundles in self.models.items():
            require(len(bundles) == 5 and len({b['split'] for b in bundles}) == 5,
                    'RF ensemble must contain five distinct random folds: ' + tag)
            for b in bundles:
                require(b['algorithm'] == 'RF' and b['protocol'] == 'random_unrestricted' and
                        b['version'] == 'V1.0' and
                        b['feature_count'] == len(b['feature_names']) == 18 and
                        b['full_sample_refit'] is False and b['output_clipping'] == 'none',
                        'Unexpected RF ensemble method')
        files = [BASE / n for n in ('pipeline.py','science.py','rf_adapter.py')]
        files += sorted((BASE / 'vendor' / 'olci_benchmark').glob('*.py'))
        self.code_hashes = {p.name: digest(p) for p in files}
        self.plan, self.plan_sha = load(self.work / 'collection_plan.json')
        require(not self.plan['conflicts'], 'Plan contains product conflicts')
        self.tasks = index(self.plan['task_index'], 'plan task')
        require(set(self.tasks) == self.expected, 'Plan does not cover exactly eligible available sources')
        require(set(index(self.plan['missing_sources'], 'missing source')) == self.missing_eligible,
                'Plan missing-source accounting differs')
        require({r['lake_id'] for r in self.plan['blocked_lakes']} == set(self.blocked),
                'Plan blocked-lake accounting differs')
        require(self.plan['requested_lakes'] == len(self.requested) and
                self.plan['eligible_lakes'] == len(self.eligible) and
                self.plan['source_annual_archives'] == len(self.expected), 'Plan totals differ')
        for k, task in self.tasks.items():
            source = self.sources[k]
            require(task['source'] == {n: source[n] for n in ('path','size_bytes','mtime_ns')},
                    'Plan source stamp differs from inventory: ' + str(k))
            require(task['boundary_sha256'] == self.boundary(k[0])[0], 'Plan boundary SHA differs')
            require(samepath(task['destination'], self.destination(k)), 'Plan destination differs')

    def boundary(self, lake):
        return self.selected_boundaries[lake]

    def destination(self, k):
        return self.output / k[0] / (str(k[1]) + '.zarr.zip')

    def verify_archive(self, row):
        k = key(row)
        lake, year = k
        path = self.destination(k)
        before = path.stat()
        require(before.st_size == row['size_bytes'], 'Annual ZIP size differs')
        archive_sha = digest(path)
        require(archive_sha == row['sha256'], 'Annual ZIP SHA256 differs')
        sidecar_path = path.with_name(path.name + '.summary.json')
        sidecar, sidecar_sha = load(sidecar_path)
        require(samepath(sidecar['destination'], path), 'Sidecar destination differs')
        for name in SUMMARY_FIELDS + ('sha256','size_bytes','verification'):
            require(sidecar[name] == row[name], 'Sidecar/manifest differs: ' + name)
        proof(sidecar['verification'])
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            require(len(members) == len(set(members)), 'Duplicate ZIP member names')
        with zarr.storage.ZipStore(str(path), mode='r') as store:
            group = zarr.open_group(store=store, mode='r')
            attrs = dict(group.attrs)
            boundary_sha, provenance = self.boundary(lake)
            require(attrs['lake_id'] == lake and attrs['year'] == year, 'Archive lake/year differs')
            require(attrs['exported_mask_parameters'] == self.parameters[lake], 'Entire mask JSON differs')
            require(attrs['boundary_sha256'] == boundary_sha and attrs['boundary_provenance'] == provenance,
                    'Exact SHP snapshot/provenance differs')
            require(attrs['models'] == self.models, 'Entire RF manifest provenance differs')
            require(attrs['code_hashes'] == self.code_hashes, 'Scientific code hashes differ')
            require(attrs['source'] == self.tasks[k]['source'], 'Archive source stamp differs')
            require(attrs['run_fingerprint'] == self.tasks[k]['fingerprint'], 'Planned fingerprint differs')
            fingerprint_input = dict(lake=lake, year=year, source=attrs['source'], feature=self.features[lake],
                exported_parameters=self.parameters[lake], boundary_sha256=boundary_sha,
                boundary_provenance=provenance, model_provenance=self.models,
                software=attrs['software'], code_hashes=self.code_hashes)
            require(hashlib.sha256(encoded(fingerprint_input)).hexdigest() == attrs['run_fingerprint'],
                    'Independent scientific fingerprint reconstruction differs')
            require(attrs['minimum_scene_pixels'] == 10 and attrs['product_clipping'] == 'none' and
                    attrs['source_validity_filter_enabled'] == self.parameters[lake]['params']['source']['enabled'],
                    'Scene/filter/clipping rule differs')
            require(attrs['shp_rasterization'] ==
                    'pixel centers; all_touched=False; complete multipart polygons and holes',
                    'SHP rasterization convention differs')
            require(set(group.array_keys()) == EXPECTED_ARRAYS, 'Unexpected annual arrays')
            meta = json.loads(group['metadata_json'][:].tobytes())
            require(meta.get('complete') is True, 'Embedded completion flag is false')
            for name in SUMMARY_FIELDS:
                require(meta['summary'][name] == row[name], 'Embedded summary differs: ' + name)
            times, counts, status = (group[n][:] for n in ('acquisition_time','valid_pixel_count','scene_status'))
            require(len(times) == row['scenes'] == len(counts) == len(status), 'Time dimensions differ')
            require(all(str(int(t))[:4] == str(year) for t in times), 'Observation outside calendar year')
            np.testing.assert_array_equal(status, np.where(counts >= 10, 0, 1))
            require(int((status == 0).sum()) == row['processed_scenes'] and
                    int((status != 0).sum()) == row['empty_scenes'], 'Scene summary counts differ')
            require(int(counts[status == 0].sum(dtype=np.uint64)) == row['valid_product_pixels'],
                    'Valid product pixel count differs')
            np.testing.assert_array_equal([r['time'] for r in meta['scenes']], times)
            np.testing.assert_array_equal([r['final_mask_pixels'] for r in meta['scenes']], counts)
            for tag in TAGS:
                a = group[tag]
                require(list(a.shape) == row['shape'] and a.dtype == np.dtype('float32') and
                        np.isnan(a.fill_value), 'Product shape/dtype/fill differs: ' + tag)
            require(group['x'].shape[0] == row['shape'][2] and group['y'].shape[0] == row['shape'][1],
                    'Coordinate dimensions differ')
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                'Archive changed during verification')
        return dict(lake=lake, year=year, destination=str(path), sha256=archive_sha,
            size_bytes=after.st_size, sidecar_sha256=sidecar_sha, boundary_sha256=boundary_sha,
            mask_parameters_sha256=self.parameter_hashes[lake],
            water_threshold=self.parameters[lake]['params']['water']['threshold'],
            ndwi_threshold=self.parameters[lake]['params']['ndwi']['threshold'],
            exact_mask_document=True, exact_shp_provenance=True, exact_rf_manifest=True,
            independent_fingerprint=True, inherited_full_chunk_evidence=True)

    def verify(self, args):
        manifest_path = self.output / 'run_manifest.json'
        manifest, manifest_sha = load(manifest_path)
        rows = index(manifest.get('completed', []), 'completed manifest row')
        require(set(rows) <= self.expected, 'Completed manifest contains out-of-scope lake/year')
        errors = []
        for k, row in rows.items():
            try:
                require(samepath(row['destination'], self.destination(k)), 'Manifest destination differs')
                proof(row['verification'])
                require(row['scenes'] == row['processed_scenes'] + row['empty_scenes'],
                        'Manifest scene accounting differs')
            except Exception as exc:
                errors.append(dict(lake=k[0], year=k[1], phase='manifest', error=str(exc)))
        failed = index(manifest.get('failed', []), 'failed manifest row')
        require(set(failed) <= self.expected and not set(failed).intersection(rows),
                'Failed rows overlap completed rows or lie outside scope')
        incremental = manifest.get('assembly_type') == 'incremental_collection'
        if incremental:
            require(manifest['tasks'] == len(self.expected), 'Incremental task total differs')
            require(manifest['requested_lakes'] == len(self.requested) and
                    manifest['eligible_lakes'] == len(self.eligible), 'Manifest lake accounting differs')
            require(set(index(manifest['missing_sources'], 'manifest missing source')) == self.missing_eligible,
                    'Manifest missing-source accounting differs')
            require({r['lake_id'] for r in manifest['blocked_lakes']} == set(self.blocked),
                    'Manifest blocked-lake accounting differs')
        candidates = list(rows)
        if args.lakes:
            require(set(args.lakes) <= self.eligible, 'Selection contains ineligible/unknown lakes')
            candidates = [k for k in candidates if k[0] in args.lakes]
        if args.years:
            require(set(args.years) <= set(self.years), 'Selection year outside requested range')
            candidates = [k for k in candidates if k[1] in args.years]
        # Local completion receipts record completion order without stat-ing NAS files.
        recency = {}
        completed_dir = self.work / 'completed'
        if completed_dir.exists():
            for p in completed_dir.glob('*.json'):
                lake, sep, year = p.stem.rpartition('_')
                if sep and year.isdigit():
                    recency[(lake, int(year))] = p.stat().st_mtime_ns
        candidates.sort(key=lambda k: (recency.get(k, 0), k[1], k[0]), reverse=True)
        selected = candidates if args.all else candidates[:args.sample]
        results = []
        for k in selected:
            try:
                result = self.verify_archive(rows[k])
                results.append(result)
                print('PASS', k[0], k[1], flush=True)
            except Exception as exc:
                errors.append(dict(lake=k[0], year=k[1], phase='archive', error=str(exc)))
                print('FAIL', k[0], k[1], str(exc), flush=True)
        remaining = self.expected - set(rows)
        finished = not remaining and not failed
        if not incremental and remaining:
            status = 'production_plan_pending_collection_manifest'
        elif failed:
            status = 'failed_archives_present'
        elif remaining:
            status = 'in_progress' if manifest.get('state') == 'running' else 'unfinished'
        elif self.blocked:
            status = 'available_eligible_scope_complete_blocked_lakes'
        elif self.missing_all:
            status = 'available_sources_complete_missing_source_years'
        else:
            status = 'all_requested_lake_years_complete'
        progress_path = self.work / 'collection_progress.json'
        progress = load(progress_path)[0] if progress_path.exists() else None
        coverage_all = set(selected) == set(rows)
        report = dict(created_utc=now(), verification_status='FAIL' if errors else 'PASS' if selected else 'NO_COMPLETED_ARCHIVES_SELECTED',
            collection_status=status, manifest_state=manifest.get('state'), configuration=str(self.config_path),
            configuration_sha256=self.config_sha, output_root=str(self.output),
            run_manifest_snapshot_sha256=manifest_sha, run_manifest_snapshot_updated=manifest.get('updated'),
            manifest_changed_during_check=load(manifest_path)[1] != manifest_sha,
            inventory_sha256=self.inventory_sha, plan_sha256=self.plan_sha,
            requested_lakes=len(self.requested), eligible_lakes=len(self.eligible), blocked_lakes=len(self.blocked),
            requested_lake_years=len(self.requested)*len(self.years),
            available_source_archives=len(self.sources), eligible_source_archives=len(self.expected),
            blocked_source_archives=sum(k[0] in self.blocked for k in self.sources),
            missing_source_years_all_lakes=len(self.missing_all),
            missing_source_years_eligible_lakes=len(self.missing_eligible),
            completed_manifest_archives=len(rows), remaining_eligible_archives=len(remaining),
            failed_manifest_archives=len(failed), eligible_scope_finished=finished,
            all_requested_lake_years_complete=finished and not self.blocked and not self.missing_all,
            verification_mode='all_completed' if args.all else 'sample_recent_completed',
            selected_archive_count=len(selected), passed_archive_count=len(results),
            all_completed_archives_hashed=coverage_all and len(results)==len(rows),
            all_requested_products_verified=coverage_all and not errors and finished and not self.blocked and not self.missing_all,
            scope_boundary_sha256=self.full_sha, scope_view_used_only_for_index=True,
            default_boundary_sha256=self.default_boundary_sha, legacy_boundary_sha256=self.legacy_sha,
            boundary_assignments=self.boundary_assignments,
            scope_boundary_provenance_sha256=self.full_prov_sha,
            default_boundary_provenance_sha256=self.default_boundary_prov_sha,
            legacy_boundary_provenance_sha256=self.legacy_prov_sha,
            rf_manifest_sha256=self.models_sha, manifest_rows_unique=True,
            per_lake_full_parameter_documents_checked=len(self.parameters),
            progress_snapshot=None if progress is None else {k:progress.get(k) for k in
                ('state','updated','completed_count','new_completed_count','remaining_count','failed_count')},
            blocked_lake_ids=sorted(self.blocked),
            missing_sources=[dict(lake=l,year=y) for l,y in sorted(self.missing_all)],
            failed_rows=list(failed.values()), errors=errors, rows=results,
            limitations=[
                'Product arrays are not recomputed or fully decompressed by this checker. Matching archive SHA256 binds the prior all-chunk/CRC/support/SHP evidence to the exact bytes checked.',
                'Raw source pixels, RF bundles and external original SHP files are not re-read; frozen manifest, exact geometry snapshot, provenance and per-product fingerprints are checked.',
                'Manifest is a live point-in-time snapshot. Products completed after that snapshot are checked on a subsequent invocation.',
                'Annual inversion products only; the existing typical14 statistics do not certify full-collection statistics.'])
        return report

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='Explicit version-2 collection config')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--sample', type=int, default=3, metavar='N')
    mode.add_argument('--all', action='store_true')
    parser.add_argument('--lakes', nargs='+')
    parser.add_argument('--years', nargs='+', type=int)
    parser.add_argument('--report', default=str(BASE/'work_all_v2/collection_verification_v2.json'))
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args(argv)
    if args.sample < 1:
        parser.error('--sample must be positive')
    try:
        check = CollectionCheck(args.config)
        report = check.verify(args)
    except Exception as exc:
        traceback.print_exc()
        report = dict(created_utc=now(), verification_status='FAIL', configuration=args.config,
                      errors=[dict(phase='configuration_or_manifest', error=str(exc))])
    destination = locate(args.report)
    # Reports are local only. Do not overwrite a production/configuration input.
    require(destination.resolve().is_relative_to(BASE.resolve()), 'Report must stay inside the local producer directory')
    require(destination.suffix == '.json' and
            (destination.name.startswith('collection_verification') or destination.name.startswith('verification_')),
            'Use a dedicated collection_verification*.json or verification_*.json report name')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in
                     ('rows','missing_sources','blocked_lake_ids','failed_rows','boundary_assignments')}, ensure_ascii=True, indent=2), flush=True)
    if report['verification_status'] == 'FAIL':
        return 1
    if args.require_complete and not report.get('all_requested_products_verified'):
        return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
