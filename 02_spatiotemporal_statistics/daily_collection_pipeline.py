"""Daily-first monthly climatology and annual statistics for frozen lake products.

The legacy observation-weighted engine is not modified. Its validated input,
cache, geometry, resource-reservation and delivery primitives are reused.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse, contextlib, hashlib, json, multiprocessing, shutil, time, traceback, uuid
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import fiona, numpy as np, psutil, rasterio, zarr
import collection_pipeline as collection
import pipeline as legacy
from pipeline import now, encoded, hash_file, write_json, assert_unchanged, require_same_grid, polygon_mask, verify_delivery
from daily_statistics_core import TAGS, UNITS, METHOD, AGGREGATION, selected_months, day_groups, iter_daily_means, add_day, daily_temporal_map

BASE = Path(__file__).resolve().parent
VERSION = '1.0.0-daily-first-collection'
OUTPUT_NAMES = legacy.OUTPUT_NAMES
read = collection.read
resolved = collection.resolved
pinned = collection.pinned
check_pins = collection.check_pins
init_worker = collection.init_worker
cache_source = collection.cache_source
validate_product = collection.validate_product
exclusive_lock = collection.exclusive_lock


def make_plan(c):
    tasks, _, pins = collection.make_plan(c)
    code = {name: hash_file(BASE / name) for name in (
        'daily_collection_pipeline.py', 'daily_statistics_core.py',
        'collection_pipeline.py', 'pipeline.py', 'statistics_core.py',
        'daily_publication.py')}
    for name, sha in code.items():
        pin = pinned(BASE / name, sha)
        if pin not in pins:
            pins.append(pin)
    for task in tasks:
        identity = {**task['identity'], 'version': VERSION, 'code_sha256': code,
                    'aggregation': AGGREGATION, 'method': METHOD,
                    'coverage_unit': 'source_acquisition_day'}
        task['identity'] = identity
        task['lake_fingerprint'] = hashlib.sha256(encoded(identity)).hexdigest()
        lake = task['feature']['properties']['lake_id']
        task['work_dir'] = str(Path(c['work_root']) / 'lakes' / lake / task['lake_fingerprint'])
        task['cache_root'] = c['source_cache_root']
        h, w = task['shape_hw']
        task['estimated_memory_bytes'] = max(task['estimated_memory_bytes'], int(h) * int(w) * 700 + 128 * 1024**2)
    run_id = hashlib.sha256(encoded({'version': VERSION,
        'lake_fingerprints': sorted(t['lake_fingerprint'] for t in tasks),
        'frozen_manifest_sha256': c['frozen_manifest_sha256']})).hexdigest()
    return tasks, run_id, pins


def output_groups(paths, grid, task, month_scenes, year_scenes, month_days, year_days):
    """Write distinct source-scene and source-day denominator coordinates."""
    stores, groups = {}, {}
    years = task['identity']['source_years']
    used = np.asarray([m in selected_months(task['feature']['properties']['region']) for m in range(1, 13)], dtype=np.uint8)
    sources = [{k: v for k, v in s.items() if k != 'read_path'} for s in task['sources']]
    common = {**task['identity'], 'lake_fingerprint': task['lake_fingerprint'],
        'lake_id': task['feature']['properties']['lake_id'], 'lake_name': task['feature']['properties']['name'],
        'region': task['feature']['properties']['region'], 'created_utc': now(),
        'crs_wkt': grid['crs_wkt'], 'transform': list(grid['transform']),
        'transform_convention': 'rasterio.Affine a,b,c,d,e,f (NOT GDAL order)',
        'sourceyears': years, 'input_fingerprints': sources, 'boundary_source': task['boundary'],
        'global_n_scenes': int(sum(year_scenes)), 'global_n_days': int(sum(year_days)),
        'month_used_definition': '1 included by regional season; 0 excluded before sums and denominators',
        'aggregation': AGGREGATION, 'coverage_unit': 'source_acquisition_day', 'complete': True}
    try:
        for name in OUTPUT_NAMES:
            stores[name] = zarr.storage.ZipStore(str(paths[name]), mode='w')
            group = groups[name] = zarr.group(store=stores[name], zarr_format=2, attributes={**common, 'statistic': name})
            legacy.write_coordinate(group, 'x', grid['x'], 'x')
            legacy.write_coordinate(group, 'y', grid['y'], 'y')
            legacy.write_coordinate(group, 'month', np.arange(1, 13, dtype=np.uint8), 'month')
            legacy.write_coordinate(group, 'month_used', used, 'month')
            if name == 'climatology_month':
                for key, values in (('n_scenes', month_scenes), ('n_days', month_days)):
                    legacy.write_coordinate(group, key, np.asarray(values, dtype=np.uint32), 'month', {'long_name': METHOD[key]})
                leading, dims = (12,), ['month', 'y', 'x']
            elif name == 'annual':
                legacy.write_coordinate(group, 'year', np.asarray(years, dtype=np.int16), 'year')
                for key, values in (('n_scenes', year_scenes), ('n_days', year_days)):
                    legacy.write_coordinate(group, key, np.asarray(values, dtype=np.uint32), 'year', {'long_name': METHOD[key]})
                leading, dims = (len(years),), ['year', 'y', 'x']
            else:
                for key, values in (('n_scenes', year_scenes), ('n_days', year_days)):
                    array = legacy.create_array(group, key, (), 'uint32', (), [], attrs={'long_name': METHOD[key]})
                    array[()] = np.uint32(sum(values))
                leading, dims = (), ['y', 'x']
            shape = leading + grid['shape']
            spatial = tuple(min(task['spatial_chunk'], size) for size in grid['shape'])
            chunks = ((1,) if leading else ()) + spatial
            for tag in TAGS:
                legacy.create_array(group, tag, shape, 'float32', chunks, dims, np.nan,
                    {'units': UNITS[tag], 'cell_methods': 'time: mean within days time: mean over days',
                     'aggregation': AGGREGATION, 'ancillary_variables': tag + '_ndays ' + tag + '_nobs'})
                legacy.create_array(group, tag + '_ndays', shape, 'uint32', chunks, dims, attrs={'long_name': METHOD['ndays']})
                legacy.create_array(group, tag + '_nobs', shape, 'uint32', chunks, dims, attrs={'long_name': METHOD['nobs']})
        return stores, groups
    except BaseException:
        for store in stores.values():
            store.close()
        raise


def summarize_archive(path, name, feature, mask, years):
    rows = legacy.summarize_archive(path, name, feature, mask, years)
    with zarr.storage.ZipStore(str(path), mode='r') as store:
        group = zarr.open_group(store=store, mode='r')
        for row in rows:
            index = row['month'] - 1 if name == 'climatology_month' else years.index(row['year']) if name == 'annual' else None
            row['ndays'] = int(group['n_days'][index] if index is not None else group['n_days'][()])
    return rows


def _calculate_lake(task):
    started = time.perf_counter()
    lake = task['feature']['properties']['lake_id']
    work = Path(task['work_dir'])
    result_dir, scratch = work / 'result', work / 'scratch'
    completion = result_dir / 'lake_result.json'
    if completion.is_file():
        result = read(completion)
        if result.get('lake_fingerprint') != task['lake_fingerprint'] or result.get('aggregation') != AGGREGATION:
            raise ValueError('Local daily statistics belong to a different scientific fingerprint')
        for name, info in result['outputs'].items():
            if hash_file(result_dir / info['filename']) != info['sha256']:
                raise ValueError(f'Completed daily statistics checksum mismatch: {lake}/{name}')
        for source in task['sources']:
            assert_unchanged(source)
        return result
    result_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    allowed = selected_months(task['feature']['properties']['region'])
    grid, by_year = None, []
    month_scenes, month_days = np.zeros(12, dtype=np.uint64), np.zeros(12, dtype=np.uint64)
    year_scenes, year_days = [], []
    for source in task['sources']:
        assert_unchanged(source)
        with zarr.storage.ZipStore(source['read_path'], mode='r') as store:
            group = zarr.open_group(store=store, mode='r')
            current, months, fp = validate_product(group, source, task)
            dates = day_groups(group['acquisition_time'][:], source['year'], allowed)
        task.setdefault('source_checks', {})[str(source['year'])] = {
            'run_fingerprint': fp, 'sha256': source['sha256'], 'local_cache': source['read_path']}
        if grid is None:
            grid = current
        else:
            require_same_grid(grid, current)
        by_year.append(dates)
        year_days.append(len(dates))
        year_scenes.append(sum(len(indices) for _, _, indices in dates))
        for _, month, indices in dates:
            month_days[month - 1] += 1
            month_scenes[month - 1] += len(indices)
        if year_scenes[-1] != int(np.isin(months, allowed).sum()):
            raise AssertionError('Daily grouping lost or duplicated selected scenes')
    if sum(year_scenes) > np.iinfo(np.uint32).max:
        raise OverflowError('Selected scene count exceeds uint32')
    mask = polygon_mask(task['feature'], grid)
    paths = {name: result_dir / (name + '.zarr.zip') for name in OUTPUT_NAMES}
    temporary = {name: path.with_name(path.name + '.partial-' + uuid.uuid4().hex) for name, path in paths.items()}
    stores, groups = output_groups(temporary, grid, task, month_scenes, year_scenes, month_days, year_days)
    try:
        for tag in TAGS:
            monthly_sum = np.zeros((12,) + grid['shape'], dtype=np.float64)
            monthly_ndays = np.zeros((12,) + grid['shape'], dtype=np.uint32)
            monthly_nobs = np.zeros((12,) + grid['shape'], dtype=np.uint32)
            total_sum = np.zeros(grid['shape'], dtype=np.float64)
            total_ndays = np.zeros(grid['shape'], dtype=np.uint32)
            total_nobs = np.zeros(grid['shape'], dtype=np.uint32)
            for source, dates in zip(task['sources'], by_year):
                annual_sum = np.zeros(grid['shape'], dtype=np.float64)
                annual_ndays = np.zeros(grid['shape'], dtype=np.uint32)
                annual_nobs = np.zeros(grid['shape'], dtype=np.uint32)
                with zarr.storage.ZipStore(source['read_path'], mode='r') as store:
                    group = zarr.open_group(store=store, mode='r')
                    for _, month, daily, raw_count in iter_daily_means(group[tag], dates, mask, 16):
                        add_day(annual_sum, annual_ndays, annual_nobs, daily, raw_count)
                        add_day(monthly_sum[month - 1], monthly_ndays[month - 1], monthly_nobs[month - 1], daily, raw_count)
                        add_day(total_sum, total_ndays, total_nobs, daily, raw_count)
                for suffix, value in (('sum', annual_sum), ('ndays', annual_ndays), ('nobs', annual_nobs)):
                    np.save(scratch / f'{tag}_{source["year"]}_{suffix}.npy', value)
                write_json(work / 'progress.json', {'lake_id': lake, 'tag': tag, 'year': source['year'],
                    'phase': 'daily_accumulated', 'updated_utc': now()})
                assert_unchanged(source)
            for total, monthly in ((total_ndays, monthly_ndays), (total_nobs, monthly_nobs)):
                if not np.array_equal(total, monthly.sum(axis=0, dtype=np.uint64)):
                    raise AssertionError('Monthly and full-period counts differ')
            for mi in range(12):
                groups['climatology_month'][tag][mi] = daily_temporal_map(monthly_sum[mi], monthly_ndays[mi],
                    int(month_days[mi]), total_ndays, sum(year_days), mask)
                groups['climatology_month'][tag + '_ndays'][mi] = monthly_ndays[mi]
                groups['climatology_month'][tag + '_nobs'][mi] = monthly_nobs[mi]
            for yi, source in enumerate(task['sources']):
                annual_sum, annual_ndays, annual_nobs = [np.load(scratch / f'{tag}_{source["year"]}_{suffix}.npy', mmap_mode='r')
                    for suffix in ('sum', 'ndays', 'nobs')]
                groups['annual'][tag][yi] = daily_temporal_map(annual_sum, annual_ndays, year_days[yi], total_ndays, sum(year_days), mask)
                groups['annual'][tag + '_ndays'][yi] = annual_ndays
                groups['annual'][tag + '_nobs'][yi] = annual_nobs
                del annual_sum, annual_ndays, annual_nobs
            groups['climatology_annual'][tag][:] = daily_temporal_map(total_sum, total_ndays, sum(year_days), total_ndays, sum(year_days), mask)
            groups['climatology_annual'][tag + '_ndays'][:] = total_ndays
            groups['climatology_annual'][tag + '_nobs'][:] = total_nobs
    finally:
        for store in stores.values():
            store.close()
    for name in OUTPUT_NAMES:
        os.replace(temporary[name], paths[name])
    rows = {name: summarize_archive(paths[name], name, task['feature'], mask, task['identity']['source_years']) for name in OUTPUT_NAMES}
    for source in task['sources']:
        assert_unchanged(source)
    result = {'complete': True, 'aggregation': AGGREGATION, 'lake_id': lake, 'lake_fingerprint': task['lake_fingerprint'],
        'result_dir': str(result_dir), 'n_lakepix': int(mask.sum()), 'rows': rows,
        'outputs': {name: {'filename': path.name, 'sha256': hash_file(path), 'size_bytes': path.stat().st_size} for name, path in paths.items()},
        'sourceyears': task['identity']['source_years'], 'input_fingerprints': [{k: v for k, v in item.items() if k != 'read_path'} for item in task['sources']],
        'elapsed_seconds': time.perf_counter() - started, 'completed_utc': now(), 'source_checks': task.get('source_checks', {}),
        'global_n_days': sum(year_days), 'global_n_scenes': sum(year_scenes)}
    result['scope'] = {key: task['identity'][key] for key in (
        'requested_years', 'source_years', 'available_years', 'missing_years', 'n_years', 'complete_requested_period', 'climatology_policy')}
    for records in rows.values():
        for record in records:
            record.update(n_years=result['scope']['n_years'], yrs_used=','.join(map(str, result['sourceyears'])),
                missing_yr=','.join(map(str, result['scope']['missing_years'])), yr_full=int(result['scope']['complete_requested_period']),
                req_start=min(result['scope']['requested_years']), req_end=max(result['scope']['requested_years']))
    write_json(completion, result)
    return result


def process_lake(task):
    task = dict(task)
    lake = task['feature']['properties']['lake_id']
    with collection.reservation(task['estimated_memory_bytes']):
        write_json(Path(task['work_dir']) / 'progress.json', {'lake_id': lake, 'phase': 'validating_source_cache', 'updated_utc': now()})
        task['sources'] = [{**source, 'read_path': cache_source(source, task['cache_root'], lake)} for source in task['sources']]
        return _calculate_lake(task)


SHP_FIELDS = {**collection.SHP_FIELDS, 'ndays': 'int:10'}


def load_config(filename):
    c=read(filename)
    if c.get('format')!='olci-daily-collection-statistics' or c.get('version')!=1:raise ValueError('Unsupported statistics config')
    for k in ('source_root','output_root','work_root','product_config','frozen_manifest','input_audit','source_cache_root'):
        c[k]=str(resolved(c[k]))
    if c.get('climatology_policy')!='available_years':raise ValueError('Explicit available_years policy required')
    if c.get('years')!=list(range(2016,2026)):raise ValueError('Requested years must be 2016-2025')
    if c.get('coverage_fraction')!=.05 or c.get('read_scenes')!=16:raise ValueError('Approved coverage/read block is 0.05/16')
    for k in ('workers','io_slots','spatial_chunk','max_attempts'):
        if isinstance(c.get(k),bool) or not isinstance(c.get(k),int) or c[k]<1:raise ValueError('Positive integer required: '+k)
    if c['max_attempts']>2 or c['io_slots']>4:raise ValueError('At most 2 attempts / 4 NAS transfers')
    if not 0<float(c['memory_fraction_available'])<=.6:raise ValueError('Memory fraction must be in (0,.6]')
    if 'lake_ids' in c:legacy.validate_lake_ids(c['lake_ids'])
    src=Path(c['source_root']);out=Path(c['output_root']);work=Path(c['work_root'])
    if out==src or out in src.parents or work==src or src in work.parents:raise ValueError('Unsafe source/output/scratch relationship')
    for k in ('product_config','frozen_manifest','input_audit'):
        if not isinstance(c.get(k+'_sha256'),str) or len(c[k+'_sha256'])!=64:raise ValueError('Missing frozen checksum: '+k)
    if c.get('aggregation') != AGGREGATION or c.get('coverage_unit') != 'source_acquisition_day':raise ValueError('Explicit daily-first aggregation/day coverage required')
    if Path(c['source_cache_root']) == src or src in Path(c['source_cache_root']).parents:raise ValueError('Source cache must be outside immutable product tree')
    return c

def write_shapefiles(destination,results,features):
    geometries={f['properties']['lake_id']:f['geometry'] for f in features}
    for name in OUTPUT_NAMES:
        path=Path(destination)/(name+'.shp')
        with fiona.open(str(path),'w',driver='ESRI Shapefile',schema={'geometry':'Polygon','properties':SHP_FIELDS},crs='EPSG:4326',encoding='UTF-8') as writer:
            for result in results:
                for row in result['rows'][name]:writer.write({'geometry':geometries[result['lake_id']],'properties':row})
        expected=sum(len(r['rows'][name]) for r in results)
        with fiona.open(str(path),encoding='UTF-8') as reader:
            if len(reader)!=expected or rasterio.crs.CRS.from_user_input(reader.crs)!=rasterio.crs.CRS.from_epsg(4326):raise ValueError('SHP scope/CRS verification failed')
            if any(len(k)>10 for k in reader.schema['properties']):raise ValueError('SHP field name too long')

def assemble_delivery(c,tasks,results,run_id,pins):
    destination=Path(c['work_root'])/'deliveries'/run_id
    if destination.exists():verify_delivery(destination,run_id);return destination
    check_pins(pins)
    staging=destination.with_name(run_id+'.partial-'+uuid.uuid4().hex);staging.mkdir(parents=True)
    for result in results:
        target=staging/result['lake_id'];target.mkdir()
        for info in result['outputs'].values():shutil.copy2(Path(result['result_dir'])/info['filename'],target/info['filename'])
        shutil.copy2(Path(result['result_dir'])/'lake_result.json',target/'lake_result.json')
    write_shapefiles(staging,results,[t['feature'] for t in tasks])
    write_json(staging/'config_used.json',c)
    scopes={t['feature']['properties']['lake_id']:{**{k:t['identity'][k] for k in ('requested_years','source_years','available_years','missing_years','n_years','complete_requested_period','climatology_policy')},'boundary_source':t['boundary'],'boundary_provenance':t['boundary_provenance']} for t in tasks}
    files={str(p.relative_to(staging)).replace('\\','/'):{'sha256':hash_file(p),'size_bytes':p.stat().st_size} for p in sorted(staging.rglob('*')) if p.is_file()}
    write_json(staging/'run_manifest.json',{'complete':True,'version':VERSION,'created_utc':now(),'run_fingerprint':run_id,'requested_years':c['years'],'climatology_policy':'available_years','aggregation':AGGREGATION,'coverage_unit':'source_acquisition_day','method':METHOD,'lake_count':len(tasks),'annual_input_count':sum(len(t['sources']) for t in tasks),'lake_scopes':scopes,'lake_fingerprints':{r['lake_id']:r['lake_fingerprint'] for r in results},'input_fingerprints':{t['feature']['properties']['lake_id']:t['sources'] for t in tasks},'input_pins':pins,'files':files})
    verify_delivery(staging,run_id);check_pins(pins);os.rename(staging,destination)
    return destination

def publish_delivery(local, output, run_id):
    from daily_publication import publish_delivery as publish_daily
    manifest = verify_delivery(local, run_id)
    check_pins(manifest['input_pins'])
    return publish_daily(local, output, run_id, workers=4)


def plan_document(c,tasks,run_id,pins):
    return {'state':'planned','updated_utc':now(),'run_id':run_id,'tasks':len(tasks),'annual_inputs':sum(len(t['sources']) for t in tasks),'missing_lake_years':sum(len(t['identity']['missing_years']) for t in tasks),'climatology_policy':'available_years','aggregation':AGGREGATION,'coverage_unit':'source_acquisition_day','workers':c['workers'],'io_slots':c['io_slots'],'input_pins':pins,'lakes':[{'lake_id':t['feature']['properties']['lake_id'],'lake_fingerprint':t['lake_fingerprint'],'source_years':t['identity']['source_years'],'missing_years':t['identity']['missing_years'],'estimated_memory_bytes':t['estimated_memory_bytes'],'local_result':str(Path(t['work_dir'])/'result/lake_result.json')} for t in tasks]}

def run_collection(c,tasks,run_id,pins,local_only=True):
    work=Path(c['work_root']);progress=work/'collection_progress.json'
    budget=int(psutil.virtual_memory().available*float(c['memory_fraction_available']))
    workers=min(c['workers'],len(tasks),psutil.cpu_count(logical=True) or c['workers'])
    if max(t['estimated_memory_bytes'] for t in tasks)>budget:raise MemoryError('Largest lake exceeds memory budget; no tasks started')
    state={'state':'running','started_utc':now(),'updated_utc':now(),'pid':os.getpid(),'run_id':run_id,'tasks':len(tasks),'workers':workers,'io_slots':c['io_slots'],'memory_budget_bytes':budget,'completed':[],'failed':[],'attempts':{},'local_delivery':None}
    write_json(progress,state);results=[];attempts={};pending={};queue=sorted(tasks,key=lambda t:t['estimated_memory_bytes'],reverse=True)
    ctx=multiprocessing.get_context('spawn')
    with ctx.Manager() as manager:
        condition=manager.Condition();used=manager.Value('q',0);gate=manager.BoundedSemaphore(c['io_slots'])
        try:
            with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_worker,initargs=(condition,used,budget,gate)) as pool:
                while queue or pending:
                    while queue and len(pending)<workers*2:
                        task=queue.pop(0);lake=task['feature']['properties']['lake_id'];attempts[lake]=attempts.get(lake,0)+1
                        pending[pool.submit(process_lake,task)]=task
                    done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                    for future in done:
                        task=pending.pop(future);lake=task['feature']['properties']['lake_id']
                        try:
                            result=future.result();results.append(result)
                            state['completed'].append({'lake_id':lake,'lake_fingerprint':result['lake_fingerprint'],'result_dir':result['result_dir'],'elapsed_seconds':result['elapsed_seconds']})
                        except Exception:
                            error=traceback.format_exc()
                            if attempts[lake]<c['max_attempts']:queue.append(task)
                            else:state['failed'].append({'lake_id':lake,'attempts':attempts[lake],'error':error})
                    state.update(updated_utc=now(),completed_count=len(results),failed_count=len(state['failed']),attempts=attempts,reserved_memory_bytes=used.value,cpu_percent=psutil.cpu_percent(),available_memory_bytes=psutil.virtual_memory().available)
                    write_json(progress,state)
            check_pins(pins)
            if state['failed']:
                state.update(state='failed',updated_utc=now());write_json(progress,state);return 1
            order={t['feature']['properties']['lake_id']:i for i,t in enumerate(tasks)}
            results.sort(key=lambda r:order[r['lake_id']])
            state.update(state='assembling',updated_utc=now());write_json(progress,state)
            delivery=assemble_delivery(c,tasks,results,run_id,pins)
            state.update(state='local_complete',local_delivery=str(delivery),updated_utc=now())
            if not local_only:
                state['publication']=publish_delivery(delivery,c['output_root'],run_id);state['state']='complete'
            state['output_root']=c['output_root'];write_json(progress,state)
            print(json.dumps({k:state[k] for k in ('state','run_id','tasks','local_delivery','output_root')},ensure_ascii=True),flush=True)
            return 0
        except BaseException:
            state.update(state='interrupted_or_failed',error=traceback.format_exc(),updated_utc=now());write_json(progress,state);raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(BASE/'config/daily_all_2016_2025.json'))
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--plan',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--local-only',action='store_true')
    args=parser.parse_args(argv);c=load_config(args.config)
    lock=exclusive_lock(c['work_root']) if args.run else contextlib.nullcontext()
    with lock:
        tasks,run_id,pins=make_plan(c);doc=plan_document(c,tasks,run_id,pins)
        write_json(Path(c['work_root'])/'collection_plan.json',doc)
        print(json.dumps({k:doc[k] for k in ('state','run_id','tasks','annual_inputs','missing_lake_years')},ensure_ascii=True),flush=True)
        if not args.run:return 0
        return run_collection(c,tasks,run_id,pins,local_only=args.local_only)

if __name__=='__main__':raise SystemExit(main())
