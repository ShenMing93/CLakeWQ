"""Available-year, observation-pooled statistics for a frozen OLCI collection.
Default is planning only. No inversion and no changes to the scientific core.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import argparse,contextlib,hashlib,json,math,multiprocessing,shutil,time,traceback,uuid
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import fiona,numpy as np,psutil,rasterio,zarr
import pipeline as legacy
from pipeline import (now,encoded,hash_file,write_json,assert_unchanged,grid_from_group,
    require_same_grid,polygon_mask,output_groups,summarize_archive,verify_delivery,publish_delivery)
from statistics_core import TAGS,UNITS,add_observations,selected_months,temporal_map
BASE=Path(__file__).resolve().parent
VERSION='1.0.0-available-year-collection'
OUTPUT_NAMES=legacy.OUTPUT_NAMES
METHOD={**legacy.METHOD,'climatology_annual':'Pool all selected observations across available source years; missing requested years are explicitly listed, never fabricated.',
        'missing_years':'Excluded from denominators; annual coordinate contains only available years; n_years and complete_requested_period describe the lake input period.'}
MEM_CONDITION=None
MEM_USED=None
MEM_BUDGET=0
IO_GATE=None

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def resolved(value,base=BASE):
    p=Path(value);return p.resolve() if p.is_absolute() else (base/p).resolve()
def normalized_path(value,base=BASE):
    # Lexical comparison only: the product root is resolved and verified once.
    p=Path(value);p=p if p.is_absolute() else base/p
    return os.path.normcase(os.path.abspath(os.path.normpath(str(p))))
def pinned(path,expected):
    p=Path(path);actual=hash_file(p)
    if actual!=expected:raise ValueError('Pinned checksum differs: '+str(p))
    s=p.stat();return {'path':str(p),'sha256':actual,'size_bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def check_pins(pins):
    for p in pins:pinned(p['path'],p['sha256'])

def load_config(filename):
    c=read(filename)
    if c.get('format')!='olci-collection-statistics' or c.get('version')!=1:raise ValueError('Unsupported statistics config')
    for k in ('source_root','output_root','work_root','product_config','frozen_manifest','input_audit'):
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
    return c

def boundary_routes(product_config,product_base):
    cache={};pins=[]
    def snap(bp,pp,bsha,psha):
        bp=resolved(bp,product_base);pp=resolved(pp,product_base);key=(str(bp),str(pp))
        if key not in cache:
            b=pinned(bp,bsha);proof=pinned(pp,psha);pins.extend((b,proof))
            fs=read(bp)['features'];lookup={f['properties']['lake_id']:f for f in fs}
            if len(lookup)!=len(fs):raise ValueError('Duplicate boundary lake ID')
            provenance=read(pp);declared=provenance.get('snapshot',{}).get('sha256') or provenance.get('output_geojson_sha256')
            if declared is not None and declared!=bsha:raise ValueError('Boundary provenance does not match snapshot')
            cache[key]=(lookup,b,proof,provenance)
        found=cache[key]
        if found[1]['sha256']!=bsha or found[2]['sha256']!=psha:raise ValueError('Conflicting boundary pins')
        return found
    c=product_config
    scope=snap(c['boundaries'],c['boundary_provenance'],c['boundary_snapshot_sha256'],c['boundary_provenance_sha256'])
    default=snap(c['default_boundaries'],c['default_boundary_provenance'],c['default_boundary_sha256'],c['default_boundary_provenance_sha256'])
    old=snap(c['legacy_boundaries'],c['legacy_boundary_provenance'],c['legacy_boundary_sha256'],c['legacy_boundary_provenance_sha256'])
    for lake,f in old[0].items():
        if default[0].get(lake)!=f:raise ValueError('Legacy feature changed: '+lake)
    routes={}
    for lake in c['lake_ids']:
        if lake in old[0]:route=old
        elif lake in default[0]:route=default
        else:
            o=c['boundary_overrides'][lake]
            route=snap(o['boundaries'],o['boundary_provenance'],o['boundary_sha256'],o['boundary_provenance_sha256'])
        if route[0][lake]!=scope[0].get(lake):raise ValueError('Scope feature differs from routed feature: '+lake)
        f=route[0][lake];selected_months(f['properties']['region'])
        if f['geometry']['type'] not in ('Polygon','MultiPolygon'):raise ValueError('Lake polygon required')
        routes[lake]={'feature':f,'boundary':route[1],'boundary_provenance':route[2],'proof':route[3]}
    return routes,pins

def make_plan(c):
    pins=[pinned(c[k],c[k+'_sha256']) for k in ('product_config','frozen_manifest','input_audit')]
    pc=read(c['product_config']);manifest=read(c['frozen_manifest']);audit=read(c['input_audit'])
    if pc.get('format')!='olci-incremental-collection' or pc.get('version')!=2 or pc.get('years')!=c['years']:raise ValueError('Unexpected frozen product config')
    if audit.get('status') not in ('PASS','pass','complete','validated'):raise ValueError('Input audit is not PASS')
    for k in ('product_config','frozen_manifest'):
        if audit[k]['sha256']!=c[k+'_sha256']:raise ValueError('Audit and config pins differ: '+k)
    if manifest.get('assembly_type')!='incremental_collection' or manifest.get('failed') or not manifest.get('available_scope_finished'):raise ValueError('Available product collection is not complete')
    product_base=Path(c['product_config']).parent.parent
    if resolved(pc['output_root'],product_base)!=Path(c['source_root']):raise ValueError('Product source root differs')
    source_roots={normalized_path(pc['output_root'],product_base),normalized_path(c['source_root'])}
    routes,bpins=boundary_routes(pc,product_base);pins.extend(bpins)
    code={n:hash_file(BASE/n) for n in ('collection_pipeline.py','pipeline.py','statistics_core.py')}
    for n,sha in code.items():pins.append(pinned(BASE/n,sha))
    model=resolved(pc['model_manifest'],product_base);pins.append(pinned(model,pc['model_manifest_sha256']));expected_models=read(model)
    selected=c.get('lake_ids',pc['lake_ids'])
    if set(selected)-set(pc['lake_ids']):raise ValueError('Statistics subset exceeds frozen product scope')
    am={x['lake_id']:x for x in audit['lakes']}
    completed={}
    for row in manifest['completed']:
        key=(row['lake'],int(row['year']))
        if key in completed:raise ValueError('Duplicate annual archive in frozen manifest')
        evidence=row.get('verification',{})
        if evidence.get('all_chunks_read') is not True or evidence.get('outside_shp_finite_pixels')!=0 or evidence.get('same_support_all_products') is not True:raise ValueError('Frozen annual manifest lacks complete input QA evidence')
        completed[key]=row
    tasks=[];mask_cache={}
    for lake in selected:
        route=routes[lake];au=am[lake]
        sources=[]
        for item in sorted(au['annual'],key=lambda x:x['year']):
            year=int(item['year']);row=completed[(lake,year)]
            expected={normalized_path(Path(root)/lake/f'{year}.zarr.zip') for root in source_roots}
            if normalized_path(row['destination']) not in expected or normalized_path(item['path']) not in expected:raise ValueError('Annual source path differs: '+lake)
            if row['sha256']!=item['sha256'] or row['size_bytes']!=item['size_bytes']:raise ValueError('Annual source hash/size disagrees with audit')
            info={k:item[k] for k in ('year','path','size_bytes','mtime_ns','sha256')}
            if item.get('run_fingerprint'):info['run_fingerprint']=item['run_fingerprint']
            if len(info['sha256'])!=64:raise ValueError('Invalid frozen annual SHA')
            sources.append(info)
        years=[s['year'] for s in sources]
        expected_years=sorted(y for l,y in completed if l==lake)
        if not years or years!=expected_years or set(years)-set(c['years']):raise ValueError('Audit is not the full available-year input set: '+lake)
        missing=sorted(set(c['years'])-set(years))
        maskvalue=pc['default_mask_parameters']
        if maskvalue not in mask_cache:
            maskpath=resolved(maskvalue,product_base);maskpin=pinned(maskpath,hash_file(maskpath))
            mask_cache[maskvalue]=read(maskpath)
            if maskpin not in pins:pins.append(maskpin)
        scope={'requested_years':c['years'],'source_years':years,'available_years':years,'missing_years':missing,'n_years':len(years),'complete_requested_period':not missing,'climatology_policy':c['climatology_policy']}
        identity={'version':VERSION,**scope,'feature':route['feature']['properties'],'sources':sources,'boundary_sha256':route['boundary']['sha256'],'boundary_provenance_sha256':route['boundary_provenance']['sha256'],'code_sha256':code,'model_manifest_sha256':pc['model_manifest_sha256'],'coverage_fraction':.05,'method':METHOD,'spatial_chunk':c['spatial_chunk']}
        fp=hashlib.sha256(encoded(identity)).hexdigest()
        h,w=au['shape_hw'];memory=max(int(au['estimated_memory_bytes']),int(h)*int(w)*600+128*1024**2)
        tasks.append({**route,'sources':sources,'identity':identity,'lake_fingerprint':fp,'work_dir':str(Path(c['work_root'])/'lakes'/lake/fp),'cache_root':str(Path(c['work_root'])/'source_cache'),'spatial_chunk':c['spatial_chunk'],'expected_mask':mask_cache[maskvalue],'expected_models':expected_models,'estimated_memory_bytes':memory,'shape_hw':[h,w]})
    run_id=hashlib.sha256(encoded({'version':VERSION,'lake_fingerprints':sorted(t['lake_fingerprint'] for t in tasks),'frozen_manifest_sha256':c['frozen_manifest_sha256']})).hexdigest()
    return tasks,run_id,pins

def init_worker(condition=None,used=None,budget=0,io_gate=None):
    global MEM_CONDITION,MEM_USED,MEM_BUDGET,IO_GATE
    legacy.init_worker();MEM_CONDITION=condition;MEM_USED=used;MEM_BUDGET=budget;IO_GATE=io_gate

@contextlib.contextmanager
def reservation(amount):
    if MEM_CONDITION is None:yield;return
    if amount>MEM_BUDGET:raise MemoryError('One lake exceeds configured memory reservation budget')
    with MEM_CONDITION:
        while MEM_USED.value+amount>MEM_BUDGET:MEM_CONDITION.wait(timeout=5)
        MEM_USED.value+=amount
    try:yield
    finally:
        with MEM_CONDITION:MEM_USED.value-=amount;MEM_CONDITION.notify_all()

@contextlib.contextmanager
def io_slot():
    if IO_GATE is None:yield;return
    IO_GATE.acquire()
    try:yield
    finally:IO_GATE.release()

def cache_source(source,cache_root,lake):
    target=Path(cache_root)/lake/f'{source["year"]}_{source["sha256"]}.zarr.zip'
    assert_unchanged(source)
    if target.is_file() and target.stat().st_size==source['size_bytes'] and hash_file(target)==source['sha256']:
        assert_unchanged(source);return str(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_name(target.name+'.partial-'+uuid.uuid4().hex)
    try:
        with io_slot():
            assert_unchanged(source);h=hashlib.sha256();n=0
            with open(source['path'],'rb') as inp,temp.open('xb') as out:
                while True:
                    block=inp.read(8*1024**2)
                    if not block:break
                    out.write(block);h.update(block);n+=len(block)
            assert_unchanged(source)
        if n!=source['size_bytes'] or h.hexdigest()!=source['sha256']:raise ValueError('Frozen source checksum differs: '+source['path'])
        if hash_file(temp)!=source['sha256']:raise ValueError('Local cache copy verification failed')
        os.replace(temp,target)
        return str(target)
    finally:
        if temp.exists():temp.unlink()

def validate_product(group,source,task):
    a=dict(group.attrs);lake=task['feature']['properties']['lake_id']
    if a.get('boundary_provenance')!=task['proof'] or a.get('exported_mask_parameters')!=task['expected_mask']:raise ValueError('Product mask/boundary provenance differs: '+lake)
    if a.get('models')!=task['expected_models']:raise ValueError('Product RF models differ from frozen model manifest: '+lake)
    original={'lake':lake,'year':source['year'],'source':a['source'],'feature':task['feature'],'exported_parameters':a['exported_mask_parameters'],'boundary_sha256':a['boundary_sha256'],'boundary_provenance':a['boundary_provenance'],'model_provenance':a['models'],'software':a['software'],'code_hashes':a['code_hashes']}
    fp=hashlib.sha256(encoded(original)).hexdigest()
    if a.get('run_fingerprint')!=fp:raise ValueError('Product scientific fingerprint cannot be reconstructed: '+lake)
    if source.get('run_fingerprint') and source['run_fingerprint']!=fp:raise ValueError('Product run fingerprint differs from audited metadata')
    grid,months=grid_from_group(group,source['year'],lake,task['boundary']['sha256'])
    if list(grid['shape'])!=task['shape_hw']:raise ValueError('Product grid size differs from input audit')
    return grid,months,fp

def _verified_grid(group,source,task):
    grid,months,fp=validate_product(group,source,task)
    task.setdefault('source_checks',{})[str(source['year'])]={'run_fingerprint':fp,'sha256':source['sha256'],'local_cache':source['read_path']}
    return grid,months

def _calculate_lake(task):
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
        with zarr.storage.ZipStore(source['read_path'], mode='r') as store:
            group = zarr.open_group(store=store, mode='r')
            current, months = _verified_grid(group, source, task)
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
                with zarr.storage.ZipStore(source['read_path'], mode='r') as store:
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
                del annual_sum, annual_count
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
              'sourceyears': task['identity']['source_years'], 'input_fingerprints': [{k:v for k,v in item.items() if k != 'read_path'} for item in task['sources']],
              'elapsed_seconds': time.perf_counter() - started, 'completed_utc': now()}
    result['source_checks'] = task.get('source_checks', {})
    result['scope'] = {k: task['identity'][k] for k in ('requested_years','source_years','available_years','missing_years','n_years','complete_requested_period','climatology_policy')}
    for records in result['rows'].values():
        for record in records:
            record.update(n_years=result['scope']['n_years'], yrs_used=','.join(map(str,result['sourceyears'])), missing_yr=','.join(map(str,result['scope']['missing_years'])), yr_full=int(result['scope']['complete_requested_period']), req_start=min(result['scope']['requested_years']), req_end=max(result['scope']['requested_years']))
    write_json(completion, result)
    return result


def output_groups(paths,grid,task,month_counts,year_counts):
    clean={**task,'sources':[{k:v for k,v in s.items() if k!='read_path'} for s in task['sources']]}
    return legacy.output_groups(paths,grid,clean,month_counts,year_counts)

def process_lake(task):
    task=dict(task)
    lake=task['feature']['properties']['lake_id']
    with reservation(task['estimated_memory_bytes']):
        write_json(Path(task['work_dir'])/'progress.json',{'lake_id':lake,'phase':'validating_source_cache','updated_utc':now()})
        task['sources']=[{**s,'read_path':cache_source(s,task['cache_root'],lake)} for s in task['sources']]
        result=_calculate_lake(task)
        return result

SHP_FIELDS={**legacy.SHP_FIELDS,'n_years':'int:3','yrs_used':'str:80','missing_yr':'str:80','yr_full':'int:1','req_start':'int:6','req_end':'int:6'}

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
    shutil.copy2(BASE/'README_COLLECTION.txt',staging/'README_COLLECTION.txt')
    write_json(staging/'config_used.json',c)
    scopes={t['feature']['properties']['lake_id']:{**{k:t['identity'][k] for k in ('requested_years','source_years','available_years','missing_years','n_years','complete_requested_period','climatology_policy')},'boundary_source':t['boundary'],'boundary_provenance':t['boundary_provenance']} for t in tasks}
    files={str(p.relative_to(staging)).replace('\\','/'):{'sha256':hash_file(p),'size_bytes':p.stat().st_size} for p in sorted(staging.rglob('*')) if p.is_file()}
    write_json(staging/'run_manifest.json',{'complete':True,'version':VERSION,'created_utc':now(),'run_fingerprint':run_id,'requested_years':c['years'],'climatology_policy':'available_years','method':METHOD,'lake_count':len(tasks),'annual_input_count':sum(len(t['sources']) for t in tasks),'lake_scopes':scopes,'lake_fingerprints':{r['lake_id']:r['lake_fingerprint'] for r in results},'input_fingerprints':{t['feature']['properties']['lake_id']:t['sources'] for t in tasks},'input_pins':pins,'files':files})
    verify_delivery(staging,run_id);check_pins(pins);os.rename(staging,destination)
    return destination

def publish_delivery(local,output,run_id):
    manifest=verify_delivery(local,run_id)
    check_pins(manifest['input_pins'])
    return legacy.publish_delivery(local,output,run_id)

@contextlib.contextmanager
def exclusive_lock(work):
    path=Path(work);path.mkdir(parents=True,exist_ok=True)
    f=(path/'collection.run.lock').open('a+b')
    try:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:f.close()

def plan_document(c,tasks,run_id,pins):
    return {'state':'planned','updated_utc':now(),'run_id':run_id,'tasks':len(tasks),'annual_inputs':sum(len(t['sources']) for t in tasks),'missing_lake_years':sum(len(t['identity']['missing_years']) for t in tasks),'climatology_policy':'available_years','workers':c['workers'],'io_slots':c['io_slots'],'input_pins':pins,'lakes':[{'lake_id':t['feature']['properties']['lake_id'],'lake_fingerprint':t['lake_fingerprint'],'source_years':t['identity']['source_years'],'missing_years':t['identity']['missing_years'],'estimated_memory_bytes':t['estimated_memory_bytes'],'local_result':str(Path(t['work_dir'])/'result/lake_result.json')} for t in tasks]}

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
    parser.add_argument('--config',default=str(BASE/'config/collection_all_2016_2025.json'))
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
