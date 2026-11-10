"""Append-compatible OLCI collection driver v2; scientific pipeline unchanged.
A merged scope view never enters task fingerprints. Each lake uses its immutable
legacy/default/explicitly overridden boundary and provenance snapshots.
Default is --plan. Only --run acquires the active-root lock and produces archives.
Existing products must match exact scientific fingerprints and verified ZIP hashes.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse, collections, contextlib, hashlib, json, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import shutil, sys, time, traceback
import psutil, zarr
import pipeline as science
from runtime import machine_info, read_cpu_usage
BASE=Path(__file__).resolve().parent
MEM_CONDITION=None
MEM_USED=None
MEM_BUDGET=0
MEM_PER_PIXEL=12000
MEM_FIXED=128*1024**2


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def path(value):
    p=Path(value);return p if p.is_absolute() else BASE/p

def load_config(filename):
    cfg=read(filename)
    if cfg.get('format')!='olci-incremental-collection' or cfg.get('version')!=2:raise ValueError('Unsupported collection config')
    cfg=dict(cfg)
    for key in ('source_root','output_root','work_root','shared_model_work_root','model_manifest','default_mask_parameters','boundaries','boundary_provenance','legacy_boundaries','legacy_boundary_provenance','default_boundaries','default_boundary_provenance'):
        cfg[key]=str(path(cfg[key]))
    overrides=cfg.get('boundary_overrides',{})
    if not isinstance(overrides,dict):raise ValueError('boundary_overrides must be an object')
    cfg['boundary_overrides']={}
    for lake,spec in overrides.items():
        if not isinstance(spec,dict):raise ValueError('Invalid boundary override: '+lake)
        cfg['boundary_overrides'][lake]={k:str(path(v)) if k in ('boundaries','boundary_provenance') else v for k,v in spec.items()}
    if cfg['years']!=list(range(2016,2026)):raise ValueError('The authorized collection years are 2016 through 2025')
    ids=cfg['lake_ids']
    if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)):raise ValueError('lake_ids must be a nonempty unique list')
    if science.sha256(cfg['default_mask_parameters'])!=cfg.get('default_mask_sha256','0bf8de8f5031080d81cb18daf9cf41786c6d4c15eb165ce1ad9296af570bc191'):
        raise ValueError('Frozen exported default mask checksum changed')
    if science.sha256(cfg['model_manifest'])!=cfg.get('model_manifest_sha256','895c827dd5acb00e8d002e2ec580e08ef285542170d9c9bcbcad52896f2eb9ae'):
        raise ValueError('Frozen RF manifest checksum changed')
    src=Path(cfg['source_root']).resolve();out=Path(cfg['output_root']).resolve();work=Path(cfg['work_root']).resolve()
    if out==src or src in out.parents or work==src or src in work.parents:
        raise ValueError('Output/scratch must not be inside the read-only raw source tree')
    default=read(cfg['default_mask_parameters'])
    if default['params']['water']['threshold']!=.2 or default['params']['ndwi']['threshold']!=.2:raise ValueError('Frozen default mask changed')
    cfg.setdefault('cpu_target',.6);cfg.setdefault('workers',24);cfg.setdefault('model_threads',2)
    cfg.setdefault('io_slots',3);cfg.setdefault('max_attempts',2);cfg.setdefault('memory_fraction_available',.6)
    cfg.setdefault('blocked_lakes',[]);cfg.setdefault('manifest_interval_seconds',10)
    if not .5<=float(cfg['cpu_target'])<=.7:raise ValueError('CPU target must be between 0.5 and 0.7')
    if not 1<=int(cfg['io_slots'])<=4:raise ValueError('I/O slots must be 1 through 4')
    if not 1<=int(cfg['max_attempts'])<=3:raise ValueError('max_attempts must be 1 through 3')
    if not .1<=float(cfg['memory_fraction_available'])<=.7:raise ValueError('Unsafe memory reservation fraction')
    boundary_routes(cfg)
    return cfg


def task_key(row):return (row['lake'],int(row['year']))
def sidecar(dest):return dest.with_name(dest.name+'.summary.json')

def inspect_existing_readonly(task):
    """Read no raw arrays, write nothing; a missing sidecar requires run-time recovery."""
    dest=Path(task['destination'])
    with zarr.storage.ZipStore(str(dest),mode='r') as store:
        g=zarr.open_group(store=store,mode='r')
        if g.attrs.get('run_fingerprint')!=task['fingerprint']:raise ValueError('Scientific fingerprint differs: '+str(dest))
        meta=json.loads(g['metadata_json'][:].tobytes())
        if meta.get('complete') is not True:raise ValueError('Annual product completion metadata is false: '+str(dest))
    if not sidecar(dest).exists():return None
    row=read(sidecar(dest));actual=science.sha256(dest)
    if row.get('sha256')!=actual or row.get('size_bytes')!=dest.stat().st_size:raise ValueError('Annual product hash/size differs: '+str(dest))
    if task_key(row)!=task_key(task):raise ValueError('Annual summary lake/year differs')
    evidence=row.get('verification',{})
    if not evidence.get('all_chunks_read') or evidence.get('outside_shp_finite_pixels')!=0 or evidence.get('same_support_all_products') is not True:
        raise ValueError('Annual summary lacks prior full-chunk verification evidence')
    row=dict(row);row['destination']=str(dest);row['resumed']=True
    return row



def boundary_routes(cfg):
    """Select immutable scientific snapshots; the merged file is only a scope view.

    Old default-snapshot lakes cannot be overridden. The legacy subset always
    selects its historical snapshot. Every new lake needs an explicit override.
    This function only reads JSON/files and does not inspect annual archives.
    """
    selected=set(cfg['lake_ids'])
    overrides=cfg.get('boundary_overrides',{})
    if not isinstance(overrides,dict):raise ValueError('boundary_overrides must be an object')
    unknown=set(overrides)-selected
    if unknown:raise ValueError('Boundary override lake outside requested scope: '+str(sorted(unknown)))
    cache={}
    def snapshot(boundary,provenance,expected_sha=None,expected_provenance_sha=None):
        boundary=str(path(boundary));provenance=str(path(provenance))
        pair=(boundary,provenance)
        if pair not in cache:
            document=read(boundary);proof=read(provenance)
            features={}
            for f in document.get('features',[]):
                lake=f.get('properties',{}).get('lake_id')
                if not isinstance(lake,str) or not lake:raise ValueError('Boundary feature lacks a lake_id: '+boundary)
                if lake in features:raise ValueError('Duplicate boundary lake ID: '+lake)
                geom=f.get('geometry') or {}
                if geom.get('type') not in ('Polygon','MultiPolygon') or not geom.get('coordinates'):
                    raise ValueError('Missing polygon geometry: '+lake)
                features[lake]=f
            if not features:raise ValueError('Boundary snapshot has no lake features: '+boundary)
            sha=science.sha256(boundary);proof_sha=science.sha256(provenance)
            declared=proof.get('snapshot',{}).get('sha256') or proof.get('output_geojson_sha256')
            if declared is not None and declared!=sha:
                raise ValueError('Boundary provenance snapshot hash differs: '+boundary)
            cache[pair]=(features,sha,proof_sha,declared)
        features,sha,proof_sha,declared=cache[pair]
        if expected_sha is not None and sha!=expected_sha:
            raise ValueError('Frozen boundary snapshot hash differs: '+boundary)
        if expected_provenance_sha is not None and proof_sha!=expected_provenance_sha:
            raise ValueError('Frozen boundary provenance hash differs: '+provenance)
        if expected_sha is None and declared is None:
            raise ValueError('Boundary snapshot requires a frozen SHA256: '+boundary)
        return features,sha
    scope,_=snapshot(cfg['boundaries'],cfg['boundary_provenance'],
                     cfg.get('boundary_snapshot_sha256'),cfg.get('boundary_provenance_sha256'))
    default,default_sha=snapshot(cfg['default_boundaries'],cfg['default_boundary_provenance'],
                     cfg.get('default_boundary_sha256'),cfg.get('default_boundary_provenance_sha256'))
    legacy,legacy_sha=snapshot(cfg['legacy_boundaries'],cfg['legacy_boundary_provenance'],
                     cfg.get('legacy_boundary_sha256'),cfg.get('legacy_boundary_provenance_sha256'))
    missing=selected-set(scope)
    if missing:raise ValueError('Requested lakes missing from scope boundaries: '+str(sorted(missing)))
    if set(legacy)-set(default):raise ValueError('Legacy boundary lakes missing from frozen default snapshot')
    for lake,f in legacy.items():
        if f!=default[lake]:raise ValueError('Legacy geometry/properties differ from frozen default: '+lake)
    if set(overrides)&set(legacy):
        raise ValueError('Legacy boundary lakes cannot be overridden: '+str(sorted(set(overrides)&set(legacy))))
    if set(overrides)&set(default):
        raise ValueError('Frozen default boundary lakes cannot be overridden: '+str(sorted(set(overrides)&set(default))))
    missing_overrides=selected-set(default)-set(overrides)
    if missing_overrides:raise ValueError('New lakes require boundary_overrides: '+str(sorted(missing_overrides)))
    routes={}
    for lake in cfg['lake_ids']:
        if lake in legacy:
            route={'boundaries':str(path(cfg['legacy_boundaries'])),
                   'boundary_provenance':str(path(cfg['legacy_boundary_provenance'])),
                   'boundary_sha256':legacy_sha}
            f=legacy[lake]
        elif lake in default:
            route={'boundaries':str(path(cfg['default_boundaries'])),
                   'boundary_provenance':str(path(cfg['default_boundary_provenance'])),
                   'boundary_sha256':default_sha}
            f=default[lake]
        else:
            spec=overrides[lake]
            required={'boundaries','boundary_provenance','boundary_sha256'}
            allowed=required|{'boundary_provenance_sha256'}
            if not isinstance(spec,dict) or not required<=set(spec) or set(spec)-allowed:
                raise ValueError('Invalid boundary override fields: '+lake)
            if not isinstance(spec['boundary_sha256'],str) or len(spec['boundary_sha256'])!=64:
                raise ValueError('Boundary override needs a 64-character SHA256: '+lake)
            subset,sha=snapshot(spec['boundaries'],spec['boundary_provenance'],
                               spec['boundary_sha256'],spec.get('boundary_provenance_sha256'))
            if lake not in subset:raise ValueError('Override snapshot missing lake: '+lake)
            route={'boundaries':str(path(spec['boundaries'])),
                   'boundary_provenance':str(path(spec['boundary_provenance'])),
                   'boundary_sha256':sha}
            f=subset[lake]
        if scope[lake]!=f:
            raise ValueError('Scope view geometry/properties differ from selected frozen snapshot: '+lake)
        routes[lake]=route
    return routes


def collect_tasks(cfg,models):
    routes=boundary_routes(cfg)
    groups=collections.defaultdict(list);missing=[]
    for lake in cfg['lake_ids']:
        folder=Path(cfg['source_root'])/lake
        available=set()
        if folder.is_dir():
            for f in folder.glob('*.zarr.zip'):
                stem=f.name[:-9]
                if stem.isdigit() and int(stem) in cfg['years'] and f.is_file():available.add(int(stem))
        years=tuple(y for y in cfg['years'] if y in available)
        missing.extend({'lake':lake,'year':y,'path':str(folder/f'{y}.zarr.zip'),'reason':'source_annual_archive_missing'} for y in cfg['years'] if y not in available)
        if not years:continue
        boundary=routes[lake]['boundaries']
        provenance=routes[lake]['boundary_provenance']
        mask=cfg['default_mask_parameters']
        groups[(boundary,provenance,mask,years)].append(lake)
    tasks=[]
    for (boundary,provenance,mask,years),lakes in groups.items():
        producer={k:cfg[k] for k in ('source_root','output_root','work_root','model_manifest')}
        producer.update(boundaries=boundary,boundary_provenance=provenance,mask_parameters=mask,years=list(years))
        tasks.extend(science.build_tasks(producer,models,lakes=lakes,years=list(years)))
    if len({task_key(t) for t in tasks})!=len(tasks):raise ValueError('Duplicate lake/year tasks')
    return sorted(tasks,key=lambda t:t['source']['size_bytes'],reverse=True),missing


def build_plan(cfg,models):
    tasks,missing=collect_tasks(cfg,models)
    output=Path(cfg['output_root']);manifest_path=output/'run_manifest.json'
    old=read(manifest_path) if manifest_path.exists() else {'completed':[],'failed':[]}
    oldrows={task_key(r):r for r in old.get('completed',[])}
    if len(oldrows)!=len(old.get('completed',[])):raise ValueError('Duplicate existing manifest rows')
    completed={};todo=[];conflicts=[]
    for i,task in enumerate(tasks):
        dest=Path(task['destination'])
        if dest.exists():
            try:
                row=inspect_existing_readonly(task)
                if row is None:todo.append(task)
                else:completed[task_key(row)]=row
            except Exception as exc:conflicts.append({'lake':task['lake'],'year':task['year'],'path':str(dest),'error':str(exc)})
        else:todo.append(task)
        if (i+1)%100==0:print('PLAN',i+1,'/',len(tasks),'reuse',len(completed),'new_or_recover',len(todo),'conflicts',len(conflicts),flush=True)
    expected={task_key(t) for t in tasks}
    outside=[r for k,r in oldrows.items() if k not in expected]
    for row in outside:
        p=output/row['lake']/f"{row['year']}.zarr.zip"
        if not p.exists() or science.sha256(p)!=row['sha256']:raise ValueError('Retained out-of-plan archive changed: '+str(p))
        row=dict(row);row['destination']=str(p);completed[task_key(row)]=row
    report={'status':'conflicts' if conflicts else 'planned','planned_utc':science.now(),
        'source_root':cfg['source_root'],'output_root':cfg['output_root'],'years':cfg['years'],
        'requested_lakes':len(cfg['lake_ids'])+len(cfg['blocked_lakes']),'eligible_lakes':len(cfg['lake_ids']),
        'source_annual_archives':len(tasks),'reused_archives':len(completed)-len(outside),'new_or_recovery_tasks':len(todo),
        'retained_out_of_plan_archives':len(outside),'missing_sources':missing,'blocked_lakes':cfg['blocked_lakes'],
        'conflicts':conflicts,'source_bytes':sum(t['source']['size_bytes'] for t in tasks),
        'source_years_complete':not missing,'all_requested_lakes_have_boundaries':not cfg['blocked_lakes'],
        'task_index':[{'lake':t['lake'],'year':t['year'],'fingerprint':t['fingerprint'],'source':t['source'],'destination':t['destination'],'boundary_sha256':t['boundary_sha256']} for t in tasks],
        'previous_manifest_sha256':science.sha256(manifest_path) if manifest_path.exists() else None}
    return report,tasks,todo,completed,old


def init_collection_worker(models,gate,threads,counter,pending,budget,workers,condition,used,memory_budget):
    global MEM_CONDITION,MEM_USED,MEM_BUDGET
    MEM_CONDITION=condition;MEM_USED=used;MEM_BUDGET=memory_budget
    science.init_worker(models,gate,threads,counter,pending,budget,workers)


def run_one(task):
    # Reserve conservatively for block arrays, RF feature copies and output buffers.
    with science.IO_GATE:
        with zarr.storage.ZipStore(task['source']['path'],mode='r') as store:
            g=zarr.open_group(store=store,mode='r');h,w=g['data'].shape[-2:]
    need=int(h)*int(w)*MEM_PER_PIXEL+MEM_FIXED
    if need>MEM_BUDGET:raise MemoryError(f'Conservative task reservation {need} exceeds memory budget {MEM_BUDGET}: {task["lake"]}/{task["year"]}')
    with MEM_CONDITION:
        while MEM_USED.value+need>MEM_BUDGET:MEM_CONDITION.wait(timeout=5)
        MEM_USED.value+=need
    try:
        row=science.process_year(task)
        row['collection_memory_reservation_bytes']=need
        return row
    finally:
        with MEM_CONDITION:
            MEM_USED.value-=need;MEM_CONDITION.notify_all()


@contextlib.contextmanager
def collection_lock(output):
    import msvcrt
    output.mkdir(parents=True,exist_ok=True)
    f=(output/'collection.run.lock').open('a+b');f.seek(0,2)
    if f.tell()==0:f.write(b'0');f.flush()
    f.seek(0)
    try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        f.close();raise RuntimeError('Another collection driver holds the active-root lock')
    try:yield
    finally:
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1);f.close()

def run_collection(cfg,models,report,tasks,todo,completed,old):
    if report['conflicts']:raise RuntimeError('Existing products conflict with exact requested fingerprints; no output was replaced. See collection_plan.json.')
    work=Path(cfg['work_root']);output=Path(cfg['output_root'])
    info=machine_info();logical=int(info['logical_cpus']);budget=max(1,int(logical*float(cfg['cpu_target'])))
    workers=min(int(cfg['workers']),max(1,len(todo)));threads=int(cfg['model_threads'])
    if workers<1 or threads<1 or workers*threads>budget:raise ValueError('Configured worker/model thread count exceeds the machine-derived CPU budget')
    # Keep the full model thread budget available during tail tasks, capped inside scientific predict().
    memory_budget=int(psutil.virtual_memory().available*float(cfg['memory_fraction_available']))
    if memory_budget<MEM_FIXED:raise MemoryError('Insufficient memory to initialize workers')
    if shutil.disk_usage(work).free<2*1024**3:raise OSError('Local scratch has less than 2 GiB free')
    initial_completed=len(completed)
    state={'state':'running','assembly_type':'incremental_collection','collection_driver_version':2,
        'started':science.now(),'updated':science.now(),'pid':os.getpid(),'source_root':cfg['source_root'],
        'years':cfg['years'],'requested_lakes':report['requested_lakes'],'eligible_lakes':report['eligible_lakes'],
        'tasks':len(tasks)+report['retained_out_of_plan_archives'],'source_annual_archives':len(tasks),
        'missing_sources':report['missing_sources'],'blocked_lakes':cfg['blocked_lakes'],
        'source_years_complete':report['source_years_complete'],'source_scope':'all available annual archives for configured lakes; missing source years are explicit, never synthesized',
        'reused_at_start':initial_completed,'machine':info,'workers':workers,'model_threads':threads,
        'cpu_thread_budget':budget,'cpu_utilization_target':cfg['cpu_target'],'io_slots':cfg['io_slots'],
        'memory_reservation_budget_bytes':memory_budget,'memory_reservation_bytes_per_spatial_pixel':MEM_PER_PIXEL,
        'memory_reservation_fixed_bytes':MEM_FIXED,'completed':[],'failed':[],
        'previous_manifest_sha256':report['previous_manifest_sha256'],'statistics_scope_note':'Existing typical14 statistics are preserved; this driver produces annual inversion archives only.'}
    config_copy=dict(cfg);config_copy['driver_sha256']=science.sha256(Path(__file__))
    science.atomic_json(work/'config_used.json',config_copy)
    science.atomic_json(output/'collection_config.json',config_copy)
    if not (work/'previous_run_manifest.json').exists():science.atomic_json(work/'previous_run_manifest.json',old)
    samples=[];failures={};attempts=collections.Counter();queue=collections.deque(todo)
    last_manifest=0.0;last_cpu=0.0
    def checkpoint(force=False):
        nonlocal last_manifest,last_cpu
        current=time.monotonic();state['updated']=science.now()
        if current-last_cpu>=60:
            cpu=read_cpu_usage();sample={'utc':science.now(),'memory_percent':psutil.virtual_memory().percent,
                'system_cpu_percent':cpu['total_percent'] if cpu['available'] and cpu['complete'] else None,
                'native_cpu':cpu};samples.append(sample);state['last_cpu']=sample;last_cpu=current
            science.atomic_json(work/'cpu_samples.json',samples)
        lightweight={k:v for k,v in state.items() if k not in ('completed','failed','missing_sources','machine')}
        lightweight.update(completed_count=len(completed),failed_count=len(failures),
            new_completed_count=len(completed)-initial_completed,remaining_count=state['tasks']-len(completed),
            failed=list(failures.values()),missing_source_count=len(state['missing_sources']))
        science.atomic_json(work/'collection_progress.json',lightweight)
        if force or current-last_manifest>=float(cfg['manifest_interval_seconds']):
            state['completed']=sorted(completed.values(),key=task_key);state['failed']=sorted(failures.values(),key=task_key)
            science.atomic_json(output/'run_manifest.json',state)
            last_manifest=current
    checkpoint(True)
    if todo:
        context=mp.get_context('spawn');gate=context.BoundedSemaphore(int(cfg['io_slots']));counter=context.Value('i',0)
        pending_count=context.Value('i',len(todo));condition=context.Condition();memory_used=context.Value('q',0)
        try:
            with ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=init_collection_worker,
                initargs=(models,gate,threads,counter,pending_count,budget,workers,condition,memory_used,memory_budget)) as pool:
                pending={}
                while queue or pending:
                    while queue and len(pending)<workers*2:
                        task=queue.popleft();attempts[task_key(task)]+=1
                        pending[pool.submit(run_one,task)]=task
                    done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                    for future in done:
                        task=pending.pop(future);key=task_key(task)
                        try:
                            row=future.result();row['destination']=task['destination']
                            completed[key]=row;failures.pop(key,None)
                            # The producer already atomically committed and verified the archive and sidecar.
                            science.atomic_json(work/'completed'/f'{task["lake"]}_{task["year"]}.json',row)
                            print('COMPLETE',task['lake'],task['year'],len(completed),'/',state['tasks'],flush=True)
                        except Exception:
                            error={'lake':task['lake'],'year':task['year'],'attempts':attempts[key],'error':traceback.format_exc()}
                            if attempts[key]<int(cfg['max_attempts']):
                                queue.append(task);print('RETRY',task['lake'],task['year'],attempts[key],flush=True)
                            else:
                                failures[key]=error;print('FAILED',task['lake'],task['year'],error['error'],flush=True)
                    pending_count.value=len(queue)+len(pending)
                    state['reserved_memory_bytes']=memory_used.value
                    checkpoint(bool(done) and not queue and not pending)
        except BaseException:
            state['state']='interrupted';state['finished']=science.now();checkpoint(True);raise
    state['finished']=science.now()
    if failures:state['state']='failed'
    elif cfg['blocked_lakes']:state['state']='incomplete_blocked_lakes'
    elif report['missing_sources']:state['state']='complete_available_sources'
    else:state['state']='complete'
    state['available_scope_finished']=not failures and len(completed)==state['tasks']
    checkpoint(True)
    print(json.dumps({'state':state['state'],'completed':len(completed),'failed':len(failures),
        'missing_sources':len(report['missing_sources']),'blocked_lakes':len(cfg['blocked_lakes'])},indent=2),flush=True)
    return 1 if failures else 2 if cfg['blocked_lakes'] else 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,help='Explicit version-2 append configuration')
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--plan',action='store_true');modes.add_argument('--run',action='store_true')
    args=parser.parse_args(argv);cfg=load_config(path(args.config));work=Path(cfg['work_root']);work.mkdir(parents=True,exist_ok=True)
    zarr.config.set({'async.concurrency':1,'threading.max_workers':1})
    lock=collection_lock(Path(cfg['output_root'])) if args.run else contextlib.nullcontext()
    with lock:
        if args.run:
            cache_cfg={'work_root':cfg['shared_model_work_root'],'model_manifest':cfg['model_manifest']}
            models=science.stage_models(cache_cfg)
        else:models=read(cfg['model_manifest'])
        report,tasks,todo,completed,old=build_plan(cfg,models)
        science.atomic_json(work/'collection_plan.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('task_index','missing_sources','blocked_lakes')},ensure_ascii=True,indent=2),flush=True)
        if not args.run:return 2 if report['conflicts'] else 0
        return run_collection(cfg,models,report,tasks,todo,completed,old)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        traceback.print_exc();raise SystemExit(1)
