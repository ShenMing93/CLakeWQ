"""Full-year OLCI RF products with exported spectral mask and exact lake boundaries.
Read-only Windows source paths; one independently committed ZIP per lake/year.
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid
import zipfile
import joblib
import numpy as np
import psutil
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from affine import Affine
from numcodecs import Blosc, blosc
import sklearn
import zarr
from science import compute_mask
from rf_adapter import FrozenRFCollection

BASE = Path(__file__).resolve().parent
TAGS = ('Chla', 'SDD', 'SPM')
UNITS = {'Chla': 'ug L-1', 'SDD': 'm', 'SPM': 'mg L-1'}
BANDS = list(range(2, 12)) + [23, 26, 27, 28, 32, 37, 41]
VERSION = '2.0.0-rfv43'
MODELS = None
IO_GATE = None
MODEL_THREADS = 1
WORKER_RUNTIME = {}
PENDING_TASKS = None
CPU_BUDGET = 1
WORKER_LIMIT = 1


def now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encoded(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':')).encode('utf-8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_bytes(encoded(obj))
    for attempt in range(20):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


def stamp(path):
    s = Path(path).stat()
    return {'path': str(path), 'size_bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def config_path(cfg, key):
    p = Path(cfg[key])
    return p if p.is_absolute() else BASE / p


def lake_mask(feature, attrs, shape):
    geom = feature['geometry']
    crs = rasterio.crs.CRS.from_wkt(attrs['crs_wkt'])
    if crs != rasterio.crs.CRS.from_epsg(4326):
        geom = transform_geom('EPSG:4326', crs, geom)
    # The source stores rasterio/Affine coefficients, NOT a GDAL geotransform.
    affine = Affine(*attrs['transform'][:6])
    result = rasterize([(geom, 1)], out_shape=shape, transform=affine,
                       fill=0, all_touched=False, dtype='uint8').astype(bool)
    if not result.any():
        raise ValueError('Lake SHP has no pixel-center intersection with input grid')
    return result


def stage_models(cfg):
    if sklearn.__version__ != '1.4.2':
        raise RuntimeError('These supplied model bundles require scikit-learn 1.4.2')
    manifest = load_json(config_path(cfg, 'model_manifest'))
    if set(manifest) != set(TAGS):
        raise ValueError('Model manifest must explicitly list Chla, SDD, SPM')
    cache = config_path(cfg, 'work_root') / 'new_rf_models'
    result = {}
    for tag in TAGS:
        if not manifest[tag]:
            raise ValueError('Empty model collection: ' + tag)
        result[tag] = []
        for entry in manifest[tag]:
            p = Path(entry['source'])
            digest = sha256(p)
            if digest != entry['sha256']:
                raise ValueError('Model changed since manifest was frozen: ' + str(p))
            dest = cache / tag / digest / 'bundle.joblib'
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or sha256(dest) != digest:
                shutil.copy2(p, dest)
            if sha256(dest) != digest:
                raise IOError('Local model copy differs from source: ' + str(p))
            result[tag].append({**entry, 'local': str(dest)})
    return result


def init_worker(model_manifest, io_gate, model_threads, counter, pending_tasks, cpu_budget, worker_limit):
    global MODELS, IO_GATE, MODEL_THREADS, WORKER_RUNTIME, PENDING_TASKS, CPU_BUDGET, WORKER_LIMIT
    IO_GATE = io_gate
    MODEL_THREADS = model_threads
    PENDING_TASKS, CPU_BUDGET, WORKER_LIMIT = pending_tasks, cpu_budget, worker_limit
    with counter.get_lock():
        ordinal = counter.value
        counter.value += 1
    try:
        from runtime import init_affinity
        WORKER_RUNTIME = init_affinity(ordinal)
    except (ImportError, AttributeError):
        WORKER_RUNTIME = {'affinity': 'OS-managed', 'worker_ordinal': ordinal}
    blosc.set_nthreads(1)
    zarr.config.set({'async.concurrency': 1, 'threading.max_workers': 1})
    MODELS = FrozenRFCollection(model_manifest, runtime_path=BASE/'vendor', model_threads=model_threads)


def predict(rrs10):
    # Each frozen bundle owns its 18-feature contract, imputation, scaling,
    # target standardization and expm1 inverse. Aggregate physical predictions.
    global MODEL_THREADS
    # Keep the total inference thread budget as annual tasks finish; never
    # exceed the original machine-derived CPU budget or 16 threads/model.
    live = max(1, min(WORKER_LIMIT, PENDING_TASKS.value))
    MODEL_THREADS = max(1, min(16, CPU_BUDGET // live))
    for bundles in MODELS.bundles.values():
        for bundle in bundles:
            bundle.estimator.set_params(n_jobs=MODEL_THREADS)
    result = MODELS.predict(rrs10)
    for tag, v in result.items():
        if not np.all(np.isfinite(v) & (v > 0)):
            raise ValueError(f'{tag}: ensemble produced a non-finite/non-positive value')
    return result


def create_array(root, name, shape, dtype, chunks, dims, fill=0, extra=None):
    return root.create_array(name, shape=shape, dtype=dtype, chunks=chunks,
        compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE),
        fill_value=fill, attributes={'_ARRAY_DIMENSIONS': dims, **(extra or {})})


def write_json_array(root, name, obj):
    raw = np.frombuffer(encoded(obj), dtype='uint8')
    arr = create_array(root, name, raw.shape, 'uint8', (max(1, min(len(raw), 1024**2)),), ['json_byte'])
    arr[:] = raw


def inspect_completed(path, task):
    with zarr.storage.ZipStore(str(path), mode='r') as store:
        g = zarr.open_group(store=store, mode='r')
        if g.attrs.get('run_fingerprint') != task['fingerprint']:
            raise ValueError('Existing product has different inputs/config; use a new output directory: ' + str(path))
        if 'metadata_json' not in g:
            raise ValueError('Existing product lacks completion metadata: ' + str(path))
        meta = json.loads(g['metadata_json'][:].tobytes())
        if not meta.get('complete'):
            raise ValueError('Existing product not complete')
        summary = meta['summary']
        for tag in TAGS:
            if tag not in g or list(g[tag].shape) != summary['shape']:
                raise ValueError('Existing product array missing or shape differs: ' + tag)
    sidecar = Path(path).with_name(Path(path).name + '.summary.json')
    digest = sha256(path)
    if sidecar.exists():
        recorded = load_json(sidecar)
        if recorded.get('sha256') != digest:
            raise ValueError('Existing product SHA-256 differs from verified completion record: ' + str(path))
        return recorded
    # A crash may occur after committing a verified ZIP but before its summary.
    # Recheck all data instead of trusting a completion flag on its own.
    with zarr.storage.ZipStore(task['source']['path'], mode='r') as source_store:
        src = zarr.open_group(store=source_store, mode='r')
        polygon = lake_mask(task['feature'], dict(src.attrs), src['data'].shape[-2:])
        expected_times = src['acquisition_time'][:]
    audit = verify_archive(path, task['fingerprint'], polygon, expected_times)
    summary.update(destination=str(path), sha256=digest, size_bytes=Path(path).stat().st_size, verification=audit)
    atomic_json(sidecar, summary)
    return summary


def verify_archive(path, expected, polygon, expected_times):
    """Read every output chunk, checking support, fill, counts and archive integrity."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP entries')
        bad = z.testzip()
        if bad:
            raise ValueError('ZIP CRC failure: ' + bad)
    with zarr.storage.ZipStore(str(path), mode='r') as store:
        g = zarr.open_group(store=store, mode='r')
        np.testing.assert_array_equal(g['acquisition_time'][:], expected_times)
        count = g['valid_pixel_count'][:]
        status = g['scene_status'][:]
        if np.any((count >= 10) != (status == 0)):
            raise ValueError('Scene-status/count mismatch')
        extrema = {t: [None, None] for t in TAGS}
        for start in range(0, len(count), 16):
            end = min(start + 16, len(count))
            common = None
            for tag in TAGS:
                a = g[tag][start:end]
                valid = np.isfinite(a)
                if np.any(valid[:, ~polygon]):
                    raise ValueError('Finite product outside the corresponding SHP')
                if np.any(np.isinf(a)) or np.any(a[valid] <= 0):
                    raise ValueError('Invalid product value')
                if common is not None and not np.array_equal(valid, common):
                    raise ValueError('Product masks disagree')
                common = valid
                expected_n = np.where(status[start:end] == 0, count[start:end], 0)
                np.testing.assert_array_equal(valid.sum(axis=(1, 2)), expected_n)
                if valid.any():
                    lo, hi = float(a[valid].min()), float(a[valid].max())
                    old_lo, old_hi = extrema[tag]
                    extrema[tag] = [lo if old_lo is None else min(lo, old_lo), hi if old_hi is None else max(hi, old_hi)]
        if g.attrs['run_fingerprint'] != expected:
            raise ValueError('Fingerprint mismatch')
    return {'all_chunks_read': True, 'zip_crc': 'passed', 'outside_shp_finite_pixels': 0,
            'same_support_all_products': True, 'ranges': extrema}


def process_year(task):
    started = time.perf_counter()
    source = Path(task['source']['path'])
    dest = Path(task['destination'])
    progress = Path(task['progress'])
    if dest.exists():
        summary = inspect_completed(dest, task)
        atomic_json(progress, {'state': 'complete', 'resumed': True, **summary})
        return {'destination': str(dest), 'resumed': True, **summary}
    local = Path(task['local'])
    local.parent.mkdir(parents=True, exist_ok=True)
    # Incomplete files belong only to this exact task, never to source archives.
    if local.exists():
        local.unlink()
    p = task['exported_parameters']['params']
    details = []
    checkpoint = {'lake': task['lake'], 'year': task['year'], 'pid': os.getpid(), 'state': 'reading', 'done': 0,
                  'worker_runtime': WORKER_RUNTIME, 'model_threads': MODEL_THREADS, 'updated': now()}
    atomic_json(progress, checkpoint)
    with zarr.storage.ZipStore(str(source), mode='r') as source_store:
        src = zarr.open_group(store=source_store, mode='r')
        data = src['data']
        n, b, h, w = data.shape
        if b != 69 or str(data.dtype) != 'float32':
            raise ValueError('Expected source (time,69,y,x), float32')
        times = src['acquisition_time'][:]
        if any(str(int(t))[:4] != str(task['year']) for t in times):
            raise ValueError('Source contains a different year')
        attrs = dict(src.attrs)
        polygon = lake_mask(task['feature'], attrs, (h, w))
        source_meta = json.loads(src['metadata_json'][:].tobytes())
        source_scenes = source_meta.get('plan', {}).get('scenes', [])
        provenance = {'pipeline_version': VERSION, 'created_utc': now(), 'run_fingerprint': task['fingerprint'],
            'lake_id': task['lake'], 'lake_name': task['feature']['properties']['name'], 'year': task['year'],
            'dimensions': ['time', 'y', 'x'], 'crs_wkt': attrs['crs_wkt'], 'transform': attrs['transform'],
            'transform_convention': 'rasterio.Affine a,b,c,d,e,f,0,0,1',
            'source': task['source'], 'source_task_fingerprint': attrs.get('task_fingerprint'),
            'source_metadata_sha256': attrs.get('metadata_sha256'), 'exported_mask_parameters': task['exported_parameters'],
            'processing_scope': 'all source acquisitions in the calendar year; export UI month/lake scope ignored by explicit user instruction',
            'mask_order': 'exported spectral conditions -> diamond erosion -> intersection with matching lake SHP',
            'shp_rasterization': 'pixel centers; all_touched=False; complete multipart polygons and holes',
            'boundary_sha256': task['boundary_sha256'], 'boundary_provenance': task['boundary_provenance'],
            'models': task['model_provenance'], 'model_formula': 'mean of frozen V1.0 RF bundle.predict in original physical units; per-bundle saved preprocessing and expm1 inverse',
            'features': 'V1.0 frozen 10 bands + 8 indices (18); each bundle computes its own features in float64 and applies saved median imputer/scaler; Rrs already sr-1, no division by pi',
            'source_validity_filter_enabled': p['source']['enabled'],
            'minimum_scene_pixels': 10, 'nodata': 'NaN (original TIFF writer used 0)',
            'product_clipping': 'none', 'time_encoding': 'YYYYMMDDhhmmss, unchanged source time convention',
            'software': task['software'], 'code_hashes': task['code_hashes']}
        counts = np.zeros(n, dtype='uint32')
        status = np.ones(n, dtype='uint8')
        with zarr.storage.ZipStore(str(local), mode='w') as out_store:
            out = zarr.group(store=out_store, zarr_format=2, attributes=provenance)
            products = {tag: create_array(out, tag, (n, h, w), 'float32', (16, h, w), ['time','y','x'], np.nan,
                        {'units': UNITS[tag], 'long_name': {'Chla':'chlorophyll-a concentration','SDD':'Secchi disk depth','SPM':'suspended particulate matter concentration'}[tag]}) for tag in TAGS}
            tm = create_array(out, 'acquisition_time', (n,), 'int64', (n,), ['time'], extra={'encoding': 'YYYYMMDDhhmmss'})
            tm[:] = times
            af = Affine(*attrs['transform'][:6])
            if af.b != 0 or af.d != 0:
                raise ValueError('Rotated grid requires explicit two-dimensional coordinate arrays')
            for key, vals, dim in [('x',af.c+(np.arange(w)+.5)*af.a,'x'), ('y',af.f+(np.arange(h)+.5)*af.e,'y')]:
                ca = create_array(out, key, vals.shape, 'float64', vals.shape, [dim])
                ca[:] = vals
            for start in range(0, n, 16):
                end = min(start + 16, n)
                block_started = time.perf_counter()
                checkpoint.update(state='reading', done=start, total=n, updated=now())
                atomic_json(progress, checkpoint)
                with IO_GATE:
                    block = np.asarray(data.oindex[start:end, BANDS, :, :], dtype='float32')
                    validity = None
                    if p['source']['enabled']:
                        validity = np.all(src['masks'].oindex[start:end, BANDS[10:], :, :] != 0, axis=1)
                        validity &= np.all(np.isfinite(block[:,10:]), axis=1)
                masks, xs = [], []
                lengths = []
                for j in range(end - start):
                    spectral = compute_mask(block[j,10:], p, None if validity is None else validity[j])
                    mask = spectral & polygon
                    count = int(mask.sum())
                    counts[start+j] = count
                    masks.append(mask)
                    lengths.append(count if count >= 10 else 0)
                    if count >= 10:
                        status[start+j] = 0
                        xs.append(block[j,:10, :, :][:,mask])
                    details.append({'index': start+j, 'time': int(times[start+j]), 'spectral_pixels': int(spectral.sum()),
                        'shp_pixels': int(polygon.sum()), 'final_mask_pixels': count,
                        'status': 'processed' if count >= 10 else 'fewer_than_10_pixels',
                        'source_filename': source_scenes[start+j].get('filename') if len(source_scenes) == n else None})
                del block
                checkpoint.update(state='predicting', updated=now())
                atomic_json(progress, checkpoint)
                values = predict(np.concatenate(xs, axis=1)) if xs else {}
                del xs
                result = {tag: np.full((end-start,h,w), np.nan, dtype='float32') for tag in TAGS}
                offset = 0
                for j, length in enumerate(lengths):
                    if length:
                        for tag in TAGS:
                            result[tag][j][masks[j]] = values[tag][offset:offset+length]
                        offset += length
                for tag in TAGS:
                    products[tag][start:end] = result[tag]
                checkpoint.update(state='working', done=end, total=n, elapsed_seconds=time.perf_counter()-started,
                    last_block_seconds=time.perf_counter()-block_started, model_threads=MODEL_THREADS, updated=now())
                atomic_json(progress, checkpoint)
            for name, a, extra in [('valid_pixel_count', counts, {'definition':'final mask pixel count before <10 scene rule'}),
                                    ('scene_status', status, {'codes':{'0':'processed','1':'fewer_than_10_final_mask_pixels'}})]:
                arr = create_array(out,name,a.shape,str(a.dtype),(n,),['time'],extra=extra)
                arr[:] = a
            summary = {'lake': task['lake'], 'year': task['year'], 'scenes': n, 'processed_scenes': int((status == 0).sum()),
                'empty_scenes': int((status != 0).sum()), 'valid_product_pixels': int(counts[status == 0].sum()),
                'shp_pixels': int(polygon.sum()), 'shape': [n,h,w], 'elapsed_seconds': time.perf_counter()-started}
            write_json_array(out, 'metadata_json', {'complete': True, 'summary': summary, 'scenes': details})
    if stamp(source) != task['source']:
        raise RuntimeError('Source changed during processing')
    checkpoint.update(state='verifying', done=n, updated=now())
    atomic_json(progress, checkpoint)
    audit = verify_archive(local, task['fingerprint'], polygon, times)
    local_sha = sha256(local)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + '.' + uuid.uuid4().hex + '.partial')
    checkpoint.update(state='copying', updated=now())
    atomic_json(progress, checkpoint)
    with IO_GATE:
        shutil.copyfile(local, temp)
        if sha256(temp) != local_sha:
            raise IOError('NAS product copy failed SHA-256 verification')
        if dest.exists():
            raise FileExistsError('Concurrent output appeared: ' + str(dest))
        os.replace(temp, dest)
    summary.update(destination=str(dest), sha256=local_sha, size_bytes=dest.stat().st_size,
        verification=audit, elapsed_seconds=time.perf_counter()-started, worker_runtime=WORKER_RUNTIME)
    atomic_json(dest.with_name(dest.name + '.summary.json'), summary)
    atomic_json(progress, {'state':'complete', 'updated':now(), **summary})
    return summary


def build_tasks(cfg, model_manifest, lakes=None, years=None):
    params = load_json(config_path(cfg,'mask_parameters'))
    features = load_json(config_path(cfg,'boundaries'))['features']
    boundary_provenance = load_json(config_path(cfg,'boundary_provenance'))
    boundary_sha = sha256(config_path(cfg,'boundaries'))
    work = config_path(cfg,'work_root')
    software = {'python': sys.version, 'numpy': np.__version__, 'scikit_learn': sklearn.__version__,
                'zarr': zarr.__version__, 'rasterio': rasterio.__version__}
    code_hashes = {p.name: sha256(p) for p in [BASE/'pipeline.py', BASE/'science.py', BASE/'rf_adapter.py'] + sorted((BASE/'vendor'/'olci_benchmark').glob('*.py'))}
    model_provenance = {tag:[{k:v for k,v in row.items() if k != 'local'} for row in rows] for tag,rows in model_manifest.items()}
    tasks = []
    for feature in features:
        lake = feature['properties']['lake_id']
        if lakes and lake not in lakes:
            continue
        for year in years or cfg['years']:
            source = Path(cfg['source_root']) / lake / f'{year}.zarr.zip'
            task = {'lake':lake,'year':year,'source':stamp(source),'feature':feature,
                'exported_parameters':params,'boundary_sha256':boundary_sha,'boundary_provenance':boundary_provenance,
                'model_provenance':model_provenance,'software':software,'code_hashes':code_hashes}
            # Fingerprint records scientific inputs, independent of destination/parallelism.
            task['fingerprint'] = hashlib.sha256(encoded(task)).hexdigest()
            task.update(destination=str(Path(cfg['output_root'])/lake/f'{year}.zarr.zip'),
                local=str(work/'archives'/lake/f'{year}.zarr.zip'),progress=str(work/'progress'/f'{lake}_{year}.json'))
            tasks.append(task)
    if not tasks:
        raise ValueError('No lake/year tasks selected')
    return sorted(tasks, key=lambda task: task['source']['size_bytes'], reverse=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default=str(BASE/'config'/'run.json'))
    ap.add_argument('--lakes',nargs='+')
    ap.add_argument('--years',nargs='+',type=int)
    ap.add_argument('--workers',type=int)
    ap.add_argument('--model-threads',type=int)
    args=ap.parse_args()
    cfg=load_json(args.config)
    work=config_path(cfg,'work_root');work.mkdir(parents=True,exist_ok=True)
    # OS-held exclusive run lock; a killed process does not leave a stale lock.
    lock=(work/'run.lock').open('a+b')
    lock.seek(0,2)
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0)
    if os.name=='nt':
        import msvcrt
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    from runtime import machine_info, choose_parallelism, read_cpu_usage
    print('STAGE_MODELS',now(),flush=True)
    model_manifest=stage_models(cfg)
    atomic_json(work/'models_manifest.json',model_manifest)
    tasks=build_tasks(cfg,model_manifest,args.lakes,args.years)
    machine=machine_info()
    parallel=choose_parallelism(len(tasks),cfg.get('cpu_target',.6))
    workers=args.workers or parallel['workers']
    model_threads=args.model_threads or parallel['model_threads']
    io_slots=cfg.get('io_slots',3)
    run={'started':now(),'state':'running','pid':os.getpid(),'machine':machine,'parallelism':parallel,
         'workers':workers,'model_threads':model_threads,'io_slots':io_slots,'tasks':len(tasks),'completed':[],'failed':[]}
    atomic_json(work/'run_progress.json',run)
    atomic_json(work/'tasks.json',tasks)
    ctx=mp.get_context('spawn')
    gate=ctx.BoundedSemaphore(io_slots);counter=ctx.Value('i',0)
    pending_count=ctx.Value('i',len(tasks))
    samples=[]
    print('RUN',len(tasks),'workers',workers,'model_threads',model_threads,flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_worker,
            initargs=(model_manifest,gate,model_threads,counter,pending_count,workers*model_threads,workers)) as pool:
        pending={pool.submit(process_year,t):t for t in tasks}
        while pending:
            done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            cpu=read_cpu_usage()
            samples.append({'utc':now(),'system_cpu_percent':cpu['total_percent'] if cpu['available'] and cpu['complete'] else None,
                            'native_cpu':cpu,'memory_percent':psutil.virtual_memory().percent})
            run['last_cpu']=samples[-1]
            for fut in done:
                task=pending.pop(fut)
                try:
                    result=fut.result();run['completed'].append(result)
                    print('COMPLETE',task['lake'],task['year'],result['processed_scenes'],'/',result['scenes'],flush=True)
                except Exception:
                    error={'lake':task['lake'],'year':task['year'],'error':traceback.format_exc()}
                    run['failed'].append(error)
                    print('FAILED',error,flush=True)
            pending_count.value=len(pending)
            run['updated']=now()
            atomic_json(work/'run_progress.json',run)
            atomic_json(work/'cpu_samples.json',samples)
    run['finished']=now();run['state']='failed' if run['failed'] else 'complete'
    atomic_json(work/'run_progress.json',run)
    output=Path(cfg['output_root']);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'run_manifest.json',run)
    print('FINISHED',run['state'],len(run['completed']),'/',len(tasks),flush=True)
    if run['failed']:return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())


