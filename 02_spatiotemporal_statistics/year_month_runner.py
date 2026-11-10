"""Produce individual year-month maps from the previously verified local cache."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import stat
import subprocess
import time
import traceback
import uuid
import zipfile
import fiona
import psutil
import collection_pipeline as collection
import daily_collection_pipeline as daily
from daily_safe_runner import safe_write_json as write
from pipeline import encoded, hash_file, now
from statistics_core import selected_months

BASE = Path(__file__).resolve().parent
VERSION = '1.0.0-daily-first-year-month'
PILOT = ['A22S304_Shunshanbao', 'F36A001_Poyang', 'F43A001_Dongting',
         'H53A302_Xingyun', 'R63D205_Xitaijinaier', 'NMB01_Huhenuoer']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def lexical_resolved(value, base=BASE):
    # Source paths remain provenance only. No NAS resolution/stat is needed.
    path = Path(value)
    return Path(os.path.abspath(str(path if path.is_absolute() else base / path)))


def build_tasks(cfg):
    import year_month_engine as monthly
    collection.resolved = lexical_resolved
    old_cfg = read(cfg['previous_config'])
    previous = read(cfg['previous_plan'])
    qa = read(cfg['previous_qa'])
    if (qa.get('status') != 'passed'
            or qa.get('source_hashes_checked') != previous.get('annual_inputs')
            or qa.get('run_id') != previous['run_id'] or qa.get('failures')):
        raise ValueError('Prior complete source-cache verification is required')
    manifest = read(cfg['previous_manifest'])
    if hash_file(cfg['previous_manifest']) != qa['manifest_sha256']:
        raise ValueError('Prior delivery manifest differs from its completed QA')
    old_tasks, old_run, pins = daily.make_plan(old_cfg)
    if old_run != previous['run_id']:
        raise ValueError('The frozen daily science/input identity has changed')
    old_lakes = {x['lake_id']: x for x in previous['lakes']}
    selected_lakes = cfg.get('lake_ids')
    if selected_lakes is not None:
        if not selected_lakes or len(selected_lakes) != len(set(selected_lakes)):
            raise ValueError('Configured lake_ids must be a non-empty unique list')
        unknown = set(selected_lakes) - set(old_lakes)
        if unknown:
            raise ValueError('Configured lake_ids are absent from the verified daily run: '
                             + ', '.join(sorted(unknown)))
        selected_lakes = set(selected_lakes)
    finished_ns = int(datetime.fromisoformat(qa['finished_local']).timestamp() * 1e9)
    code = {name: hash_file(BASE / name) for name in
            ('year_month_engine.py', 'year_month_runner.py', 'daily_statistics_core.py',
             'collection_pipeline.py', 'pipeline.py', 'statistics_core.py')}
    tasks = []
    for task in old_tasks:
        lake = task['feature']['properties']['lake_id']
        if selected_lakes is not None and lake not in selected_lakes:
            continue
        prior = old_lakes[lake]
        result = read(prior['local_result'])
        if task['lake_fingerprint'] != prior['lake_fingerprint'] or result['lake_fingerprint'] != prior['lake_fingerprint']:
            raise ValueError('Prior checkpoint identity differs: ' + lake)
        info = result['outputs']['climatology_annual']
        if info['sha256'] != manifest['files'][lake + '/' + info['filename']]['sha256']:
            raise ValueError('Global day-count archive is not the verified archive')
        task['daily_lake_fingerprint'] = task['lake_fingerprint']
        task['global_archive'] = str(Path(result['result_dir']) / info['filename'])
        task['global_archive_sha256'] = info['sha256']
        task['prior_result'] = prior['local_result']
        new_sources = []
        for source in task['sources']:
            cache = Path(old_cfg['source_cache_root']) / lake / f'{source["year"]}_{source["sha256"]}.zarr.zip'
            if lexical_resolved(result['source_checks'][str(source['year'])]['local_cache']) != lexical_resolved(cache):
                raise ValueError('Prior validated cache route differs')
            s = cache.lstat()
            if (not stat.S_ISREG(s.st_mode) or s.st_size != source['size_bytes']
                    or s.st_mtime_ns > finished_ns or s.st_ctime_ns > finished_ns
                    or getattr(s, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)):
                raise ValueError('Cache changed or missing since completed verification: ' + str(cache))
            new_sources.append({**source, 'read_path': str(cache), 'cache_mtime_ns': s.st_mtime_ns})
        task['sources'] = new_sources
        identity = {**task['identity'], 'version': VERSION, 'statistic': 'year_month', 'method': monthly.METHOD,
                    'code_sha256': code, 'previous_daily_run_id': old_run,
                    'daily_lake_fingerprint': task['daily_lake_fingerprint'],
                    'global_archive_sha256': info['sha256']}
        task['identity'] = identity
        task['lake_fingerprint'] = hashlib.sha256(encoded(identity)).hexdigest()
        task['work_dir'] = str(Path(cfg['work_root']) / 'lakes' / lake / task['lake_fingerprint'])
        tasks.append(task)
    if not tasks:
        raise ValueError('The selected lake scope contains no verified daily inputs')
    if selected_lakes is not None and {task['feature']['properties']['lake_id'] for task in tasks} != selected_lakes:
        raise ValueError('The selected lake scope does not match the verified daily inputs')
    run = hashlib.sha256(encoded({'version': VERSION, 'fingerprints': [t['lake_fingerprint'] for t in tasks]})).hexdigest()
    work = Path(cfg['work_root'])
    write(work / 'tasks.json', tasks)
    doc = {'state': 'planned', 'run_id': run, 'lake_count': len(tasks),
           'annual_inputs': sum(len(t['sources']) for t in tasks),
           'expected_month_rows': sum(len(t['sources']) * len(selected_months(t['feature']['properties']['region'])) for t in tasks),
           'previous_qa': cfg['previous_qa'], 'cache_validation': 'Reuse completed SHA256 QA; require unchanged local size and timestamps; check read stability.',
           'output_root': cfg['output_root'], 'updated_utc': now(), 'code_sha256': code}
    write(work / 'plan.json', doc)
    return tasks, run, doc


def checked_process(task):
    import year_month_engine as engine
    def check():
        for source in task['sources']:
            s = Path(source['read_path']).stat()
            if (s.st_size, s.st_mtime_ns) != (source['size_bytes'], source['cache_mtime_ns']):
                raise ValueError('Local cache changed during processing: ' + source['read_path'])
    with collection.reservation(task['estimated_memory_bytes']):
        check()
        result = engine.process_lake(task)
        check()
        return result


def calculate(cfg, tasks, run, pilot=False):
    work = Path(cfg['work_root'])
    chosen = [t for t in tasks if t['feature']['properties']['lake_id'] in PILOT] if pilot else tasks
    workers = min(cfg['workers'], len(chosen))
    budget = int(psutil.virtual_memory().available * cfg['memory_fraction_available'])
    state = {'state': 'calculating', 'run_id': run, 'pid': os.getpid(), 'started_utc': now(),
             'tasks': len(chosen), 'workers': workers, 'completed': [], 'failed': []}
    progress = work / ('pilot_progress.json' if pilot else 'progress.json')
    results, queue, pending = [], sorted(chosen, key=lambda t: t['estimated_memory_bytes'], reverse=True), {}
    ctx = multiprocessing.get_context('spawn')
    with ctx.Manager() as manager:
        condition, used = manager.Condition(), manager.Value('q', 0)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=collection.init_worker,
                                 initargs=(condition, used, budget, None)) as pool:
            while queue or pending:
                while queue and len(pending) < workers:
                    task = queue.pop(0)
                    pending[pool.submit(checked_process, task)] = task
                done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    task = pending.pop(future)
                    lake = task['feature']['properties']['lake_id']
                    try:
                        result = future.result()
                        results.append(result)
                        state['completed'].append(lake)
                    except Exception:
                        state['failed'].append({'lake_id': lake, 'error': traceback.format_exc()})
                state.update(updated_utc=now(), completed_count=len(results), failed_count=len(state['failed']),
                             cpu_percent=psutil.cpu_percent(), available_memory_bytes=psutil.virtual_memory().available)
                write(progress, state)
    state.update(state='failed' if state['failed'] else 'local_calculated', updated_utc=now())
    write(progress, state)
    if state['failed']:
        raise RuntimeError('Year-month calculation failed; see ' + str(progress))
    return results


def assemble(cfg, tasks, results, run):
    destination = Path(cfg['work_root']) / 'deliveries' / run
    if destination.exists():
        if read(destination / 'run_manifest.json')['run_fingerprint'] != run:
            raise ValueError('Existing delivery belongs to a different run')
        return destination
    staging = destination.with_name(run + '.partial-' + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    files = {}
    for result in sorted(results, key=lambda r: r['lake_id']):
        target = staging / result['lake_id']
        target.mkdir()
        for info in result['outputs'].values():
            source = Path(result['result_dir']) / info['filename']
            os.link(source, target / source.name)
            files[result['lake_id'] + '/' + source.name] = info
        shutil.copyfile(Path(result['result_dir']) / 'lake_result.json', target / 'lake_result.json')
    geometries = {t['feature']['properties']['lake_id']: t['feature']['geometry'] for t in tasks}
    fields = {**daily.SHP_FIELDS, 'ym': 'int:8'}
    shp = staging / 'year_month.shp'
    with fiona.open(str(shp), 'w', driver='ESRI Shapefile',
                    schema={'geometry': 'Polygon', 'properties': fields}, crs='EPSG:4326', encoding='UTF-8') as writer:
        for result in results:
            for row in result['rows']:
                writer.write({'geometry': geometries[result['lake_id']], 'properties': row})
    count = sum(len(r['rows']) for r in results)
    with fiona.open(str(shp), encoding='UTF-8') as reader:
        if len(reader) != count:
            raise ValueError('SHP record count mismatch')
    scope_keys = ('requested_years', 'source_years', 'available_years', 'missing_years',
                  'n_years', 'complete_requested_period', 'climatology_policy')
    scopes = {t['feature']['properties']['lake_id']: {key: t['identity'][key] for key in scope_keys} for t in tasks}
    write(staging / 'lake_scopes.json', scopes)
    write(staging / 'missing_years.json', {lake: scope['missing_years'] for lake, scope in scopes.items() if scope['missing_years']})
    write(staging / 'config_used.json', cfg)
    write(staging / 'run_manifest.json', {'complete': True, 'version': VERSION, 'run_fingerprint': run,
          'lake_count': len(results), 'annual_input_count': sum(len(t['sources']) for t in tasks),
          'year_month_rows': count, 'aggregation': 'daily_mean_then_period_mean',
          'coverage_fraction': .05, 'coverage_unit': 'source_acquisition_day',
          'previous_qa': cfg['previous_qa'], 'files': files, 'created_utc': now(),
          'lake_fingerprints': {r['lake_id']: r['lake_fingerprint'] for r in results}})
    os.rename(staging, destination)
    return destination


def package(cfg, destination):
    names = ['year_month_engine.py', 'year_month_runner.py', 'verify_year_month_sample.py',
             'daily_statistics_core.py', 'daily_safe_runner.py', 'daily_collection_pipeline.py',
             'collection_pipeline.py', 'pipeline.py', 'statistics_core.py', 'daily_publication.py',
             'config/year_month_all_2016_2025.json',
             'tests/test_year_month_engine.py', 'tests/test_year_month_verifier.py']
    with zipfile.ZipFile(destination / 'source_code_year_month.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(BASE / name, name)
        archive.write(Path(cfg['work_root']) / 'tasks.json', 'execution/tasks.json')
    shutil.copyfile(Path(cfg['work_root']) / 'sample_validation.json', destination / 'sample_validation.json')


def publish(cfg, destination, run):
    work = Path(cfg['work_root'])
    source = destination.resolve(strict=True)
    if source.parent != (work / 'deliveries').resolve() or source.name != run:
        raise ValueError('Unexpected local move source')
    output = Path(cfg['io_output_root'])
    parent = output.parent.resolve(strict=True)
    if output.resolve().parent != parent or output.name != Path(cfg['output_root']).name:
        raise ValueError('Move destination differs from configured product folder')
    if output.exists():
        raise FileExistsError('Final year-month output already exists')
    staging = parent / (output.name + '.partial-' + run[:16])
    staging.mkdir(exist_ok=True)
    if staging.resolve().parent != parent:
        raise ValueError('Staging target escapes the intended directory')
    manifest = read(source / 'run_manifest.json')
    report = read(work / 'sample_validation.json')
    if (report.get('status') not in ('passed', 'pass') or report.get('shapefile_verified') is not True
            or report.get('task_fingerprints') != manifest['lake_fingerprints']
            or report.get('shapefile_rows') != manifest['year_month_rows']
            or manifest['run_fingerprint'] != run):
        raise ValueError('Sample and SHP validation must match this completed year-month delivery')
    write(work / 'delivery_manifest.json', manifest)
    total_files = sum(len(fs) for _, _, fs in os.walk(source))
    total_bytes = sum((Path(root) / f).stat().st_size for root, _, fs in os.walk(source) for f in fs)
    state = {'state': 'moving', 'run_id': run, 'pid': os.getpid(), 'lake_count': len(manifest['lake_fingerprints']),
             'total_files': total_files, 'total_bytes': total_bytes, 'output_root': cfg['output_root'],
             'io_output_root': str(output), 'local_delivery': str(source), 'started_utc': now()}
    write(work / 'progress.json', state)
    log = work / 'robocopy_move.log'
    command = [str(Path(os.environ['SystemRoot']) / 'System32/robocopy.exe'), str(source), str(staging),
               '/MOVE', '/E', '/MT:16', '/R:2', '/W:2', '/COPY:DAT', '/DCOPY:T', '/IS', '/IT', '/XJ', '/NP', '/BYTES', '/UNILOG:' + str(log)]
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    while proc.poll() is None:
        remaining = sum(len(fs) for _, _, fs in os.walk(source))
        state.update(files_moved=total_files-remaining, updated_utc=now(), robocopy_pid=proc.pid)
        write(work / 'progress.json', state)
        time.sleep(5)
    code = proc.wait()
    remaining = sum(len(fs) for _, _, fs in os.walk(source))
    if code >= 8 or remaining:
        state.update(state='move_failed', robocopy_exit_code=code, remaining_files=remaining, updated_utc=now())
        write(work / 'progress.json', state)
        raise RuntimeError('Native MOVE incomplete; see ' + str(log))
    os.rename(staging, output)
    state.update(state='complete', files_moved=total_files, robocopy_exit_code=code, finished_utc=now(),
                 year_month_rows=manifest['year_month_rows'], annual_inputs=manifest['annual_input_count'])
    write(work / 'progress.json', state)
    write(output / 'workflow_completion.json', state)
    print(json.dumps(state, ensure_ascii=True), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(BASE / 'config/year_month_all_2016_2025.json'))
    p.add_argument('--mode', choices=('plan', 'pilot', 'run', 'publish'), default='plan')
    args = p.parse_args(argv)
    cfg = read(args.config)
    work = Path(cfg['work_root'])
    with collection.exclusive_lock(work):
        if args.mode == 'publish':
            plan = read(work / 'plan.json')
            destination = work / 'deliveries' / plan['run_id']
            output = Path(cfg['io_output_root'])
            if output.exists():
                completion = read(output / 'workflow_completion.json')
                if completion.get('state') == 'complete' and completion.get('run_id') == plan['run_id']:
                    print(json.dumps(completion, ensure_ascii=True), flush=True)
                    return 0
                raise FileExistsError('Final output exists without matching completion evidence')
            if not (destination / 'run_manifest.json').exists():
                snapshot = read(work / 'delivery_manifest.json')
                if snapshot['run_fingerprint'] != plan['run_id']:
                    raise ValueError('Move-resume manifest differs from planned run')
                destination.mkdir(parents=True, exist_ok=True)
                write(destination / 'run_manifest.json', snapshot)
            package(cfg, destination)
            publish(cfg, destination, plan['run_id'])
            return 0
        tasks, run, doc = build_tasks(cfg)
        print(json.dumps(doc, ensure_ascii=True), flush=True)
        if args.mode == 'plan':
            return 0
        results = calculate(cfg, tasks, run, pilot=args.mode == 'pilot')
        if args.mode == 'run':
            destination = assemble(cfg, tasks, results, run)
            print(json.dumps({'local_delivery': str(destination), 'run_id': run}), flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
