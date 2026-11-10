"""Run daily-first statistics, independent QA, and verified atomic publication."""
from __future__ import annotations
import argparse, contextlib, hashlib, json, os, subprocess, sys, time, traceback, zipfile, shutil
from datetime import datetime, timezone
from pathlib import Path
import psutil
BASE=Path(__file__).resolve().parent

def now():return datetime.now(timezone.utc).isoformat()
def local(value):
    p=Path(value);return p if p.is_absolute() else BASE/p
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp-'+str(os.getpid()))
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    # Windows readers/antivirus can briefly deny renaming a status sidecar.
    for attempt in range(12):
        try:
            os.replace(temp,p)
            break
        except PermissionError:
            if attempt==11:raise
            time.sleep(min(.05*(2**attempt),.5))
@contextlib.contextmanager
def lock(work):
    import msvcrt
    work.mkdir(parents=True,exist_ok=True);f=(work/'daily_workflow.lock').open('a+b');f.seek(0,2)
    if not f.tell():f.write(b'0');f.flush()
    f.seek(0)
    try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:f.close();raise RuntimeError('Statistics workflow already running')
    try:yield
    finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1);f.close()
def count(value):
    return len(value) if isinstance(value,(dict,list)) else int(value or 0)
def copy_verified(src,dst):
    dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    tmp=dst.with_name('.publishing-'+dst.name)
    shutil.copy2(src,tmp)
    if sha(src)!=sha(tmp):raise IOError('Copy checksum differs: '+str(dst))
    os.replace(tmp,dst)
def package_sources(work,config):
    parent=BASE.parent
    paths=set()
    for pattern in ('*.py','*.ps1','*.txt','config/*.json','tests/*.py','tests/*.json'):
        paths.update(p for p in BASE.glob(pattern) if p.is_file())
    paths.add(Path(config))
    cfg=read(config)
    for key in ('frozen_manifest','input_audit'):
        paths.add(local(cfg[key]))
    frozen=local(cfg['frozen_manifest'])
    provenance=frozen.with_name(frozen.stem+'_provenance.json')
    if provenance.exists():paths.add(provenance)
    product_config=local(cfg['product_config']).resolve()
    prod=product_config.parent.parent
    product=read(product_config)
    paths.add(product_config)
    for key in ('boundaries','boundary_provenance','default_boundaries','default_boundary_provenance',
                'legacy_boundaries','legacy_boundary_provenance','source_inventory',
                'default_mask_parameters','model_manifest'):
        item=Path(product[key])
        paths.add(item if item.is_absolute() else prod/item)
    for spec in product.get('boundary_overrides',{}).values():
        for key in ('boundaries','boundary_provenance'):
            p=Path(spec[key]);paths.add(p if p.is_absolute() else prod/p)
    for name in ('runtime.py','collection_driver_v2.py','pipeline.py','science.py','rf_adapter.py','requirements.txt'):
        paths.add(prod/name)
    paths.update((prod/'vendor/olci_benchmark').glob('*.py'))
    paths={p.resolve() for p in paths}
    if any(not p.is_relative_to(parent.resolve()) for p in paths):
        raise ValueError('Source package dependency outside the project')
    manifest={p.relative_to(parent).as_posix():sha(p) for p in sorted(paths)}
    archive=work/'source_code_statistics_daily.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(paths):z.write(p,p.relative_to(parent).as_posix())
        z.writestr('source_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise IOError('Source archive CRC failure')
        for name,digest in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise IOError(name)
    return archive,len(paths)

def validate_completion(cfg,work,started):
    product=read(local(cfg['product_config']))
    expected=cfg.get('lake_ids',product['lake_ids'])
    if not expected or len(expected)!=len(set(expected)):
        raise ValueError('Invalid requested lake scope')
    progress_path=work/'collection_progress.json';progress=read(progress_path)
    if progress_path.stat().st_mtime<started-2:raise ValueError('Statistics progress is stale')
    actual=[r['lake_id'] for r in progress.get('completed',[])]
    if (progress.get('state')!='local_complete' or progress.get('tasks')!=len(expected)
        or len(actual)!=len(expected) or set(actual)!=set(expected) or progress.get('failed')):
        raise ValueError('Statistics lake scope did not complete')
    plan=read(work/'collection_plan.json')
    if (progress.get('run_id')!=plan.get('run_id') or not progress.get('run_id')
        or plan.get('tasks')!=len(expected)
        or {r['lake_id'] for r in plan['lakes']}!=set(expected)):
        raise ValueError('Completed scope differs from the execution plan')
    delivery=Path(progress['local_delivery']).resolve();run_id=progress['run_id']
    if delivery!=work.resolve()/'deliveries'/run_id:
        raise ValueError('Unexpected local delivery directory')
    manifest=read(delivery/'run_manifest.json')
    if (manifest.get('complete') is not True or manifest.get('run_fingerprint')!=run_id
        or manifest.get('lake_count')!=len(expected)
        or set(manifest.get('lake_scopes',{}))!=set(expected)
        or set(manifest.get('lake_fingerprints',{}))!=set(expected)):
        raise ValueError('Delivery manifest scope/identity differs')
    for row in progress['completed']:
        if row['lake_fingerprint']!=manifest['lake_fingerprints'][row['lake_id']]:
            raise ValueError('Completed lake identity differs from manifest')
    return progress,delivery,run_id,expected

def validate_qa(report,started,config,delivery,run_id,expected):
    if report.stat().st_mtime<started:raise ValueError('Independent QA report is stale')
    qa=read(report)
    if (qa.get('status')!='passed' or qa.get('config_sha256')!=sha(config)
        or qa.get('run_id')!=run_id or Path(qa.get('output_root','')).resolve()!=delivery.resolve()
        or qa.get('manifest_sha256')!=sha(delivery/'run_manifest.json')
        or len(qa.get('lake_ids',[]))!=len(expected) or set(qa.get('lake_ids',[]))!=set(expected)):
        raise ValueError('Independent QA did not pass for this exact delivery and lake scope')
    return qa

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',required=True)
    ap.add_argument('--local-only',action='store_true',help='Compute and independently verify without publishing')
    args=ap.parse_args(argv);config=local(args.config).resolve();cfg=read(config)
    work=local(cfg['work_root']);output=local(cfg['output_root'])
    inputs=[config,BASE/'daily_collection_pipeline.py',BASE/'daily_statistics_core.py',BASE/'daily_safe_runner.py',BASE/'daily_cached_runner.py',
            BASE/'collection_pipeline.py',BASE/'statistics_core.py',BASE/'pipeline.py',
            BASE/'verify_daily_statistics.py',BASE/'verify_collection_statistics.py',BASE/'verify_statistics.py',
            BASE/'daily_publication.py',Path(__file__).resolve()]
    pins={str(p):sha(p) for p in inputs}
    def unchanged():
        for p,d in pins.items():
            if sha(p)!=d:raise ValueError('Workflow code/config changed: '+p)
    with lock(work):
        state=dict(state='starting',pid=os.getpid(),process_create_time=psutil.Process().create_time(),
                   started=now(),config=str(config),config_sha256=sha(config),output_root=str(output),
                   local_only=args.local_only,pinned_sources=pins)
        def save(label,**extra):
            state.update(state=label,updated=now(),**extra);write(work/'workflow_progress.json',state)
        def command(label,script,arguments):
            unchanged();save(label)
            with (work/(label+'.stdout.log')).open('a',encoding='utf-8') as out,(work/(label+'.stderr.log')).open('a',encoding='utf-8') as err:
                out.write('\n'+json.dumps({'workflow_invocation':state['started'],'phase':label})+'\n');out.flush()
                proc=subprocess.Popen([sys.executable,str(BASE/script),*arguments],cwd=BASE,stdout=out,stderr=err,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env={**os.environ,'PYTHONIOENCODING':'utf-8'})
                save(label,child_launcher_pid=proc.pid)
                code=proc.wait()
            if code:raise RuntimeError(label+' failed with exit code '+str(code)+'; inspect '+label+'.stderr.log')
        try:
            started=time.time()
            command('calculating','daily_cached_runner.py',['--config',str(config),'--run','--local-only'])
            progress,delivery,run_id,expected=validate_completion(cfg,work,started)
            save('computed',local_delivery=str(delivery),run_id=run_id,lakes=int(progress['tasks']))
            report=work/'daily_verification.json'
            qa_started=time.time()
            command('verifying','verify_daily_statistics.py',
                    ['--config',str(config),'--output-root',str(delivery),'--report',str(report)])
            qa=validate_qa(report,qa_started,config,delivery,run_id,expected)
            unchanged()
            if args.local_only:
                save('verified_local',verification_report=str(report),finished=now());return 0
            save('packaging',verification_report=str(report))
            archive,nfiles=package_sources(work,config)
            unchanged()
            save('publishing',source_files=nfiles)
            import daily_collection_pipeline as engine
            published=engine.publish_delivery(delivery,output,run_id)
            copy_verified(report,output/'daily_verification.json')
            copy_verified(archive,output/'source_code_statistics_daily.zip')
            # Shared root sidecar is supplemental; the statistics manifest remains inside its own directory.
            copy_verified(archive,output.parent/'source_code_statistics_daily.zip')
            save('complete',publication=published,source_archive=str(output/'source_code_statistics_daily.zip'),
                 source_archive_sha256=sha(archive),source_files=nfiles,finished=now())
            copy_verified(work/'workflow_progress.json',output/'workflow_completion.json')
            print(json.dumps({k:state[k] for k in ('state','output_root','lakes','run_id','finished')},ensure_ascii=True),flush=True)
            return 0
        except BaseException:
            save('failed',error=traceback.format_exc(),finished=now());traceback.print_exc();return 1
if __name__=='__main__':raise SystemExit(main())

