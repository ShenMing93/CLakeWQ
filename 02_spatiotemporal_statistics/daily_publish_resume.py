"""Resume only publication of an already independently verified daily delivery."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback

import psutil

import daily_workflow as wf
from daily_status import shared_read

BASE = Path(__file__).resolve().parent
ORIGINAL_FILES = (
    'daily_collection_pipeline.py', 'daily_statistics_core.py',
    'daily_safe_runner.py', 'daily_cached_runner.py', 'collection_pipeline.py',
    'statistics_core.py', 'pipeline.py', 'verify_daily_statistics.py',
    'verify_collection_statistics.py', 'verify_statistics.py',
    'daily_publication.py', 'daily_workflow.py',
)


def lexical(path):
    """Compare Windows paths without querying remote filesystems."""
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def check_hashes(pins):
    for path, digest in pins.items():
        if wf.sha(path) != digest:
            raise ValueError('Pinned dependency changed: ' + str(path))


def prior_process_running(prior):
    try:
        proc = psutil.Process(int(prior['pid']))
        return abs(proc.create_time() - float(prior['process_create_time'])) < .01
    except psutil.NoSuchProcess:
        return False


def validate_resume(config, prior_path):
    """Read-only identity checks; fresh computation/QA are deliberately not run."""
    config = wf.local(config).resolve()
    prior = wf.read(prior_path)
    cfg = wf.read(config)
    work, output = wf.local(cfg['work_root']), wf.local(cfg['output_root'])
    if (prior.get('state') != 'publishing' or prior.get('local_only') is not False
            or lexical(prior.get('config', '')) != lexical(config)
            or prior.get('config_sha256') != wf.sha(config)
            or lexical(prior.get('output_root', '')) != lexical(output)):
        raise ValueError('Prior workflow is not this exact publishing run')
    required = {str(config)} | {str(BASE / name) for name in ORIGINAL_FILES}
    original_pins = prior.get('pinned_sources', {})
    if set(original_pins) != required:
        raise ValueError('Prior workflow pin set is incomplete or different')
    check_hashes(original_pins)
    original_start = datetime.fromisoformat(prior['started']).timestamp()
    progress, delivery, run_id, expected = wf.validate_completion(cfg, work, original_start)
    if (prior.get('run_id') != run_id or prior.get('lakes') != len(expected)
            or lexical(prior.get('local_delivery', '')) != lexical(delivery)):
        raise ValueError('Prior workflow and completed delivery differ')
    report = work / 'daily_verification.json'
    if (wf.sha(report) != prior.get('resume_verified_qa_sha256')
            or wf.sha(delivery / 'run_manifest.json') != prior.get('resume_manifest_sha256')):
        raise ValueError('Frozen QA report or manifest differs from resume snapshot')
    qa = wf.validate_qa(report, original_start, config, delivery, run_id, expected)
    if (qa.get('failures') != [] or qa.get('full_config_selected') is not True
            or qa.get('source_sampling_enabled') is not True
            or qa.get('source_and_output_size_mtime_unchanged') is not True
            or qa.get('lakes_verified') != len(expected)
            or qa.get('zarr_archives_verified') != len(expected) * 3
            or qa.get('checks_passed', 0) <= 0
            or not qa.get('source_samples')):
        raise ValueError('QA is incomplete for publication resume')
    manifest = wf.read(delivery / 'run_manifest.json')
    if (manifest.get('aggregation') != 'daily_mean_then_period_mean'
            or manifest.get('coverage_unit') != 'source_acquisition_day'
            or qa.get('source_hashes_checked') != manifest.get('annual_input_count')
            or qa.get('aggregation') != manifest.get('aggregation')
            or qa.get('coverage_unit') != manifest.get('coverage_unit')):
        raise ValueError('QA aggregation or source scope differs')
    input_pins = {item['path']: item['sha256'] for item in manifest['input_pins']}
    if len(input_pins) != len(manifest['input_pins']) or not input_pins:
        raise ValueError('Duplicate or absent input pins')
    check_hashes(input_pins)
    pins = dict(original_pins)
    pins.update(input_pins)
    pins[str(report)] = prior['resume_verified_qa_sha256']
    pins[str(delivery / 'run_manifest.json')] = prior['resume_manifest_sha256']
    return dict(config=config, cfg=cfg, prior=prior, work=work, output=output,
                delivery=delivery, run_id=run_id, expected=expected, report=report,
                qa=qa, pins=pins, original_start=original_start)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--prior-state', required=True)
    parser.add_argument('--staging', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    config, prior_path = wf.local(args.config).resolve(), wf.local(args.prior_state).resolve()
    context = validate_resume(config, prior_path)
    if args.check_only:
        print(json.dumps({'state': 'resume_validated', 'run_id': context['run_id'],
                          'lakes': len(context['expected']),
                          'qa_checks': context['qa']['checks_passed']}, ensure_ascii=True), flush=True)
        return 0
    if prior_process_running(context['prior']):
        raise RuntimeError('Prior publication process is still running')
    import daily_parallel_publication as publisher
    work = context['work']
    with wf.lock(work):
        # Revalidate after acquiring the lock, so the snapshot cannot race a writer.
        context = validate_resume(config, prior_path)
        if prior_process_running(context['prior']):
            raise RuntimeError('Prior publication process is still running')
        current = json.loads(shared_read(work / 'workflow_progress.json'))
        if (current.get('run_id') != context['run_id']
                or current.get('state') not in ('publishing', 'failed')):
            raise ValueError('Unexpected live workflow identity or state')
        pins = context['pins']
        for path in (Path(__file__).resolve(), BASE / 'daily_parallel_publication.py', prior_path):
            pins[str(path)] = wf.sha(path)
        state = dict(state='resume_validated', pid=os.getpid(),
                     process_create_time=psutil.Process().create_time(), started=wf.now(),
                     original_started=context['prior']['started'], resumed_from=str(prior_path),
                     config=str(config), config_sha256=wf.sha(config),
                     run_id=context['run_id'], lakes=len(context['expected']), local_only=False,
                     local_delivery=str(context['delivery']), output_root=str(context['output']),
                     verification_report=str(context['report']),
                     verification_report_sha256=wf.sha(context['report']),
                     manifest_sha256=wf.sha(context['delivery'] / 'run_manifest.json'),
                     pinned_sources=pins, publication_workers=args.workers,
                     staging=str(args.staging))

        def save(label, **extra):
            state.update(state=label, updated=wf.now(), **extra)
            wf.write(work / 'workflow_progress.json', state)

        last_progress = [0., None]

        def progress(event):
            now = time.monotonic()
            phase = event['phase']
            if (phase != last_progress[1] or now - last_progress[0] >= 1
                    or event.get('completed') == event.get('total')):
                save('publishing', publication_progress=event)
                last_progress[:] = [now, phase]
                if event.get('completed') == event.get('total'):
                    print(json.dumps(event, ensure_ascii=True), flush=True)

        try:
            save('packaging')
            archive, nfiles = wf.package_sources(work, config)
            check_hashes(pins)
            save('publishing', source_files=nfiles)
            result = publisher.publish_delivery(context['delivery'], context['output'],
                                                context['run_id'], workers=args.workers,
                                                staging=Path(args.staging), progress=progress)
            check_hashes(pins)
            save('finalizing', publication=result)
            output = context['output']
            wf.copy_verified(context['report'], output / 'daily_verification.json')
            wf.copy_verified(archive, output / 'source_code_statistics_daily.zip')
            wf.copy_verified(archive, output.parent / 'source_code_statistics_daily.zip')
            check_hashes(pins)
            complete = dict(state, state='complete', updated=wf.now(), finished=wf.now(),
                            source_archive=str(output / 'source_code_statistics_daily.zip'),
                            source_archive_sha256=wf.sha(archive), source_files=nfiles)
            candidate = work / 'publication_completion_candidate.json'
            wf.write(candidate, complete)
            wf.copy_verified(candidate, output / 'workflow_completion.json')
            wf.write(work / 'workflow_progress.json', complete)
            if wf.sha(candidate) != wf.sha(work / 'workflow_progress.json'):
                raise ValueError('Local completion record differs')
            print(json.dumps({key: complete[key] for key in
                              ('state', 'output_root', 'lakes', 'run_id', 'finished')},
                             ensure_ascii=True), flush=True)
            return 0
        except BaseException:
            save('failed', error=traceback.format_exc(), finished=wf.now())
            traceback.print_exc()
            return 1


if __name__ == '__main__':
    raise SystemExit(main())

