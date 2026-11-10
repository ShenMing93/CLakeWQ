"""Bounded parallel publication/resumption; frozen scientific files stay untouched.

All manifest-listed bytes are independently verified locally, in staging, and
at the final destination. Directory preparation and copying also use the bounded
pool. Network file size checks use the opened byte stream, avoiding per-file
Path.resolve/stat round trips. Existing final output is never overwritten.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import uuid

MAX_WORKERS = 16
BLOCK_BYTES = 8 * 1024**2


def _workers(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('workers must be a positive integer')
    return min(value, MAX_WORKERS)


def _absolute(path):
    return Path(os.path.abspath(os.path.normpath(str(path))))


def _key(path):
    return os.path.normcase(str(_absolute(path)))


def _notify(callback, phase, completed, total, **extra):
    if callback is not None:
        callback({'phase': phase, 'completed': completed, 'total': total, **extra})


def _parallel(items, operation, workers, progress, phase, offset=0, phase_total=None, **extra):
    """All callback invocations occur serially on the calling thread."""
    items = list(items)
    total = len(items) if phase_total is None else phase_total
    _notify(progress, phase, offset, total, **extra)
    with ThreadPoolExecutor(max_workers=_workers(workers)) as pool:
        pending = {pool.submit(operation, item): item for item in items}
        try:
            for completed, future in enumerate(as_completed(pending), 1):
                future.result()
                _notify(progress, phase, offset + completed, total, **extra)
        except BaseException:
            for future in pending:
                future.cancel()
            raise


def _validate_relative(relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError('Manifest relative path must be a nonempty string')
    path = PurePosixPath(relative)
    if (not path.parts or path.is_absolute() or '..' in path.parts or '\\' in relative
            or ':' in relative or '\x00' in relative or str(path) != relative
            or relative.casefold() == 'run_manifest.json'):
        raise ValueError('Unsafe manifest relative path: ' + relative)
    for part in path.parts:
        if (part in ('.', '..') or part.rstrip(' .') != part or any(ord(c) < 32 for c in part)
                or re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', part, re.IGNORECASE)):
            raise ValueError('Ambiguous Windows manifest path: ' + relative)
    return path


def manifest_at(directory, run_id):
    directory = _absolute(directory)
    if not isinstance(run_id, str) or re.fullmatch('[0-9a-f]{64}', run_id) is None:
        raise ValueError('Expected a frozen SHA256 run ID')
    raw = (directory / 'run_manifest.json').read_bytes()
    manifest = json.loads(raw.decode('utf-8-sig'))
    if manifest.get('complete') is not True or manifest.get('run_fingerprint') != run_id:
        raise FileExistsError('Different or incomplete publication run: ' + str(directory))
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Missing publication checksums')
    seen = set()
    for relative, expected in files.items():
        _validate_relative(relative)
        key = relative.casefold()
        if key in seen:
            raise ValueError('Duplicate Windows manifest path: ' + relative)
        seen.add(key)
        if (not isinstance(expected, dict)
                or not isinstance(expected.get('sha256'), str)
                or re.fullmatch('[0-9a-f]{64}', expected['sha256']) is None
                or isinstance(expected.get('size_bytes'), bool)
                or not isinstance(expected.get('size_bytes'), int) or expected['size_bytes'] < 0):
            raise ValueError('Malformed publication checksum: ' + relative)
    for relative in files:
        if any(str(parent).casefold() in seen for parent in PurePosixPath(relative).parents if str(parent) != '.'):
            raise ValueError('Manifest file/directory collision: ' + relative)
    return manifest, hashlib.sha256(raw).hexdigest()


def _verify_file(path, expected):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b''):
            size += len(block)
            digest.update(block)
    if size != expected['size_bytes'] or digest.hexdigest() != expected['sha256']:
        raise ValueError('Publication size/SHA256 differs: ' + str(path))


def verify_delivery(directory, run_id, workers=16, expected_manifest_sha=None,
                    progress=None, phase='verify'):
    directory = _absolute(directory)
    manifest, digest = manifest_at(directory, run_id)
    if expected_manifest_sha is not None and digest != expected_manifest_sha:
        raise ValueError('Publication manifest checksum differs')
    def check(item):
        relative, expected = item
        _verify_file(directory / relative, expected)
    _parallel(manifest['files'].items(), check, workers, progress, phase, path=str(directory))
    _, final_digest = manifest_at(directory, run_id)
    if final_digest != digest:
        raise RuntimeError('Publication manifest changed during verification')
    return manifest


def _check_staging(staging, output):
    staging, output = _absolute(staging), _absolute(output)
    if (_key(staging.parent) != _key(output.parent)
            or re.fullmatch(re.escape(output.name) + r'\.partial-[0-9a-f]{32}', staging.name) is None):
        raise ValueError('Staging must be an exact output sibling named output.partial-<32 hex>')
    if staging.is_symlink():
        raise ValueError('Staging must not be a symbolic link')
    return staging


def _check_directory(path):
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode)
            or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)):
        raise ValueError('Staging directory must be a regular directory: ' + str(path))


def _prepare_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    _check_directory(path)


def _copy_file(local, staging, relative):
    """Always recopy; atomic replacement avoids trusting existing staged files."""
    source, target = local / relative, staging / relative
    temporary = target.with_name(target.name + '.copy-' + uuid.uuid4().hex)
    # Exclusive temporary creation and replacement also avoid following a
    # pre-existing destination-file symlink. No source metadata is modified.
    with source.open('rb') as incoming, temporary.open('xb') as outgoing:
        for block in iter(lambda: incoming.read(BLOCK_BYTES), b''):
            outgoing.write(block)
    os.replace(temporary, target)


def publish_delivery(local, output, run_id, workers=16, staging=None, progress=None):
    """Publish or resume a precise staging sibling; return published/already_verified.

    Progress is a callable accepting dictionaries with phase/completed/total,
    plus path or stage_path. Existing staged files are never trusted or skipped.
    On any failure staging is preserved, and no final rename is attempted.
    """
    workers = _workers(workers)
    local, output = _absolute(local), _absolute(output)
    if (_key(local) == _key(output) or _key(local) in {_key(p) for p in output.parents}
            or _key(output) in {_key(p) for p in local.parents}):
        raise ValueError('Local delivery and output must be separate nonoverlapping trees')
    if staging is not None:
        staging = _check_staging(staging, output)
        if (_key(local) == _key(staging) or _key(local) in {_key(p) for p in staging.parents}
                or _key(staging) in {_key(p) for p in local.parents}):
            raise ValueError('Local delivery and staging must not overlap')
    _, digest = manifest_at(local, run_id)
    manifest = verify_delivery(local, run_id, workers, digest, progress, 'verify_local')
    if output.exists():
        verify_delivery(output, run_id, workers, digest, progress, 'verify_existing')
        _notify(progress, 'complete', 1, 1, output=str(output), result='already_verified')
        return 'already_verified'
    output.parent.mkdir(parents=True, exist_ok=True)
    if staging is None:
        staging = output.with_name(output.name + '.partial-' + uuid.uuid4().hex)
        staging.mkdir(exist_ok=False)
    else:
        staging.mkdir(exist_ok=True)
        old_manifest = staging / 'run_manifest.json'
        if old_manifest.exists():
            _, old_digest = manifest_at(staging, run_id)
            if old_digest != digest:
                raise FileExistsError('Existing staging belongs to a different manifest')
    _check_directory(staging)
    entries = list(manifest['files']) + ['run_manifest.json']
    directories = {parent for relative in entries for parent in Path(relative).parents if str(parent) != '.'}
    # Process equal-depth directories concurrently. Each parent is checked
    # before children are created, preventing writes through existing links.
    offset = 0
    for depth in sorted({len(path.parts) for path in directories}):
        level = sorted((path for path in directories if len(path.parts) == depth), key=str)
        _parallel(level, lambda relative: _prepare_directory(staging / relative),
                  workers, progress, 'prepare_directories', offset=offset,
                  phase_total=len(directories), stage_path=str(staging))
        offset += len(level)
    _parallel(entries, lambda relative: _copy_file(local, staging, relative),
              workers, progress, 'copy_files', stage_path=str(staging))
    verify_delivery(staging, run_id, workers, digest, progress, 'verify_staging')
    # Explicitly reject existing outputs, including a destination created while
    # copying. Windows os.rename additionally refuses an existing directory.
    if output.exists():
        raise FileExistsError('Final destination appeared while publishing: ' + str(output))
    _notify(progress, 'rename', 0, 1, stage_path=str(staging), output=str(output))
    os.rename(staging, output)
    _notify(progress, 'rename', 1, 1, output=str(output))
    verify_delivery(output, run_id, workers, digest, progress, 'verify_final')
    _notify(progress, 'complete', 1, 1, output=str(output), result='published')
    return 'published'
