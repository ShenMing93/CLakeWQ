"""Operational local-cache runner with batched frozen-source metadata checks.

The numerical engine, input identity and scientific fingerprints are unchanged.
Each lake's source directory is enumerated before and after processing. Only
already-existing local ZIPs matching every frozen size and SHA256 may be read.
No NAS copy or cache repair is attempted by this entry point.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import stat

import daily_collection_pipeline as engine
import daily_safe_runner as safe

_original_process_lake = engine.process_lake
_current_lake = ContextVar('daily_cached_runner_lake', default=None)
_scandir = os.scandir


def _normalized(path):
    """Lexical Windows path comparison; never resolves a NAS path per file."""
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def _frozen_identity(source):
    return tuple(source.get(key) for key in ('path', 'year', 'size_bytes', 'mtime_ns', 'sha256', 'run_fingerprint'))


@dataclass
class LakeGuard:
    lake: str
    directory: Path
    sources: dict
    filenames: dict
    cache_stats: dict = field(default_factory=dict)


def lake_guard(task):
    """Reject ambiguous source routing before any data calculation or resume."""
    lake = task['feature']['properties']['lake_id']
    if not isinstance(lake, str) or not lake or Path(lake).name != lake:
        raise ValueError('Invalid canonical lake ID')
    sources, filenames = {}, {}
    directory = None
    for source in task['sources']:
        path = Path(source['path'])
        year = source['year']
        if not path.is_absolute() or '..' in path.parts or '.' in path.parts:
            raise ValueError('Frozen source path must be absolute and unambiguous')
        if isinstance(year, bool) or not isinstance(year, int) or not 1000 <= year <= 9999:
            raise ValueError('Frozen source year is invalid')
        if path.name != f'{year}.zarr.zip' or path.parent.name != lake:
            raise ValueError('Frozen source path does not match its lake/year')
        if directory is None:
            directory = path.parent
        elif _normalized(directory) != _normalized(path.parent):
            raise ValueError('All frozen lake sources must share one directory')
        if (not isinstance(source['sha256'], str)
                or re.fullmatch('[0-9a-f]{64}', source['sha256']) is None):
            raise ValueError('Frozen source SHA256 is invalid')
        for key in ('size_bytes', 'mtime_ns'):
            if isinstance(source[key], bool) or not isinstance(source[key], int) or source[key] < 0:
                raise ValueError('Frozen source metadata is invalid: ' + key)
        key = _normalized(path)
        if key in sources or path.name in filenames:
            raise ValueError('Duplicate frozen source file')
        sources[key] = dict(source)
        filenames[path.name] = dict(source)
    if directory is None:
        raise ValueError('Lake has no frozen source records')
    return LakeGuard(lake, directory, sources, filenames)


def check_source_directory(guard):
    """One directory enumeration per boundary; DirEntry uses cached NAS metadata."""
    observed = set()
    with _scandir(guard.directory) as entries:
        for entry in entries:
            source = guard.filenames.get(entry.name)
            if source is None:
                continue
            if entry.name in observed:
                raise RuntimeError('Duplicate directory entry for frozen source')
            observed.add(entry.name)
            # follow_symlinks=False rejects source links rather than silently
            # substituting bytes from a different location.
            info = entry.stat(follow_symlinks=False)
            if (not stat.S_ISREG(info.st_mode) or entry.is_symlink()
                    or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)):
                raise RuntimeError('Frozen source is not a regular file: ' + source['path'])
            if (info.st_size, info.st_mtime_ns) != (source['size_bytes'], source['mtime_ns']):
                raise RuntimeError('Frozen source changed during cached processing: ' + source['path'])
    missing = set(guard.filenames) - observed
    if missing:
        raise RuntimeError('Frozen source files absent from lake directory: ' + ', '.join(sorted(missing)))


def _guarded_source(source):
    guard = _current_lake.get()
    if guard is None:
        raise RuntimeError('Source validation attempted outside a guarded lake task')
    frozen = guard.sources.get(_normalized(source.get('path', '')))
    if frozen is None or _frozen_identity(source) != _frozen_identity(frozen):
        raise RuntimeError('Source identity differs from the active frozen lake task')
    return guard, frozen


def _check_cache_stats(guard):
    for path, expected in guard.cache_stats.items():
        info = Path(path).stat()
        if (info.st_size, info.st_mtime_ns) != expected:
            raise RuntimeError('Verified local cache changed while processing: ' + path)


def assert_guarded_source(source):
    """Check task identity and local-cache stability inside the source boundaries."""
    guard, _ = _guarded_source(source)
    if source.get('read_path'):
        key = _normalized(source['read_path'])
        expected = guard.cache_stats.get(key)
        if expected is None:
            raise RuntimeError('Reading an unverified local cache file')
        info = Path(source['read_path']).stat()
        if (info.st_size, info.st_mtime_ns) != expected:
            raise RuntimeError('Verified local cache changed while processing: ' + source['read_path'])


def verified_local_cache(source, cache_root, lake):
    """Hash an existing cache ZIP completely; missing/corrupt data is fatal."""
    guard, frozen = _guarded_source(source)
    if lake != guard.lake:
        raise RuntimeError('Cache request belongs to a different lake')
    target = Path(cache_root) / lake / f'{frozen["year"]}_{frozen["sha256"]}.zarr.zip'
    if not target.is_absolute() or str(target.drive).startswith('\\\\') or '..' in target.parts:
        raise RuntimeError('Cache path must be an absolute local path')
    if _normalized(target) == _normalized(frozen['path']) or _normalized(guard.directory) in [_normalized(p) for p in target.parents]:
        raise RuntimeError('Local cache must be separate from original source files')
    before = target.lstat()  # Missing cache is deliberately not repaired here.
    if (not stat.S_ISREG(before.st_mode) or before.st_size != frozen['size_bytes']
            or getattr(before, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)):
        raise ValueError('Local cache size/type differs from frozen source: ' + str(target))
    actual = engine.hash_file(target)
    after = target.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Local cache changed during SHA256 verification: ' + str(target))
    if actual != frozen['sha256']:
        raise ValueError('Local cache SHA256 differs from frozen source: ' + str(target))
    guard.cache_stats[_normalized(target)] = (after.st_size, after.st_mtime_ns)
    return str(target)


def cached_process_lake(task):
    """A post-check failure prevents this lake from becoming a successful result."""
    if _current_lake.get() is not None:
        raise RuntimeError('Nested cached lake tasks are not supported')
    guard = lake_guard(task)
    check_source_directory(guard)
    token = _current_lake.set(guard)
    try:
        result = _original_process_lake(task)
        _check_cache_stats(guard)
        check_source_directory(guard)
        return result
    finally:
        _current_lake.reset(token)


def make_plan_with_runner_pins(config):
    tasks, run_id, pins = safe.make_plan_with_runner_pin(config)
    path = Path(__file__).resolve()
    pin = engine.pinned(path, engine.hash_file(path))
    pins = list(pins)
    if pin not in pins:
        pins.append(pin)
    return tasks, run_id, pins


def cached_init_worker(*args):
    install()
    safe._original_init_worker(*args)


def install():
    """Install only this daily engine's operational bindings, including spawn."""
    safe.install()
    engine.make_plan = make_plan_with_runner_pins
    engine.init_worker = cached_init_worker
    engine.process_lake = cached_process_lake
    engine.cache_source = verified_local_cache
    engine.assert_unchanged = assert_guarded_source


def main(argv=None):
    install()
    return engine.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
