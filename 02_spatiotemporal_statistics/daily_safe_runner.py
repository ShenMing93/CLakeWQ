"""Operational entry point for transient Windows JSON replacement locks.

The frozen scientific engine, source identities, lake fingerprints and run ID
are unchanged. Only this engine's JSON writer gains bounded PermissionError
retries. The entry point is pinned separately in the execution/delivery record.
"""
from __future__ import annotations

import os
from pathlib import Path
import time
import uuid

import daily_collection_pipeline as engine

REPLACE_RETRY_SECONDS = 8.0
_replace = os.replace
_clock = time.monotonic
_sleep = time.sleep
_original_make_plan = engine.make_plan
_original_init_worker = engine.init_worker


def safe_write_json(path, value):
    """Keep the original bytes and atomic rename, retrying only transient locks.

    The target is never truncated or written in place. A persistent error is
    raised, leaving the old target and complete temporary file for diagnosis.
    Errors in serialization, creation or writing are not hidden or retried.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    temporary.write_bytes(engine.encoded(value))
    deadline = _clock() + REPLACE_RETRY_SECONDS
    delay = .05
    attempts = 0
    while True:
        attempts += 1
        try:
            _replace(temporary, path)
            return
        except PermissionError as error:
            remaining = deadline - _clock()
            if remaining <= 0:
                error.add_note(
                    f'Atomic JSON replace failed after {attempts} attempts over '
                    f'{REPLACE_RETRY_SECONDS:g}s: {path}; preserved temporary file: {temporary}')
                raise
            _sleep(min(delay, remaining))
            delay = min(delay * 2, .5)


def make_plan_with_runner_pin(config):
    """Add operational provenance without altering the scientific identities."""
    tasks, run_id, pins = _original_make_plan(config)
    runner = Path(__file__).resolve()
    pin = engine.pinned(runner, engine.hash_file(runner))
    pins = list(pins)
    if pin not in pins:
        pins.append(pin)
    return tasks, run_id, pins


def safe_init_worker(*args):
    """Install inside every Windows spawn process before any lake task runs."""
    install()
    _original_init_worker(*args)


def install():
    """Idempotently bind this engine only; legacy modules remain untouched."""
    engine.write_json = safe_write_json
    engine.init_worker = safe_init_worker
    engine.make_plan = make_plan_with_runner_pin


def main(argv=None):
    install()
    return engine.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
