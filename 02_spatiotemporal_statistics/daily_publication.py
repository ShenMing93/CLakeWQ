"""Bounded concurrent copying with complete checksums and atomic publication."""
from __future__ import annotations
import hashlib, json, os, shutil, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(block)
    return digest.hexdigest()

def manifest_at(directory, run_id):
    directory = Path(directory)
    path = directory/'run_manifest.json'
    raw = path.read_bytes()
    manifest = json.loads(raw.decode('utf-8-sig'))
    if manifest.get('complete') is not True or manifest.get('run_fingerprint') != run_id:
        raise FileExistsError('Destination has a different or incomplete statistics run: '+str(directory))
    files = manifest.get('files')
    if not isinstance(files,dict) or not files:
        raise ValueError('Missing publication checksums')
    for relative, expected in files.items():
        rel = PurePosixPath(relative)
        if (not relative or rel.is_absolute() or '..' in rel.parts or '\\' in relative
            or ':' in relative or relative=='run_manifest.json'
            or not isinstance(expected.get('sha256'),str) or len(expected['sha256'])!=64
            or not isinstance(expected.get('size_bytes'),int) or expected['size_bytes']<0):
            raise ValueError('Unsafe or malformed manifest entry: '+str(relative))
    return manifest, hashlib.sha256(raw).hexdigest()

def verify_delivery(directory, run_id, workers=4, expected_manifest_sha=None):
    manifest, digest = manifest_at(directory, run_id)
    if expected_manifest_sha is not None and digest != expected_manifest_sha:
        raise ValueError('Publication manifest checksum differs')
    def check(item):
        relative, expected = item
        path = Path(directory)/relative
        if path.stat().st_size != expected['size_bytes'] or sha(path) != expected['sha256']:
            raise ValueError('Publication checksum differs: '+str(path))
    with ThreadPoolExecutor(max_workers=min(4,max(1,workers))) as pool:
        list(pool.map(check,manifest['files'].items()))
    return manifest

def publish_delivery(local, output, run_id, workers=4):
    local, output = Path(local), Path(output)
    _, digest = manifest_at(local, run_id)
    manifest = verify_delivery(local, run_id, workers, digest)
    if output.exists():
        verify_delivery(output, run_id, workers, digest)
        return 'already_verified'
    output.parent.mkdir(parents=True,exist_ok=True)
    staging = output.with_name(output.name+'.partial-'+uuid.uuid4().hex)
    staging.mkdir(exist_ok=False)
    entries = list(manifest['files'])+['run_manifest.json']
    directories = {Path(relative).parent for relative in entries}
    for relative in sorted(directories,key=lambda p:len(p.parts)):
        (staging/relative).mkdir(parents=True,exist_ok=True)
    def copy(relative):
        shutil.copy2(local/relative,staging/relative)
    with ThreadPoolExecutor(max_workers=min(4,max(1,workers))) as pool:
        list(pool.map(copy,entries))
    verify_delivery(staging,run_id,workers,digest)
    # os.rename refuses an existing Windows directory; prior products remain intact.
    os.rename(staging,output)
    verify_delivery(output,run_id,workers,digest)
    return 'published'
