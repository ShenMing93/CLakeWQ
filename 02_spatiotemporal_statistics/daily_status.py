"""Read live daily-statistics status with Windows delete-sharing enabled."""
from __future__ import annotations
import argparse, ctypes, json, os
from pathlib import Path
import psutil
BASE = Path(__file__).resolve().parent

def shared_read(path):
    path = Path(path)
    if os.name != 'nt':
        return path.read_text(encoding='utf-8-sig')
    from ctypes import wintypes
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                               wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    api.CreateFileW.restype = wintypes.HANDLE
    api.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                            ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = api.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        pieces = []
        buffer = ctypes.create_string_buffer(65536)
        size = wintypes.DWORD()
        while True:
            if not api.ReadFile(handle, buffer, len(buffer), ctypes.byref(size), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if not size.value:
                break
            pieces.append(buffer.raw[:size.value])
        return b''.join(pieces).decode('utf-8-sig')
    finally:
        api.CloseHandle(handle)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(BASE/'config/daily_all_2016_2025.json'))
    args = parser.parse_args()
    config = json.loads(shared_read(args.config))
    work = Path(config['work_root'])
    result = {'output_root':config['output_root'],
              'available_memory_gib':round(psutil.virtual_memory().available/1024**3, 2),
              'system_cpu_percent':psutil.cpu_percent(interval=1)}
    for filename in ('workflow_progress.json','collection_progress.json','verification_progress.json'):
        path = work/filename
        if not path.exists():
            continue
        data = json.loads(shared_read(path))
        summary = {k:v for k,v in data.items()
                   if k not in ('completed','failed','attempts','pinned_sources','input_pins')}
        for key in ('completed','failed'):
            if key in data:
                value = data[key]
                summary[key+'_count'] = len(value) if isinstance(value,(list,dict)) else value
        if data.get('failed'):
            summary['failures'] = data['failed']
        pid = data.get('pid')
        if pid:
            try:
                process = psutil.Process(pid)
                summary['process_alive'] = bool(process.is_running()
                    and (not data.get('process_create_time') or abs(process.create_time()-data['process_create_time'])<1)
                    and any('daily_' in arg for arg in process.cmdline()))
                if summary['process_alive']:
                    children = process.children(recursive=True)
                    summary['live_child_processes'] = len(children)
                    summary['rss_gib'] = round(sum(p.memory_info().rss for p in [process,*children] if p.is_running())/1024**3,2)
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                summary['process_alive'] = False
        result[filename] = summary
    print(json.dumps(result, ensure_ascii=True, indent=2))

if __name__ == '__main__':
    main()
