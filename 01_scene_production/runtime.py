"""Read-only CPU/memory discovery and conservative bounded scheduling.

Never changes environment, process priority or affinity. Windows 11 / Server
2022 default multigroup scheduling is a capability, not proof of both-socket
workload utilization. See https://learn.microsoft.com/en-us/windows/win32/procthread/processor-groups
"""
from __future__ import annotations
import base64
import ctypes
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
try:
    import psutil
except ImportError:
    psutil = None


def _groups():
    result = dict(processor_groups=[], process_groups=None,
                  current_thread_group=None, current_processor=None,
                  group_query_error=None)
    if os.name != 'nt':
        return result
    try:
        from ctypes import wintypes as wt
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.GetActiveProcessorGroupCount.argtypes = []
        k.GetActiveProcessorGroupCount.restype = wt.WORD
        k.GetActiveProcessorCount.argtypes = [wt.WORD]
        k.GetActiveProcessorCount.restype = wt.DWORD
        n = k.GetActiveProcessorGroupCount()
        if not n:
            raise ctypes.WinError(ctypes.get_last_error())
        result['processor_groups'] = [dict(group=i, logical_cpus=int(k.GetActiveProcessorCount(i))) for i in range(n)]
        if any(g['logical_cpus'] == 0 for g in result['processor_groups']):
            raise RuntimeError('GetActiveProcessorCount returned zero')
        k.GetCurrentProcess.argtypes = []
        k.GetCurrentProcess.restype = wt.HANDLE
        k.GetProcessGroupAffinity.argtypes = [wt.HANDLE, ctypes.POINTER(wt.WORD), ctypes.POINTER(wt.WORD)]
        k.GetProcessGroupAffinity.restype = wt.BOOL
        a = (wt.WORD * n)()
        length = wt.WORD(n)
        if k.GetProcessGroupAffinity(k.GetCurrentProcess(), ctypes.byref(length), a):
            result['process_groups'] = list(a[:length.value])
        class GA(ctypes.Structure):
            _fields_ = [('Mask', ctypes.c_size_t), ('Group', wt.WORD), ('Reserved', wt.WORD * 3)]
        k.GetCurrentThread.argtypes = []
        k.GetCurrentThread.restype = wt.HANDLE
        k.GetThreadGroupAffinity.argtypes = [wt.HANDLE, ctypes.POINTER(GA)]
        k.GetThreadGroupAffinity.restype = wt.BOOL
        ga = GA()
        if k.GetThreadGroupAffinity(k.GetCurrentThread(), ctypes.byref(ga)):
            result['current_thread_group'] = int(ga.Group)
            result['current_thread_affinity_mask_hex'] = hex(ga.Mask)
        class PN(ctypes.Structure):
            _fields_ = [('Group', wt.WORD), ('Number', wt.BYTE), ('Reserved', wt.BYTE)]
        k.GetCurrentProcessorNumberEx.argtypes = [ctypes.POINTER(PN)]
        k.GetCurrentProcessorNumberEx.restype = None
        pn = PN()
        k.GetCurrentProcessorNumberEx(ctypes.byref(pn))
        result['current_processor'] = dict(group=int(pn.Group), number=int(pn.Number))
    except (OSError, AttributeError, RuntimeError) as exc:
        result['group_query_error'] = str(exc)
    return result


@lru_cache(maxsize=1)
def _cim():
    if os.name != 'nt':
        return {}
    shell = Path(os.environ.get('SystemRoot', r'C:\Windows')) / r'System32\WindowsPowerShell\v1.0\powershell.exe'
    command = ('[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;'
               '$taskProcessors=@(Get-CimInstance Win32_Processor -ErrorAction Stop | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors);'
               '$taskOS=Get-CimInstance Win32_OperatingSystem -ErrorAction Stop | Select-Object Caption,BuildNumber,ProductType;'
               '$taskJSON=[pscustomobject]@{processors=$taskProcessors;os=$taskOS} | ConvertTo-Json -Depth 4 -Compress;'
               '[Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($taskJSON))')
    try:
        p = subprocess.run([str(shell), '-NoProfile', '-NonInteractive', '-Command', command],
                           capture_output=True, text=True, encoding='utf-8-sig', errors='replace',
                           timeout=8, check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return json.loads(base64.b64decode(p.stdout.strip()).decode('utf-8-sig'))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {'cim_query_error': str(exc)}


def machine_info():
    """Group topology is not socket topology; reported placement is a snapshot."""
    groups = _groups()
    cim = _cim()
    processors = cim.get('processors') or []
    if isinstance(processors, dict):
        processors = [processors]
    group_total = sum(g['logical_cpus'] for g in groups['processor_groups'])
    cim_total = sum(p.get('NumberOfLogicalProcessors') or 0 for p in processors)
    logical = group_total or cim_total or os.cpu_count() or 1
    physical = sum(p.get('NumberOfCores') or 0 for p in processors) or None
    if physical is None and psutil is not None:
        physical = psutil.cpu_count(logical=False)
    win = cim.get('os') or {}
    build = int(win.get('BuildNumber') or 0)
    product = int(win.get('ProductType') or 0)
    multigroup = ((product == 1 and build >= 22000) or (product in (2, 3) and build >= 20348)) if build and product else None
    total = available = None
    if psutil is not None:
        mem = psutil.virtual_memory()
        total, available = int(mem.total), int(mem.available)
    warnings = []
    if len(groups['processor_groups']) > 1:
        warnings.append('Groups do not identify sockets. Actual both-socket workload utilization is unverified; affinity is unchanged.')
        if multigroup is not True:
            warnings.append('OS default cross-group scheduling is unconfirmed; the requested CPU budget may be unattainable.')
    if groups.get('group_query_error'):
        warnings.append(groups['group_query_error'])
    if cim.get('cim_query_error'):
        warnings.append(cim['cim_query_error'])
    if total is None:
        warnings.append('Memory discovery unavailable: psutil is not installed.')
    return dict(platform=platform.platform(), logical_cpus=int(logical), physical_cores=physical,
                sockets=len(processors) if processors else None, processors=processors,
                logical_cpu_source='Windows processor groups' if group_total else ('CIM' if cim_total else 'os.cpu_count'),
                os_cpu_count=os.cpu_count(), **groups,
                processor_group_count=len(groups['processor_groups']) or None,
                windows_os=win, os_default_multigroup_scheduling=multigroup,
                affinity_policy='unchanged_os_scheduler', dual_socket_utilization_verified=False,
                memory_total_bytes=total, memory_available_bytes=available, warnings=warnings)


def choose_parallelism(task_count: int, target: float = .6):
    """Plan a thread budget, never promise an actual measured CPU percentage.

    80 logical CPUs with >=24 tasks -> 24 workers x 2 model threads.
    Caller bounds array memory, BLAS/OpenMP threads and network I/O separately.
    """
    if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count < 0:
        raise ValueError('task_count must be a non-negative integer')
    if not math.isfinite(target) or not .5 <= target <= .7:
        raise ValueError('target must be a finite fraction between 0.5 and 0.7')
    info = machine_info()
    logical = info['logical_cpus']
    budget = max(1, math.floor(logical * target))
    threads = 2 if budget >= 2 else 1
    workers = min(task_count, max(1, budget // threads)) if task_count else 0
    planned = workers * threads
    io = min(workers, 4, max(1, (info['sockets'] or 1) * 2))
    available = info['memory_available_bytes']
    warnings = list(info['warnings'])
    if planned < budget:
        warnings.append('Too few tasks or integer rounding to reach the requested CPU thread budget.')
    return dict(workers=workers, model_threads=threads, io_slots=io,
                task_count=task_count, target=float(target), logical_cpus=logical,
                physical_cores=info['physical_cores'], sockets=info['sockets'],
                processor_group_count=info['processor_group_count'],
                cpu_thread_budget=budget, planned_model_threads=planned,
                planned_logical_cpu_fraction=planned/logical,
                measured_cpu_utilization_guaranteed=False,
                memory_budget_bytes=int(available*.75) if available else None,
                memory_budget_note='Caller must bound blocks and workers to this budget.',
                affinity_policy='unchanged_os_scheduler',
                dual_socket_utilization_verified=False, warnings=warnings)


def init_affinity(workerordinal: int):
    """Inspect worker placement, leaving all present/future threads unrestricted.

    An initializer-thread-only binding would not prove sklearn thread placement.
    GetThreadGroupAffinity on newer Windows reports the primary group; a single
    reported group is not proof that default scheduling is single-group.
    """
    if isinstance(workerordinal, bool) or not isinstance(workerordinal, int) or workerordinal < 0:
        raise ValueError('workerordinal must be a non-negative integer')
    return dict(worker_ordinal=workerordinal, pid=os.getpid(), affinity_changed=False,
                policy='unchanged_os_scheduler', dual_socket_utilization_verified=False, **_groups())


def cpu_throttle(ceiling: float = 70, max_wait: float = 1.0):
    """Bounded between-block backoff; proceed even if other apps remain busy.

    Do not loop on timed_out. This is cooperative scheduling, not a hard cap.
    """
    if not math.isfinite(ceiling) or not 0 < ceiling <= 100:
        raise ValueError('ceiling must be between 0 and 100')
    if not math.isfinite(max_wait) or not 0 <= max_wait <= 5:
        raise ValueError('max_wait must be between 0 and 5 seconds')
    started = time.monotonic()
    observed = None
    timed_out = False
    if psutil is not None and max_wait:
        deadline = started + max_wait
        while True:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            observed = psutil.cpu_percent(interval=min(.1, remaining))
            if observed <= ceiling:
                break
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            time.sleep(min(.15, remaining))
    return dict(proceed=True, observed_system_cpu_percent=observed, ceiling=ceiling,
                waited_seconds=round(time.monotonic()-started, 4), timed_out=timed_out,
                monitor_available=psutil is not None)


def read_cpu_usage() -> dict:
    """Read Windows group-aware PercentProcessorTime, with a bounded query.

    psutil 5.9 may report zero values for a whole processor group. On Windows
    this function therefore never substitutes psutil values after a CIM failure.
    A failed/incomplete sample is explicit and must not be interpreted as 0%.
    Groups are not socket IDs. This is system telemetry, not attribution of CPU
    activity to this pipeline. No affinity or process settings are changed.
    """
    started = time.monotonic()
    result = dict(source='Win32_PerfFormattedData_Counters_ProcessorInformation',
                  metric='PercentProcessorTime', available=False, complete=False,
                  total_percent=None, groups=[], per_group_percent={},
                  per_cpu=[], per_cpu_percent=[], elapsed_seconds=None,
                  error=None, affinity_changed=False)
    if os.name != 'nt':
        if psutil is not None:
            values = list(psutil.cpu_percent(interval=.1, percpu=True))
            result.update(source='psutil_non_windows', available=bool(values),
                          complete=bool(values), per_cpu_percent=values,
                          total_percent=sum(values)/len(values) if values else None)
        else:
            result['error'] = 'psutil unavailable on non-Windows host'
        result['elapsed_seconds'] = round(time.monotonic()-started, 4)
        return result
    shell = Path(os.environ.get('SystemRoot', r'C:\Windows')) / r'System32\WindowsPowerShell\v1.0\powershell.exe'
    command = ("$ErrorActionPreference='Stop';"
               '$taskRows=@(Get-CimInstance -ClassName Win32_PerfFormattedData_Counters_ProcessorInformation '
               '-OperationTimeoutSec 5 -ErrorAction Stop | Select-Object Name,PercentProcessorTime);'
               '$taskJSON=ConvertTo-Json -InputObject $taskRows -Compress;'
               '[Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($taskJSON))')
    try:
        completed = subprocess.run(
            [str(shell), '-NoProfile', '-NonInteractive', '-Command', command],
            capture_output=True, text=True, encoding='utf-8-sig', errors='replace',
            timeout=7.0, check=True,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        rows = json.loads(base64.b64decode(completed.stdout.strip()).decode('utf-8-sig'))
        if not isinstance(rows, list) or not rows:
            raise ValueError('CIM returned no processor counter instances')
        counters = {}
        cpus = {}
        group_totals = {}
        for row in rows:
            name = str(row['Name'])
            raw = row.get('PercentProcessorTime')
            if raw is None:
                raise ValueError('Missing PercentProcessorTime for '+name)
            value = float(raw)
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError('Invalid PercentProcessorTime for '+name+': '+str(raw))
            if name in counters:
                raise ValueError('Duplicate processor counter instance: '+name)
            counters[name] = value
            if name == '_Total':
                continue
            parts = name.split(',')
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            group = int(parts[0])
            if parts[1] == '_Total':
                group_totals[group] = value
            elif parts[1].isdigit():
                cpus[(group,int(parts[1]))] = value
        if not cpus:
            raise ValueError('CIM returned no group,processor instances')
        topology = _groups()
        expected = {g['group']:g['logical_cpus'] for g in topology['processor_groups']}
        observed = sorted({g for g,_ in cpus})
        groups = []
        for group in observed:
            values = [v for (g,_),v in cpus.items() if g == group]
            native_total = group_totals.get(group)
            groups.append(dict(group=group, percent=sum(values)/len(values),
                               logical_cpus=len(values), counter_instance='mean_of_group_cpus',
                               native_total_percent=native_total,
                               per_cpu_mean_percent=sum(values)/len(values)))
        ordered = [dict(group=g, number=n, percent=cpus[(g,n)]) for g,n in sorted(cpus)]
        native_total = counters.get('_Total')
        # Some formatted _Total instances lag or disagree with their CPU rows.
        # Aggregate the complete, explicit group/CPU sample with equal weights.
        total = sum(cpus.values())/len(cpus)
        total_source = 'mean_of_all_observed_cpus'
        complete = bool(expected) and set(expected) == set(observed) and all(
            {n for g,n in cpus if g == group} == set(range(count))
            for group,count in expected.items())
        result.update(available=True, complete=complete, total_percent=total,
                      total_counter_instance=total_source, native_total_percent=native_total,
                      native_aggregate_discrepancy_percent=(native_total-total) if native_total is not None else None, groups=groups,
                      per_group_percent={str(g['group']):g['percent'] for g in groups},
                      per_cpu=ordered, per_cpu_percent=[c['percent'] for c in ordered],
                      expected_processor_groups=topology['processor_groups'],
                      observed_group_ids=observed, instance_count=len(counters),
                      group_topology_error=topology['group_query_error'])
        if not complete:
            result['error'] = 'Counter sample does not cover the independently queried active processor topology'
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
    result['elapsed_seconds'] = round(time.monotonic()-started, 4)
    return result
