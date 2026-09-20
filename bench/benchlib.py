"""Shared helpers for the benchmarks: timing, percentiles, RSS and threshold checking.

Dependency-free on purpose (standard library only), so the benches run anywhere the package
runs and add nothing to the runtime dependency set.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "ROOT",
    "check_thresholds",
    "machine",
    "percentile",
    "print_table",
    "rss_mb",
    "timed",
    "write_results",
]

ROOT = Path(__file__).resolve().parents[1]


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile (no interpolation, no numpy). ``fraction`` is in [0, 1]."""
    if not values:
        return 0.0
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be within [0, 1]")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(fraction * len(ordered) + 0.5)))
    return float(ordered[rank - 1])


def timed(function: Callable[[], object], runs: int, warmup: int = 5) -> dict[str, float]:
    """Run ``function`` ``runs`` times and report p50/p95 in microseconds."""
    for _ in range(warmup):
        function()
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1_000.0)
    return {
        "runs": float(runs),
        "p50_us": round(percentile(samples, 0.50), 1),
        "p95_us": round(percentile(samples, 0.95), 1),
    }


class _WindowsCounters(ctypes.Structure):
    _fields_ = (
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    )


def rss_mb() -> float:
    """Resident set size of this process in MiB (0.0 when the platform will not say).

    Windows reports the current working set through psapi and Linux through /proc/self/statm;
    both are good enough for the overload bench's memory ceiling. Other platforms report 0.0,
    and the threshold check then passes trivially -- the number is only meaningful where the
    README says it was measured.
    """
    if os.name == "nt":
        counters = _WindowsCounters()
        counters.cb = ctypes.sizeof(_WindowsCounters)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi")
        handle = ctypes.c_void_p(kernel32.GetCurrentProcess())
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return round(counters.WorkingSetSize / 1024 / 1024, 1) if ok else 0.0
    return _proc_rss_mb()


def _proc_rss_mb() -> float:
    """Linux RSS from /proc/self/statm. Platforms without it report 0.0 (documented)."""
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
    except (OSError, IndexError, ValueError):
        return 0.0
    sysconf = getattr(os, "sysconf", None)
    page_size = int(sysconf("SC_PAGE_SIZE")) if sysconf is not None else 4096
    return round(pages * page_size / 1024 / 1024, 1)


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def write_results(path: Path, results: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"machine": machine(), "results": results}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print("| " + " | ".join(str(cell) for cell in row) + " |")


def check_thresholds(
    results: Mapping[str, Mapping[str, float]], thresholds_path: Path
) -> list[str]:
    """Compare results with a thresholds file.

    A threshold is either a number (an upper bound, kept for the baseline benchmarks) or an
    object with ``max`` and/or ``min`` keys, for metrics where more is better (goodput).
    """
    thresholds: dict[str, dict[str, Any]] = json.loads(thresholds_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for name, limits in thresholds.items():
        if name not in results:
            problems.append(f"{name}: benchmark missing")
            continue
        for metric, limit in limits.items():
            if metric not in results[name]:
                problems.append(f"{name}.{metric}: metric missing")
                continue
            value = float(results[name][metric])
            bounds = limit if isinstance(limit, dict) else {"max": limit}
            upper = bounds.get("max")
            lower = bounds.get("min")
            if upper is not None and value > float(upper):
                problems.append(f"{name}.{metric} = {value} exceeds max {upper}")
            if lower is not None and value < float(lower):
                problems.append(f"{name}.{metric} = {value} is below min {lower}")
    return problems
