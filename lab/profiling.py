"""Lightweight timing for every lab function, saved with each experiment.

Decorate a function with @profiled, or wrap a block with `with timed("name"):`.
For a full call-level breakdown, run an experiment with --cprofile.
"""

from __future__ import annotations

import functools
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Iterator

_lock = threading.Lock()
_samples: dict[str, list[float]] = defaultdict(list)


def record(name: str, seconds: float) -> None:
    with _lock:
        _samples[name].append(seconds)


@contextmanager
def timed(name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        record(name, time.perf_counter() - start)


def profiled(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with timed(fn.__qualname__):
            return fn(*args, **kwargs)

    return wrapper


def reset() -> None:
    with _lock:
        _samples.clear()


def snapshot() -> dict[str, dict]:
    """Per-function stats in milliseconds, slowest total first."""
    with _lock:
        items = {name: sorted(values) for name, values in _samples.items()}
    stats = {}
    for name, values in sorted(items.items(), key=lambda item: -sum(item[1])):
        count = len(values)
        total = sum(values)
        stats[name] = {
            "calls": count,
            "total_ms": round(total * 1000, 2),
            "mean_ms": round(total / count * 1000, 3),
            "p95_ms": round(values[min(count - 1, int(count * 0.95))] * 1000, 3),
            "max_ms": round(values[-1] * 1000, 3),
        }
    return stats


def format_table(stats: dict[str, dict]) -> str:
    lines = [f"{'function':<34}{'calls':>9}{'total ms':>12}{'mean ms':>11}{'p95 ms':>11}{'max ms':>11}"]
    for name, row in stats.items():
        lines.append(
            f"{name:<34}{row['calls']:>9}{row['total_ms']:>12.1f}{row['mean_ms']:>11.3f}"
            f"{row['p95_ms']:>11.3f}{row['max_ms']:>11.3f}"
        )
    return "\n".join(lines)
