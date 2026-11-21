"""Per-table time budget (official limit: 15 s per table) and the triage path.

parse_table() starts the clock; the expensive optional work (second reader,
retries at other scales, per-phrase re-reads) checks expired() and is skipped
once the budget is used up, so a hard table still returns its best result
in time instead of timing out with nothing.

Triage (tableparser/triage.py) then picks the path: T0/T1 (clean scans) take
the fast path with a 4 s budget, T2/T3 (photos, faxes) the full path with 12 s.

Large tables (more than LARGE_CELLS grid cells) read every cell once: no second
reader, no retries, no per-phrase re-reads. For large tables only, after HARD_S
seconds in total the remaining cells are not read at all, so a huge table still ends
near 12 s. Normal tables are always read to the end: the official limit is a total
(5 min + 15 s per table), and on a slow machine cutting them costs more than it saves.

TABLEPARSER_TIME_SCALE (measurement only, default 1) stretches every budget by
that factor: with N processes sharing the CPUs, a table gets about the same
amount of work done as in a single-process run. The submission never sets it.
"""
import os
import time

_SCALE = float(os.environ.get("TABLEPARSER_TIME_SCALE", "1") or 1)


def _clock():
    return time.monotonic() / _SCALE

TABLE_BUDGET_S = 12.0
FAST_BUDGET_S = 4.0
HARD_S = 12.0
LARGE_CELLS = 150
_large = False
_unread = 0
_start = None
_deadline = None
_fast = False
_skipped = 0


def start(seconds: float = TABLE_BUDGET_S):
    global _start, _deadline, _skipped, _fast, _large, _unread
    _start = _clock()
    _deadline = _start + seconds
    _skipped = 0
    _fast = False
    _large = False
    _unread = 0


def set_large(n_cells: int):
    """Mark the table as large (first read only) once its grid size is known."""
    global _large
    _large = n_cells > LARGE_CELLS


def large() -> bool:
    return _large


def hard_expired() -> bool:
    """Only large tables have a hard limit (see the module docstring)."""
    return _large and _start is not None and _clock() > _start + HARD_S


def set_path(fast: bool):
    """Choose the path after triage; the budget counts from start()."""
    global _deadline, _fast
    _fast = bool(fast)
    if _start is not None:
        _deadline = _start + (FAST_BUDGET_S if _fast else TABLE_BUDGET_S)


def fast() -> bool:
    return _fast


def expired() -> bool:
    return _deadline is not None and _clock() > _deadline


def skip():
    """Count one skipped optional step (reported in meta)."""
    global _skipped
    _skipped += 1


def skipped() -> int:
    return _skipped


def unread():
    """Count one cell left unread because the hard limit was reached."""
    global _unread
    _unread += 1


def n_unread() -> int:
    return _unread
