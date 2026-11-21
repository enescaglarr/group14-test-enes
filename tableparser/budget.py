"""Per-table time budget (official limit: 15 s per table).

parse_table() starts the clock; the expensive optional work (second reader,
retries at other scales, per-phrase re-reads) checks expired() and is skipped
once the budget is used up, so a hard table still returns its best result
in time instead of timing out with nothing.
"""
import time

TABLE_BUDGET_S = 12.0
_deadline = None
_skipped = 0


def start(seconds: float = TABLE_BUDGET_S):
    global _deadline, _skipped
    _deadline = time.monotonic() + seconds
    _skipped = 0


def expired() -> bool:
    return _deadline is not None and time.monotonic() > _deadline


def skip():
    """Count one skipped optional step (reported in meta)."""
    global _skipped
    _skipped += 1


def skipped() -> int:
    return _skipped
