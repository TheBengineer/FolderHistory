"""Metadata-based similarity signals for probabilistic file matching."""

from __future__ import annotations

from folderhistory.types import FileRecord

# Nanoseconds in one hour and one day
_NS_IN_HOUR: int = 3_600_000_000_000
_NS_IN_DAY: int = 86_400_000_000_000


def metadata_similarity(f_a: FileRecord, f_b: FileRecord) -> float:
    """Return a metadata-dissimilarity cost based on mtime and permission bits.

    *mtimd diff*:
      < 1 hour  → 0.0
      < 24 hours → 0.5
      >= 24 hours → 1.0

    *permission match*:
      same bits → 0.0
      different  → 0.5

    Returns the average of both sub-scores.
    """
    # ── mtime proximity ──────────────────────────────────────────────────────
    mtime_delta: int = f_a.mtime_ns - f_b.mtime_ns
    mtime_delta = mtime_delta if mtime_delta >= 0 else -mtime_delta

    if mtime_delta < _NS_IN_HOUR:
        mtime_score: float = 0.0
    elif mtime_delta < _NS_IN_DAY:
        mtime_score = 0.5
    else:
        mtime_score = 1.0

    # ── Permission similarity (lower 12 bits of mode) ────────────────────────
    perm_score: float = 0.0 if (f_a.mode & 0o7777) == (f_b.mode & 0o7777) else 0.5

    return (mtime_score + perm_score) / 2.0
