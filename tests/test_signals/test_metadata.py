"""Tests for ``folderhistory/signals/metadata.py``."""

from __future__ import annotations

from folderhistory.signals.metadata import metadata_similarity
from folderhistory.types import FileRecord

_NS_IN_HOUR: int = 3_600_000_000_000
_NS_IN_DAY: int = 86_400_000_000_000


def _make(mtime_ns: int = 1_000_000, mode: int = 0o100644) -> FileRecord:
    return FileRecord(
        path="f.txt",
        size=100,
        mode=mode,
        mtime_ns=mtime_ns,
        ctime_ns=mtime_ns,
        raw_blake3="aaa",
        normalized_blake3="aaa",
        line_ending="lf",
        xxhash64="xxx",
        is_symlink=False,
    )


class TestMetadataSimilarity:
    """Suite for ``metadata_similarity``."""

    def test_same_mtime_and_perm_returns_zero(self) -> None:
        """Identical metadata → cost 0.0."""
        a = _make()
        b = _make()
        assert metadata_similarity(a, b) == 0.0

    def test_mtime_within_hour_plus_same_perm(self) -> None:
        """mtime diff < 1 h, same perms → cost dominated by mtime=0, perm=0 → 0.0."""
        a = _make(mtime_ns=1_000_000)
        b = _make(mtime_ns=1_800_000_000_000)  # 30 min later
        assert metadata_similarity(a, b) == 0.0

    def test_mtime_between_one_and_24_hours(self) -> None:
        """mtime diff between 1 h and 24 h → mtime_score=0.5, perm same → avg=0.25."""
        a = _make(mtime_ns=1_000_000)
        b = _make(mtime_ns=1_000_000 + _NS_IN_HOUR * 6)  # 6 h later
        assert metadata_similarity(a, b) == 0.25

    def test_mtime_beyond_24_hours(self) -> None:
        """mtime diff >= 24 h → mtime_score=1.0, perm same → avg=0.5."""
        a = _make(mtime_ns=1_000_000)
        b = _make(mtime_ns=1_000_000 + _NS_IN_DAY * 2)  # 2 days later
        assert metadata_similarity(a, b) == 0.5

    def test_different_permissions_adds_cost(self) -> None:
        """Different permissions → perm_score=0.5 adds to total."""
        a = _make(mtime_ns=1_000_000, mode=0o100644)
        b = _make(mtime_ns=1_000_000, mode=0o100755)
        # mtime_score=0, perm_score=0.5 → avg=0.25
        assert metadata_similarity(a, b) == 0.25

    def test_everything_different(self) -> None:
        """mtime >= 24 h + different perms → (1.0 + 0.5)/2 = 0.75."""
        a = _make(mtime_ns=1_000_000, mode=0o100644)
        b = _make(mtime_ns=1_000_000 + _NS_IN_DAY * 3, mode=0o100755)
        assert metadata_similarity(a, b) == 0.75
