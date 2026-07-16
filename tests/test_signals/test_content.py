"""Tests for ``folderhistory/signals/content.py``."""

from __future__ import annotations

from folderhistory.signals.content import content_similarity
from folderhistory.types import FileRecord


def _make(path: str = "a.txt", size: int = 100, blake3: str = "aaa", xxhash64: str = "xxx") -> FileRecord:
    """Convenience factory for small FileRecord instances."""
    return FileRecord(
        path=path,
        size=size,
        mode=0o100644,
        mtime_ns=1_000_000,
        ctime_ns=1_000_000,
        raw_blake3=blake3,
        normalized_blake3=blake3,
        line_ending="lf",
        xxhash64=xxhash64,
        is_symlink=False,
    )


class TestContentSimilarity:
    """Suite for ``content_similarity``."""

    def test_identical_blake3_returns_zero(self) -> None:
        """Same blake3 → cost 0.0."""
        a = _make(blake3="deadbeef", xxhash64="abc123")
        b = _make(blake3="deadbeef", xxhash64="abc123")
        assert content_similarity(a, b) == 0.0

    def test_different_sizes_returns_one(self) -> None:
        """Different size → cost 1.0 regardless of hashes."""
        a = _make(size=100, blake3="aaa")
        b = _make(size=200, blake3="aaa")
        assert content_similarity(a, b) == 1.0

    def test_size_match_xxhash_match_blake3_differs_returns_three(self) -> None:
        """Same size + same xxhash64 (blake3 differs) → cost 0.3."""
        a = _make(size=100, blake3="aaa", xxhash64="xxx")
        b = _make(size=100, blake3="bbb", xxhash64="xxx")
        assert content_similarity(a, b) == 0.3

    def test_size_match_only_returns_six(self) -> None:
        """Same size, no hash match → cost 0.6."""
        a = _make(size=100, blake3="aaa", xxhash64="xxx")
        b = _make(size=100, blake3="bbb", xxhash64="yyy")
        assert content_similarity(a, b) == 0.6

    def test_identity_is_zero(self) -> None:
        """Same record compared to itself → 0.0."""
        rec = _make()
        assert content_similarity(rec, rec) == 0.0
