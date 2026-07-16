"""Tests for ``folderhistory/signals/path.py``."""

from __future__ import annotations

from folderhistory.signals.path import path_similarity
from folderhistory.types import FileRecord


def _make(path: str) -> FileRecord:
    return FileRecord(
        path=path,
        size=100,
        mode=0o100644,
        mtime_ns=1_000_000,
        ctime_ns=1_000_000,
        raw_blake3="aaa",
        normalized_blake3="aaa",
        line_ending="lf",
        xxhash64="xxx",
        is_symlink=False,
    )


class TestPathSimilarity:
    """Suite for ``path_similarity``."""

    def test_identical_path_returns_zero(self) -> None:
        """Same path → cost 0.0."""
        a = _make("src/main.py")
        b = _make("src/main.py")
        assert path_similarity(a, b) == 0.0

    def test_completely_different_returns_one(self) -> None:
        """Completely different paths → cost 1.0."""
        a = _make("")
        b = _make("zzzz")
        assert path_similarity(a, b) == 1.0

    def test_same_basename_different_dir(self) -> None:
        """Same file name in different directories → cost between 0 and 1."""
        a = _make("src/main.py")
        b = _make("tests/main.py")
        cost = path_similarity(a, b)
        assert 0.0 < cost < 1.0

    def test_similar_paths_have_lower_cost(self) -> None:
        """More similar paths produce a lower cost."""
        near = path_similarity(_make("src/utils/helper.py"), _make("src/utils/helper.ts"))
        far = path_similarity(_make("src/utils/helper.py"), _make("docs/index.md"))
        assert near < far

    def test_empty_strings(self) -> None:
        """Empty string paths."""
        a = _make("")
        b = _make("")
        assert path_similarity(a, b) == 0.0
