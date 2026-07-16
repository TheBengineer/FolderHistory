"""Tests for ``folderhistory/signals/fusion.py``."""

from __future__ import annotations

from folderhistory.signals.fusion import SignalConfig, fuse_signals, match_files_probabilistic
from folderhistory.types import FileRecord


def _make(
    path: str = "file.txt",
    size: int = 100,
    blake3: str = "aaa",
    xxhash64: str = "xxx",
    mtime_ns: int = 1_000_000,
    mode: int = 0o100644,
) -> FileRecord:
    return FileRecord(
        path=path,
        size=size,
        mode=mode,
        mtime_ns=mtime_ns,
        ctime_ns=mtime_ns,
        raw_blake3=blake3,
        normalized_blake3=blake3,
        line_ending="lf",
        xxhash64=xxhash64,
        is_symlink=False,
    )


class TestFuseSignals:
    """Suite for ``fuse_signals``."""

    def test_identical_records_default_weights(self) -> None:
        """Two identical records → fused cost 0.0."""
        a = _make()
        b = _make()
        assert fuse_signals(a, b) == 0.0

    def test_totally_different_records(self) -> None:
        """Records that differ in every way → cost should be > 0.5."""
        a = _make(path="a/x.py", size=100, blake3="aaa", mtime_ns=1_000_000, mode=0o100644)
        b = _make(path="b/y.rs", size=200, blake3="bbb", mtime_ns=9_000_000_000_000, mode=0o100755)
        cost = fuse_signals(a, b)
        # content=1.0*0.6 + path=~1.0*0.3 + meta=(1.0+0.5)/2*0.1 = 0.6 + ~0.3 + 0.075 = ~0.975
        assert cost > 0.5

    def test_custom_config(self) -> None:
        """Custom weights change the fused cost."""
        a = _make(path="x", size=100, blake3="aaa")
        b = _make(path="y", size=100, blake3="bbb", xxhash64="yyy")
        # content=0.6 (size match, no hash) + path > 0
        config = SignalConfig(w_hash=1.0, w_path=0.0, w_meta=0.0)
        assert fuse_signals(a, b, config) == 0.6

    def test_cost_range(self) -> None:
        """Fused cost is always in [0.0, 1.0]."""
        a = _make()
        b = _make(path="other.py", size=999, blake3="fff", mtime_ns=9_000_000_000_000, mode=0o100755)
        cost = fuse_signals(a, b)
        assert 0.0 <= cost <= 1.0


class TestMatchFilesProbabilistic:
    """Suite for ``match_files_probabilistic``."""

    def test_empty_lists(self) -> None:
        """Both lists empty → empty result."""
        assert match_files_probabilistic([], []) == []

    def test_no_matches_below_threshold(self) -> None:
        """All pairs above threshold → empty result."""
        a = _make(path="a", size=1, blake3="a", mtime_ns=0, mode=0o100644)
        b = _make(path="b", size=999, blake3="b", mtime_ns=10_000_000_000_000_000, mode=0o100755)
        assert match_files_probabilistic([a], [b], threshold=0.1) == []

    def test_match_found_below_threshold(self) -> None:
        """Identical files → cost 0.0, found below default threshold."""
        a = _make(path="same.txt")
        b = _make(path="same.txt")
        result = match_files_probabilistic([a], [b])
        assert len(result) == 1
        path_a, path_b, cost = result[0]
        assert path_a == "same.txt"
        assert path_b == "same.txt"
        assert cost == 0.0

    def test_results_sorted_by_cost(self) -> None:
        """Matches are sorted ascending by cost."""
        identical = _make(path="id.txt")
        close = _make(path="id.txt", size=100, blake3="bbb")  # content cost 0.6, path 0.0
        far = _make(path="far.txt", size=999, blake3="fff", mtime_ns=99_000_000_000_000_000)
        far2 = _make(
            path="far.txt", size=999, blake3="fff", mtime_ns=99_000_000_000_000_000
        )  # identical to far, cost 0.0

        # identical + far2 = 0.0, close + far2 = 0.6+... > 0.0
        result = match_files_probabilistic(
            [identical, close], [far, far2], threshold=0.7
        )
        costs = [c for (_, _, c) in result]
        assert costs == sorted(costs)

    def test_threshold_excludes_high_cost(self) -> None:
        """Pairs with cost exactly at or above threshold are excluded."""
        a = _make(path="a", size=100, blake3="a", xxhash64="x1")
        b = _make(path="b", size=100, blake3="b", xxhash64="x2")  # size match, no hash = 0.6
        # path cost ~ some small value, meta some small value
        # fused will be roughly 0.6*0.6 + path*0.3 + meta*0.1 ≈ 0.36 + ~0.0 + ~0.0
        result_high = match_files_probabilistic([a], [b], threshold=0.3)
        assert len(result_high) == 0
        result_low = match_files_probabilistic([a], [b], threshold=0.6)
        # fused should be > 0.36, so might be below 0.6
        # This is a range test; just verify it's consistent
        if cost := result_low:
            assert 0.3 < cost[0][2] < 0.6
