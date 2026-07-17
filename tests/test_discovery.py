"""Tests for :mod:`folderhistory.core.discovery`."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from folderhistory.core.discovery import SnapshotDir, discover_snapshots


# ── Helpers ──────────────────────────────────────────────────────────────────


def _can_chmod() -> bool:
    """Check whether running as root (where chmod restrictions don't apply)."""
    try:
        return os.geteuid() != 0
    except AttributeError:
        return True


def _make_file(path: Path) -> None:
    """Create a file at *path* with all parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text("data")


# ── Empty / Non-existent root ────────────────────────────────────────────────


class TestEdgeRoot:
    """Edge cases for the root argument."""

    def test_nonexistent_root(self: TestEdgeRoot, tmp_path: Path) -> None:
        assert discover_snapshots(tmp_path / "nonexistent") == []

    def test_file_as_root(self: TestEdgeRoot, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        _ = f.write_text("hello")
        assert discover_snapshots(f) == []

    def test_empty_directory(self: TestEdgeRoot, tmp_path: Path) -> None:
        assert discover_snapshots(tmp_path) == []


# ── max_depth=0 ──────────────────────────────────────────────────────────────


class TestMaxDepth0:
    """``max_depth=0`` — root only."""

    def test_root_with_files(self: TestMaxDepth0, tmp_path: Path) -> None:
        _make_file(tmp_path / "a.txt")
        result = discover_snapshots(tmp_path, max_depth=0)
        assert len(result) == 1
        assert result[0].path == tmp_path.resolve()
        assert result[0].relative_id == "."

    def test_root_with_only_subdirs(
        self: TestMaxDepth0, tmp_path: Path,
    ) -> None:
        (tmp_path / "sub").mkdir()
        assert discover_snapshots(tmp_path, max_depth=0) == []

    def test_root_with_mixed(self: TestMaxDepth0, tmp_path: Path) -> None:
        _make_file(tmp_path / "a.txt")
        (tmp_path / "sub").mkdir()
        result = discover_snapshots(tmp_path, max_depth=0)
        assert len(result) == 1
        assert result[0].relative_id == "."

    def test_root_empty(self: TestMaxDepth0, tmp_path: Path) -> None:
        assert discover_snapshots(tmp_path, max_depth=0) == []


# ── max_depth=1 (flat, matches current app.py:178) ──────────────────────────


class TestMaxDepth1Flat:
    """``max_depth=1`` — flat, matches current ``app.py:178``."""

    def test_immediate_subdirs(self: TestMaxDepth1Flat, tmp_path: Path) -> None:
        (tmp_path / "v1").mkdir()
        (tmp_path / "v2").mkdir()
        result = discover_snapshots(tmp_path, max_depth=1)
        assert len(result) == 2
        assert result[0].relative_id == "v1"
        assert result[1].relative_id == "v2"

    def test_empty_subdir(self: TestMaxDepth1Flat, tmp_path: Path) -> None:
        """Empty subdirectory is still returned (matches current behaviour)."""
        (tmp_path / "empty").mkdir()
        result = discover_snapshots(tmp_path, max_depth=1)
        assert len(result) == 1
        assert result[0].relative_id == "empty"

    def test_subdir_with_files(self: TestMaxDepth1Flat, tmp_path: Path) -> None:
        d = tmp_path / "v1"
        d.mkdir()
        _make_file(d / "f.txt")
        result = discover_snapshots(tmp_path, max_depth=1)
        assert len(result) == 1
        assert result[0].relative_id == "v1"

    def test_mixed_subdir(self: TestMaxDepth1Flat, tmp_path: Path) -> None:
        """Subdir with both files and subdirectories (flat = still returned)."""
        d = tmp_path / "v1"
        d.mkdir()
        _make_file(d / "f.txt")
        (d / "sub").mkdir()
        result = discover_snapshots(tmp_path, max_depth=1)
        assert len(result) == 1
        assert result[0].relative_id == "v1"

    def test_container_subdir(self: TestMaxDepth1Flat, tmp_path: Path) -> None:
        """Subdir with only subdirectories (flat = still returned)."""
        d = tmp_path / "v1"
        d.mkdir()
        (d / "inner").mkdir()
        result = discover_snapshots(tmp_path, max_depth=1)
        assert len(result) == 1
        assert result[0].relative_id == "v1"

    def test_sorted_order(self: TestMaxDepth1Flat, tmp_path: Path) -> None:
        for name in ("c", "a", "b"):
            (tmp_path / name).mkdir()
        result = discover_snapshots(tmp_path, max_depth=1)
        assert [s.relative_id for s in result] == ["a", "b", "c"]

    def test_matches_current_behavior(
        self: TestMaxDepth1Flat, tmp_path: Path,
    ) -> None:
        """``discover_snapshots(d, 1)`` must match ``app.py:178``."""
        (tmp_path / "v1").mkdir()
        (tmp_path / "v2").mkdir()
        _make_file(tmp_path / "v1" / "f.txt")

        result = discover_snapshots(tmp_path, max_depth=1)
        current = sorted(
            [d for d in tmp_path.resolve().iterdir() if d.is_dir()],
        )

        result_paths = sorted(s.path for s in result)
        current_paths = [p.resolve() for p in current]
        assert result_paths == current_paths


# ── Recursive (max_depth > 1 or max_depth == -1) ────────────────────────────


class TestRecursive:
    """Recursive discovery (``max_depth > 1`` or ``-1``)."""

    def test_nested_snapshots(self: TestRecursive, tmp_path: Path) -> None:
        _make_file(tmp_path / "project" / "v1" / "f.txt")
        _make_file(tmp_path / "project" / "v2" / "f.txt")
        result = discover_snapshots(tmp_path, max_depth=2)
        assert len(result) == 2
        assert result[0].relative_id == "project/v1"
        assert result[1].relative_id == "project/v2"

    def test_empty_containers_skipped(
        self: TestRecursive, tmp_path: Path,
    ) -> None:
        """Container directories (only subdirs, no files) are not returned."""
        (tmp_path / "project" / "v1").mkdir(parents=True)
        (tmp_path / "project" / "v2").mkdir(parents=True)
        assert discover_snapshots(tmp_path, max_depth=3) == []

    def test_container_recurse_into(
        self: TestRecursive, tmp_path: Path,
    ) -> None:
        """Container directories are recursed into to find snapshots."""
        _make_file(tmp_path / "project" / "v1" / "f.txt")
        _make_file(tmp_path / "project" / "v2" / "f.txt")
        result = discover_snapshots(tmp_path, max_depth=2)
        assert len(result) == 2
        assert {s.relative_id for s in result} == {"project/v1", "project/v2"}

    def test_mixed_dir_is_snapshot(
        self: TestRecursive, tmp_path: Path,
    ) -> None:
        """Mixed directories are snapshots and are not recursed past."""
        d = tmp_path / "project" / "v1"
        d.mkdir(parents=True)
        _make_file(d / "f.txt")
        (d / "sub").mkdir()
        _make_file(d / "sub" / "deep.txt")
        result = discover_snapshots(tmp_path, max_depth=3)
        assert len(result) == 1
        assert result[0].relative_id == "project/v1"

    def test_max_depth_2(self: TestRecursive, tmp_path: Path) -> None:
        _make_file(tmp_path / "a" / "b" / "f.txt")
        result = discover_snapshots(tmp_path, max_depth=2)
        assert len(result) == 1
        assert result[0].relative_id == "a/b"

    def test_max_depth_3(self: TestRecursive, tmp_path: Path) -> None:
        """Depth 2 finds ``a/b``. At depth 3 ``a/b`` is mixed → snapshot stop."""
        _make_file(tmp_path / "a" / "b" / "f.txt")
        result_d2 = discover_snapshots(tmp_path, max_depth=2)
        assert len(result_d2) == 1
        assert result_d2[0].relative_id == "a/b"

        _make_file(tmp_path / "a" / "b" / "c" / "f.txt")
        result_d3 = discover_snapshots(tmp_path, max_depth=3)
        assert len(result_d3) == 1  # a/b is MIXED → snapshot, not recursed
        assert result_d3[0].relative_id == "a/b"

    def test_infinite_depth(self: TestRecursive, tmp_path: Path) -> None:
        _make_file(tmp_path / "a" / "b" / "c" / "f.txt")
        result = discover_snapshots(tmp_path, max_depth=-1)
        assert len(result) == 1
        assert result[0].relative_id == "a/b/c"

    def test_sibling_independence(
        self: TestRecursive, tmp_path: Path,
    ) -> None:
        """Mixed siblings do not block discovery of container siblings."""
        v1 = tmp_path / "v1"
        v1.mkdir()
        _make_file(v1 / "f.txt")
        (v1 / "sub").mkdir()
        _make_file(v1 / "sub" / "deep.txt")

        _make_file(tmp_path / "v2" / "inner" / "f.txt")

        result = discover_snapshots(tmp_path, max_depth=3)
        ids = {s.relative_id for s in result}
        assert "v1" in ids
        assert "v2/inner" in ids
        assert "v1/sub" not in ids

    def test_three_level_nested(
        self: TestRecursive, tmp_path: Path,
    ) -> None:
        """Deep nesting with containers at each level."""
        _make_file(tmp_path / "l1" / "l2" / "l3" / "f.txt")
        result = discover_snapshots(tmp_path, max_depth=4)
        assert len(result) == 1
        assert result[0].relative_id == "l1/l2/l3"


# ── Ordering ─────────────────────────────────────────────────────────────────


class TestOrdering:
    """Results are sorted by ``relative_id``."""

    def test_flat_ordering(self: TestOrdering, tmp_path: Path) -> None:
        for name in ("z", "m", "a"):
            (tmp_path / name).mkdir()
        result = discover_snapshots(tmp_path, max_depth=1)
        assert [s.relative_id for s in result] == ["a", "m", "z"]

    def test_recursive_ordering(
        self: TestOrdering, tmp_path: Path,
    ) -> None:
        _make_file(tmp_path / "b" / "f.txt")
        _make_file(tmp_path / "a" / "v1" / "f.txt")
        _make_file(tmp_path / "a" / "v2" / "f.txt")
        result = discover_snapshots(tmp_path, max_depth=3)
        assert [s.relative_id for s in result] == ["a/v1", "a/v2", "b"]


# ── SnapshotDir dataclass ────────────────────────────────────────────────────


class TestSnapshotDirType:
    """Verify ``SnapshotDir`` is a frozen dataclass with correct fields."""

    def test_frozen(self: TestSnapshotDirType) -> None:
        s = SnapshotDir(path=Path("/a"), relative_id="a")
        with pytest.raises(AttributeError):
            s.path = Path("/b")  # type: ignore[misc]  # pyright: ignore[reportAttributeAccessIssue]

    def test_equality(self: TestSnapshotDirType) -> None:
        a = SnapshotDir(path=Path("/x"), relative_id="x")
        b = SnapshotDir(path=Path("/x"), relative_id="x")
        assert a == b

    def test_hashable(self: TestSnapshotDirType) -> None:
        s = SnapshotDir(path=Path("/x"), relative_id="x")
        assert {s} == {s}


# ── Permission errors ────────────────────────────────────────────────────────


class TestPermissionErrors:
    """Permission-denied directories are skipped gracefully."""

    def test_no_execute_in_recursive(
        self: TestPermissionErrors,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Blocked container subtree is skipped; accessible subtree found."""
        if not _can_chmod():
            pytest.skip("Running as root: permission restrictions bypassed")
        _make_file(tmp_path / "accessible" / "v1" / "f.txt")
        blocked = tmp_path / "blocked"
        blocked.mkdir()
        _make_file(blocked / "v1" / "f.txt")
        blocked.chmod(0o000)
        try:
            with caplog.at_level(logging.WARNING):
                result = discover_snapshots(tmp_path, max_depth=3)
            ids = {s.relative_id for s in result}
            assert ids == {"accessible/v1"}
            assert any(
                "Permission denied" in r.message for r in caplog.records
            )
        finally:
            blocked.chmod(0o755)


# ── relative_id format ───────────────────────────────────────────────────────


class TestRelativeId:
    """Verify ``relative_id`` is a POSIX relative path."""

    def test_root_relative_id(self: TestRelativeId, tmp_path: Path) -> None:
        _make_file(tmp_path / "f.txt")
        result = discover_snapshots(tmp_path, max_depth=0)
        assert result[0].relative_id == "."

    def test_nested_posix_format(
        self: TestRelativeId, tmp_path: Path,
    ) -> None:
        _make_file(tmp_path / "sub" / "v1" / "f.txt")
        result = discover_snapshots(tmp_path, max_depth=3)
        assert result[0].relative_id == "sub/v1"
