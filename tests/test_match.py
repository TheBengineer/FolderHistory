"""Tests for ``folderhistory.core.match`` — pairwise exact-match engine."""

from __future__ import annotations

from pathlib import Path

from folderhistory.core.match import (
    match_snapshots_exact,
    match_snapshots_exact_all_pairs,
)
from folderhistory.types import FileRecord, Snapshot


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_snapshot(
    snap_id: str,
    file_specs: list[tuple[str, str]],
) -> Snapshot:
    """Create a Snapshot from a list of ``(path, blake3_hash)`` tuples.

    Other ``FileRecord`` fields are filled with test-safe defaults.
    """
    files: list[FileRecord] = []
    for path, blake3 in file_specs:
        files.append(FileRecord(
            path=path,
            size=len(blake3),
            mode=0o100644,
            mtime_ns=1_000_000_000,
            ctime_ns=1_000_000_000,
            raw_blake3=blake3,
            normalized_blake3=blake3,
            line_ending="lf",
            xxhash64=blake3[:16],  # deterministic test stub
            is_symlink=False,
            target_path=None,
        ))
    return Snapshot(
        id=snap_id,
        timestamp=1_000_000_000.0,
        source_path=Path(f"/fake/{snap_id}"),
        files=files,
    )


# ── match_snapshots_exact ─────────────────────────────────────────────────────


class TestMatchSnapshotsExact:
    """Tests for :func:`match_snapshots_exact`."""

    def test_exact_match_two_snaps(self) -> None:
        """Two snapshots with identical files → all hashes matched."""
        snap_a = _make_snapshot("v1", [
            ("a.txt", "hash_aaa"),
            ("b.txt", "hash_bbb"),
        ])
        snap_b = _make_snapshot("v2", [
            ("a.txt", "hash_aaa"),
            ("b.txt", "hash_bbb"),
        ])

        result = match_snapshots_exact([snap_a, snap_b])
        assert len(result) == 1  # one pair

        pair_map = result[0]
        assert pair_map["hash_aaa"] == "default:a.txt"
        assert pair_map["hash_bbb"] == "default:b.txt"

    def test_empty_snapshots_list(self) -> None:
        """Empty list → empty result."""
        assert match_snapshots_exact([]) == []

    def test_single_snapshot(self) -> None:
        """Single snapshot (no pairs) → empty result."""
        snap = _make_snapshot("v1", [("a.txt", "hash_a")])
        assert match_snapshots_exact([snap]) == []

    def test_no_shared_hashes(self) -> None:
        """Two snapshots with completely different content → empty dict."""
        snap_a = _make_snapshot("v1", [("a.txt", "hash_aaa")])
        snap_b = _make_snapshot("v2", [("b.txt", "hash_bbb")])
        result = match_snapshots_exact([snap_a, snap_b])
        assert len(result) == 1
        assert result[0] == {}

    def test_partial_match(self) -> None:
        """Some files match, some don't."""
        snap_a = _make_snapshot("v1", [
            ("shared.txt", "hash_xyz"),
            ("only_a.txt", "hash_aaa"),
        ])
        snap_b = _make_snapshot("v2", [
            ("shared.txt", "hash_xyz"),
            ("only_b.txt", "hash_bbb"),
        ])
        result = match_snapshots_exact([snap_a, snap_b])
        assert len(result) == 1
        # Only hash_xyz is shared.
        assert "hash_xyz" in result[0]
        assert result[0]["hash_xyz"] == "default:shared.txt"
        assert "hash_aaa" not in result[0]
        assert "hash_bbb" not in result[0]

    def test_three_snapshots_two_pairs(self) -> None:
        """Three snapshots → two pairwise dicts."""
        snap_a = _make_snapshot("v1", [("f.txt", "hash_x")])
        snap_b = _make_snapshot("v2", [("f.txt", "hash_x")])
        snap_c = _make_snapshot("v3", [("f.txt", "hash_y")])  # content changed

        result = match_snapshots_exact([snap_a, snap_b, snap_c])
        assert len(result) == 2

        # Pair (v1, v2): hash_x matches
        assert result[0]["hash_x"] == "default:f.txt"
        # Pair (v2, v3): no shared hash (v2 has hash_x, v3 has hash_y)
        assert result[1] == {}

    def test_rename_shared_hash(self) -> None:
        """Same content at different paths → hash appears in both snapshots."""
        snap_a = _make_snapshot("v1", [("old_name.txt", "hash_abc")])
        snap_b = _make_snapshot("v2", [("new_name.txt", "hash_abc")])

        result = match_snapshots_exact([snap_a, snap_b])
        assert len(result) == 1
        # The hash links the two paths.
        assert "hash_abc" in result[0]

    def test_multiple_files_shared_hash(self) -> None:
        """Two files with the same content hash in each snapshot."""
        snap_a = _make_snapshot("v1", [
            ("dup1.txt", "hash_dup"),
            ("dup2.txt", "hash_dup"),
        ])
        snap_b = _make_snapshot("v2", [
            ("dup1.txt", "hash_dup"),
            ("dup2.txt", "hash_dup"),
        ])
        result = match_snapshots_exact([snap_a, snap_b])
        assert len(result) == 1
        assert "hash_dup" in result[0]

    def test_symlink_skipped(self) -> None:
        """Symlinks (empty raw_blake3) are not included in hash matching."""
        snap_a = _make_snapshot("v1", [("real.txt", "hash_real")])
        snap_b = Snapshot(
            id="v2",
            timestamp=2_000_000_000.0,
            source_path=Path("/fake/v2"),
            files=[
                FileRecord(
                    path="link.lnk",
                    size=0,
                    mode=0o120777,
                    mtime_ns=1_000_000_000,
                    ctime_ns=1_000_000_000,
                    raw_blake3="",
                    normalized_blake3=None,
                    line_ending="unknown",
                    xxhash64="",
                    is_symlink=True,
                    target_path="real.txt",
                ),
            ],
        )
        result = match_snapshots_exact([snap_a, snap_b])
        assert len(result) == 1
        assert result[0] == {}  # no hash match for symlink


class TestMatchSnapshotsExactAllPairs:
    """Tests for :func:`match_snapshots_exact_all_pairs`."""

    def test_three_snapshots_three_pairs(self) -> None:
        """Three snapshots → three unordered pairs."""
        snap_a = _make_snapshot("v1", [("f.txt", "hash_x")])
        snap_b = _make_snapshot("v2", [("f.txt", "hash_x")])
        snap_c = _make_snapshot("v3", [("f.txt", "hash_y")])

        result = match_snapshots_exact_all_pairs([snap_a, snap_b, snap_c])
        assert len(result) == 3  # (0,1), (0,2), (1,2)

    def test_all_pairs_rename_detected(self) -> None:
        """Rename is detected across all pairs."""
        snap_a = _make_snapshot("v1", [("old.txt", "hash_abc")])
        snap_b = _make_snapshot("v2", [("new.txt", "hash_abc")])

        result = match_snapshots_exact_all_pairs([snap_a, snap_b])
        assert len(result) == 1
        assert "hash_abc" in result[0]

    def test_all_pairs_empty(self) -> None:
        """Empty list → empty result."""
        assert match_snapshots_exact_all_pairs([]) == []
