"""Tests for ``folderhistory.core.consistency``."""

from __future__ import annotations

from pathlib import Path

from folderhistory.core.consistency import (
    ConflictRecord,
    ConsistencyReport,
    check_consistency,
    check_cross_location_consistency,
    check_with_inner_loop,
)
from folderhistory.types import EditOperation, FileRecord, IdentityCluster, Snapshot


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_file(
    path: str,
    blake3: str,
    *,
    size: int | None = None,
    xxhash64: str | None = None,
) -> FileRecord:
    """Create a :class:`FileRecord` with minimal test defaults."""
    return FileRecord(
        path=path,
        size=size if size is not None else len(blake3),
        mode=0o100644,
        mtime_ns=1_000_000_000,
        ctime_ns=1_000_000_000,
        raw_blake3=blake3,
        normalized_blake3=blake3,
        line_ending="lf",
        xxhash64=xxhash64 or blake3,
        is_symlink=False,
        target_path=None,
    )


def _make_snapshot(snap_id: str, *records: FileRecord) -> Snapshot:
    """Create a :class:`Snapshot` from ``FileRecord`` entries."""
    return Snapshot(
        id=snap_id,
        timestamp=1_000_000_000.0,
        source_path=Path(f"/fake/{snap_id}"),
        files=list(records),
    )


def _most_common_path(observations: list[tuple[str, str]]) -> str | None:
    """Return the most frequent path from observation pairs."""
    counts: dict[str, int] = {}
    for _, p in observations:
        counts[p] = counts.get(p, 0) + 1
    if not counts:
        return None
    best: str | None = None
    best_count = -1
    for path_str, count in counts.items():
        if count > best_count:
            best_count = count
            best = path_str
    return best


def _make_identity(
    uid: str,
    observations: list[tuple[str, str]],
    confidence: float = 1.0,
) -> tuple[str, IdentityCluster]:
    """Build a ``(uid, IdentityCluster)`` pair for inserting into a dict."""
    return (
        uid,
        IdentityCluster(
            uid=uid,
            observations=observations,
            canonical_path=_most_common_path(observations),
            confidence=confidence,
        ),
    )


# ── ConflictRecord ────────────────────────────────────────────────────────────


class TestConflictRecord:
    """Tests for :class:`ConflictRecord`."""

    def test_default_construction(self) -> None:
        """Default-constructed record has sensible defaults."""
        r = ConflictRecord()
        assert r.hash == ""
        assert r.source_paths == []
        assert r.target_paths == []
        assert r.conflict_type == "identity_mismatch"
        assert r.confidence_a == 0.0
        assert r.confidence_b == 0.0

    def test_all_fields_populated(self) -> None:
        """All fields can be populated correctly."""
        r = ConflictRecord(
            hash="abc123",
            source_paths=["/a/file.txt"],
            target_paths=["/b/file.txt"],
            conflict_type="path_mismatch",
            confidence_a=0.8,
            confidence_b=0.2,
        )
        assert r.hash == "abc123"
        assert r.source_paths == ["/a/file.txt"]
        assert r.target_paths == ["/b/file.txt"]
        assert r.conflict_type == "path_mismatch"
        assert r.confidence_a == 0.8
        assert r.confidence_b == 0.2


# ── ConsistencyReport ─────────────────────────────────────────────────────────


class TestConsistencyReport:
    """Tests for :class:`ConsistencyReport`."""

    def test_default_construction(self) -> None:
        """Default-constructed report passes with no conflicts."""
        r = ConsistencyReport()
        assert r.passes is True
        assert r.conflicts == []
        assert r.iterations_used == 0
        assert r.relaxed_params is None

    def test_with_conflicts(self) -> None:
        """Report with conflicts correctly reflects non-passing state."""
        c = ConflictRecord(hash="abc")
        r = ConsistencyReport(passes=False, conflicts=[c], iterations_used=1)
        assert r.passes is False
        assert len(r.conflicts) == 1
        assert r.conflicts[0].hash == "abc"
        assert r.iterations_used == 1


# ── check_consistency ────────────────────────────────────────────────────────


class TestCheckConsistency:
    """Tests for :func:`check_consistency`."""

    def test_clean_data(self) -> None:
        """Clean identity assignments → passes=True."""
        f = _make_file("a.txt", "hash_aaa")
        snap = _make_snapshot("v1", f)
        identities = dict([_make_identity("root_a", [("v1", "a.txt")])])

        report = check_consistency([snap], identities)
        assert report.passes is True
        assert report.conflicts == []

    def test_clean_data_multiple_snapshots(self) -> None:
        """Multiple snapshots with all files covered → passes=True."""
        f1 = _make_file("a.txt", "hash_aaa")
        f2 = _make_file("b.txt", "hash_bbb")
        snap_a = _make_snapshot("v1", f1, f2)
        snap_b = _make_snapshot("v2", f1, f2)
        identities = dict(
            [
                _make_identity("root_a", [("v1", "a.txt"), ("v2", "a.txt")]),
                _make_identity("root_b", [("v1", "b.txt"), ("v2", "b.txt")]),
            ]
        )

        report = check_consistency([snap_a, snap_b], identities)
        assert report.passes is True
        assert report.conflicts == []

    def test_empty_identities(self) -> None:
        """Empty identities dict with non-empty snapshots → conflicts."""
        f = _make_file("a.txt", "hash_aaa")
        snap = _make_snapshot("v1", f)
        identities: dict[str, IdentityCluster] = {}

        report = check_consistency([snap], identities)
        assert report.passes is False
        # One conflict for the uncovered file
        assert len(report.conflicts) == 1
        assert report.conflicts[0].source_paths == ["a.txt"]

    def test_empty_snapshots(self) -> None:
        """Empty snapshots and empty identities → passes=True."""
        report = check_consistency([], {})
        assert report.passes is True
        assert report.conflicts == []

    def test_missing_file_in_identity(self) -> None:
        """File present in snapshot but missing from identities → conflict."""
        f_covered = _make_file("known.txt", "hash_aaa")
        f_missing = _make_file("unknown.txt", "hash_bbb")
        snap = _make_snapshot("v1", f_covered, f_missing)
        identities = dict([_make_identity("root_a", [("v1", "known.txt")])])

        report = check_consistency([snap], identities)
        assert report.passes is False
        assert len(report.conflicts) == 1
        assert report.conflicts[0].source_paths == ["unknown.txt"]

    def test_empty_cluster_observations(self) -> None:
        """Cluster with no observations → conflict."""
        snap = _make_snapshot("v1", _make_file("a.txt", "hash_aaa"))
        identities = {
            "empty_cluster": IdentityCluster(
                uid="empty_cluster",
                observations=[],
                canonical_path=None,
                confidence=0.0,
            ),
        }

        report = check_consistency([snap], identities)
        assert report.passes is False

    def test_with_valid_operations(self) -> None:
        """Valid operations on covered files → passes=True."""
        f = _make_file("a.txt", "hash_aaa")
        snap_a = _make_snapshot("v1", f)
        snap_b = _make_snapshot("v2", _make_file("a.txt", "hash_bbb"))
        identities = dict(
            [
                _make_identity("root_a", [("v1", "a.txt"), ("v2", "a.txt")]),
            ]
        )
        ops: list[list[EditOperation]] = [
            [
                EditOperation(
                    op_type="modify",
                    file_id="a.txt",
                    source_path="a.txt",
                    target_path="a.txt",
                    old_hash="hash_aaa",
                    new_hash="hash_bbb",
                    confidence=0.9,
                ),
            ],
        ]

        report = check_consistency([snap_a, snap_b], identities, operations=ops)
        assert report.passes is True

    def test_invalid_operation_type(self) -> None:
        """Operation with invalid op_type → conflict."""
        snap_a = _make_snapshot("v1", _make_file("a.txt", "hash_aaa"))
        snap_b = _make_snapshot("v2", _make_file("a.txt", "hash_bbb"))
        identities = dict(
            [
                _make_identity("root_a", [("v1", "a.txt"), ("v2", "a.txt")]),
            ]
        )
        # Use a valid literal to construct, then override via __setattr__
        # to bypass the frozen dataclass type constraint.
        op = EditOperation(
            op_type="create",  # placeholder — will be replaced below
            file_id="a.txt",
            source_path="a.txt",
            target_path="b.txt",
            old_hash="hash_aaa",
            new_hash="hash_bbb",
            confidence=0.5,
        )
        object.__setattr__(op, "op_type", "INVALID")
        ops: list[list[EditOperation]] = [[op]]

        report = check_consistency([snap_a, snap_b], identities, operations=ops)
        assert report.passes is False
        conflict = report.conflicts[0]
        assert conflict.conflict_type == "path_mismatch"
        assert conflict.confidence_a == 0.5

    def test_duplicate_file_id_in_pair(self) -> None:
        """Duplicate file_id within a snapshot pair → conflict."""
        snap_a = _make_snapshot("v1", _make_file("a.txt", "hash_aaa"))
        snap_b = _make_snapshot("v2", _make_file("a.txt", "hash_bbb"))
        identities = dict(
            [
                _make_identity("root_a", [("v1", "a.txt"), ("v2", "a.txt")]),
            ]
        )
        ops: list[list[EditOperation]] = [
            [
                EditOperation(
                    op_type="modify",
                    file_id="a.txt",
                    confidence=0.9,
                ),
                EditOperation(
                    op_type="modify",
                    file_id="a.txt",  # duplicate
                    confidence=0.8,
                ),
            ],
        ]

        report = check_consistency([snap_a, snap_b], identities, operations=ops)
        assert report.passes is False
        assert len(report.conflicts) == 1
        assert report.conflicts[0].conflict_type == "temporal_overlap"


# ── check_with_inner_loop ────────────────────────────────────────────────────


class TestCheckWithInnerLoop:
    """Tests for :func:`check_with_inner_loop`."""

    def test_no_conflicts_passes_immediately(self) -> None:
        """No conflicts → passes on first iteration, iterations_used=0."""
        f = _make_file("a.txt", "hash_aaa")
        snap = _make_snapshot("v1", f)
        identities = dict([_make_identity("root_a", [("v1", "a.txt")])])

        report = check_with_inner_loop([snap], identities)
        assert report.passes is True
        assert report.iterations_used == 0
        assert report.relaxed_params is None

    def test_iterations_tracked_when_no_relaxation(self) -> None:
        """Clean data → no iterations needed, iterations_used=0."""
        f = _make_file("a.txt", "hash_aaa")
        snap = _make_snapshot("v1", f)
        identities = dict(
            [
                _make_identity("root_a", [("v1", "a.txt")], confidence=1.0),
            ]
        )

        report = check_with_inner_loop(
            [snap],
            identities,
            max_iterations=2,
            threshold_step=0.05,
        )
        assert report.passes is True
        assert report.iterations_used == 0

    def test_initial_conflict_resolved_by_relaxation(self) -> None:
        """Low-confidence conflicts are filtered after one relaxation."""
        f = _make_file("main.py", "hash_old")
        snap_a = _make_snapshot("v1", f)
        snap_b = _make_snapshot("v2", f)

        # Invalid op_type with very low confidence — will be filtered
        # after threshold relaxation.
        op = EditOperation(
            op_type="create",  # placeholder
            file_id="main.py",
            source_path="main.py",
            target_path="main.py",
            old_hash="hash_old",
            new_hash="hash_old",
            confidence=0.02,
        )
        object.__setattr__(op, "op_type", "INVALID")
        ops: list[list[EditOperation]] = [[op]]

        identities = dict(
            [
                _make_identity("root", [("v1", "main.py"), ("v2", "main.py")]),
            ]
        )

        report = check_with_inner_loop(
            [snap_a, snap_b],
            identities,
            operations=ops,
            max_iterations=2,
            threshold_step=0.05,
        )

        # After 1 iteration: threshold = 1.0 - 0.05 = 0.95
        # Conflict has confidence_a=0.02 which is < 0.95 → filtered → pass
        assert report.iterations_used == 1
        assert report.passes is True
        assert report.relaxed_params is not None
        match_threshold = report.relaxed_params.get("match_threshold")
        assert match_threshold is not None and isinstance(match_threshold, float)
        assert abs(match_threshold - 0.95) < 1e-9

    def test_exhausted_iterations(self) -> None:
        """High-confidence conflicts persist even after full relaxation."""
        f = _make_file("a.txt", "hash_aaa")
        snap_a = _make_snapshot("v1", f)
        snap_b = _make_snapshot("v2", f)

        # Duplicate file_id ops with high confidence
        op1 = EditOperation(
            op_type="modify",
            file_id="a.txt",
            confidence=0.95,
        )
        op2 = EditOperation(
            op_type="modify",
            file_id="a.txt",
            confidence=0.95,
        )
        ops: list[list[EditOperation]] = [[op1, op2]]

        identities = dict(
            [
                _make_identity("root", [("v1", "a.txt"), ("v2", "a.txt")]),
            ]
        )

        report = check_with_inner_loop(
            [snap_a, snap_b],
            identities,
            operations=ops,
            max_iterations=2,
            threshold_step=0.05,
        )

        # After 2 iterations: threshold = 0.90
        # Both conflicts have confidence_a=0.95 which is >= 0.90 → not filtered
        assert report.iterations_used == 2
        assert report.passes is False
        assert report.relaxed_params is not None


# ── check_cross_location_consistency ─────────────────────────────────────────


class TestCheckCrossLocationConsistency:
    """Tests for :func:`check_cross_location_consistency`."""

    def test_no_issues(self) -> None:
        """No cross-location issues → empty list."""
        f1 = _make_file("a.txt", "hash_aaa")
        f2 = _make_file("b.txt", "hash_bbb")
        snap_a = Snapshot(
            id="loc1_v1",
            timestamp=1_000_000_000.0,
            source_path=Path("/root/location1"),
            files=[f1],
        )
        snap_b = Snapshot(
            id="loc2_v1",
            timestamp=2_000_000_000.0,
            source_path=Path("/root/location2"),
            files=[f2],
        )
        project_matches: dict[str, list[str]] = {}

        conflicts = check_cross_location_consistency(
            [snap_a, snap_b], project_matches,
        )
        assert conflicts == []

    def test_path_discontinuity(self) -> None:
        """Same relative path from different source roots → discontinuity."""
        f = _make_file("readme.md", "hash_abc")
        snap_a = Snapshot(
            id="loc1_v1",
            timestamp=1_000_000_000.0,
            source_path=Path("/root/location1"),
            files=[f],
        )
        snap_b = Snapshot(
            id="loc2_v1",
            timestamp=2_000_000_000.0,
            source_path=Path("/root/location2"),
            files=[f],
        )
        project_matches: dict[str, list[str]] = {}

        conflicts = check_cross_location_consistency(
            [snap_a, snap_b], project_matches,
        )
        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.conflict_type == "path_mismatch"
        assert "readme.md" in conflict.target_paths
        assert conflict.confidence_a == 0.5
        assert conflict.confidence_b == 0.5

    def test_temporal_overlap(self) -> None:
        """Matched snapshots with same timestamp → temporal overlap."""
        f1 = _make_file("a.txt", "hash_aaa")
        f2 = _make_file("b.txt", "hash_bbb")
        snap_a = Snapshot(
            id="loc1_v1",
            timestamp=1_000_000_000.0,
            source_path=Path("/root/location1"),
            files=[f1],
        )
        snap_b = Snapshot(
            id="loc2_v1",
            timestamp=1_000_000_000.0,  # same timestamp
            source_path=Path("/root/location2"),
            files=[f2],
        )
        project_matches: dict[str, list[str]] = {
            "loc1_v1": ["loc2_v1"],
        }

        conflicts = check_cross_location_consistency(
            [snap_a, snap_b], project_matches,
        )
        assert len(conflicts) >= 1
        temporal = [c for c in conflicts if c.conflict_type == "temporal_overlap"]
        assert len(temporal) >= 1
        assert temporal[0].confidence_a == 0.3
        assert temporal[0].confidence_b == 0.3

    def test_high_content_overlap_different_paths(self) -> None:
        """Same content hash with different paths → possible duplicate."""
        f1 = _make_file("original.txt", "hash_abc")
        f2 = _make_file("copy.txt", "hash_abc")  # same hash, different path
        snap_a = Snapshot(
            id="loc1_v1",
            timestamp=1_000_000_000.0,
            source_path=Path("/root/location1"),
            files=[f1],
        )
        snap_b = Snapshot(
            id="loc2_v1",
            timestamp=2_000_000_000.0,
            source_path=Path("/root/location2"),
            files=[f2],
        )
        project_matches: dict[str, list[str]] = {}

        conflicts = check_cross_location_consistency(
            [snap_a, snap_b], project_matches,
        )
        identity_conflicts = [
            c for c in conflicts if c.conflict_type == "identity_mismatch"
        ]
        assert len(identity_conflicts) >= 1
        assert identity_conflicts[0].hash == "hash_abc"
        assert "original.txt" in identity_conflicts[0].target_paths
        assert "copy.txt" in identity_conflicts[0].target_paths
