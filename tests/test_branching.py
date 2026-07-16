"""Tests for branching DAG timeline support.

Covers:
- :class:`BranchingNode` and :class:`BranchingTimeline` dataclasses
- :func:`infer_temporal_order` ordering inference
- :func:`build_branching_timeline` DAG construction
- :func:`format_branching_dag` DOT output
"""

from __future__ import annotations

from pathlib import Path

from folderhistory.core.timeline import (
    BranchingNode,
    BranchingTimeline,
    build_branching_timeline,
    infer_temporal_order,
)
from folderhistory.io.output import format_branching_dag
from folderhistory.types import EditOperation, FileRecord, Snapshot


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_file(path: str, blake3: str = "0000") -> FileRecord:
    """Create a minimal FileRecord for testing."""
    return FileRecord(
        path=path,
        size=len(blake3),
        mode=0o100644,
        mtime_ns=1_000_000_000,
        ctime_ns=1_000_000_000,
        raw_blake3=blake3,
        normalized_blake3=blake3,
        line_ending="lf",
        xxhash64=blake3,
        is_symlink=False,
        target_path=None,
    )


def _make_snapshot(
    snap_id: str,
    timestamp: float | None = 1_000_000_000.0,
) -> Snapshot:
    """Create a minimal Snapshot for testing."""
    return Snapshot(
        id=snap_id,
        timestamp=timestamp,
        source_path=Path(f"/fake/{snap_id}"),
        files=[],
    )


def _make_op(
    op_type: str = "modify",
    file_id: str = "f.txt",
    **kwargs: object,
) -> EditOperation:
    """Create a minimal EditOperation for testing."""
    return EditOperation(
        op_type=op_type,  # type: ignore  [arg-type]
        file_id=file_id,
        source_path=kwargs.get("source_path", file_id),  # type: ignore  [arg-type]
        target_path=kwargs.get("target_path"),  # type: ignore  [arg-type]
        old_hash=kwargs.get("old_hash"),  # type: ignore  [arg-type]
        new_hash=kwargs.get("new_hash"),  # type: ignore  [arg-type]
        confidence=float(kwargs.get("confidence", 1.0)),
    )


# ── BranchingNode tests ────────────────────────────────────────────────────────


class TestBranchingNode:
    """Tests for the :class:`BranchingNode` dataclass."""

    def test_all_fields_populated(self) -> None:
        """All fields can be set via the constructor."""
        ops = [_make_op("modify", "main.py")]
        node = BranchingNode(
            snapshot_id="v2",
            operations=ops,
            parent_ids=["v1"],
            children_ids=["v3"],
        )
        assert node.snapshot_id == "v2"
        assert len(node.operations) == 1
        assert node.operations[0].op_type == "modify"
        assert node.parent_ids == ["v1"]
        assert node.children_ids == ["v3"]

    def test_default_fields(self) -> None:
        """Fields without explicit values use their defaults."""
        node = BranchingNode(snapshot_id="v1")
        assert node.snapshot_id == "v1"
        assert node.operations == []
        assert node.parent_ids == []
        assert node.children_ids == []

    def test_multiple_parents_merge_point(self) -> None:
        """Multiple parent IDs indicate a merge point."""
        node = BranchingNode(
            snapshot_id="merge",
            parent_ids=["branch_a", "branch_b"],
        )
        assert len(node.parent_ids) == 2
        assert "branch_a" in node.parent_ids
        assert "branch_b" in node.parent_ids

    def test_multiple_children_fork_point(self) -> None:
        """Multiple children IDs indicate a fork point."""
        node = BranchingNode(
            snapshot_id="fork",
            children_ids=["child_a", "child_b"],
        )
        assert len(node.children_ids) == 2
        assert "child_a" in node.children_ids
        assert "child_b" in node.children_ids


# ── BranchingTimeline tests ────────────────────────────────────────────────────


class TestBranchingTimeline:
    """Tests for the :class:`BranchingTimeline` dataclass."""

    def test_empty_timeline(self) -> None:
        """Empty timeline has no nodes, roots, or leaves."""
        tl = BranchingTimeline(nodes={}, roots=[], leaves=[])
        assert tl.nodes == {}
        assert tl.roots == []
        assert tl.leaves == []

    def test_single_node(self) -> None:
        """Single node is both root and leaf."""
        n = BranchingNode(snapshot_id="v1")
        tl = BranchingTimeline(nodes={"v1": n}, roots=["v1"], leaves=["v1"])
        assert len(tl.nodes) == 1
        assert tl.roots == ["v1"]
        assert tl.leaves == ["v1"]

    def test_linear_chain(self) -> None:
        """Three nodes in a chain: v1 root, v3 leaf."""
        v1 = BranchingNode(snapshot_id="v1", children_ids=["v2"])
        v2 = BranchingNode(snapshot_id="v2", parent_ids=["v1"], children_ids=["v3"])
        v3 = BranchingNode(snapshot_id="v3", parent_ids=["v2"])
        tl = BranchingTimeline(
            nodes={"v1": v1, "v2": v2, "v3": v3},
            roots=["v1"],
            leaves=["v3"],
        )
        assert set(tl.nodes.keys()) == {"v1", "v2", "v3"}
        assert tl.roots == ["v1"]
        assert tl.leaves == ["v3"]

    def test_fork_timeline(self) -> None:
        """Fork: one root, two leaves."""
        v1 = BranchingNode(snapshot_id="v1", children_ids=["v2a", "v2b"])
        v2a = BranchingNode(snapshot_id="v2a", parent_ids=["v1"])
        v2b = BranchingNode(snapshot_id="v2b", parent_ids=["v1"])
        tl = BranchingTimeline(
            nodes={"v1": v1, "v2a": v2a, "v2b": v2b},
            roots=["v1"],
            leaves=["v2a", "v2b"],
        )
        assert tl.roots == ["v1"]
        assert set(tl.leaves) == {"v2a", "v2b"}


# ── infer_temporal_order tests ─────────────────────────────────────────────────


class TestInferTemporalOrder:
    """Tests for :func:`infer_temporal_order`."""

    def test_empty_snapshots(self) -> None:
        """Empty list produces empty ordering."""
        assert infer_temporal_order([]) == []

    def test_ordered_timestamps_one_group(self) -> None:
        """Timestamps close together form a single group."""
        snaps = [
            _make_snapshot("v1", 1_000_000_000.0),
            _make_snapshot("v2", 1_000_000_100.0),
            _make_snapshot("v3", 1_000_000_200.0),
        ]
        groups = infer_temporal_order(snaps)
        assert len(groups) == 1
        assert set(groups[0]) == {"v1", "v2", "v3"}

    def test_unordered_timestamps_multiple_groups(self) -> None:
        """Timestamps far apart form separate groups."""
        snaps = [
            _make_snapshot("v1", 1_000_000_000.0),
            _make_snapshot("v2", 1_000_000_050.0),
            _make_snapshot("v3", 2_000_000_000.0),
            _make_snapshot("v4", 2_000_000_050.0),
        ]
        groups = infer_temporal_order(snaps)
        assert len(groups) == 2
        assert set(groups[0]) == {"v1", "v2"}
        assert set(groups[1]) == {"v3", "v4"}

    def test_snapshots_without_timestamps(self) -> None:
        """Snapshots without timestamps are appended to the last group."""
        snaps = [
            _make_snapshot("v1", 1_000_000_000.0),
            _make_snapshot("v2", None),
            _make_snapshot("v3", None),
        ]
        groups = infer_temporal_order(snaps)
        # v1 (with timestamp) and v2/v3 (untimestamped) all in one group
        assert len(groups) == 1
        assert groups[0] == ["v1", "v2", "v3"]

    def test_all_without_timestamps(self) -> None:
        """All snapshots without timestamps → single group."""
        snaps = [
            _make_snapshot("v1", None),
            _make_snapshot("v2", None),
        ]
        groups = infer_temporal_order(snaps)
        assert len(groups) == 1
        assert set(groups[0]) == {"v1", "v2"}

    def test_three_groups(self) -> None:
        """Snapshots forming three temporal clusters."""
        snaps = [
            _make_snapshot("v1", 1_000_000_000.0),
            _make_snapshot("v2", 2_000_000_000.0),
            _make_snapshot("v3", 2_000_000_100.0),
            _make_snapshot("v4", 3_000_000_000.0),
        ]
        groups = infer_temporal_order(snaps)
        assert len(groups) == 3
        assert groups[0] == ["v1"]
        assert set(groups[1]) == {"v2", "v3"}
        assert groups[2] == ["v4"]


# ── build_branching_timeline tests ─────────────────────────────────────────────


class TestBuildBranchingTimeline:
    """Tests for :func:`build_branching_timeline`."""

    def test_no_snapshots(self) -> None:
        """Empty snapshots list with empty operation_sets."""
        tl = build_branching_timeline([], {})
        assert tl.nodes == {}
        assert tl.roots == []
        assert tl.leaves == []

    def test_linear_chain(self) -> None:
        """v1 → v2 → v3 produces a single root and single leaf."""
        op12 = [_make_op("modify", "f.txt")]
        op23 = [_make_op("delete", "f.txt")]
        snaps = [
            _make_snapshot("v1", 1_000_000_000.0),
            _make_snapshot("v2", 1_000_000_100.0),
            _make_snapshot("v3", 1_000_000_200.0),
        ]
        ops: dict[tuple[str, str], list[EditOperation]] = {
            ("v1", "v2"): op12,
            ("v2", "v3"): op23,
        }
        tl = build_branching_timeline(snaps, ops)
        assert set(tl.nodes.keys()) == {"v1", "v2", "v3"}
        assert tl.roots == ["v1"]
        assert tl.leaves == ["v3"]
        # Check chain consistency.
        assert tl.nodes["v1"].children_ids == ["v2"]
        assert tl.nodes["v2"].parent_ids == ["v1"]
        assert tl.nodes["v2"].children_ids == ["v3"]
        assert tl.nodes["v3"].parent_ids == ["v2"]
        # Operations assigned to children.
        assert tl.nodes["v2"].operations == op12
        assert tl.nodes["v3"].operations == op23
        # v1 has no operations.
        assert tl.nodes["v1"].operations == []

    def test_fork_one_root_two_leaves(self) -> None:
        """v1 → v2a, v1 → v2b: one root, two leaves."""
        op_a = [_make_op("modify", "a.txt")]
        op_b = [_make_op("create", "b.txt", target_path="b.txt")]
        snaps = [
            _make_snapshot("v1", 1_000_000_000.0),
            _make_snapshot("v2a", 1_000_000_100.0),
            _make_snapshot("v2b", 1_000_000_200.0),
        ]
        ops: dict[tuple[str, str], list[EditOperation]] = {
            ("v1", "v2a"): op_a,
            ("v1", "v2b"): op_b,
        }
        tl = build_branching_timeline(snaps, ops)
        assert set(tl.nodes.keys()) == {"v1", "v2a", "v2b"}
        assert tl.roots == ["v1"]
        assert set(tl.leaves) == {"v2a", "v2b"}
        # v1 has two children (fork point).
        assert set(tl.nodes["v1"].children_ids) == {"v2a", "v2b"}
        assert tl.nodes["v2a"].parent_ids == ["v1"]
        assert tl.nodes["v2b"].parent_ids == ["v1"]

    def test_merge_two_roots_one_leaf(self) -> None:
        """v1 → v3, v2 → v3: two roots, one leaf (merge)."""
        op13 = [_make_op("modify", "f.txt")]
        op23 = [_make_op("modify", "g.txt")]
        snaps = [
            _make_snapshot("v1", 1_000_000_000.0),
            _make_snapshot("v2", 1_000_000_100.0),
            _make_snapshot("v3", 1_000_000_200.0),
        ]
        ops: dict[tuple[str, str], list[EditOperation]] = {
            ("v1", "v3"): op13,
            ("v2", "v3"): op23,
        }
        tl = build_branching_timeline(snaps, ops)
        assert set(tl.nodes.keys()) == {"v1", "v2", "v3"}
        assert set(tl.roots) == {"v1", "v2"}
        assert tl.leaves == ["v3"]
        # v3 has two parents (merge point).
        assert set(tl.nodes["v3"].parent_ids) == {"v1", "v2"}
        assert tl.nodes["v1"].children_ids == ["v3"]
        assert tl.nodes["v2"].children_ids == ["v3"]
        # Operations from both branches present on v3.
        ops_on_v3 = tl.nodes["v3"].operations
        assert len(ops_on_v3) == 2  # both op sets combined

    def test_fork_then_merge(self) -> None:
        """v1 → v2a, v1 → v2b → v3 (v2a→v3, v2b→v3): fork then merge."""
        op_a = [_make_op("modify", "a.txt")]
        op_b = [_make_op("modify", "b.txt")]
        op_a_to_merge = [_make_op("modify", "a.txt")]
        op_b_to_merge = [_make_op("modify", "b.txt")]
        snaps = [
            _make_snapshot("v1", 1_000_000_000.0),
            _make_snapshot("v2a", 1_000_000_100.0),
            _make_snapshot("v2b", 1_000_000_200.0),
            _make_snapshot("v3", 1_000_000_300.0),
        ]
        ops: dict[tuple[str, str], list[EditOperation]] = {
            ("v1", "v2a"): op_a,
            ("v1", "v2b"): op_b,
            ("v2a", "v3"): op_a_to_merge,
            ("v2b", "v3"): op_b_to_merge,
        }
        tl = build_branching_timeline(snaps, ops)
        assert set(tl.nodes.keys()) == {"v1", "v2a", "v2b", "v3"}
        assert tl.roots == ["v1"]
        assert tl.leaves == ["v3"]
        # v1 has two children (fork).
        assert set(tl.nodes["v1"].children_ids) == {"v2a", "v2b"}
        # v3 has two parents (merge).
        assert set(tl.nodes["v3"].parent_ids) == {"v2a", "v2b"}
        # Operations on merge node.
        ops_on_v3 = tl.nodes["v3"].operations
        assert len(ops_on_v3) == 2

    def test_with_ordering_linear(self) -> None:
        """Ordering parameter defines level structure."""
        op12 = [_make_op("modify", "f.txt")]
        snaps = [
            _make_snapshot("v1"),
            _make_snapshot("v2"),
        ]
        ops: dict[tuple[str, str], list[EditOperation]] = {("v1", "v2"): op12}
        tl = build_branching_timeline(snaps, ops, ordering=[["v1"], ["v2"]])
        assert tl.roots == ["v1"]
        assert tl.leaves == ["v2"]

    def test_isolated_snapshots_no_ops(self) -> None:
        """Snapshots without operation edges are isolated nodes."""
        snaps = [
            _make_snapshot("v1", 1_000_000_000.0),
            _make_snapshot("v2", 2_000_000_000.0),
        ]
        tl = build_branching_timeline(snaps, {})
        assert set(tl.nodes.keys()) == {"v1", "v2"}
        # Both are roots (no parents) and leaves (no children).
        assert set(tl.roots) == {"v1", "v2"}
        assert set(tl.leaves) == {"v1", "v2"}


# ── format_branching_dag tests ─────────────────────────────────────────────────


class TestFormatBranchingDag:
    """Tests for :func:`format_branching_dag`."""

    def test_empty_timeline(self) -> None:
        """Empty timeline produces a minimal DOT graph."""
        tl = BranchingTimeline(nodes={}, roots=[], leaves=[])
        dot = format_branching_dag(tl)
        assert "digraph BranchingTimeline" in dot
        assert "node [shape=box, style=rounded]" in dot
        assert dot.strip().endswith("}")

    def test_single_node(self) -> None:
        """Single node renders with its label."""
        n = BranchingNode(snapshot_id="v1")
        tl = BranchingTimeline(nodes={"v1": n}, roots=["v1"], leaves=["v1"])
        dot = format_branching_dag(tl)
        assert '"v1"' in dot
        assert dot.strip().endswith("}")

    def test_linear_chain(self) -> None:
        """Linear chain renders as sequential edges."""
        v1 = BranchingNode(snapshot_id="v1", children_ids=["v2"])
        v2 = BranchingNode(
            snapshot_id="v2", parent_ids=["v1"], children_ids=["v3"]
        )
        v3 = BranchingNode(snapshot_id="v3", parent_ids=["v2"])
        tl = BranchingTimeline(
            nodes={"v1": v1, "v2": v2, "v3": v3},
            roots=["v1"],
            leaves=["v3"],
        )
        dot = format_branching_dag(tl)
        assert '"v1" -> "v2"' in dot
        assert '"v2" -> "v3"' in dot

    def test_fork_nodes_render_correctly(self) -> None:
        """Fork point renders multiple outgoing edges."""
        v1 = BranchingNode(
            snapshot_id="v1", children_ids=["v2a", "v2b"]
        )
        v2a = BranchingNode(snapshot_id="v2a", parent_ids=["v1"])
        v2b = BranchingNode(snapshot_id="v2b", parent_ids=["v1"])
        tl = BranchingTimeline(
            nodes={"v1": v1, "v2a": v2a, "v2b": v2b},
            roots=["v1"],
            leaves=["v2a", "v2b"],
        )
        dot = format_branching_dag(tl)
        assert '"v1" -> "v2a"' in dot
        assert '"v1" -> "v2b"' in dot

    def test_merge_nodes_render_correctly(self) -> None:
        """Merge point renders multiple incoming edges."""
        v1 = BranchingNode(snapshot_id="v1", children_ids=["v3"])
        v2 = BranchingNode(snapshot_id="v2", children_ids=["v3"])
        v3 = BranchingNode(
            snapshot_id="v3", parent_ids=["v1", "v2"]
        )
        tl = BranchingTimeline(
            nodes={"v1": v1, "v2": v2, "v3": v3},
            roots=["v1", "v2"],
            leaves=["v3"],
        )
        dot = format_branching_dag(tl)
        assert '"v1" -> "v3"' in dot
        assert '"v2" -> "v3"' in dot

    def test_operations_appear_in_label(self) -> None:
        """Operation count appears in the node label."""
        op = _make_op("modify", "f.txt")
        n = BranchingNode(
            snapshot_id="v2",
            operations=[op],
            parent_ids=["v1"],
        )
        tl = BranchingTimeline(
            nodes={"v1": BranchingNode(snapshot_id="v1"), "v2": n},
            roots=["v1"],
            leaves=["v2"],
        )
        dot = format_branching_dag(tl)
        assert "1 ops" in dot

    def test_produces_valid_dot_structure(self) -> None:
        """Output has valid DOT brace structure."""
        v1 = BranchingNode(snapshot_id="v1", children_ids=["v2"])
        v2 = BranchingNode(snapshot_id="v2", parent_ids=["v1"])
        tl = BranchingTimeline(
            nodes={"v1": v1, "v2": v2},
            roots=["v1"],
            leaves=["v2"],
        )
        dot = format_branching_dag(tl)
        lines = [l for l in dot.split("\n") if l.strip()]
        assert lines[0].startswith("digraph")
        assert lines[-1] == "}"
        # Every opening brace has a matching close.
        opens = dot.count("{")
        closes = dot.count("}")
        assert opens == closes
