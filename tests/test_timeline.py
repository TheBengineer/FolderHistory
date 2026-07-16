"""Tests for ``folderhistory.core.timeline`` and ``folderhistory.io.output``."""

from __future__ import annotations

import json as json_module
from pathlib import Path

import orjson

from folderhistory.core.timeline import Timeline, TimelineNode, build_timeline, detect_root_changes
from folderhistory.io.output import format_gitlog, format_json, format_jsonlines
from folderhistory.types import EditOperation, FileRecord, IdentityCluster, Snapshot


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_file(path: str, blake3: str) -> FileRecord:
    """Create a :class:`FileRecord` with minimal test defaults."""
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
    *records: FileRecord,
    source_path: Path | None = None,
) -> Snapshot:
    """Create a :class:`Snapshot` from ``FileRecord`` entries."""
    return Snapshot(
        id=snap_id,
        timestamp=1_000_000_000.0,
        source_path=source_path or Path(f"/fake/{snap_id}"),
        files=list(records),
    )


def _make_op(
    op_type: str,
    file_id: str,
    **kwargs: object,
) -> EditOperation:
    """Create an :class:`EditOperation` with the given fields."""
    return EditOperation(
        op_type=op_type,  # type: ignore  [arg-type]
        file_id=file_id,
        source_path=kwargs.get("source_path", file_id),  # type: ignore  [arg-type]
        target_path=kwargs.get("target_path"),  # type: ignore  [arg-type]
        old_hash=kwargs.get("old_hash"),  # type: ignore  [arg-type]
        new_hash=kwargs.get("new_hash"),  # type: ignore  [arg-type]
        confidence=float(kwargs.get("confidence", 1.0)),
    )


def _make_cluster(
    uid: str,
    observations: list[tuple[str, str]],
    *,
    confidence: float = 1.0,
) -> IdentityCluster:
    """Create an :class:`IdentityCluster` with the given observations."""
    return IdentityCluster(
        uid=uid,
        observations=observations,
        canonical_path=observations[0][1] if observations else None,
        confidence=confidence,
    )


# ── TimelineNode tests ─────────────────────────────────────────────────────────


class TestTimelineNode:
    """Tests for the :class:`TimelineNode` dataclass."""

    def test_all_fields_populated(self) -> None:
        """All fields can be set via the constructor."""
        ops = [_make_op("modify", "main.py")]
        node = TimelineNode(
            snapshot_id="v2",
            timestamp=2_000_000_000.0,
            source_path="/real/v2",
            operations=ops,
            parent_id="v1",
            root_change="/real/v1 → /real/v2",
        )
        assert node.snapshot_id == "v2"
        assert node.timestamp == 2_000_000_000.0
        assert node.source_path == "/real/v2"
        assert node.parent_id == "v1"
        assert node.root_change == "/real/v1 → /real/v2"
        assert len(node.operations) == 1
        assert node.operations[0].op_type == "modify"

    def test_default_fields(self) -> None:
        """Fields without explicit values use their defaults."""
        node = TimelineNode(snapshot_id="v1")
        assert node.snapshot_id == "v1"
        assert node.timestamp is None
        assert node.source_path == ""
        assert node.operations == []
        assert node.parent_id is None
        assert node.root_change is None


# ── Timeline tests ─────────────────────────────────────────────────────────────


class TestTimeline:
    """Tests for the :class:`Timeline` dataclass."""

    def test_empty_timeline(self) -> None:
        """A Timeline can be created with no nodes."""
        tl = Timeline()
        assert tl.nodes == []
        assert tl.root_changes == []

    def test_with_nodes(self) -> None:
        """A Timeline holds an ordered list of nodes."""
        node1 = TimelineNode(snapshot_id="v1")
        node2 = TimelineNode(snapshot_id="v2", parent_id="v1")
        tl = Timeline(nodes=[node1, node2])
        assert len(tl.nodes) == 2
        assert tl.nodes[0].snapshot_id == "v1"
        assert tl.nodes[1].parent_id == "v1"


# ── build_timeline tests ──────────────────────────────────────────────────────


class TestBuildTimeline:
    """Tests for :func:`build_timeline`."""

    def test_empty_snapshots(self) -> None:
        """Empty snapshots list produces an empty timeline."""
        tl = build_timeline([], [])
        assert tl.nodes == []
        assert tl.root_changes == []

    def test_single_snapshot(self) -> None:
        """Single snapshot → one node, no operations, no parent."""
        snap = _make_snapshot("v1")
        tl = build_timeline([snap], [])
        assert len(tl.nodes) == 1
        assert tl.nodes[0].snapshot_id == "v1"
        assert tl.nodes[0].parent_id is None
        assert tl.nodes[0].operations == []
        assert tl.nodes[0].root_change is None

    def test_two_snapshots_one_change(self) -> None:
        """Two snapshots with one modify operation."""
        f1 = _make_file("main.py", "hash_old")
        f2 = _make_file("main.py", "hash_new")
        snap_a = _make_snapshot("v1", f1)
        snap_b = _make_snapshot("v2", f2)

        op = _make_op("modify", "main.py", old_hash="hash_old", new_hash="hash_new")
        operations = [[op]]

        tl = build_timeline([snap_a, snap_b], operations)

        assert len(tl.nodes) == 2

        # First node: no operations, no parent
        assert tl.nodes[0].snapshot_id == "v1"
        assert tl.nodes[0].parent_id is None
        assert tl.nodes[0].operations == []

        # Second node: one modify operation, parent is v1
        assert tl.nodes[1].snapshot_id == "v2"
        assert tl.nodes[1].parent_id == "v1"
        assert len(tl.nodes[1].operations) == 1
        assert tl.nodes[1].operations[0].op_type == "modify"
        assert tl.nodes[1].operations[0].file_id == "main.py"

    def test_three_snapshots_chain(self) -> None:
        """Three snapshots with operations in each pair."""
        snap_a = _make_snapshot("v1")
        snap_b = _make_snapshot("v2")
        snap_c = _make_snapshot("v3")

        op1 = _make_op("create", "new.txt", target_path="new.txt")
        op2 = _make_op("delete", "new.txt", source_path="new.txt")
        operations = [[op1], [op2]]

        tl = build_timeline([snap_a, snap_b, snap_c], operations)

        assert len(tl.nodes) == 3

        # v1 → v2 transition: create
        assert tl.nodes[0].snapshot_id == "v1"
        assert tl.nodes[0].operations == []

        assert tl.nodes[1].snapshot_id == "v2"
        assert tl.nodes[1].parent_id == "v1"
        assert len(tl.nodes[1].operations) == 1
        assert tl.nodes[1].operations[0].op_type == "create"

        # v2 → v3 transition: delete
        assert tl.nodes[2].snapshot_id == "v3"
        assert tl.nodes[2].parent_id == "v2"
        assert len(tl.nodes[2].operations) == 1
        assert tl.nodes[2].operations[0].op_type == "delete"

    def test_no_operations_empty_list(self) -> None:
        """Passing empty operations for each pair works."""
        snap_a = _make_snapshot("v1")
        snap_b = _make_snapshot("v2")
        tl = build_timeline([snap_a, snap_b], [[]])
        assert len(tl.nodes) == 2
        assert tl.nodes[0].operations == []
        assert tl.nodes[1].operations == []

    def test_timestamps_and_source_paths_carried_over(self) -> None:
        """Timestamps and source_paths from snapshots appear on nodes."""
        src = Path("/custom/path/v1")
        snap = _make_snapshot("v1", source_path=src)
        tl = build_timeline([snap], [])
        assert tl.nodes[0].timestamp == 1_000_000_000.0
        assert tl.nodes[0].source_path == str(src.resolve())


# ── detect_root_changes tests ─────────────────────────────────────────────────


class TestDetectRootChanges:
    """Tests for :func:`detect_root_changes`."""

    def test_empty_snapshots(self) -> None:
        """Empty snapshots → empty list."""
        assert detect_root_changes([]) == []

    def test_single_snapshot(self) -> None:
        """Single snapshot → no root change."""
        snap = _make_snapshot("v1")
        assert detect_root_changes([snap]) == []

    def test_same_root(self) -> None:
        """Same source_path across snapshots → empty list."""
        path = Path("/same/root")
        snap_a = _make_snapshot("v1", source_path=path)
        snap_b = _make_snapshot("v2", source_path=path)
        assert detect_root_changes([snap_a, snap_b]) == []

    def test_different_root(self) -> None:
        """Different source_path → change detected."""
        snap_a = _make_snapshot("v1", source_path=Path("/root/one"))
        snap_b = _make_snapshot("v2", source_path=Path("/root/two"))
        changes = detect_root_changes([snap_a, snap_b])
        assert len(changes) == 1
        entry = changes[0]
        assert entry["snapshot_id"] == "v2"
        assert "from_root" in entry
        assert "to_root" in entry
        assert "two" in entry["to_root"]

    def test_root_change_middle_snapshot(self) -> None:
        """Root changes only for the snapshot that differs."""
        path_a = Path("/root/a")
        path_b = Path("/root/b")
        snap_a = _make_snapshot("v1", source_path=path_a)
        snap_b = _make_snapshot("v2", source_path=path_b)
        snap_c = _make_snapshot("v3", source_path=path_b)  # same as v2
        changes = detect_root_changes([snap_a, snap_b, snap_c])
        assert len(changes) == 1
        assert changes[0]["snapshot_id"] == "v2"

    def test_multiple_root_changes(self) -> None:
        """Multiple root changes across the chain."""
        path_a = Path("/root/a")
        path_b = Path("/root/b")
        path_c = Path("/root/c")
        snap_a = _make_snapshot("v1", source_path=path_a)
        snap_b = _make_snapshot("v2", source_path=path_b)
        snap_c = _make_snapshot("v3", source_path=path_c)
        changes = detect_root_changes([snap_a, snap_b, snap_c])
        assert len(changes) == 2
        assert changes[0]["snapshot_id"] == "v2"
        assert changes[1]["snapshot_id"] == "v3"

    def test_root_change_annotated_in_timeline(self) -> None:
        """Root changes are annotated as root_change on the TimelineNode."""
        path_a = Path("/root/a")
        path_b = Path("/root/b")
        snap_a = _make_snapshot("v1", source_path=path_a)
        snap_b = _make_snapshot("v2", source_path=path_b)
        tl = build_timeline([snap_a, snap_b], [[]])
        assert tl.nodes[0].root_change is None
        assert tl.nodes[1].root_change is not None
        assert "→" in tl.nodes[1].root_change  # type: ignore  [operator]


# ── format_json tests ─────────────────────────────────────────────────────────


class TestFormatJson:
    """Tests for :func:`format_json`."""

    def test_produces_valid_json(self) -> None:
        """Output can be parsed as JSON."""
        tl = Timeline()
        raw = format_json(tl)
        parsed = orjson.loads(raw)
        assert isinstance(parsed, dict)

    def test_empty_timeline(self) -> None:
        """Empty timeline produces JSON with empty arrays."""
        tl = Timeline()
        raw = format_json(tl)
        parsed = orjson.loads(raw)
        assert parsed["root_changes"] == []
        assert parsed["nodes"] == []

    def test_single_node_timeline(self) -> None:
        """A single node appears in the JSON output."""
        node = TimelineNode(
            snapshot_id="v1",
            timestamp=1_000_000_000.0,
            source_path="/fake/v1",
        )
        tl = Timeline(nodes=[node])
        raw = format_json(tl)
        parsed = orjson.loads(raw)
        assert len(parsed["nodes"]) == 1
        n = parsed["nodes"][0]
        assert n["snapshot_id"] == "v1"
        assert n["parent_id"] is None
        assert n["root_change"] is None
        assert n["operations"] == []

    def test_node_with_operations(self) -> None:
        """Operations are serialised in the JSON output."""
        op = _make_op("create", "new.txt", target_path="new.txt")
        node = TimelineNode(
            snapshot_id="v2",
            parent_id="v1",
            operations=[op],
        )
        tl = Timeline(nodes=[node])
        raw = format_json(tl)
        parsed = orjson.loads(raw)
        ops = parsed["nodes"][0]["operations"]
        assert len(ops) == 1
        assert ops[0]["op_type"] == "create"
        assert ops[0]["file_id"] == "new.txt"

    def test_root_changes_serialised(self) -> None:
        """Root changes are included in the JSON output."""
        tl = Timeline(
            nodes=[TimelineNode(snapshot_id="v2")],
            root_changes=[{"snapshot_id": "v2", "from_root": "/a", "to_root": "/b"}],
        )
        raw = format_json(tl)
        parsed = orjson.loads(raw)
        assert len(parsed["root_changes"]) == 1
        assert parsed["root_changes"][0]["snapshot_id"] == "v2"


# ── format_gitlog tests ───────────────────────────────────────────────────────


class TestFormatGitlog:
    """Tests for :func:`format_gitlog`."""

    def test_empty_timeline(self) -> None:
        """Empty timeline produces an empty string."""
        tl = Timeline()
        assert format_gitlog(tl) == ""

    def test_single_node_no_timestamp(self) -> None:
        """A node without a timestamp renders 'unknown'."""
        node = TimelineNode(snapshot_id="abc123")
        tl = Timeline(nodes=[node])
        output = format_gitlog(tl)
        assert "commit abc123" in output
        assert "Date:   unknown" in output
        assert "0 operations" in output

    def test_single_node_with_timestamp(self) -> None:
        """A node with a timestamp renders a formatted date."""
        node = TimelineNode(
            snapshot_id="v1",
            timestamp=1_000_000_000.0,
            source_path="/fake/v1",
        )
        tl = Timeline(nodes=[node])
        output = format_gitlog(tl)
        assert "commit v1" in output
        assert "Date:" in output
        assert "root: /fake/v1" in output

    def test_node_with_root_change(self) -> None:
        """Root change annotation appears in the output."""
        node = TimelineNode(
            snapshot_id="v2",
            root_change="/a → /b",
        )
        tl = Timeline(nodes=[node])
        output = format_gitlog(tl)
        assert "Root change: /a → /b" in output

    def test_node_with_operations(self) -> None:
        """Operation count is rendered."""
        ops = [_make_op("modify", "f.py")]
        node = TimelineNode(
            snapshot_id="v2",
            operations=ops,
        )
        tl = Timeline(nodes=[node])
        output = format_gitlog(tl)
        assert "1 operation" in output

    def test_multiple_operations_plural(self) -> None:
        """Multiple operations use plural form."""
        ops = [
            _make_op("create", "a.txt", target_path="a.txt"),
            _make_op("modify", "b.txt"),
        ]
        node = TimelineNode(
            snapshot_id="v2",
            operations=ops,
        )
        tl = Timeline(nodes=[node])
        output = format_gitlog(tl)
        assert "2 operations" in output

    def test_multiple_nodes(self) -> None:
        """Multiple nodes all appear in the output."""
        node1 = TimelineNode(snapshot_id="v1")
        node2 = TimelineNode(snapshot_id="v2", parent_id="v1")
        tl = Timeline(nodes=[node1, node2])
        output = format_gitlog(tl)
        assert output.count("commit ") == 2


# ── format_jsonlines tests ────────────────────────────────────────────────────


class TestFormatJsonlines:
    """Tests for :func:`format_jsonlines`."""

    def test_empty_timeline(self) -> None:
        """Empty timeline produces empty output."""
        tl = Timeline()
        assert format_jsonlines(tl) == ""

    def test_one_operation_per_line(self) -> None:
        """Each operation produces a single JSON line."""
        ops = [
            _make_op("create", "a.txt", target_path="a.txt"),
            _make_op("delete", "b.txt", source_path="b.txt"),
        ]
        node = TimelineNode(
            snapshot_id="v2",
            operations=ops,
        )
        tl = Timeline(nodes=[node])
        output = format_jsonlines(tl)
        lines = output.strip().split("\n")
        assert len(lines) == 2

        # Each line is valid JSON.
        for line in lines:
            parsed = json_module.loads(line)
            assert isinstance(parsed, dict)

    def test_operation_fields_present(self) -> None:
        """Each operation JSON includes expected fields."""
        op = _make_op("modify", "main.py", old_hash="abc", new_hash="def")
        node = TimelineNode(
            snapshot_id="v2",
            operations=[op],
        )
        tl = Timeline(nodes=[node])
        output = format_jsonlines(tl)
        lines = output.strip().split("\n")
        assert len(lines) == 1
        parsed = json_module.loads(lines[0])
        assert parsed["snapshot_id"] == "v2"
        assert parsed["op_type"] == "modify"
        assert parsed["file_id"] == "main.py"
        assert parsed["old_hash"] == "abc"
        assert parsed["new_hash"] == "def"

    def test_no_operations_no_lines(self) -> None:
        """Nodes without operations produce no lines."""
        node = TimelineNode(snapshot_id="v1")
        tl = Timeline(nodes=[node])
        assert format_jsonlines(tl) == ""

    def test_operations_across_multiple_nodes(self) -> None:
        """Operations from multiple nodes appear in order."""
        op1 = _make_op("create", "a.txt", target_path="a.txt")
        op2 = _make_op("delete", "a.txt", source_path="a.txt")
        node1 = TimelineNode(snapshot_id="v2", operations=[op1])
        node2 = TimelineNode(snapshot_id="v3", operations=[op2])
        tl = Timeline(nodes=[node1, node2])
        output = format_jsonlines(tl)
        lines = output.strip().split("\n")
        assert len(lines) == 2
        parsed1 = json_module.loads(lines[0])
        parsed2 = json_module.loads(lines[1])
        assert parsed1["snapshot_id"] == "v2"
        assert parsed1["op_type"] == "create"
        assert parsed2["snapshot_id"] == "v3"
        assert parsed2["op_type"] == "delete"


# ── Integration tests (build_timeline + output) ────────────────────────────────


class TestTimelineIntegration:
    """End-to-end tests combining timeline construction and formatting."""

    def test_analyze_pipeline_with_output(self) -> None:
        """End-to-end: snapshots → identities → operations → timeline → format."""
        from folderhistory.core.diff import derive_operations
        from folderhistory.core.identity import assign_identities_exact

        f1_v1 = _make_file("readme.md", "hash_abc")
        f1_v2 = _make_file("readme.md", "hash_def")
        f2_v2 = _make_file("new.txt", "hash_xyz")

        snap_a = _make_snapshot("v1", f1_v1)
        snap_b = _make_snapshot("v2", f1_v2, f2_v2)

        identities = assign_identities_exact([snap_a, snap_b])
        operations = derive_operations([snap_a, snap_b], identities)
        tl = build_timeline([snap_a, snap_b], operations)

        assert len(tl.nodes) == 2

        # v2 should have exactly two operations: modify (readme.md) + create (new.txt)
        assert len(tl.nodes[1].operations) == 2
        op_types = {op.op_type for op in tl.nodes[1].operations}
        assert "modify" in op_types
        assert "create" in op_types

        # All output formats produce content.
        assert len(format_json(tl)) > 0
        assert len(format_gitlog(tl)) > 0
        assert len(format_jsonlines(tl)) > 0
