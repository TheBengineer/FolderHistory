"""Output formatters for FolderHistory timeline data.

Provides JSON, git-log-style, and JSON-lines serialisation of
:class:`~folderhistory.core.timeline.Timeline` objects.
"""

from __future__ import annotations

import datetime

import orjson

from folderhistory.core.timeline import BranchingTimeline, Timeline
from folderhistory.types import EditOperation


def format_json(timeline: Timeline) -> str:
    """Return pretty-printed JSON string of the timeline.

    The output includes root changes and an ordered list of nodes, each
    containing snapshot metadata and its edit operations.
    """
    payload = _timeline_to_dict(timeline)
    return orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode()


def format_gitlog(timeline: Timeline) -> str:
    """Return git-style log string.

    Format per snapshot::

        commit <snapshot_id>
        Date:   <timestamp>
        root: <source_path>

            <count> operations
    """
    lines: list[str] = []

    for node in timeline.nodes:
        lines.append(f"commit {node.snapshot_id}")

        if node.timestamp is not None:
            lines.append(f"Date:   {_format_timestamp(node.timestamp)}")
        else:
            lines.append("Date:   unknown")

        lines.append(f"root: {node.source_path}")

        if node.root_change:
            lines.append(f"Root change: {node.root_change}")

        count = len(node.operations)
        lines.append("")
        lines.append(f"\t{count} {'operation' if count == 1 else 'operations'}")
        lines.append("")

    return "\n".join(lines)


def format_jsonlines(timeline: Timeline) -> str:
    """Return JSON-lines format, one operation per line.

    Each line is a JSON object containing the snapshot identifier and the
    operation fields.
    """
    json_lines: list[str] = []

    for node in timeline.nodes:
        for op in node.operations:
            op_dict = _operation_to_dict(node.snapshot_id, op)
            json_lines.append(orjson.dumps(op_dict).decode())

    return "\n".join(json_lines)


# ── Internal helpers ───────────────────────────────────────────────────────────


def _timeline_to_dict(timeline: Timeline) -> dict[str, object]:
    """Convert a Timeline to a JSON-serialisable dict."""
    nodes_data: list[dict[str, object]] = []
    for node in timeline.nodes:
        node_dict: dict[str, object] = {
            "snapshot_id": node.snapshot_id,
            "timestamp": node.timestamp,
            "source_path": node.source_path,
            "parent_id": node.parent_id,
            "root_change": node.root_change,
            "operations": [_operation_to_dict(None, op) for op in node.operations],
        }
        nodes_data.append(node_dict)

    return {
        "root_changes": timeline.root_changes,
        "nodes": nodes_data,
    }


def _operation_to_dict(
    snapshot_id: str | None,
    op: EditOperation,
) -> dict[str, object]:
    """Convert an EditOperation to a JSON-serialisable dict."""
    d: dict[str, object] = {
        "op_type": op.op_type,
        "file_id": op.file_id,
        "source_path": op.source_path,
        "target_path": op.target_path,
        "old_hash": op.old_hash,
        "new_hash": op.new_hash,
        "confidence": op.confidence,
    }
    if snapshot_id is not None:
        d["snapshot_id"] = snapshot_id
    return d


def _format_timestamp(timestamp: float) -> str:
    """Format a Unix timestamp as a human-readable date-time string."""
    dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    return dt.strftime("%a %b %d %H:%M:%S %Y %z")


# ── Branching DAG output ────────────────────────────────────────────────────────


def format_branching_dag(timeline: BranchingTimeline) -> str:
    """Return a DOT graph string suitable for rendering with Graphviz.

    Each :class:`BranchingNode` becomes a box-shaped node labelled with its
    snapshot ID.  Edges represent parent-child relationships.  Fork points
    (multiple outgoing edges) and merge points (multiple incoming edges) are
    visually distinct by construction.

    The output is valid DOT that can be piped directly into ``dot``,
    ``neato``, or rendered with ``xdot``.
    """
    lines: list[str] = [
        "digraph BranchingTimeline {",
        "    node [shape=box, style=rounded];",
        "",
    ]

    # Emit nodes.
    for sid, node in timeline.nodes.items():
        label = sid
        op_count = len(node.operations)
        if op_count > 0:
            label += f"\\n{op_count} ops"
        lines.append(f'    "{sid}" [label="{label}"];')

    lines.append("")

    # Emit edges.
    for sid, node in timeline.nodes.items():
        for child_id in node.children_ids:
            lines.append(f'    "{sid}" -> "{child_id}";')

    lines.append("}")
    return "\n".join(lines)
