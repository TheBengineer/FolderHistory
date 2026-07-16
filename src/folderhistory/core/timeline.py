"""Linear timeline construction for FolderHistory.

Chains per-pair edit operations into a linear timeline and provides
root-change detection across ordered snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from folderhistory.types import EditOperation, Snapshot


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass
class TimelineNode:
    """A single node in the folder-history timeline.

    :param snapshot_id: Unique identifier for the snapshot at this node.
    :param timestamp: Capture timestamp of the snapshot, or ``None``.
    :param source_path: Filesystem path of the snapshot root.
    :param operations: Operations that produced this node from its parent.
    :param parent_id: Snapshot identifier of the parent node, or ``None``
        for the first node in the timeline.
    :param root_change: Human-readable annotation describing a change in
        the root directory, e.g. ``"~/Documents/ → ~/Dropbox/"``, or
        ``None`` if the root path did not change.
    """

    snapshot_id: str
    timestamp: float | None = None
    source_path: str = ""
    operations: list[EditOperation] = field(default_factory=list)
    parent_id: str | None = None
    root_change: str | None = None


@dataclass
class Timeline:
    """An ordered linear sequence of timeline nodes.

    :param nodes: Chronologically ordered nodes, one per snapshot.
    :param root_changes: Detected root-directory changes (raw form).
    """

    nodes: list[TimelineNode] = field(default_factory=list)
    root_changes: list[dict[str, str]] = field(default_factory=list)


# ── Core construction ──────────────────────────────────────────────────────────


def build_timeline(
    snapshots: list[Snapshot],
    operations: list[list[EditOperation]],
) -> Timeline:
    """Build a linear timeline from ordered snapshots and per-pair operations.

    For each snapshot in *snapshots* a :class:`TimelineNode` is created.
    Operations from ``operations[i-1]`` (the edit script that produced
    snapshot *i* from snapshot *i-1*) are attached to node *i*.
    Root-directory changes between consecutive snapshots are detected and
    annotated.

    Parameters
    ----------
    snapshots:
        Ordered list of snapshots forming the history.
    operations:
        One list of :class:`EditOperation` per consecutive pair, as returned
        by :func:`~folderhistory.core.diff.derive_operations`.  May be empty
        when there are fewer than two snapshots.

    Returns
    -------
    A :class:`Timeline` with one node per snapshot.
    """
    root_changes = detect_root_changes(snapshots)
    root_change_map: dict[str, str] = {}
    for rc in root_changes:
        sid: str = rc["snapshot_id"]
        annotation = f"{rc['from_root']} → {rc['to_root']}"
        root_change_map[sid] = annotation

    nodes: list[TimelineNode] = []
    for i, snap in enumerate(snapshots):
        parent_id: str | None = snapshots[i - 1].id if i > 0 else None

        pair_ops: list[EditOperation] = []
        if i > 0 and i - 1 < len(operations):
            pair_ops = operations[i - 1]

        rc_annotation: str | None = root_change_map.get(snap.id)

        nodes.append(
            TimelineNode(
                snapshot_id=snap.id,
                timestamp=snap.timestamp,
                source_path=str(snap.source_path),
                operations=pair_ops,
                parent_id=parent_id,
                root_change=rc_annotation,
            ),
        )

    return Timeline(nodes=nodes, root_changes=root_changes)


# ── Root-change detection ──────────────────────────────────────────────────────


def detect_root_changes(
    snapshots: list[Snapshot],
) -> list[dict[str, str]]:
    """Detect root path changes between consecutive snapshots.

    A root change is recorded when a snapshot's ``source_path`` differs
    from the previous snapshot's ``source_path``.

    Parameters
    ----------
    snapshots:
        Ordered list of snapshots to inspect.

    Returns
    -------
    List of ``{"snapshot_id": str, "from_root": str, "to_root": str}``
    entries, one per snapshot (starting from index 1) whose root differs
    from its immediate predecessor.
    """
    changes: list[dict[str, str]] = []

    if len(snapshots) < 2:
        return changes

    for i in range(1, len(snapshots)):
        prev_root = _resolve_path(snapshots[i - 1].source_path)
        curr_root = _resolve_path(snapshots[i].source_path)

        if prev_root != curr_root:
            changes.append(
                {
                    "snapshot_id": snapshots[i].id,
                    "from_root": prev_root,
                    "to_root": curr_root,
                },
            )

    return changes


def _resolve_path(path: Path) -> str:
    """Resolve a path to an absolute string, handling relative paths."""
    try:
        return str(path.resolve())
    except (OSError, RuntimeError):
        return str(path.absolute())
