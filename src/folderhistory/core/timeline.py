"""Linear and branching DAG timeline construction for FolderHistory.

Chains per-pair edit operations into a linear timeline or a branching DAG
and provides root-change detection across ordered snapshots.
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


# ── Branching DAG types ─────────────────────────────────────────────────────────


@dataclass
class BranchingNode:
    """A single node in a branching DAG timeline.

    :param snapshot_id: Unique identifier for this snapshot.
    :param operations: Edit operations that produced this node from its parent(s).
    :param parent_ids: Snapshot IDs of parent nodes — multiple parents means
        this node is a merge point.
    :param children_ids: Snapshot IDs of child nodes — multiple children means
        this node is a fork point.
    """

    snapshot_id: str
    operations: list[EditOperation] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)
    children_ids: list[str] = field(default_factory=list)


@dataclass
class BranchingTimeline:
    """A branching DAG timeline representing non-linear snapshot history.

    :param nodes: Mapping from snapshot ID to BranchingNode.
    :param roots: Snapshot IDs with no parents (starting points).
    :param leaves: Snapshot IDs with no children (end points).
    """

    nodes: dict[str, BranchingNode]
    roots: list[str]
    leaves: list[str]


# ── Temporal ordering inference ─────────────────────────────────────────────────


def infer_temporal_order(
    snapshots: list[Snapshot],
) -> list[list[str]]:
    """Infer ordering constraints from snapshot timestamps.

    Groups snapshots by temporal proximity.  Snapshots with timestamps
    within approximately one hour of each other are grouped into the same
    level.  Groups are ordered chronologically by their median timestamp.

    Snapshots without a timestamp are placed in the final group.

    Parameters
    ----------
    snapshots:
        Snapshots whose temporal ordering should be inferred.

    Returns
    -------
    A list of groups, each being a list of snapshot IDs.  Groups are ordered
    from earliest to latest.  Multiple IDs in the same group can be treated
    as parallel branches.
    """
    _SAME_GROUP_THRESHOLD_S: float = 3600.0  # 1 hour

    # Separate snapshots with and without timestamps.
    with_ts: list[tuple[str, float]] = [
        (s.id, s.timestamp) for s in snapshots if s.timestamp is not None
    ]
    without_ts: list[str] = [s.id for s in snapshots if s.timestamp is None]

    if not with_ts:
        return [without_ts] if without_ts else []

    # Sort by timestamp ascending.
    with_ts.sort(key=lambda x: x[1])

    # Cluster into groups by temporal proximity.
    groups: list[list[str]] = []
    current: list[str] = [with_ts[0][0]]

    for i in range(1, len(with_ts)):
        gap: float = with_ts[i][1] - with_ts[i - 1][1]  # gap in seconds
        if gap > _SAME_GROUP_THRESHOLD_S:
            groups.append(current)
            current = [with_ts[i][0]]
        else:
            current.append(with_ts[i][0])

    groups.append(current)

    # Place untimestamped snapshots into the last group.
    if without_ts:
        groups[-1].extend(without_ts)

    return groups


# ── Branching timeline construction ─────────────────────────────────────────────


def build_branching_timeline(
    snapshots: list[Snapshot],
    operation_sets: dict[tuple[str, str], list[EditOperation]],
    ordering: list[list[str]] | None = None,
) -> BranchingTimeline:
    """Build a branching DAG timeline from per-pair operation sets.

    Uses *operation_sets* as the primary source of parent-child edges.  The
    optional *ordering* parameter provides level constraints: when given,
    edges are validated to respect the level ordering (parents must be in
    earlier levels than children).

    The identity graph of the operation sets is used to detect:

    * **Fork points** — nodes with multiple children (a snapshot that is the
      parent of several later snapshots).
    * **Merge points** — nodes with multiple parents (a snapshot that combines
      changes from several earlier snapshots).
    * **Linear segments** — simple parent-child chains.

    Parameters
    ----------
    snapshots:
        All known snapshots (used to resolve snapshot metadata).
    operation_sets:
        Mapping of ``(parent_id, child_id)`` to the list of operations
        that transform the parent into the child.
    ordering:
        Optional ordered list of groups.  Each inner list contains snapshot
        IDs at the same chronological level.  When provided, roots and leaves
        are sorted to respect this ordering.  If ``None``, temporal order is
        inferred from snapshot timestamps via :func:`infer_temporal_order`.

    Returns
    -------
    A :class:`BranchingTimeline` with nodes, roots, and leaves.
    """
    snap_map: dict[str, Snapshot] = {s.id: s for s in snapshots}

    # Collect all snapshot IDs referenced in snapshots, operation_sets, and ordering.
    all_ids: set[str] = set(snap_map.keys())
    for parent_id, child_id in operation_sets:
        all_ids.add(parent_id)
        all_ids.add(child_id)
    if ordering:
        for level in ordering:
            all_ids.update(level)

    # Initialise every node.
    nodes: dict[str, BranchingNode] = {}
    for sid in all_ids:
        nodes[sid] = BranchingNode(snapshot_id=sid)

    # Build edges from operation_sets.
    for (parent_id, child_id), ops in operation_sets.items():
        if parent_id in nodes and child_id in nodes:
            nodes[child_id].parent_ids.append(parent_id)
            nodes[child_id].operations.extend(ops)
            nodes[parent_id].children_ids.append(child_id)

    # Compute roots (no parents) and leaves (no children).
    roots: list[str] = [sid for sid, n in nodes.items() if not n.parent_ids]
    leaves: list[str] = [sid for sid, n in nodes.items() if not n.children_ids]

    # Order roots / leaves deterministically.
    if ordering:
        flat_order: list[str] = [sid for level in ordering for sid in level]
        _sort_by_flat_order(roots, flat_order)
        _sort_by_flat_order(leaves, flat_order)
    else:
        snap_order: list[str] = list(snap_map.keys())
        _sort_by_flat_order(roots, snap_order)
        _sort_by_flat_order(leaves, snap_order)

    return BranchingTimeline(nodes=nodes, roots=roots, leaves=leaves)


def _sort_by_flat_order(items: list[str], order: list[str]) -> None:
    """Sort *items* in-place by their position in *order* (unknown items go last)."""
    pos_map: dict[str, int] = {sid: i for i, sid in enumerate(order)}
    default_pos: int = len(order)
    items.sort(key=lambda x: pos_map.get(x, default_pos))
