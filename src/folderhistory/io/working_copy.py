"""Working Copy output format — intuitive directory-tree export.

Content/ stores the latest snapshot's full file tree (clean working copy with real names).
Snapshots/<id>/ stores CHANGES.txt + symlinks into Content/ for changed files only.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import orjson

from folderhistory.core.timeline import Timeline, TimelineNode
from folderhistory.types import EditOperation, Snapshot


class ContentStore:
    """Working tree store — Content/ mirrors the latest snapshot's file tree.
    Layout: Content/<relative_path> for each file in the latest snapshot.
    Older snapshots symlink into Content/ for their changed files.
    """
    _root: Path
    _content_dir: Path

    def __init__(self, root: Path) -> None:
        self._root = root
        self._content_dir = root / "Content"
        self._content_dir.mkdir(parents=True, exist_ok=True)

    def store_latest(self, snap: Snapshot) -> None:
        """Populate Content/ with the full file tree of the latest snapshot."""
        for fr in snap.files:
            src = snap.source_path / fr.path
            if src.exists():
                dest = self._content_dir / fr.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(str(src), str(dest))
                except PermissionError:
                    continue

    def store_snapshot_file(self, src: Path, rel_path: str) -> None:
        """Copy a single file into Content/ at the given relative path."""
        dest = self._content_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(src), str(dest))
        except PermissionError:
            pass

    def link_to(self, rel_path: str, target: Path) -> None:
        """Create a symlink to Content/<rel_path> at the target location."""
        target.parent.mkdir(parents=True, exist_ok=True)
        content_target = self._content_dir / rel_path
        if not content_target.exists():
            return
        try:
            os.symlink(os.path.relpath(str(content_target), str(target.parent)), str(target))
        except OSError:
            pass


def _write_changes_file(snap_dir: Path, ops: list[EditOperation], snap: Snapshot) -> None:
    machine: list[dict[str, object]] = []
    human: list[str] = [f"Snapshot: {snap.id}"]
    if snap.timestamp is not None:
        human.append(f"Timestamp: {snap.timestamp}")
    human.append("")
    human.append("Changes:")
    for op in ops:
        e: dict[str, object] = {"op": op.op_type}
        if op.source_path is not None:
            e["old_path"] = op.source_path
        if op.target_path is not None:
            e["new_path"] = op.target_path
        if op.old_hash is not None:
            e["old_hash"] = op.old_hash
        if op.new_hash is not None:
            e["new_hash"] = op.new_hash
        machine.append(e)
        if op.op_type == "create":
            human.append(f"  + {op.target_path} (created)")
        elif op.op_type == "delete":
            human.append(f"  - {op.source_path} (deleted)")
        elif op.op_type == "modify":
            human.append(f"  ~ {op.source_path} (modified)")
        elif op.op_type in ("rename", "move"):
            human.append(f'  > {op.source_path} -> {op.target_path}')
        elif op.op_type == "copy":
            human.append(f'  = {op.source_path} -> {op.target_path} (copied)')
    human.append(f"\n{len(ops)} operations")
    payload: dict[str, object] = {"machine": machine, "human": "\n".join(human)}
    (snap_dir / "CHANGES.txt").write_text(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode())


def _write_timeline_json(output_dir: Path, timeline: Timeline) -> None:
    snaps: list[dict[str, object]] = []
    ops_out: list[dict[str, object]] = []
    for node in timeline.nodes:
        snaps.append({
            "id": node.snapshot_id,
            "timestamp": getattr(node, "timestamp", None),
            "parent_id": node.parent_id,
            "op_count": len(node.operations),
        })
        for op in node.operations:
            e: dict[str, object] = {
                "op_type": op.op_type,
                "file_id": op.file_id,
                "snapshot_id": node.snapshot_id,
            }
            if op.source_path is not None:
                e["source_path"] = op.source_path
            if op.target_path is not None:
                e["target_path"] = op.target_path
            ops_out.append(e)
    payload: dict[str, object] = {
        "version": 1, "format": "working-copy",
        "snapshots": snaps, "operations": ops_out,
    }
    output_dir.joinpath("timeline.json").write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def export_working_copy(
    timeline: Timeline,
    snapshots: dict[str, Snapshot],
    output_dir: Path,
    *,
    exclude_deleted: bool = False,
) -> Path:
    """Export timeline as Working Copy layout.
    
    Content/ stores the latest snapshot's full file tree (clean working copy).
    Snapshots/<id>/ stores CHANGES.txt + symlinks into Content/ for changed files.
    """
    store = ContentStore(output_dir)
    sroot = output_dir / "Snapshots"
    sroot.mkdir(parents=True, exist_ok=True)
    latest_id = timeline.nodes[-1].snapshot_id if timeline.nodes else None

    for node in timeline.nodes:
        snap = snapshots.get(node.snapshot_id)
        if snap is None:
            continue
        sdir = sroot / node.snapshot_id
        sdir.mkdir(parents=True, exist_ok=True)
        is_latest = node.snapshot_id == latest_id

        if is_latest:
            store.store_latest(snap)
            for fr in snap.files:
                store.link_to(fr.path, sdir / fr.path)
        else:
            for op in node.operations:
                if op.op_type == "create" and op.new_hash and op.target_path:
                    src = snap.source_path / op.target_path
                    if src.exists():
                        store.store_snapshot_file(src, op.target_path)
                        store.link_to(op.target_path, sdir / op.target_path)
                elif op.op_type == "modify" and op.new_hash and op.source_path:
                    src = snap.source_path / op.source_path
                    if src.exists():
                        store.store_snapshot_file(src, op.source_path)
                        store.link_to(op.source_path, sdir / op.source_path)
                elif op.op_type == "delete" and not exclude_deleted and op.source_path:
                    content_src = store._content_dir / op.source_path
                    if content_src.exists():
                        dd = sdir / ".fh-deleted"
                        dd.mkdir(parents=True, exist_ok=True)
                        dst = dd / op.source_path
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.copy2(str(content_src), str(dst))
                        except OSError:
                            pass
                elif op.op_type in ("rename", "move") and op.new_hash:
                    if op.target_path:
                        src = snap.source_path / op.target_path
                        if src.exists():
                            store.store_snapshot_file(src, op.target_path)
                            store.link_to(op.target_path, sdir / op.target_path)
                    if op.source_path:
                        rd = sdir / ".fh-renamed"
                        rd.mkdir(parents=True, exist_ok=True)
                        stub = rd / op.source_path
                        stub.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            os.symlink(
                                os.path.relpath(str(sdir / op.target_path), str(stub.parent)),
                                str(stub),
                            )
                        except OSError:
                            pass
                elif op.op_type == "copy" and op.new_hash and op.target_path:
                    src = snap.source_path / op.target_path
                    if src.exists():
                        store.store_snapshot_file(src, op.target_path)
                        store.link_to(op.target_path, sdir / op.target_path)

        _write_changes_file(sdir, node.operations, snap)

    if latest_id is not None:
        ll = sroot / "latest"
        if ll.is_symlink() or ll.exists():
            ll.unlink()
        try:
            os.symlink(latest_id, ll)
        except OSError:
            pass

    _write_timeline_json(output_dir, timeline)
    return output_dir / "timeline.json"


def load_working_copy(timeline_json_path: Path) -> Timeline:
    """Reconstruct a Timeline from a Working Copy timeline.json."""
    data = json.loads(timeline_json_path.read_text())
    nodes: list[TimelineNode] = []
    for s in data["snapshots"]:
        sid: str = s["id"]
        ol: list[Any] = [o for o in data["operations"] if o["snapshot_id"] == sid]
        ops: list[EditOperation] = [
            EditOperation(
                op_type=o["op_type"],
                file_id=o.get("file_id", ""),
                source_path=o.get("source_path"),
                target_path=o.get("target_path"),
                old_hash=o.get("old_hash"),
                new_hash=o.get("new_hash"),
                confidence=o.get("confidence", 1.0),
            )
            for o in ol
        ]
        nodes.append(TimelineNode(
            snapshot_id=sid, operations=ops, parent_id=s.get("parent_id"),
            root_change=None, timestamp=s.get("timestamp"), source_path="",
        ))
    return Timeline(nodes=nodes)


def round_trip(output_dir: Path, new_snapshot_path: Path) -> Path:
    """Re-ingest content from a new snapshot, append to timeline."""
    from folderhistory.core.diff import derive_operations_between
    from folderhistory.core.identity import assign_identities_exact
    from folderhistory.core.ingest import ingest_snapshot as _ingest

    new_snap: Snapshot = _ingest(new_snapshot_path, snapshot_id=new_snapshot_path.name)
    timeline: Timeline = load_working_copy(output_dir / "timeline.json")

    if not timeline.nodes:
        return export_working_copy(
            Timeline(nodes=[TimelineNode(
                snapshot_id=new_snap.id, operations=[], parent_id=None,
                root_change=None, timestamp=new_snap.timestamp,
                source_path=str(new_snapshot_path),
            )]),
            {new_snap.id: new_snap},
            output_dir,
        )

    latest_node: TimelineNode = timeline.nodes[-1]
    latest_dir: Path = output_dir / "Snapshots" / latest_node.snapshot_id

    if latest_dir.exists():
        latest_snap: Snapshot = _ingest(latest_dir, snapshot_id=latest_node.snapshot_id)
    else:
        latest_snap = Snapshot(
            id=latest_node.snapshot_id,
            timestamp=getattr(latest_node, "timestamp", None),
            source_path=Path(latest_node.source_path or "."),
            files=[],
        )

    identities: dict[str, Any] = assign_identities_exact([latest_snap, new_snap])
    ops: list[EditOperation] = derive_operations_between(latest_snap, new_snap, identities)

    new_node: TimelineNode = TimelineNode(
        snapshot_id=new_snap.id, operations=ops,
        parent_id=latest_node.snapshot_id, root_change=None,
        timestamp=new_snap.timestamp, source_path=str(new_snapshot_path),
    )
    new_timeline: Timeline = Timeline(nodes=list(timeline.nodes) + [new_node])
    all_snaps: dict[str, Snapshot] = {}
    for n in new_timeline.nodes:
        if n.snapshot_id == new_snap.id:
            all_snaps[n.snapshot_id] = new_snap
        else:
            d = output_dir / "Snapshots" / n.snapshot_id
            if d.exists():
                all_snaps[n.snapshot_id] = _ingest(d, snapshot_id=n.snapshot_id)
            else:
                all_snaps[n.snapshot_id] = Snapshot(
                    id=n.snapshot_id, timestamp=getattr(n, "timestamp", None),
                    source_path=Path(n.source_path or "."), files=[],
                )
    return export_working_copy(new_timeline, all_snaps, output_dir)
