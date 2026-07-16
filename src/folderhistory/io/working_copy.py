"""Working Copy output format — intuitive directory-tree export."""
from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path

import orjson

from folderhistory.core.timeline import Timeline
from folderhistory.types import EditOperation, Snapshot


class ContentStore:
    """Deduplicated content store keyed by BLAKE3 hash."""
    _root: Path
    _content_dir: Path

    def __init__(self, root: Path) -> None:
        self._root = root
        self._content_dir = root / "Content"
        self._content_dir.mkdir(parents=True, exist_ok=True)

    def _hash_path(self, hash_hex: str) -> Path:
        return self._content_dir / hash_hex[:2] / hash_hex

    def store_bytes(self, data: bytes) -> str:
        import blake3 as _b3
        h: str = _b3.blake3(data).hexdigest()
        p: Path = self._hash_path(h)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)  # noqa
        return h

    def store_file(self, src: Path) -> str:
        return self.store_bytes(src.read_bytes())

    def get_path(self, hash_hex: str) -> Path:
        p: Path = self._hash_path(hash_hex)
        if not p.exists():
            raise KeyError(f"Content not found: {hash_hex}")
        return p

    def link_to(self, hash_hex: str, target: Path) -> None:
        src: Path = self._hash_path(hash_hex)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(os.path.relpath(str(src), str(target.parent)), str(target))
        except OSError:
            try:
                os.link(str(src), str(target))
            except OSError:
                shutil.copy2(str(src), str(target))

    def has(self, hash_hex: str) -> bool:
        return self._hash_path(hash_hex).exists()


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
    (snap_dir / "CHANGES.txt").write_text(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode()
    )


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
    output_dir.joinpath("timeline.json").write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2)
    )


def export_working_copy(
    timeline: Timeline, snapshots: Mapping[str, Snapshot],
    output_dir: Path, *, exclude_deleted: bool = False,
) -> Path:
    store = ContentStore(output_dir)
    sroot = output_dir / "Snapshots"
    sroot.mkdir(parents=True, exist_ok=True)
    latest_id: str | None = timeline.nodes[-1].snapshot_id if timeline.nodes else None
    for node in timeline.nodes:
        snap = snapshots.get(node.snapshot_id)
        if snap is None:
            continue
        sdir = sroot / node.snapshot_id
        sdir.mkdir(parents=True, exist_ok=True)
        is_latest = node.snapshot_id == latest_id
        if is_latest:
            for fr in snap.files:
                src = snap.source_path / fr.path
                if src.exists():
                    _ = store.store_file(src)
                    _ = store.link_to(fr.raw_blake3, sdir / fr.path)
        else:
            for op in node.operations:
                if op.op_type == "create" and op.new_hash and op.target_path:
                    src = snap.source_path / op.target_path
                    if src.exists():
                        _ = store.store_file(src)
                        _ = store.link_to(op.new_hash, sdir / op.target_path)
                elif op.op_type == "modify" and op.new_hash and op.source_path:
                    src = snap.source_path / op.source_path
                    if src.exists():
                        _ = store.store_file(src)
                        _ = store.link_to(op.new_hash, sdir / op.source_path)
                elif op.op_type == "delete" and not exclude_deleted and op.old_hash and op.source_path:
                    try:
                        dd = sdir / ".fh-deleted"
                        dd.mkdir(parents=True, exist_ok=True)
                        dst = dd / op.source_path
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        _ = store.link_to(op.old_hash, dst)
                    except KeyError:
                        pass
                elif op.op_type in ("rename", "move") and op.new_hash:
                    if op.target_path:
                        src = snap.source_path / op.target_path
                        if src.exists():
                            _ = store.store_file(src)
                            _ = store.link_to(op.new_hash, sdir / op.target_path)
                    if op.source_path:
                        rd = sdir / ".fh-renamed"
                        rd.mkdir(parents=True, exist_ok=True)
                        stub = rd / op.source_path
                        stub.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            os.symlink(
                                os.path.relpath(str(sdir / str(op.target_path)), str(stub.parent)),
                                str(stub),
                            )
                        except OSError:
                            pass
                elif op.op_type == "copy" and op.new_hash and op.target_path:
                    src = snap.source_path / op.target_path
                    if src.exists():
                        _ = store.store_file(src)
                        _ = store.link_to(op.new_hash, sdir / op.target_path)
        _write_changes_file(sdir, node.operations, snap)
    if latest_id is not None:
        ll = sroot / "latest"
        if ll.is_symlink() or ll.exists():
            ll.unlink()
        try:
            os.symlink(latest_id, str(ll))
        except OSError:
            pass
    _write_timeline_json(output_dir, timeline)
    return output_dir / "timeline.json"
