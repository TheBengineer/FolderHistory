"""Git repository export for FolderHistory — Timeline → real git repos."""
from __future__ import annotations

import json
import os
import shutil
import stat as _stat
import subprocess
import tempfile
from pathlib import Path  # noqa

from folderhistory.core.timeline import Timeline, TimelineNode
from folderhistory.io.working_copy import ContentStore
from folderhistory.types import EditOperation, Snapshot, FileRecord

NOTES_METADATA = "refs/notes/fh/metadata"
NOTES_CONFIDENCE = "refs/notes/fh/confidence"
NOTES_EMPTYDIRS = "refs/notes/fh/emptydirs"


def _git(repo_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + list(args),
        cwd=str(repo_path),
        capture_output=True,
        text=True,
    )


def export_git(
    timeline: Timeline,
    snapshots: dict[str, Snapshot],
    output_dir: Path,
    *,
    author_name: str = "FolderHistory",
    author_email: str = "fh@localhost",
    branch: str = "main",
) -> Path:
    """Export timeline as git repo. Returns path to repo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # git init
    _git(output_dir, "init", "-b", branch)
    _git(output_dir, "config", "user.name", author_name)
    _git(output_dir, "config", "user.email", author_email)
    _git(output_dir, "config", "diff.renames", "copies")
    
    store = ContentStore(output_dir)
    
    for node in timeline.nodes:
        snap = snapshots.get(node.snapshot_id)
        if snap is None:
            continue
        
        # Write all snapshot files into working tree
        for fr in snap.files:
            src = snap.source_path / fr.path
            if src.exists():
                store.store_file(src)
                dest = output_dir / fr.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(str(src), str(dest))
                except PermissionError:
                    continue
        
        # git add -A
        _git(output_dir, "add", "-A")
        
        # Build commit message with file-level details
        ops_by_type: dict[str, list[str]] = {}
        for op in node.operations:
            ops_by_type.setdefault(op.op_type, [])
            path = op.target_path or op.source_path or op.file_id
            if path not in ops_by_type[op.op_type]:
                ops_by_type[op.op_type].append(path)
        summary = ", ".join(f"{len(v)} {k}" for k, v in sorted(ops_by_type.items())) or "no changes"
        details_list: list[str] = []
        for op_type in ["create", "modify", "rename", "move", "delete", "copy"]:
            if op_type in ops_by_type:
                details_list.append(op_type.title() + ":")
                for p in ops_by_type[op_type]:
                    details_list.append("  " + p)
        details = "\n" + "\n".join(details_list) if details_list else ""
        msg = f"Snapshot: {node.snapshot_id}\n\n{summary}{details}"
        r = _git(output_dir, "commit", "-m", msg, "--allow-empty")
        if r.returncode != 0:
            continue
        
        # Get commit hash
        r2 = _git(output_dir, "rev-parse", "HEAD")
        commit_hash = r2.stdout.strip()
        
        # Attach git notes
        _attach_metadata_notes(output_dir, commit_hash, node, snap)
        _attach_confidence_notes(output_dir, commit_hash, node)
    
    return output_dir


def _attach_metadata_notes(repo_path: Path, commit_hash: str, node: TimelineNode, snap: Snapshot) -> None:
    """Attach metadata note (ctime, mtime, mode, empty dirs)."""
    metadata: dict[str, Any] = {"files": {}, "empty_dirs": []}
    for fr in snap.files:
        metadata["files"][fr.path] = {
            "ctime_ns": fr.ctime_ns,
            "mtime_ns": fr.mtime_ns,
            "mode": fr.mode,
        }
    note_json = json.dumps(metadata, indent=2)
    _git(repo_path, "notes", "--ref=" + NOTES_METADATA, "add", "-m", note_json, commit_hash)


def _attach_confidence_notes(repo_path: Path, commit_hash: str, node: TimelineNode) -> None:
    """Attach confidence note for probabilistic operations."""
    confs: dict[str, float] = {}
    for op in node.operations:
        confs[op.file_id] = op.confidence
    if confs:
        note_json = json.dumps(confs, indent=2)
        _git(repo_path, "notes", "--ref=" + NOTES_CONFIDENCE, "add", "-m", note_json, commit_hash)


def export_git_from_timeline_json(timeline_json_path: Path, snapshots_dir: Path, output_dir: Path) -> Path:
    """Load a timeline.json and export as git repo."""
    from folderhistory.core.ingest import ingest_snapshot
    from folderhistory.io.working_copy import load_working_copy
    
    timeline = load_working_copy(timeline_json_path)
    snapshots: dict[str, Snapshot] = {}
    for node in timeline.nodes:
        sd = snapshots_dir / node.snapshot_id
        if sd.exists():
            snapshots[node.snapshot_id] = ingest_snapshot(sd, snapshot_id=node.snapshot_id)
    
    return export_git(timeline, snapshots, output_dir)
