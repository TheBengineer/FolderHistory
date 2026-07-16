"""Tests for the Working Copy output format."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from folderhistory.core.timeline import Timeline, TimelineNode
from folderhistory.io.working_copy import ContentStore, export_working_copy
from folderhistory.types import EditOperation, FileRecord, Snapshot


class TestContentStore:
    def test_store_and_get(self):
        with tempfile.TemporaryDirectory() as td:
            store = ContentStore(Path(td))
            h = store.store_bytes(b"hello world")
            p = store.get_path(h)
            assert p.exists()
            assert p.read_bytes() == b"hello world"

    def test_dedup(self):
        with tempfile.TemporaryDirectory() as td:
            store = ContentStore(Path(td))
            h1 = store.store_bytes(b"data")
            h2 = store.store_bytes(b"data")
            assert h1 == h2

    def test_has(self):
        with tempfile.TemporaryDirectory() as td:
            store = ContentStore(Path(td))
            h = store.store_bytes(b"test")
            assert store.has(h)
            assert not store.has("nonexistent")


class TestExportWorkingCopy:
    def test_empty_timeline(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            result = export_working_copy(Timeline(nodes=[]), {}, out)
            assert result.name == "timeline.json"

    def test_single_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            snap_dir = Path(td) / "snap"
            snap_dir.mkdir()
            (snap_dir / "hello.txt").write_text("world")
            snap = Snapshot(id="S0", timestamp=1000.0, source_path=snap_dir, files=[
                FileRecord(path="hello.txt", size=5, mode=0o644, mtime_ns=1000, ctime_ns=1000,
                           raw_blake3="abc123", normalized_blake3=None, line_ending="lf",
                           xxhash64="def456", is_symlink=False, target_path=None),
            ])
            node = TimelineNode(snapshot_id="S0", operations=[], parent_id=None, root_change=None,
                                timestamp=1000.0, source_path=str(snap_dir))
            out = Path(td) / "out"
            result = export_working_copy(Timeline(nodes=[node]), {"S0": snap}, out)
            assert result.exists()
            tj = json.loads(result.read_text())
            assert tj["format"] == "working-copy"
            assert len(tj["snapshots"]) == 1

    def test_changes_file(self):
        with tempfile.TemporaryDirectory() as td:
            snap_dir = Path(td) / "snap"
            snap_dir.mkdir()
            snap = Snapshot(id="S0", timestamp=1000.0, source_path=snap_dir, files=[])
            ops = [EditOperation(op_type="create", file_id="f1", target_path="new.txt",
                                 new_hash="abc", confidence=1.0)]
            from folderhistory.io.working_copy import _write_changes_file
            sd = Path(td) / "snap_out"
            sd.mkdir()
            _write_changes_file(sd, ops, snap)
            assert (sd / "CHANGES.txt").exists()
            data = json.loads((sd / "CHANGES.txt").read_text())
            assert len(data["machine"]) == 1
            assert "(created)" in data["human"]
