"""Tests for git export."""
from __future__ import annotations
import json, tempfile
from pathlib import Path
import pytest
from folderhistory.core.timeline import Timeline, TimelineNode
from folderhistory.types import EditOperation, Snapshot, FileRecord
from folderhistory.io.git_export import export_git

class TestGitExport:
    def test_empty_repo(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "repo"
            r = export_git(Timeline(nodes=[]), {}, out)
            assert r.exists()
    
    def test_single_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            sd = Path(td) / "snap"
            sd.mkdir()
            (sd / "f.txt").write_text("hello")
            snap = Snapshot(id="S0", timestamp=1000.0, source_path=sd, files=[
                FileRecord(path="f.txt", size=5, mode=0o644, mtime_ns=1000, ctime_ns=1000,
                           raw_blake3="abc", normalized_blake3=None, line_ending="lf",
                           xxhash64="def", is_symlink=False, target_path=None),
            ])
            node = TimelineNode(snapshot_id="S0", operations=[], parent_id=None, root_change=None,
                               timestamp=1000.0, source_path=str(sd))
            out = Path(td) / "repo"
            export_git(Timeline(nodes=[node]), {"S0": snap}, out)
            r = subprocess.run(["git","log","--oneline"], cwd=str(out), capture_output=True, text=True)
            assert len(r.stdout.strip().split("
")) >= 1
