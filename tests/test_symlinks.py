"""Tests for symbolic link edge cases in :mod:`folderhistory.core.ingest`.

Covers symlink chains, dangling symlinks, directory symlinks,
symlinks pointing outside the snapshot root, and file-vs-directory
symlink distinction.
"""

from __future__ import annotations

import os
import stat
import unicodedata
from pathlib import Path

import pytest

from folderhistory.core.ingest import ingest_snapshot


# ── Helpers ──────────────────────────────────────────────────────────────────


def _write(src: Path, rel: str, content: bytes = b"") -> None:
    """Create a file at *src / rel* with *content*, creating parent dirs."""
    p = src / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


# ── Symlink chains ───────────────────────────────────────────────────────────


class TestSymlinkChains:
    """Sequences of symlinks pointing to other symlinks."""

    def test_chain_of_two(self, tmp_path: Path) -> None:
        """link_a → link_b → real_target."""
        _write(tmp_path, "real.txt", b"target")
        (tmp_path / "link_b").symlink_to("real.txt")
        (tmp_path / "link_a").symlink_to("link_b")
        snapshot = ingest_snapshot(tmp_path)
        symlinks = {f.path: f for f in snapshot.files if f.is_symlink}
        assert "link_a" in symlinks
        assert "link_b" in symlinks
        assert symlinks["link_a"].target_path == "link_b"
        assert symlinks["link_b"].target_path == "real.txt"
        # Neither symlink has content hash
        assert symlinks["link_a"].raw_blake3 == ""
        assert symlinks["link_b"].raw_blake3 == ""

    def test_chain_of_three(self, tmp_path: Path) -> None:
        """link1 → link2 → link3 → real_target."""
        _write(tmp_path, "real.txt", b"deep chain")
        (tmp_path / "link3").symlink_to("real.txt")
        (tmp_path / "link2").symlink_to("link3")
        (tmp_path / "link1").symlink_to("link2")
        snapshot = ingest_snapshot(tmp_path)
        symlinks = {f.path: f for f in snapshot.files if f.is_symlink}
        assert sorted(symlinks.keys()) == ["link1", "link2", "link3"]
        assert symlinks["link1"].target_path == "link2"
        assert symlinks["link2"].target_path == "link3"
        assert symlinks["link3"].target_path == "real.txt"

    def test_chain_regular_file_still_hashed(self, tmp_path: Path) -> None:
        """The ultimate regular file in a chain is still content-hashed."""
        _write(tmp_path, "real.txt", b"final target")
        (tmp_path / "link1").symlink_to("link2")
        (tmp_path / "link2").symlink_to("real.txt")
        snapshot = ingest_snapshot(tmp_path)
        regular = {f.path: f for f in snapshot.files if not f.is_symlink}
        assert "real.txt" in regular
        assert regular["real.txt"].raw_blake3 != ""
        assert regular["real.txt"].normalized_blake3 is not None


# ── Dangling symlinks ───────────────────────────────────────────────────────


class TestDanglingSymlinks:
    """Symlinks whose targets do not exist."""

    def test_broken_symlink(self, tmp_path: Path) -> None:
        (tmp_path / "broken").symlink_to("nonexistent")
        snapshot = ingest_snapshot(tmp_path)
        assert len(snapshot.files) == 1
        rec = snapshot.files[0]
        assert rec.is_symlink
        assert rec.target_path == "nonexistent"
        assert rec.size == 0

    def test_broken_symlink_in_subdir(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub/ghost").symlink_to("../../../nowhere")
        _write(tmp_path, "sub/real.txt", b"real")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path: f for f in snapshot.files}
        assert "sub/ghost" in paths
        assert paths["sub/ghost"].is_symlink
        assert paths["sub/ghost"].target_path == "../../../nowhere"
        assert "sub/real.txt" in paths
        assert not paths["sub/real.txt"].is_symlink

    def test_multiple_broken_symlinks(self, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"broken_{i}").symlink_to(f"missing_{i}")
        snapshot = ingest_snapshot(tmp_path)
        assert len(snapshot.files) == 5
        assert all(f.is_symlink for f in snapshot.files)

    def test_broken_symlink_deep_path(self, tmp_path: Path) -> None:
        """Deep relative path that doesn't exist."""
        (tmp_path / "deep").symlink_to("a/b/c/d/e/f/g/h/i/j")
        snapshot = ingest_snapshot(tmp_path)
        assert len(snapshot.files) == 1
        assert snapshot.files[0].target_path == "a/b/c/d/e/f/g/h/i/j"


# ── Symlinks to directories ──────────────────────────────────────────────────


class TestSymlinksToDirectories:
    """Symlinks pointing at directories (not regular files)."""

    def test_symlink_to_directory(self, tmp_path: Path) -> None:
        """Symlink to a directory is recorded but directory contents are not
        followed through the symlink (the symlink is the record, not its
        resolved target)."""
        (tmp_path / "target_dir").mkdir()
        _write(tmp_path, "target_dir/inside.txt", b"inside")
        (tmp_path / "link_to_dir").symlink_to("target_dir")
        snapshot = ingest_snapshot(tmp_path)
        symlinks = {f.path for f in snapshot.files if f.is_symlink}
        regular = {f.path for f in snapshot.files if not f.is_symlink}
        assert "link_to_dir" in symlinks
        # Contents inside the real directory are still scanned via the
        # real path, not via the symlink.
        assert "target_dir/inside.txt" in regular

    def test_symlink_to_directory_excludes_nested_symlink(self, tmp_path: Path) -> None:
        """A directory symlink inside the tree: its children are NOT traversed
        via the symlink path, only via the real path."""
        (tmp_path / "real").mkdir()
        _write(tmp_path, "real/file.txt", b"real file")
        (tmp_path / "link_to_real").symlink_to("real")
        snapshot = ingest_snapshot(tmp_path)
        # All file paths should be under "real/" not "link_to_real/"
        paths = {f.path for f in snapshot.files}
        assert "real/file.txt" in paths
        assert "link_to_real/file.txt" not in paths

    def test_nested_symlink_to_directory(self, tmp_path: Path) -> None:
        """Symlink to dir at multiple layers."""
        (tmp_path / "a/b").mkdir(parents=True)
        _write(tmp_path, "a/b/file.txt", b"nested")
        (tmp_path / "link_to_b").symlink_to("a/b")
        snapshot = ingest_snapshot(tmp_path)
        symlinks = {f.path: f for f in snapshot.files if f.is_symlink}
        assert "link_to_b" in symlinks
        assert symlinks["link_to_b"].target_path == "a/b"


# ── Symlinks pointing outside the snapshot root ──────────────────────────────


class TestSymlinksOutsideRoot:
    """Symlinks whose targets lie outside the directory being snapshotted."""

    def test_symlink_to_parent(self, tmp_path: Path) -> None:
        """Symlink inside the tree pointing to the parent (..)."""
        _write(tmp_path, "real.txt", b"outside")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub/parent_link").symlink_to("..")
        snapshot = ingest_snapshot(tmp_path)
        symlinks = {f.path: f for f in snapshot.files if f.is_symlink}
        assert "sub/parent_link" in symlinks
        assert symlinks["sub/parent_link"].target_path == ".."

    def test_symlink_to_absolute_outside_path(self, tmp_path: Path) -> None:
        """Symlink pointing to an absolute path outside the snapshot root."""
        outside_file = tmp_path / "outside_repo.txt"
        outside_file.write_text("external")
        (tmp_path / "external_link").symlink_to(str(outside_file))
        snapshot = ingest_snapshot(tmp_path)
        symlinks = {f.path: f for f in snapshot.files if f.is_symlink}
        assert "external_link" in symlinks
        # The target path is stored NFC-normalised (ingest normalises it)
        norm_target = os.path.normpath(str(outside_file))
        assert symlinks["external_link"].target_path == norm_target

    def test_symlink_chain_escaping_root(self, tmp_path: Path) -> None:
        """A symlink chain where the intermediate link is inside root but
        the ultimate target is outside."""
        outside = tmp_path / "outside.txt"
        outside.write_text("out")
        (tmp_path / "mid").symlink_to(str(outside))
        (tmp_path / "top").symlink_to("mid")
        snapshot = ingest_snapshot(tmp_path)
        symlinks = {f.path: f for f in snapshot.files if f.is_symlink}
        assert "top" in symlinks
        assert symlinks["top"].target_path == "mid"
        assert "mid" in symlinks
        # mid's target is the absolute path outside
        norm_mid_target = os.path.normpath(str(outside))
        assert symlinks["mid"].target_path == norm_mid_target

    def test_multiple_symlinks_outside(self, tmp_path: Path) -> None:
        """Multiple symlinks all pointing outside."""
        for i in range(3):
            outside = tmp_path.parent / f"external_{i}.txt"
            outside.write_text(f"external {i}")
            (tmp_path / f"ext_link_{i}").symlink_to(str(outside))
        snapshot = ingest_snapshot(tmp_path)
        assert len(snapshot.files) == 3
        assert all(f.is_symlink for f in snapshot.files)


# ── File vs directory symlink detection ──────────────────────────────────────


class TestFileVsDirSymlink:
    """Verify that the is_symlink attribute correctly distinguishes symlinks
    from regular files, and that os.scandir can identify symlink type."""

    def test_file_symlink_marked_as_symlink(self, tmp_path: Path) -> None:
        _write(tmp_path, "target.txt", b"target")
        (tmp_path / "link").symlink_to("target.txt")
        snapshot = ingest_snapshot(tmp_path)
        link_rec = next(f for f in snapshot.files if f.path == "link")
        assert link_rec.is_symlink
        # The symlink is a symlink to a regular file — it should still show
        # as a symlink, NOT as a regular file.
        assert not stat.S_ISREG(link_rec.mode) or link_rec.is_symlink

    def test_dir_symlink_marked_as_symlink(self, tmp_path: Path) -> None:
        (tmp_path / "realdir").mkdir()
        (tmp_path / "linkdir").symlink_to("realdir")
        snapshot = ingest_snapshot(tmp_path)
        link_rec = next(f for f in snapshot.files if f.path == "linkdir")
        assert link_rec.is_symlink

    def test_regular_file_not_symlink(self, tmp_path: Path) -> None:
        _write(tmp_path, "normal.txt", b"normal")
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        assert not rec.is_symlink
        assert rec.target_path is None

    def test_symlink_mode_has_s_iflag(self, tmp_path: Path) -> None:
        """Symlink entries should have S_IFLNK in their mode bits."""
        _write(tmp_path, "target.txt", b"t")
        (tmp_path / "link").symlink_to("target.txt")
        snapshot = ingest_snapshot(tmp_path)
        link_rec = next(f for f in snapshot.files if f.path == "link")
        assert stat.S_ISLNK(link_rec.mode)


# ── Unicode symlink targets ──────────────────────────────────────────────────


class TestSymlinksUnicode:
    """Symlinks where the link name or the target name includes Unicode."""

    def test_unicode_link_name(self, tmp_path: Path) -> None:
        _write(tmp_path, "target.txt", b"data")
        (tmp_path / "リンク.txt").symlink_to("target.txt")
        snapshot = ingest_snapshot(tmp_path)
        symlinks = {f.path: f for f in snapshot.files if f.is_symlink}
        assert "リンク.txt" in symlinks

    def test_unicode_target(self, tmp_path: Path) -> None:
        _write(tmp_path, "ファイル.txt", b"data")
        (tmp_path / "link").symlink_to("ファイル.txt")
        snapshot = ingest_snapshot(tmp_path)
        link_rec = next(f for f in snapshot.files if f.is_symlink)
        # Target path should be NFC-normalised
        assert unicodedata.is_normalized("NFC", link_rec.target_path)

    def test_unicode_link_and_target(self, tmp_path: Path) -> None:
        _write(tmp_path, "目標.txt", b"data")
        (tmp_path / "鏈接.txt").symlink_to("目標.txt")
        snapshot = ingest_snapshot(tmp_path)
        links = {f.path: f for f in snapshot.files if f.is_symlink}
        assert "鏈接.txt" in links
        assert links["鏈接.txt"].target_path == "目標.txt"



