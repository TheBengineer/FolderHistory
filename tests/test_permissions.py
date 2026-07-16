"""Tests for permission edge cases in :mod:`folderhistory.core.ingest`.

Covers permission-denied directories, unreadable files,
mixed permission trees, and files with various restricted modes.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import pytest

from folderhistory.core.ingest import ingest_snapshot


# ── Helpers ──────────────────────────────────────────────────────────────────


def _write(src: Path, rel: str, content: bytes = b"") -> None:
    """Create a file at *src / rel* with *content*, creating parent dirs."""
    p = src / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def _can_chmod() -> bool:
    """Check whether running as root (where chmod restrictions don't apply)."""
    try:
        return os.geteuid() != 0
    except AttributeError:
        return True


# ── Permission-denied directories ────────────────────────────────────────────


class TestPermissionDeniedDirectories:
    """Directories that cannot be listed due to missing +x / +r bits."""

    def test_no_execute_on_directory(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Directory without +x cannot be entered; children are skipped."""
        if not _can_chmod():
            pytest.skip("Running as root: permission restrictions are bypassed")
        secret = tmp_path / "no_access"
        secret.mkdir()
        _write(secret, "hidden.txt", b"secret")
        # Remove all permissions
        secret.chmod(0o000)
        try:
            with caplog.at_level(logging.WARNING):
                snapshot = ingest_snapshot(tmp_path)
            assert snapshot.files == [], "No files should be discoverable"
            assert any("Permission denied reading directory" in r.message for r in caplog.records)
        finally:
            secret.chmod(0o755)

    def test_no_read_on_directory(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Directory without +r but with +x: can be entered but not listed."""
        if not _can_chmod():
            pytest.skip("Running as root: permission restrictions are bypassed")
        secret = tmp_path / "no_read"
        secret.mkdir()
        _write(secret, "file.txt", b"data")
        # +x but no +r
        secret.chmod(0o111)
        try:
            with caplog.at_level(logging.WARNING):
                snapshot = ingest_snapshot(tmp_path)
            # On Linux, os.scandir on a +x-only dir returns EACCES
            assert "file.txt" not in {f.path for f in snapshot.files}
            assert any("Permission denied" in r.message for r in caplog.records)
        finally:
            secret.chmod(0o755)

    def test_nested_permission_denied_directory(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A permission-denied directory deep in the tree blocks its subtree
        but leaves sibling subtrees accessible."""
        if not _can_chmod():
            pytest.skip("Running as root: permission restrictions are bypassed")
        _write(tmp_path, "public/a.txt", b"public")
        secret = tmp_path / "secret"
        secret.mkdir()
        _write(secret, "sub/private.txt", b"private")
        # Block all access to the secret directory
        secret.chmod(0o000)
        try:
            with caplog.at_level(logging.WARNING):
                snapshot = ingest_snapshot(tmp_path)
            paths = {f.path for f in snapshot.files}
            assert "public/a.txt" in paths
            assert "secret/sub/private.txt" not in paths
        finally:
            secret.chmod(0o755)

    def test_multiple_restricted_directories(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Multiple blocked directories all produce warnings."""
        if not _can_chmod():
            pytest.skip("Running as root: permission restrictions are bypassed")
        for name in ("blocked_a", "blocked_b", "blocked_c"):
            d = tmp_path / name
            d.mkdir()
            _write(d, "f.txt", b"content")
            d.chmod(0o000)
        try:
            with caplog.at_level(logging.WARNING):
                snapshot = ingest_snapshot(tmp_path)
            assert snapshot.files == []
            # Count distinct directories in warning messages
            blocked_dirs = sum(1 for r in caplog.records if "Permission denied reading directory" in r.message)
            assert blocked_dirs == 3
        finally:
            for name in ("blocked_a", "blocked_b", "blocked_c"):
                (tmp_path / name).chmod(0o755)


# ── Permission-denied files ──────────────────────────────────────────────────


class TestUnreadableFiles:
    """Files that cannot be read but are discoverable (stat succeeds, read fails)."""

    def test_file_with_no_read_permission(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """File exists, can be stat'd, but cannot be read."""
        if not _can_chmod():
            pytest.skip("Running as root: permission restrictions are bypassed")
        _write(tmp_path, "public.txt", b"ok")
        _write(tmp_path, "secret.bin", b"hidden")
        (tmp_path / "secret.bin").chmod(0o000)
        try:
            with caplog.at_level(logging.WARNING):
                snapshot = ingest_snapshot(tmp_path)
            paths = {f.path for f in snapshot.files}
            assert "public.txt" in paths
            assert "secret.bin" not in paths
            assert any("Permission denied reading file" in r.message for r in caplog.records)
        finally:
            (tmp_path / "secret.bin").chmod(0o644)

    def test_file_with_only_execute_permission(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """File with --x--x--x mode cannot be read for hashing."""
        if not _can_chmod():
            pytest.skip("Running as root: permission restrictions are bypassed")
        _write(tmp_path, "exec_only.sh", b"#!/bin/sh\necho hi\n")
        (tmp_path / "exec_only.sh").chmod(0o111)
        try:
            with caplog.at_level(logging.WARNING):
                snapshot = ingest_snapshot(tmp_path)
            assert "exec_only.sh" not in {f.path for f in snapshot.files}
            assert any("Permission denied" in r.message for r in caplog.records)
        finally:
            (tmp_path / "exec_only.sh").chmod(0o644)

    def test_file_with_write_only_permission(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """File with -w--w--w- cannot be read."""
        if not _can_chmod():
            pytest.skip("Running as root: permission restrictions are bypassed")
        _write(tmp_path, "write_only.txt", b"data")
        (tmp_path / "write_only.txt").chmod(0o222)
        try:
            with caplog.at_level(logging.WARNING):
                snapshot = ingest_snapshot(tmp_path)
            assert "write_only.txt" not in {f.path for f in snapshot.files}
        finally:
            (tmp_path / "write_only.txt").chmod(0o644)

    def test_all_files_unreadable(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """All files in the tree are unreadable; result should be empty."""
        if not _can_chmod():
            pytest.skip("Running as root: permission restrictions are bypassed")
        for name in ("a.txt", "b.txt", "c.txt"):
            _write(tmp_path, name, b"content")
            (tmp_path / name).chmod(0o000)
        try:
            with caplog.at_level(logging.WARNING):
                snapshot = ingest_snapshot(tmp_path)
            assert snapshot.files == []
        finally:
            for name in ("a.txt", "b.txt", "c.txt"):
                (tmp_path / name).chmod(0o644)


# ── Mixed permission trees ───────────────────────────────────────────────────


class TestMixedPermissionTrees:
    """Various combinations of accessible and restricted entries."""

    def test_accessible_file_in_restricted_subdir(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A file with normal permissions inside a directory without +x
        should NOT be discoverable because the directory cannot be entered."""
        if not _can_chmod():
            pytest.skip("Running as root: permission restrictions are bypassed")
        sub = tmp_path / "sub"
        sub.mkdir()
        _write(sub, "file.txt", b"accessible")
        _write(tmp_path, "root.txt", b"root")
        sub.chmod(0o000)
        try:
            with caplog.at_level(logging.WARNING):
                snapshot = ingest_snapshot(tmp_path)
            paths = {f.path for f in snapshot.files}
            assert "root.txt" in paths
            assert "sub/file.txt" not in paths
        finally:
            sub.chmod(0o755)

    def test_readable_file_in_executable_only_dir(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A directory with +x but no +r: cannot list children via scandir,
        so files inside are invisible."""
        if not _can_chmod():
            pytest.skip("Running as root: permission restrictions are bypassed")
        sub = tmp_path / "sub"
        sub.mkdir()
        _write(sub, "file.txt", b"data")
        sub.chmod(0o111)  # --x--x--x
        try:
            with caplog.at_level(logging.WARNING):
                snapshot = ingest_snapshot(tmp_path)
            assert "sub/file.txt" not in {f.path for f in snapshot.files}
        finally:
            sub.chmod(0o755)

    def test_large_mixed_permission_tree(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A tree with a mix of accessible and blocked branches."""
        if not _can_chmod():
            pytest.skip("Running as root: permission restrictions are bypassed")
        # Accessible branch
        _write(tmp_path, "good/a/1.txt", b"1")
        _write(tmp_path, "good/a/2.txt", b"2")
        _write(tmp_path, "good/b.txt", b"b")
        # Blocked branch
        blocked = tmp_path / "bad"
        blocked.mkdir()
        _write(blocked, "x.txt", b"x")
        _write(blocked, "y.txt", b"y")
        blocked.chmod(0o000)
        try:
            with caplog.at_level(logging.WARNING):
                snapshot = ingest_snapshot(tmp_path)
            paths = {f.path for f in snapshot.files}
            assert "good/a/1.txt" in paths
            assert "good/a/2.txt" in paths
            assert "good/b.txt" in paths
            assert "bad/x.txt" not in paths
            assert "bad/y.txt" not in paths
        finally:
            blocked.chmod(0o755)


# ── Read-only files (ingest behaviour) ───────────────────────────────────────


class TestReadOnlyFiles:
    """Read-only files should be fully ingested (readable on Unix)."""

    def test_read_only_file(self, tmp_path: Path) -> None:
        _write(tmp_path, "readonly.txt", b"immutable")
        (tmp_path / "readonly.txt").chmod(0o444)
        snapshot = ingest_snapshot(tmp_path)
        assert len(snapshot.files) == 1
        rec = snapshot.files[0]
        assert rec.path == "readonly.txt"
        assert rec.size == 9
        assert rec.raw_blake3 != ""

    def test_read_only_directory(self, tmp_path: Path) -> None:
        """A read-only directory (no +w) should still be traversable."""
        sub = tmp_path / "readonly_dir"
        sub.mkdir()
        _write(sub, "file.txt", b"data")
        sub.chmod(0o555)  # r-xr-xr-x
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert "readonly_dir/file.txt" in paths

    def test_read_only_files_in_subdirs(self, tmp_path: Path) -> None:
        _write(tmp_path, "a/b/c.txt", b"nested readonly")
        (tmp_path / "a/b/c.txt").chmod(0o444)
        snapshot = ingest_snapshot(tmp_path)
        assert len(snapshot.files) == 1
        assert snapshot.files[0].path == "a/b/c.txt"


# ── Immutable / special attributes (Linux chattr) ────────────────────────────


class TestSpecialAttributes:
    """Tests for files with immutable or special filesystem attributes."""

    def test_file_is_ingested_regardless_of_sticky_bit(self, tmp_path: Path) -> None:
        """Sticky bit on a file should not affect ingest."""
        _write(tmp_path, "sticky.txt", b"sticky")
        st = (tmp_path / "sticky.txt").stat()
        # Add sticky bit
        os.chmod(str(tmp_path / "sticky.txt"), st.st_mode | stat.S_ISVTX)
        snapshot = ingest_snapshot(tmp_path)
        assert len(snapshot.files) == 1

    def test_setuid_file(self, tmp_path: Path) -> None:
        """Setuid bit on a file (common for executables) — normal ingest."""
        _write(tmp_path, "suid_prog", b"binary content")
        st = (tmp_path / "suid_prog").stat()
        os.chmod(str(tmp_path / "suid_prog"), st.st_mode | stat.S_ISUID)
        snapshot = ingest_snapshot(tmp_path)
        assert len(snapshot.files) == 1
        assert snapshot.files[0].path == "suid_prog"
