"""Tests for :mod:`folderhistory.core.ingest`."""

from __future__ import annotations

import logging
import os
import unicodedata
from pathlib import Path

import orjson
import pytest

from folderhistory.core.ingest import (
    ingest_manifest,
    ingest_snapshot,
    save_manifest,
)
from folderhistory.core.hash import hash_bytes
from folderhistory.types import FileRecord


# ── Helpers ──────────────────────────────────────────────────────────────────


def _write(src: Path, rel: str, content: bytes = b"") -> None:
    """Create a file at *src / rel* with *content*, creating parent dirs."""
    p = src / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# ── Basic tree walk ──────────────────────────────────────────────────────────


class TestTreeWalk:
    def test_empty_directory(self, tmp_path: Path) -> None:
        snapshot = ingest_snapshot(tmp_path, snapshot_id="empty")
        assert snapshot.id == "empty"
        assert snapshot.files == []

    def test_single_file(self, tmp_path: Path) -> None:
        _write(tmp_path, "hello.txt", b"hello")
        snapshot = ingest_snapshot(tmp_path)
        assert len(snapshot.files) == 1
        assert snapshot.files[0].path == "hello.txt"
        assert snapshot.files[0].size == 5

    def test_nested_directories(self, tmp_path: Path) -> None:
        _write(tmp_path, "a.txt", b"a")
        _write(tmp_path, "sub/b.txt", b"bb")
        _write(tmp_path, "sub/deep/c.txt", b"ccc")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert paths == {"a.txt", "sub/b.txt", "sub/deep/c.txt"}

    def test_files_are_sorted_by_path(self, tmp_path: Path) -> None:
        _write(tmp_path, "z.txt", b"z")
        _write(tmp_path, "a.txt", b"a")
        _write(tmp_path, "m.txt", b"m")
        snapshot = ingest_snapshot(tmp_path)
        paths = [f.path for f in snapshot.files]
        assert paths == sorted(paths)

    def test_directory_not_included_in_files(self, tmp_path: Path) -> None:
        _write(tmp_path, "sub/a.txt", b"a")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert "sub" not in paths
        assert "sub/a.txt" in paths

    def test_snapshot_id_defaults_to_dirname(self, tmp_path: Path) -> None:
        dir_name = tmp_path.name
        snapshot = ingest_snapshot(tmp_path)
        assert snapshot.id == dir_name

    def test_snapshot_source_path_is_resolved(self, tmp_path: Path) -> None:
        snapshot = ingest_snapshot(tmp_path)
        assert snapshot.source_path == tmp_path.resolve()

    def test_timestamp_is_set(self, tmp_path: Path) -> None:
        snapshot = ingest_snapshot(tmp_path)
        assert snapshot.timestamp is not None
        assert isinstance(snapshot.timestamp, float)


# ── Metadata correctness ─────────────────────────────────────────────────────


class TestMetadata:
    def test_file_size(self, tmp_path: Path) -> None:
        _write(tmp_path, "f.bin", b"\x00" * 1234)
        snapshot = ingest_snapshot(tmp_path)
        assert snapshot.files[0].size == 1234

    def test_file_mode(self, tmp_path: Path) -> None:
        _write(tmp_path, "f.bin", b"data")
        mode = (tmp_path / "f.bin").stat().st_mode
        snapshot = ingest_snapshot(tmp_path)
        assert snapshot.files[0].mode == mode

    def test_mtime_and_ctime(self, tmp_path: Path) -> None:
        _write(tmp_path, "f.bin", b"data")
        st = (tmp_path / "f.bin").stat()
        snapshot = ingest_snapshot(tmp_path)
        assert snapshot.files[0].mtime_ns == st.st_mtime_ns
        assert snapshot.files[0].ctime_ns == st.st_ctime_ns

    def test_hash_fields_populated(self, tmp_path: Path) -> None:
        _write(tmp_path, "f.txt", b"hello")
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        assert rec.raw_blake3 != ""
        assert rec.xxhash64 != ""
        assert len(rec.raw_blake3) == 64  # BLAKE3 hex digest

    def test_non_regular_file_is_skipped(self, tmp_path: Path) -> None:
        """FIFO / device nodes are skipped (not a regular file)."""
        fifo = tmp_path / "pipe"
        os.mkfifo(str(fifo))
        _write(tmp_path, "regular.txt", b"ok")
        snapshot = ingest_snapshot(tmp_path)
        assert len(snapshot.files) == 1
        assert snapshot.files[0].path == "regular.txt"


# ── Symlinks ─────────────────────────────────────────────────────────────────


class TestSymlinks:
    def test_symlink_recorded_not_followed(self, tmp_path: Path) -> None:
        _write(tmp_path, "real.txt", b"secret content")
        (tmp_path / "link.txt").symlink_to("real.txt")
        snapshot = ingest_snapshot(tmp_path)
        symlinks = [f for f in snapshot.files if f.is_symlink]
        assert len(symlinks) == 1
        assert symlinks[0].path == "link.txt"
        assert symlinks[0].target_path == "real.txt"
        # Symlink content is NOT hashed
        assert symlinks[0].raw_blake3 == ""
        assert symlinks[0].xxhash64 == ""
        assert symlinks[0].normalized_blake3 is None

    def test_regular_file_still_hashed(self, tmp_path: Path) -> None:
        _write(tmp_path, "real.txt", b"data")
        (tmp_path / "link.txt").symlink_to("real.txt")
        snapshot = ingest_snapshot(tmp_path)
        regular = [f for f in snapshot.files if not f.is_symlink]
        assert len(regular) == 1
        assert regular[0].path == "real.txt"
        assert regular[0].raw_blake3 != ""

    def test_broken_symlink(self, tmp_path: Path) -> None:
        (tmp_path / "broken").symlink_to("nowhere")
        snapshot = ingest_snapshot(tmp_path)
        assert len(snapshot.files) == 1
        rec = snapshot.files[0]
        assert rec.is_symlink
        assert rec.target_path == "nowhere"

    def test_symlink_to_directory(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        _write(tmp_path, "sub/inside.txt", b"x")
        (tmp_path / "link_to_sub").symlink_to("sub")
        snapshot = ingest_snapshot(tmp_path)
        # Symlink to directory should be recorded as a symlink
        symlinks = [f for f in snapshot.files if f.is_symlink]
        assert len(symlinks) == 1
        assert symlinks[0].path == "link_to_sub"
        # The regular file inside the real directory should also be found
        regular = [f for f in snapshot.files if not f.is_symlink]
        assert len(regular) == 1
        assert regular[0].path == "sub/inside.txt"


# ── Permission denied ────────────────────────────────────────────────────────


class TestPermissionDenied:
    def test_unreadable_file_is_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write(tmp_path, "readable.txt", b"ok")
        _write(tmp_path, "secret.bin", b"hidden")
        secret_path = tmp_path / "secret.bin"
        # Remove read permission
        secret_path.chmod(0o000)
        try:
            with caplog.at_level(logging.WARNING):
                snapshot = ingest_snapshot(tmp_path)
            paths = {f.path for f in snapshot.files}
            assert "readable.txt" in paths
            assert "secret.bin" not in paths
            assert any("Permission denied reading file" in r.message for r in caplog.records)
        finally:
            secret_path.chmod(0o644)

    def test_unreadable_directory_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write(tmp_path, "ok.txt", b"ok")
        secret_dir = tmp_path / "secret"
        secret_dir.mkdir()
        _write(tmp_path, "secret/hidden.txt", b"hidden")
        secret_dir.chmod(0o000)
        try:
            with caplog.at_level(logging.WARNING):
                snapshot = ingest_snapshot(tmp_path)
            paths = {f.path for f in snapshot.files}
            assert "ok.txt" in paths
            assert "secret/hidden.txt" not in paths
            assert any("Permission denied reading directory" in r.message for r in caplog.records)
        finally:
            secret_dir.chmod(0o755)


# ── NFC normalisation ────────────────────────────────────────────────────────


class TestNFCNormalization:
    def test_nfd_filename_becomes_nfc(self, tmp_path: Path) -> None:
        # "café" in NFD = "cafe" + combining acute accent
        nfd_name = "cafe\u0301.txt"
        nfc_name = _nfc(nfd_name)
        _write(tmp_path, nfd_name, b"hello")
        snapshot = ingest_snapshot(tmp_path)
        # Path should appear as NFC
        paths = {f.path for f in snapshot.files}
        assert nfc_name in paths
        assert nfd_name not in paths  # The raw NFD form is NOT present

    def test_nfd_nested_path(self, tmp_path: Path) -> None:
        # "dé" in NFD
        nfd_dir = "de\u0301"
        nfc_dir = _nfc(nfd_dir)
        _write(tmp_path, f"{nfd_dir}/a.txt", b"a")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        expected = f"{nfc_dir}/a.txt"
        assert expected in paths


# ── Content hashing integration ──────────────────────────────────────────────


class TestContentHashing:
    def test_text_file_has_line_ending(self, tmp_path: Path) -> None:
        _write(tmp_path, "hello.txt", b"hello\nworld\n")
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        assert rec.line_ending == "lf"
        assert rec.normalized_blake3 is not None

    def test_text_file_extension_overrides_null_byte(self, tmp_path: Path) -> None:
        """Files with known-text extensions bypass null-byte binary detection."""
        _write(tmp_path, "script.py", b"#!/usr/bin/env python3\x00stuff\n")
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        # Even though there's a null byte, .py is known text
        assert rec.normalized_blake3 is not None
        assert rec.line_ending == "lf"

    def test_unknown_extension_with_null_byte_is_binary(self, tmp_path: Path) -> None:
        _write(tmp_path, "data.bin", b"real\x00data\n")
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        assert rec.normalized_blake3 is None
        assert rec.line_ending == "binary"

    def test_lf_file_hash_integrity(self, tmp_path: Path) -> None:
        content = b"hello\nworld\n"
        _write(tmp_path, "f.txt", content)
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        raw_b3, _ = hash_bytes(content)
        assert rec.raw_blake3 == raw_b3
        # LF file → normalized == raw
        assert rec.normalized_blake3 == raw_b3


# ── Manifest round-trip ──────────────────────────────────────────────────────


class TestManifestRoundTrip:
    def test_round_trip(self, tmp_path: Path) -> None:
        # Create a source directory
        src = tmp_path / "source"
        src.mkdir()
        _write(src, "a.txt", b"hello")
        _write(src, "sub/b.ts", b"const x = 1;\n")
        (src / "link").symlink_to("a.txt")
        # Non-ASCII name
        _write(src, "caf\u00e9.txt", b"data")

        snapshot = ingest_snapshot(src, snapshot_id="test-rt")

        manifest_path = tmp_path / "manifest.json"
        save_manifest(snapshot, manifest_path)
        assert manifest_path.exists()

        loaded = ingest_manifest(manifest_path)

        # Core identity
        assert loaded.id == snapshot.id
        assert loaded.timestamp == snapshot.timestamp
        assert loaded.source_path == snapshot.source_path

        # Files — compare as sets of tuples
        def record_key(r: FileRecord) -> str:
            return r.path

        orig_by_path = {record_key(r): r for r in snapshot.files}
        loaded_by_path = {record_key(r): r for r in loaded.files}

        assert orig_by_path.keys() == loaded_by_path.keys()

        for path_str, orig in orig_by_path.items():
            loaded_rec = loaded_by_path[path_str]
            assert orig == loaded_rec, (
                f"Mismatch for {path_str}: orig={orig}, loaded={loaded_rec}"
            )

    def test_manifest_json_structure(self, tmp_path: Path) -> None:
        _write(tmp_path, "f.txt", b"data")
        snapshot = ingest_snapshot(tmp_path)
        manifest_path = tmp_path / "manifest.json"
        save_manifest(snapshot, manifest_path)

        import orjson
        raw = orjson.loads(manifest_path.read_bytes())
        assert "id" in raw
        assert "timestamp" in raw
        assert "source_path" in raw
        assert "files" in raw
        assert isinstance(raw["files"], list)
        if raw["files"]:
            f0 = raw["files"][0]
            for key in ("path", "size", "mode", "raw_blake3", "line_ending",
                        "is_symlink"):
                assert key in f0, f"Missing key: {key}"

    def test_empty_snapshot_round_trip(self, tmp_path: Path) -> None:
        src = tmp_path / "empty"
        src.mkdir()
        snapshot = ingest_snapshot(src, snapshot_id="empty")
        manifest_path = tmp_path / "manifest.json"
        save_manifest(snapshot, manifest_path)
        loaded = ingest_manifest(manifest_path)
        assert loaded.id == "empty"
        assert loaded.files == []
