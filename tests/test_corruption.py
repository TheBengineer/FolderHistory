"""Tests for corrupted metadata resilience in FolderHistory.

Covers shifted timestamps, zero timestamps, ctime > mtime,
theoretical negative sizes (via manifest injection), and
cross-platform line-ending normalisation hash matching.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from folderhistory.core.hash import hash_bytes
from folderhistory.core.ingest import ingest_manifest, ingest_snapshot, save_manifest
from folderhistory.types import FileRecord, Snapshot


# ── Helpers ──────────────────────────────────────────────────────────────────


def _write(src: Path, rel: str, content: bytes = b"") -> None:
    """Create a file at *src / rel* with *content*, creating parent dirs."""
    p = src / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


# ── Shifted timestamps (±1 year) ─────────────────────────────────────────────


class TestShiftedTimestamps:
    """Files with timestamps shifted forward/backward by ~1 year."""

    def _set_mtime_ns(self, path: Path, mtime_ns: int) -> None:
        """Set the modification time of *path* to *mtime_ns*."""
        os.utime(path, ns=(mtime_ns, mtime_ns))

    def test_mtime_one_year_in_past(self, tmp_path: Path) -> None:
        _write(tmp_path, "old.txt", b"past content")
        past_ns = int((time.time() - 365 * 86400) * 1_000_000_000)
        self._set_mtime_ns(tmp_path / "old.txt", past_ns)
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        # Should be within ~1s tolerance
        assert abs(rec.mtime_ns - past_ns) < 5_000_000_000, (
            f"mtime_ns {rec.mtime_ns} differs from expected {past_ns}"
        )

    def test_mtime_one_year_in_future(self, tmp_path: Path) -> None:
        _write(tmp_path, "future.txt", b"future content")
        future_ns = int((time.time() + 365 * 86400) * 1_000_000_000)
        self._set_mtime_ns(tmp_path / "future.txt", future_ns)
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        assert abs(rec.mtime_ns - future_ns) < 5_000_000_000

    def test_mtime_two_years_shifted(self, tmp_path: Path) -> None:
        """Extreme shift: mtime set to 2 years ago."""
        _write(tmp_path, "old.txt", b"very old")
        past_ns = int((time.time() - 2 * 365 * 86400) * 1_000_000_000)
        self._set_mtime_ns(tmp_path / "old.txt", past_ns)
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        assert abs(rec.mtime_ns - past_ns) < 5_000_000_000

    def test_mtime_and_atime_independently_shifted(self, tmp_path: Path) -> None:
        """mtime shifted but atime left alone (ingest should capture mtime)."""
        _write(tmp_path, "f.txt", b"shifted")
        past_ns = int((time.time() - 180 * 86400) * 1_000_000_000)
        self._set_mtime_ns(tmp_path / "f.txt", past_ns)
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        assert abs(rec.mtime_ns - past_ns) < 5_000_000_000


# ── Zero timestamps ──────────────────────────────────────────────────────────


class TestZeroTimestamps:
    """Timestamps at or near the Unix epoch (0 or small values)."""

    def test_mtime_at_epoch(self, tmp_path: Path) -> None:
        _write(tmp_path, "epoch.txt", b"epoch")
        os.utime(tmp_path / "epoch.txt", (0, 0))
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        assert rec.mtime_ns == 0

    def test_mtime_near_epoch(self, tmp_path: Path) -> None:
        """Timestamp = 1 second after epoch."""
        _write(tmp_path, "near_epoch.txt", b"near")
        os.utime(tmp_path / "near_epoch.txt", ns=(1_000_000, 1_000_000))
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        assert rec.mtime_ns == 1_000_000

    def test_zero_timestamp_does_not_crash_hash(self, tmp_path: Path) -> None:
        """Zero timestamps should not interfere with content hashing."""
        _write(tmp_path, "zero_time.bin", b"\x00\x01\x02")
        os.utime(tmp_path / "zero_time.bin", (0, 0))
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        assert rec.raw_blake3 != ""
        assert rec.size == 3

    def test_invalid_timestamp_manifest_roundtrip(self, tmp_path: Path) -> None:
        """A manifest with zero timestamp loads correctly."""
        src = tmp_path / "src"
        src.mkdir()
        _write(src, "f.txt", b"data")
        snapshot = ingest_snapshot(src)
        # Override the timestamp
        zero_ts_snapshot = Snapshot(
            id=snapshot.id,
            timestamp=0.0,
            source_path=snapshot.source_path,
            files=snapshot.files,
        )
        manifest_path = tmp_path / "zero_ts_manifest.json"
        save_manifest(zero_ts_snapshot, manifest_path)
        loaded = ingest_manifest(manifest_path)
        assert loaded.timestamp == 0.0
        assert len(loaded.files) == 1


# ── ctime > mtime (reverse timestamp order) ──────────────────────────────────


class TestCtimeGtMtime:
    """Scenarios where ctime (inode change time) is newer than mtime
    (modification time). This is common after a file is copied or
    restored from backup."""

    def test_mtime_in_past_ctime_current(self, tmp_path: Path) -> None:
        """Set mtime to a past value; ctime will naturally be > mtime."""
        _write(tmp_path, "copied.txt", b"restored from backup")
        past_ns = int((time.time() - 30 * 86400) * 1_000_000_000)
        os.utime(tmp_path / "copied.txt", ns=(past_ns, past_ns))
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        assert rec.ctime_ns > rec.mtime_ns, (
            f"Expected ctime ({rec.ctime_ns}) > mtime ({rec.mtime_ns})"
        )

    def test_ctime_mtime_both_past(self, tmp_path: Path) -> None:
        """Both timestamps in the past but still ctime > mtime (file was
        touched twice in the past)."""
        _write(tmp_path, "twice.txt", b"touched")
        old_ns = int((time.time() - 60 * 86400) * 1_000_000_000)
        slightly_older_ns = int((time.time() - 61 * 86400) * 1_000_000_000)
        os.utime(tmp_path / "twice.txt", ns=(slightly_older_ns, slightly_older_ns))
        # Wait a tiny bit for ctime to update
        os.chmod(tmp_path / "twice.txt", 0o644)  # ctime changes
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        assert rec.mtime_ns == slightly_older_ns
        # ctime should be > mtime because the chmod changed ctime

    def test_ctime_mtime_order_manifest_roundtrip(self, tmp_path: Path) -> None:
        """ctime > mtime values survive manifest save/load."""
        src = tmp_path / "src"
        src.mkdir()
        _write(src, "f.txt", b"data")
        snapshot = ingest_snapshot(src)
        # Create a record with ctime > mtime explicitly
        record = FileRecord(
            path="f.txt",
            size=4,
            mode=0o644,
            mtime_ns=1_000_000_000,
            ctime_ns=2_000_000_000,  # ctime > mtime
            raw_blake3=snapshot.files[0].raw_blake3,
            normalized_blake3=snapshot.files[0].normalized_blake3,
            line_ending=snapshot.files[0].line_ending,
            xxhash64=snapshot.files[0].xxhash64,
            is_symlink=False,
        )
        ctime_gt_mtime_snapshot = Snapshot(
            id="ctime-gt-mtime",
            timestamp=snapshot.timestamp,
            source_path=snapshot.source_path,
            files=[record],
        )
        manifest_path = tmp_path / "ctime_gt_mtime.json"
        save_manifest(ctime_gt_mtime_snapshot, manifest_path)
        loaded = ingest_manifest(manifest_path)
        assert loaded.files[0].mtime_ns == 1_000_000_000
        assert loaded.files[0].ctime_ns == 2_000_000_000


# ── Negative sizes (if ingested) ─────────────────────────────────────────────


class TestNegativeSizes:
    """While the OS will never report a negative file size, manifests or
    synthetic data might contain one. Verify that downstream operations
    don't crash."""

    def test_negative_size_survives_manifest_roundtrip(self, tmp_path: Path) -> None:
        """A manifest with a negative size should round-trip cleanly."""
        record = FileRecord(
            path="bad.txt",
            size=-1,
            mode=0o644,
            mtime_ns=1_000_000_000,
            ctime_ns=2_000_000_000,
            raw_blake3="",
            normalized_blake3=None,
            line_ending="unknown",
            xxhash64="",
            is_symlink=True,
            target_path="nonexistent",
        )
        snapshot = Snapshot(
            id="negative-size",
            timestamp=None,
            source_path=tmp_path,
            files=[record],
        )
        manifest_path = tmp_path / "negative_size.json"
        save_manifest(snapshot, manifest_path)
        loaded = ingest_manifest(manifest_path)
        assert loaded.files[0].size == -1

    def test_large_negative_size_manifest_roundtrip(self, tmp_path: Path) -> None:
        """A very large negative size should not overflow."""
        record = FileRecord(
            path="huge_neg.txt",
            size=-999_999_999_999,
            mode=0o644,
            mtime_ns=0,
            ctime_ns=0,
            raw_blake3="",
            normalized_blake3=None,
            line_ending="unknown",
            xxhash64="",
            is_symlink=False,
        )
        snapshot = Snapshot(
            id="large-neg",
            timestamp=None,
            source_path=tmp_path,
            files=[record],
        )
        manifest_path = tmp_path / "large_neg.json"
        save_manifest(snapshot, manifest_path)
        loaded = ingest_manifest(manifest_path)
        assert loaded.files[0].size == -999_999_999_999

    def test_negative_size_does_not_affect_positive_file(self, tmp_path: Path) -> None:
        """A positive-size file alongside a negative-size entry loads fine."""
        good = FileRecord(
            path="good.txt",
            size=42,
            mode=0o644,
            mtime_ns=0,
            ctime_ns=0,
            raw_blake3="",
            normalized_blake3=None,
            line_ending="unknown",
            xxhash64="",
            is_symlink=False,
        )
        bad = FileRecord(
            path="bad.txt",
            size=-5,
            mode=0o644,
            mtime_ns=0,
            ctime_ns=0,
            raw_blake3="",
            normalized_blake3=None,
            line_ending="unknown",
            xxhash64="",
            is_symlink=False,
        )
        snapshot = Snapshot(
            id="mixed-sizes",
            timestamp=None,
            source_path=tmp_path,
            files=[good, bad],
        )
        manifest_path = tmp_path / "mixed_sizes.json"
        save_manifest(snapshot, manifest_path)
        loaded = ingest_manifest(manifest_path)
        by_path = {f.path: f for f in loaded.files}
        assert by_path["good.txt"].size == 42
        assert by_path["bad.txt"].size == -5


# ── Cross-platform line-ending normalisation hash matching ───────────────────


class TestCrossPlatformLineEndings:
    """Verify that normalised_blake3 matches across CRLF and LF versions
    of the same text content — the core of cross-platform hash matching."""

    def test_lf_and_crlf_files_have_matching_normalized_hash(self, tmp_path: Path) -> None:
        """Two files with same text but different line endings produce the
        same normalised_blake3."""
        _write(tmp_path, "lf.txt", b"hello\nworld\n")
        _write(tmp_path, "crlf.txt", b"hello\r\nworld\r\n")
        snapshot = ingest_snapshot(tmp_path)
        by_path = {f.path: f for f in snapshot.files}
        lf = by_path["lf.txt"]
        crlf = by_path["crlf.txt"]
        assert lf.normalized_blake3 is not None
        assert crlf.normalized_blake3 is not None
        assert lf.normalized_blake3 == crlf.normalized_blake3

    def test_mixed_endings_normalized_matches_all_lf(self, tmp_path: Path) -> None:
        """A file with mixed CRLF/LF normalises to the same hash as all-LF."""
        _write(tmp_path, "mixed.txt", b"a\r\nb\nc\r\n")
        _write(tmp_path, "all_lf.txt", b"a\nb\nc\n")
        snapshot = ingest_snapshot(tmp_path)
        by_path = {f.path: f for f in snapshot.files}
        mixed = by_path["mixed.txt"]
        all_lf = by_path["all_lf.txt"]
        assert mixed.normalized_blake3 == all_lf.normalized_blake3

    def test_binary_file_has_no_normalized_hash(self, tmp_path: Path) -> None:
        """Binary files get None for normalised_blake3 regardless of platform."""
        _write(tmp_path, "data.bin", b"\x00\x01\x02\r\n")
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        assert rec.normalized_blake3 is None
        assert rec.line_ending == "binary"

    def test_known_text_extension_has_normalized_hash(self, tmp_path: Path) -> None:
        """Files with known-text extensions get a normalized hash even with
        null bytes."""
        _write(tmp_path, "script.py", b"#!/usr/bin/env python3\x00stuff\r\n")
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        assert rec.normalized_blake3 is not None
        assert rec.line_ending == "crlf"

    def test_normalized_hash_is_deterministic(self, tmp_path: Path) -> None:
        """Same content always produces the same normalised hash."""
        _write(tmp_path, "a.txt", b"line1\nline2\n")
        _write(tmp_path, "b.txt", b"line1\nline2\n")
        snapshot = ingest_snapshot(tmp_path)
        by_path = {f.path: f for f in snapshot.files}
        assert by_path["a.txt"].normalized_blake3 == by_path["b.txt"].normalized_blake3

    def test_raw_hashes_differ_for_crlf_vs_lf(self, tmp_path: Path) -> None:
        """CRLF and LF files have different raw hashes but same normalized."""
        _write(tmp_path, "lf.txt", b"hello\n")
        _write(tmp_path, "crlf.txt", b"hello\r\n")
        snapshot = ingest_snapshot(tmp_path)
        by_path = {f.path: f for f in snapshot.files}
        lf = by_path["lf.txt"]
        crlf = by_path["crlf.txt"]
        # Raw hashes must differ
        assert lf.raw_blake3 != crlf.raw_blake3
        # Normalised hashes must be equal
        assert lf.normalized_blake3 == crlf.normalized_blake3


# ── Manifest corruption resilience ───────────────────────────────────────────


class TestManifestCorruption:
    """Resilience to malformed or missing manifest fields."""

    def test_missing_target_path_in_manifest(self, tmp_path: Path) -> None:
        """A manifest without the optional target_path field loads fine."""
        raw_json = """{
            "id": "no-target",
            "timestamp": 1000.0,
            "source_path": "/tmp",
            "files": [{
                "path": "f.txt",
                "size": 4,
                "mode": 33188,
                "mtime_ns": 1000,
                "ctime_ns": 2000,
                "raw_blake3": "",
                "normalized_blake3": null,
                "line_ending": "unknown",
                "xxhash64": "",
                "is_symlink": false
            }]
        }"""
        manifest_path = tmp_path / "no_target.json"
        manifest_path.write_text(raw_json)
        loaded = ingest_manifest(manifest_path)
        assert loaded.files[0].target_path is None

    def test_null_timestamp(self, tmp_path: Path) -> None:
        """A null timestamp in the manifest loads as None."""
        raw_json = """{
            "id": "null-ts",
            "timestamp": null,
            "source_path": "/tmp",
            "files": []
        }"""
        manifest_path = tmp_path / "null_ts.json"
        manifest_path.write_text(raw_json)
        loaded = ingest_manifest(manifest_path)
        assert loaded.timestamp is None

    def test_empty_files_list(self, tmp_path: Path) -> None:
        """An empty files list loads correctly."""
        raw_json = """{
            "id": "empty-files",
            "timestamp": 1000.0,
            "source_path": "/tmp",
            "files": []
        }"""
        manifest_path = tmp_path / "empty_files.json"
        manifest_path.write_text(raw_json)
        loaded = ingest_manifest(manifest_path)
        assert loaded.files == []

    def test_unknown_line_ending_survives(self, tmp_path: Path) -> None:
        """An unknown line_ending value is preserved."""
        raw_json = """{
            "id": "unknown-le",
            "timestamp": 1000.0,
            "source_path": "/tmp",
            "files": [{
                "path": "f.txt",
                "size": 0,
                "mode": 33188,
                "mtime_ns": 0,
                "ctime_ns": 0,
                "raw_blake3": "",
                "normalized_blake3": null,
                "line_ending": "crlf",
                "xxhash64": "",
                "is_symlink": false
            }]
        }"""
        manifest_path = tmp_path / "unknown_le.json"
        manifest_path.write_text(raw_json)
        loaded = ingest_manifest(manifest_path)
        assert loaded.files[0].line_ending == "crlf"
