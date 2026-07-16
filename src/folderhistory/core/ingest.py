"""Directory-tree ingestion engine for FolderHistory.

Provides the primary entry point ``ingest_snapshot()`` that walks a directory
tree, hashes every regular file, and produces a ``Snapshot`` value object.
"""

from __future__ import annotations

import logging
import os
import stat
import time
import typing
import unicodedata
from pathlib import Path

import orjson

from folderhistory.core.hash import compute_normalized_hash, hash_bytes
from folderhistory.types import FileRecord, LineEnding, Snapshot

logger = logging.getLogger(__name__)

#: File extensions that are always treated as text, overriding null-byte
#: binary-content detection.
_TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".py",
    ".ts",
    ".rs",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
})


# ── Internal helpers ─────────────────────────────────────────────────────────


def _nfc_rel_path(entry_path: Path, anchor: Path) -> str:
    """Return the POSIX relative path of *entry_path* relative to *anchor*,
    with every component NFC-normalised.

    File operations use the raw filesystem name, so *entry_path* and *anchor*
    retain the OS-provided bytes.  Only the recorded path string is
    NFC-normalised to produce a consistent, portable identifier.
    """
    rel = entry_path.relative_to(anchor)
    parts = [unicodedata.normalize("NFC", p) for p in rel.parts]
    return "/".join(parts)


def _walk_and_hash(source: Path) -> list[FileRecord]:
    """Recursively walk *source* and produce a list of ``FileRecord``."""
    files: list[FileRecord] = []

    def recurse(dir_path: Path) -> None:
        try:
            scan_iter = os.scandir(str(dir_path))
        except PermissionError:
            logger.warning("Permission denied reading directory: %s", dir_path)
            return
        except OSError as exc:
            logger.warning("Cannot scan directory %s: %s", dir_path, exc)
            return

        with scan_iter as entries:
            for entry in entries:
                raw_name = entry.name
                entry_path = dir_path / raw_name

                try:
                    st = entry.stat(follow_symlinks=False)
                except PermissionError:
                    logger.warning("Permission denied: %s", entry_path)
                    continue
                except OSError as exc:
                    logger.warning("Cannot stat %s: %s", entry_path, exc)
                    continue

                # Recurse into subdirectories.
                if stat.S_ISDIR(st.st_mode):
                    recurse(entry_path)
                    continue

                # NFC-normalised relative path (for stable identification).
                rel_path_str = _nfc_rel_path(entry_path, source)

                # ── Symbolic link ──────────────────────────────────────
                if stat.S_ISLNK(st.st_mode):
                    try:
                        target = unicodedata.normalize(
                            "NFC",
                            os.readlink(entry.path),
                        )
                    except OSError as exc:
                        logger.warning(
                            "Cannot read symlink target: %s: %s",
                            entry_path,
                            exc,
                        )
                        target = ""

                    files.append(FileRecord(
                        path=rel_path_str,
                        size=0,
                        mode=st.st_mode,
                        mtime_ns=st.st_mtime_ns,
                        ctime_ns=st.st_ctime_ns,
                        raw_blake3="",
                        normalized_blake3=None,
                        line_ending="unknown",
                        xxhash64="",
                        is_symlink=True,
                        target_path=target,
                    ))
                    continue

                # Skip non-regular files (devices, fifos, sockets, …).
                if not stat.S_ISREG(st.st_mode):
                    continue

                # ── Regular file: read & hash ──────────────────────────
                try:
                    data = entry_path.read_bytes()
                except PermissionError:
                    logger.warning(
                        "Permission denied reading file: %s",
                        entry_path,
                    )
                    continue
                except OSError as exc:
                    logger.warning(
                        "Cannot read file %s: %s",
                        entry_path,
                        exc,
                    )
                    continue

                raw_blake3, xxh64 = hash_bytes(data)

                ext = entry_path.suffix.lower()
                known_text = ext in _TEXT_EXTENSIONS
                normalized_blake3, line_ending = compute_normalized_hash(
                    data,
                    known_text=known_text,
                )

                files.append(FileRecord(
                    path=rel_path_str,
                    size=st.st_size,
                    mode=st.st_mode,
                    mtime_ns=st.st_mtime_ns,
                    ctime_ns=st.st_ctime_ns,
                    raw_blake3=raw_blake3,
                    normalized_blake3=normalized_blake3,
                    line_ending=line_ending,
                    xxhash64=xxh64,
                    is_symlink=False,
                    target_path=None,
                ))

    recurse(source)
    return files


# ── Public API ───────────────────────────────────────────────────────────────


def ingest_snapshot(
    source: Path,
    snapshot_id: str | None = None,
) -> Snapshot:
    """Walk *source*, hash all files, and return a ``Snapshot``.

    Parameters
    ----------
    source:
        Root of the directory tree to snapshot.
    snapshot_id:
        Optional identifier for the snapshot.  Defaults to the directory name
        of *source*.

    Returns
    -------
    A fully populated ``Snapshot`` with a ``FileRecord`` per discovered file.
    """
    source = source.resolve()
    snapshot_id = snapshot_id or source.name

    files = _walk_and_hash(source)
    files.sort(key=lambda r: r.path)

    return Snapshot(
        id=snapshot_id,
        timestamp=time.time(),
        source_path=source,
        files=files,
    )


def ingest_manifest(path: Path) -> Snapshot:
    """Load a pre-computed snapshot from a JSON manifest file.

    Parameters
    ----------
    path:
        Path to the manifest JSON file (as written by :func:`save_manifest`).

    Returns
    -------
    A ``Snapshot`` reconstructed from the serialised data.
    """
    raw: dict[str, object] = typing.cast(
        "dict[str, object]",
        orjson.loads(path.read_bytes()),
    )

    snapshot_id: str = typing.cast(str, raw["id"])
    timestamp: float | None = typing.cast("float | None", raw["timestamp"])
    source_path_str: str = typing.cast(str, raw["source_path"])

    raw_files: list[object] = typing.cast("list[object]", raw["files"])
    files: list[FileRecord] = []
    for raw_file_obj in raw_files:
        raw_file: dict[str, object] = typing.cast(
            "dict[str, object]",
            raw_file_obj,
        )
        files.append(FileRecord(
            path=typing.cast(str, raw_file["path"]),
            size=typing.cast(int, raw_file["size"]),
            mode=typing.cast(int, raw_file["mode"]),
            mtime_ns=typing.cast(int, raw_file["mtime_ns"]),
            ctime_ns=typing.cast(int, raw_file["ctime_ns"]),
            raw_blake3=typing.cast(str, raw_file["raw_blake3"]),
            normalized_blake3=typing.cast(
                "str | None",
                raw_file["normalized_blake3"],
            ),
            line_ending=typing.cast(LineEnding, raw_file["line_ending"]),
            xxhash64=typing.cast(str, raw_file["xxhash64"]),
            is_symlink=typing.cast(bool, raw_file["is_symlink"]),
            target_path=typing.cast(
                "str | None",
                raw_file.get("target_path"),
            ),
        ))

    return Snapshot(
        id=snapshot_id,
        timestamp=timestamp,
        source_path=Path(source_path_str),
        files=files,
    )


def save_manifest(snapshot: Snapshot, path: Path) -> None:
    """Serialise *snapshot* to a JSON manifest file.

    Parameters
    ----------
    snapshot:
        The snapshot to persist.
    path:
        Destination file path.  The parent directory must exist.
    """
    payload = {
        "id": snapshot.id,
        "timestamp": snapshot.timestamp,
        "source_path": str(snapshot.source_path),
        "files": [
            {
                "path": f.path,
                "size": f.size,
                "mode": f.mode,
                "mtime_ns": f.mtime_ns,
                "ctime_ns": f.ctime_ns,
                "raw_blake3": f.raw_blake3,
                "normalized_blake3": f.normalized_blake3,
                "line_ending": f.line_ending,
                "xxhash64": f.xxhash64,
                "is_symlink": f.is_symlink,
                "target_path": f.target_path,
            }
            for f in snapshot.files
        ],
    }

    _written: int = path.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2),
    )
