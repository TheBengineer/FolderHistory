"""Core data types for FolderHistory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path


# ── Line ending classification ──────────────────────────────────────────────

LineEnding = Literal["crlf", "lf", "mixed", "binary", "unknown"]


# ── Core data structures ────────────────────────────────────────────────────


@dataclass(kw_only=True, frozen=True)
class FileRecord:
    """A single file entry within a snapshot.

    :param path: Normalised (NFC) relative path within the snapshot.
    :param size: File size in bytes.
    :param mode: Unix file mode (st_mode) bits.
    :param mtime_ns: Modification timestamp as nanoseconds since epoch.
    :param ctime_ns: Creation / metadata-change timestamp as nanoseconds since epoch.
    :param raw_blake3: BLAKE3 hex digest of the raw file content.
    :param normalized_blake3: BLAKE3 hex digest after CRLF->LF normalisation, or
        ``None`` when the file is binary.
    :param line_ending: Classified line-ending style of the file.
    :param xxhash64: XXH64 hex digest for fast deduplication pre-check.
    :param is_symlink: Whether this entry is a symbolic link.
    :param target_path: For symlinks, the link target; otherwise ``None``.
    """

    path: str
    size: int
    mode: int
    mtime_ns: int
    ctime_ns: int
    raw_blake3: str
    normalized_blake3: str | None
    line_ending: LineEnding
    xxhash64: str
    is_symlink: bool
    target_path: str | None = None


@dataclass(kw_only=True, frozen=True)
class Snapshot:
    """A point-in-time copy of a directory tree.

    :param id: Unique identifier for this snapshot (usually the folder name).
    :param timestamp: When the snapshot was taken, or ``None`` if unknown.
    :param source_path: Filesystem path to the snapshot directory.
    :param files: All files discovered in the snapshot.
    """

    id: str
    timestamp: float | None
    source_path: Path
    files: list[FileRecord] = field(default_factory=list)


@dataclass(kw_only=True, frozen=True)
class EditOperation:
    """A single atomic filesystem operation inferred between two snapshots.

    :param op_type: The kind of operation.
    :param file_id: Unique identifier for the affected file (usually its path
        in the earlier snapshot).
    :param source_path: For rename / move / copy, the original path.
    :param target_path: For rename / move / copy / create, the new path.
    :param old_hash: ``raw_blake3`` value before the operation, or ``None`` for
        a creation.
    :param new_hash: ``raw_blake3`` value after the operation, or ``None`` for
        a deletion.
    :param confidence: How certain we are this operation actually occurred in
        the real history (0.0 - 1.0).
    """

    op_type: Literal["create", "delete", "modify", "rename", "move", "copy"]
    file_id: str
    source_path: str | None = None
    target_path: str | None = None
    old_hash: str | None = None
    new_hash: str | None = None
    confidence: float = 1.0


@dataclass(kw_only=True, frozen=True)
class IdentityCluster:
    """A cluster of observations believed to refer to the same logical file.

    :param uid: Unique identifier for this cluster.
    :param observations: Pairs of ``(snapshot_id, file_path)`` that are
        hypothesised to represent the same file across snapshots.
    :param canonical_path: The most representative path for this file, or
        ``None`` if not yet determined.
    :param confidence: How strongly the cluster's members are believed to be
        the same logical file (0.0 - 1.0).
    """

    uid: str
    observations: list[tuple[str, str]]
    canonical_path: str | None = None
    confidence: float = 0.0
