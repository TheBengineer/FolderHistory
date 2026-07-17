"""Recursive snapshot discovery for FolderHistory.

Walks directories and identifies which directories are "snapshots"
(contain files directly) vs "containers" (contain only subdirectories,
recurse into them).
"""

from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotDir:
    """A discovered snapshot directory.

    Attributes:
        path: Absolute resolved path to the snapshot directory.
        relative_id: POSIX relative path from the discovery root.
            ``"."`` for the root itself, ``"sub/v1"`` for a nested snapshot.

    """

    path: Path
    relative_id: str


class _DirContent(enum.Enum):
    """Classification of a directory's direct contents after one scan."""

    EMPTY = "empty"
    FILES_ONLY = "files_only"
    DIRS_ONLY = "dirs_only"
    MIXED = "mixed"


# ── Internal helpers ─────────────────────────────────────────────────────────


def _classify(directory: Path) -> _DirContent:
    """Classify *directory* by its direct entries (one scan).

    Scans the directory once to determine whether it contains files,
    subdirectories, both, or neither.  Symlinks and special entries
    are ignored.
    """
    has_files = False
    has_dirs = False
    try:
        for entry in os.scandir(directory):
            try:
                if entry.is_file(follow_symlinks=False):
                    has_files = True
                elif entry.is_dir(follow_symlinks=False):
                    has_dirs = True
            except OSError:
                continue
    except PermissionError:
        logger.warning("Permission denied: %s", directory)
        return _DirContent.EMPTY
    except OSError:
        return _DirContent.EMPTY

    if has_files and has_dirs:
        return _DirContent.MIXED
    if has_files:
        return _DirContent.FILES_ONLY
    if has_dirs:
        return _DirContent.DIRS_ONLY
    return _DirContent.EMPTY


def _discover_flat(root: Path, snapshots: list[SnapshotDir]) -> None:
    """Collect all immediate subdirectories of *root* (flat discovery).

    This mirrors the current behaviour of ``app.py:178``:
    ``[d for d in root.iterdir() if d.is_dir()]``.
    """
    try:
        for entry in os.scandir(root):
            try:
                if entry.is_dir(follow_symlinks=False):
                    snapshots.append(
                        SnapshotDir(
                            path=Path(entry.path),
                            relative_id=entry.name,
                        ),
                    )
            except OSError:
                continue
    except PermissionError:
        logger.warning("Permission denied: %s", root)
    except OSError:
        pass


def _discover_recursive(
    root: Path,
    current: Path,
    max_depth: int,
    depth: int,
    snapshots: list[SnapshotDir],
) -> None:
    """Walk *current* recursively and discover snapshot directories.

    Directories containing files directly are classified as snapshots.
    Directories containing only subdirectories are containers
    — recurse into them (bounded by *max_depth*).

    Mixed directories (files + subdirectories) are treated as snapshots
    and are *not* recursed past.
    """
    if depth > 0:
        content = _classify(current)
        if content in (_DirContent.FILES_ONLY, _DirContent.MIXED):
            rel = current.relative_to(root)
            snapshots.append(
                SnapshotDir(
                    path=current,
                    relative_id=rel.as_posix(),
                ),
            )
            # Snapshots are leaf nodes — do not recurse past them.
            return

    if max_depth != -1 and depth >= max_depth:
        return

    try:
        for entry in os.scandir(current):
            try:
                if entry.is_dir(follow_symlinks=False):
                    _discover_recursive(
                        root,
                        Path(entry.path),
                        max_depth,
                        depth + 1,
                        snapshots,
                    )
            except OSError:
                continue
    except PermissionError:
        logger.warning("Permission denied: %s", current)
    except OSError:
        pass


# ── Public API ───────────────────────────────────────────────────────────────


def discover_snapshots(root: Path, max_depth: int = 1) -> list[SnapshotDir]:
    """Discover snapshot directories recursively.

    A directory is a *snapshot* if it contains at least one regular file
    directly.  A directory is a *container* if it contains only
    subdirectories — recurse into it.

    Args:
        root: Root directory to start discovery from.
        max_depth: Maximum recursion depth.
            0 = return root only (must be a snapshot)
            1 = only immediate subdirectories (current flat behaviour)
            -1 = infinite recursion
            N = up to N levels deep

    Returns:
        List of :class:`SnapshotDir`, ordered by ``relative_id``.

    """
    root_resolved = root.resolve()
    if not root_resolved.is_dir():
        return []

    snapshots: list[SnapshotDir] = []

    if max_depth == 0:
        if _classify(root_resolved) in (
            _DirContent.FILES_ONLY,
            _DirContent.MIXED,
        ):
            snapshots.append(SnapshotDir(path=root_resolved, relative_id="."))
    elif max_depth == 1:
        _discover_flat(root_resolved, snapshots)
    else:
        _discover_recursive(
            root_resolved, root_resolved, max_depth, 0, snapshots,
        )

    snapshots.sort(key=lambda s: s.relative_id)
    return snapshots
