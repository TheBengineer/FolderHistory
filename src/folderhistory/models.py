"""Data model layer for FolderHistory — project-level tracking dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime

LineEnding = Literal["crlf", "lf", "mixed", "binary", "unknown"]
SnapshotId = str
RelativePath = str


@dataclass
class SnapshotMetadata:
    """Metadata about a single snapshot in a project's history."""

    snapshot_id: str
    timestamp: datetime | None = None
    original_root_path: str = ""
    acquisition_device: str | None = None
    total_files: int = 0
    total_size: int = 0


@dataclass
class ProjectRecord:
    """Tracks a project identity across a collection of snapshots."""

    project_uid: str
    display_name: str = ""
    snapshots: list[SnapshotMetadata] = field(default_factory=list)
    created: datetime | None = None
    last_seen: datetime | None = None
    confidence: float = 0.0
    git_remote: str | None = None


@dataclass
class ProjectMatch:
    """A scored match between a project identity and a snapshot subtree."""

    project_uid: str
    snapshot_id: SnapshotId
    root_path: RelativePath
    confidence: float = 0.0
    idf_jaccard_score: float = 0.0


@dataclass
class DirectoryPreAlignment:
    """Preliminary alignment of snapshots against candidate projects."""

    project_matches: list[ProjectMatch] = field(default_factory=list)
    global_hash_freq: dict[str, int] = field(default_factory=dict)
    snapshot_clusters: dict[str, list[SnapshotId]] = field(default_factory=dict)
