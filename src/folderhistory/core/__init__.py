"""Core ingestion and hashing engine for FolderHistory."""

from __future__ import annotations

from folderhistory.core.hash import (
    compute_normalized_hash,
    hash_bytes,
    hash_file,
)
from folderhistory.core.ingest import (
    ingest_manifest,
    ingest_snapshot,
    save_manifest,
)

__all__ = [
    "compute_normalized_hash",
    "hash_bytes",
    "hash_file",
    "ingest_manifest",
    "ingest_snapshot",
    "save_manifest",
]
