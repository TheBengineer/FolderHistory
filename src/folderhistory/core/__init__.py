"""Core ingestion, hashing, matching, and identity engine for FolderHistory."""

from __future__ import annotations

from folderhistory.core.hash import (
    compute_normalized_hash,
    hash_bytes,
    hash_file,
)
from folderhistory.core.identity import (
    DisjointSet,
    assign_identities_exact,
    assign_identities_with_blocking,
)
from folderhistory.core.ingest import (
    ingest_manifest,
    ingest_snapshot,
    save_manifest,
)
from folderhistory.core.match import (
    match_snapshots_exact,
    match_snapshots_exact_all_pairs,
)

__all__ = [
    "assign_identities_exact",
    "assign_identities_with_blocking",
    "compute_normalized_hash",
    "DisjointSet",
    "hash_bytes",
    "hash_file",
    "ingest_manifest",
    "ingest_snapshot",
    "match_snapshots_exact",
    "match_snapshots_exact_all_pairs",
    "save_manifest",
]
