"""Content-based similarity signals for probabilistic file matching."""

from __future__ import annotations

from folderhistory.types import FileRecord


def content_similarity(f_a: FileRecord, f_b: FileRecord) -> float:
    """Return a content-dissimilarity cost between two file records.

    * 0.0 if ``raw_blake3`` matches (identical content)
    * 0.3 if *size* and *xxhash64* match (likely small edit)
    * 0.6 if size matches but no content hash matches
    * 1.0 if sizes differ
    """
    if f_a.size != f_b.size:
        return 1.0

    if f_a.raw_blake3 == f_b.raw_blake3:
        return 0.0

    if f_a.xxhash64 == f_b.xxhash64:
        return 0.3

    return 0.6
