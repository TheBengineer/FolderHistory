"""Path-based similarity signals for probabilistic file matching."""

from __future__ import annotations

from jaro import jaro_winkler_metric  # pyright: ignore[reportUnknownVariableType]

from folderhistory.types import FileRecord


def path_similarity(f_a: FileRecord, f_b: FileRecord) -> float:
    """Return a path-dissimilarity cost using Jaro-Winkler distance.

    Returns 0.0 for identical paths, 1.0 for completely different paths.
    """
    similarity: float = jaro_winkler_metric(f_a.path, f_b.path)
    return 1.0 - similarity
