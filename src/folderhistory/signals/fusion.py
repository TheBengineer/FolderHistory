"""Weighted-signal fusion for probabilistic identity assignment."""

from __future__ import annotations

from dataclasses import dataclass

from folderhistory.signals.content import content_similarity
from folderhistory.signals.metadata import metadata_similarity
from folderhistory.signals.path import path_similarity
from folderhistory.types import FileRecord


@dataclass(frozen=True)
class SignalConfig:
    """Weight configuration for the multi-signal fusion.

    Default weights assume content hash is the strongest signal, followed
    by path similarity, with metadata as a weak tie-breaker.
    """

    w_hash: float = 0.6
    w_path: float = 0.3
    w_meta: float = 0.1


def fuse_signals(
    f_a: FileRecord,
    f_b: FileRecord,
    config: SignalConfig | None = None,
) -> float:
    """Compute a fused cost from all three signal channels.

    ``cost = w_hash * cost_hash + w_path * cost_path + w_meta * cost_meta``

    Returns a value in ``[0.0, 1.0]`` where 0.0 means "same file" and 1.0
    means "completely different".
    """
    cfg: SignalConfig = config if config is not None else SignalConfig()
    return (
        cfg.w_hash * content_similarity(f_a, f_b)
        + cfg.w_path * path_similarity(f_a, f_b)
        + cfg.w_meta * metadata_similarity(f_a, f_b)
    )


def match_files_probabilistic(
    files_a: list[FileRecord],
    files_b: list[FileRecord],
    config: SignalConfig | None = None,
    threshold: float = 0.5,
) -> list[tuple[str, str, float]]:
    """Yield all inter-list pairs whose fused cost is below *threshold*.

    Each result is a ``(path_a, path_b, cost)`` tuple, sorted ascending by
    cost (best matches first).
    """
    cfg: SignalConfig = config if config is not None else SignalConfig()
    matches: list[tuple[str, str, float]] = []

    for a in files_a:
        for b in files_b:
            cost: float = fuse_signals(a, b, cfg)
            if cost < threshold:
                matches.append((a.path, b.path, cost))

    matches.sort(key=lambda x: x[2])
    return matches
