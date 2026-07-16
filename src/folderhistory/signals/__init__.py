"""Multi-signal identity assignment — probabilistic matching using content,
path, and metadata similarity signals."""

from __future__ import annotations

from folderhistory.signals.content import content_similarity as content_similarity
from folderhistory.signals.fusion import (
    SignalConfig as SignalConfig,
    fuse_signals as fuse_signals,
    match_files_probabilistic as match_files_probabilistic,
)
from folderhistory.signals.metadata import metadata_similarity as metadata_similarity
from folderhistory.signals.path import path_similarity as path_similarity

__all__ = [
    "SignalConfig",
    "content_similarity",
    "path_similarity",
    "metadata_similarity",
    "fuse_signals",
    "match_files_probabilistic",
]
