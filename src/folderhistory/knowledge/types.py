"""KB-specific datatypes for FolderHistory's iterative feedback loop."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObservationRecord:
    """A single observation of a file at a point in time."""

    content_hash: str
    snapshot_id: str
    relative_path: str
    confidence: float
    run_id: str
    algorithm_version: str


@dataclass
class ContradictionRecord:
    """A recorded contradiction between two identity clusters."""

    hash: str
    identity_a: str
    identity_b: str
    strategy_applied: str  # "merge_with_rename" | "split" | "confidence_weight" | "evidence_weight"
    resolved_at: str = ""  # ISO timestamp
    resolution_confidence: float = 0.0


@dataclass
class VersionMeta:
    """Schema and algorithm version metadata for the knowledge base."""

    schema_version: int = 1
    algorithm_version: str = "1.0.0"
    last_run_id: str = ""
    run_count: int = 0


@dataclass
class FeedbackReport:
    """Summary of a single feedback-loop iteration."""

    run_id: str = ""
    new_identities: int = 0
    resolved_contradictions: int = 0
    quality_metrics: dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    warnings: list[str] = field(default_factory=list)
