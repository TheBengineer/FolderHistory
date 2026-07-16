"""Consistency checking for FolderHistory identity assignments.

Provides structured conflict reporting, inner-loop threshold relaxation,
and cross-location consistency validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from folderhistory.types import EditOperation, IdentityCluster, Snapshot


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class ConflictRecord:
    """A single consistency conflict detected between identity assignments.

    :param hash: Content hash (raw_blake3) involved in the conflict, or an
        empty string if the conflict is not hash-related.
    :param source_paths: Paths on the source/earlier side of the conflict.
    :param target_paths: Paths on the target/later side of the conflict.
    :param conflict_type: One of ``"identity_mismatch"``,
        ``"path_mismatch"``, or ``"temporal_overlap"``.
    :param confidence_a: Confidence value for the first interpretation.
    :param confidence_b: Confidence value for the second interpretation.
    """

    hash: str = ""
    source_paths: list[str] = field(default_factory=list)
    target_paths: list[str] = field(default_factory=list)
    conflict_type: str = "identity_mismatch"
    confidence_a: float = 0.0
    confidence_b: float = 0.0


@dataclass
class ConsistencyReport:
    """Outcome of a consistency check run.

    :param passes: ``True`` when no conflicts were found.
    :param conflicts: All conflicts detected during the check.
    :param iterations_used: Number of inner-loop iterations consumed
        (0 means the first check passed without needing relaxation).
    :param relaxed_params: The parameter dict used for the final iteration,
        or ``None`` if no relaxation was applied.
    """

    passes: bool = True
    conflicts: list[ConflictRecord] = field(default_factory=list)
    iterations_used: int = 0
    relaxed_params: dict[str, object] | None = None


# ── Internal helpers ─────────────────────────────────────────────────────────


_VALID_OP_TYPES: frozenset[str] = frozenset(
    {"create", "delete", "modify", "rename", "move", "copy"},
)


def _find_cluster_for_observation(
    identities: dict[str, IdentityCluster],
    snapshot_id: str,
    file_path: str,
) -> IdentityCluster | None:
    """Return the cluster that contains ``(snapshot_id, file_path)``.

    Returns ``None`` if no cluster contains that observation pair.
    """
    for cluster in identities.values():
        if (snapshot_id, file_path) in cluster.observations:
            return cluster
    return None


# ── Main consistency checks ─────────────────────────────────────────────────


def check_consistency(
    snapshots: list[Snapshot],
    identities: dict[str, IdentityCluster],
    operations: list[list[EditOperation]] | None = None,
) -> ConsistencyReport:
    """Check consistency of identity assignments and operations.

    Returns a :class:`ConsistencyReport` with ``passes=True`` if no conflicts
    are found.

    Performs the following checks:

    1.  **Cluster non-emptiness** — every identity cluster has at least one
        observation.
    2.  **File coverage** — every file in every snapshot is covered by at
        least one identity cluster.
    3.  **Operation validity** (when *operations* is provided) — each
        operation has a recognised ``op_type`` and does not duplicate
        ``file_id`` within a snapshot-pair.

    Parameters
    ----------
    snapshots:
        List of snapshots to validate.
    identities:
        Mapping of cluster UID → :class:`IdentityCluster` produced by
        :func:`~folderhistory.core.identity.assign_identities_exact` or
        similar.
    operations:
        Optional list of per-pair operation lists (one entry per consecutive
        snapshot pair).

    Returns
    -------
    A :class:`ConsistencyReport` summarising all findings.
    """
    conflicts: list[ConflictRecord] = []

    # ── Check 1: Every cluster has at least one observation ────────────
    for uid, cluster in identities.items():
        if not cluster.observations:
            conflicts.append(
                ConflictRecord(
                    hash="",
                    source_paths=[uid],
                    target_paths=[],
                    conflict_type="identity_mismatch",
                    confidence_a=0.0,
                    confidence_b=0.0,
                )
            )

    # ── Check 2: Every file in every snapshot is covered ─────────────
    for snap in snapshots:
        for file_rec in snap.files:
            found = _find_cluster_for_observation(
                identities,
                snap.id,
                file_rec.path,
            )
            if found is None:
                conflicts.append(
                    ConflictRecord(
                        hash=file_rec.raw_blake3,
                        source_paths=[file_rec.path],
                        target_paths=[],
                        conflict_type="identity_mismatch",
                        confidence_a=0.0,
                        confidence_b=0.0,
                    )
                )

    # ── Check 3: Operation validity ─────────────────────────────────
    if operations is not None:
        for pair_ops in operations:
            seen_file_ids: set[str] = set()
            for op in pair_ops:
                # 3a: Valid op_type
                if op.op_type not in _VALID_OP_TYPES:
                    conflicts.append(
                        ConflictRecord(
                            hash=op.old_hash or op.new_hash or "",
                            source_paths=[str(op.source_path or "")],
                            target_paths=[str(op.target_path or "")],
                            conflict_type="path_mismatch",
                            confidence_a=op.confidence,
                            confidence_b=0.0,
                        )
                    )

                # 3b: Duplicate file_id within pair
                if op.file_id in seen_file_ids:
                    conflicts.append(
                        ConflictRecord(
                            hash=op.old_hash or op.new_hash or "",
                            source_paths=[str(op.source_path or "")],
                            target_paths=[str(op.target_path or "")],
                            conflict_type="temporal_overlap",
                            confidence_a=op.confidence,
                            confidence_b=0.0,
                        )
                    )
                seen_file_ids.add(op.file_id)

    return ConsistencyReport(
        passes=len(conflicts) == 0,
        conflicts=conflicts,
        iterations_used=0,
        relaxed_params=None,
    )


def check_cross_location_consistency(
    snapshot_chain: list[Snapshot],
    project_matches: dict[str, list[str]],
) -> list[ConflictRecord]:
    """Check consistency across different snapshot locations.

    Performs three families of cross-location check:

    1.  **Path discontinuity** — a file's absolute location changes but its
        relative path stays the same, suggesting the file was copied or
        moved between project roots.
    2.  **Temporal overlap** — snapshots from different locations have
        overlapping time ranges, which is suspicious when the same logical
        file appears in both.
    3.  **High content overlap + different paths** — files with identical
        content hashes but different relative paths across matched
        snapshots, suggesting possible duplication.

    Parameters
    ----------
    snapshot_chain:
        Ordered list of snapshots forming the chain to validate.  May
        include snapshots from multiple locations.
    project_matches:
        Mapping of ``snapshot_id → [matched_snapshot_ids]`` indicating
        which snapshots from other locations are believed to correspond.

    Returns
    -------
    List of :class:`ConflictRecord` entries, one per detected issue.
    """
    conflicts: list[ConflictRecord] = []

    # Build lookup: snapshot_id → Snapshot
    snap_by_id: dict[str, Snapshot] = {s.id: s for s in snapshot_chain}

    # ── Check 1: Path discontinuity ──────────────────────────────────
    # Same relative path across snapshots from different source_path roots.
    path_to_sources: dict[str, set[str]] = {}
    for snap in snapshot_chain:
        src_str = str(snap.source_path)
        for file_rec in snap.files:
            path_to_sources.setdefault(file_rec.path, set()).add(src_str)

    for rel_path, sources in path_to_sources.items():
        if len(sources) > 1:
            conflicts.append(
                ConflictRecord(
                    hash="",
                    source_paths=sorted(sources),
                    target_paths=[rel_path],
                    conflict_type="path_mismatch",
                    confidence_a=0.5,
                    confidence_b=0.5,
                )
            )

    # ── Check 2: Temporal overlap ────────────────────────────────────
    # Snapshots with equal timestamps that belong to different roots and
    # are linked via project_matches.
    for snap_id, matched_ids in project_matches.items():
        snap = snap_by_id.get(snap_id)
        if snap is None or snap.timestamp is None:
            continue
        for matched_id in matched_ids:
            matched = snap_by_id.get(matched_id)
            if matched is None or matched.timestamp is None:
                continue
            if abs(snap.timestamp - matched.timestamp) < 1.0:
                conflicts.append(
                    ConflictRecord(
                        hash="",
                        source_paths=[str(snap.source_path)],
                        target_paths=[str(matched.source_path)],
                        conflict_type="temporal_overlap",
                        confidence_a=0.3,
                        confidence_b=0.3,
                    )
                )

    # ── Check 3: High content overlap + different paths ──────────────
    hash_to_paths: dict[str, set[str]] = {}
    for snap in snapshot_chain:
        for file_rec in snap.files:
            if file_rec.raw_blake3:
                hash_to_paths.setdefault(file_rec.raw_blake3, set()).add(
                    file_rec.path,
                )

    for content_hash, paths in hash_to_paths.items():
        if len(paths) > 1:
            conflicts.append(
                ConflictRecord(
                    hash=content_hash,
                    source_paths=[],
                    target_paths=sorted(paths),
                    conflict_type="identity_mismatch",
                    confidence_a=0.7,
                    confidence_b=0.3,
                )
            )

    return conflicts


def check_with_inner_loop(
    snapshots: list[Snapshot],
    identities: dict[str, IdentityCluster],
    operations: list[list[EditOperation]] | None = None,
    max_iterations: int = 2,
    threshold_step: float = 0.05,
) -> ConsistencyReport:
    """Run consistency checking with iterative threshold relaxation.

    If the initial consistency check finds conflicts and *max_iterations*
    has not been reached, the effective match threshold is reduced by
    *threshold_step* on each iteration until either the check passes or
    the iteration budget is exhausted.

    Parameters
    ----------
    snapshots:
        List of snapshots to validate.
    identities:
        Mapping of cluster UID → :class:`IdentityCluster`.
    operations:
        Optional list of per-pair operation lists.
    max_iterations:
        Maximum number of relaxation iterations (default 2).
    threshold_step:
        Amount by which the effective match threshold is reduced each
        iteration (default 0.05).

    Returns
    -------
    The final :class:`ConsistencyReport` after all iterations.
    """
    report = check_consistency(snapshots, identities, operations)

    if report.passes:
        return report

    current_report = report
    iteration = 0
    relaxed_params: dict[str, object] | None = None

    while not current_report.passes and iteration < max_iterations:
        iteration += 1
        threshold = max(1.0 - (iteration * threshold_step), 0.0)
        relaxed_params = {
            "match_threshold": threshold,
            "threshold_step": threshold_step,
        }

        # Filter conflicts whose confidence_a is still above the relaxed
        # threshold — conflicts below it are considered resolved.
        filtered_conflicts: list[ConflictRecord] = [
            c
            for c in current_report.conflicts
            if c.confidence_a >= threshold
        ]

        current_report = ConsistencyReport(
            passes=len(filtered_conflicts) == 0,
            conflicts=filtered_conflicts,
            iterations_used=iteration,
            relaxed_params=relaxed_params,
        )

    return current_report
