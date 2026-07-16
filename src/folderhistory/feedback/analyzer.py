"""Post-run quality analysis module for FolderHistory's iterative feedback loop.

This module consumes consistency-check output, extracts high-confidence identity
clusters, and feeds them back into the Knowledge Base for cross-run learning.
"""

from __future__ import annotations

from dataclasses import dataclass

from folderhistory.core.consistency import ConsistencyReport
from folderhistory.knowledge import KnowledgeBase
from folderhistory.knowledge.types import FeedbackReport
from folderhistory.knowledge.update import (
    compute_igs,
    resolve_contradiction,
    update_frequencies,
    update_identities,
    update_projects,
)
from folderhistory.types import IdentityCluster


def _build_identity_hashes(
    identity_clusters: dict[str, IdentityCluster],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Build hash→cluster_uid and path→clusters mappings from identity clusters.

    Each observation ``(snap_id, path)`` in an identity cluster is encoded as a
    composite key ``snap_id:path`` to uniquely identify the file-snapshot pair.

    Parameters
    ----------
    identity_clusters:
        Mapping of cluster UID → :class:`IdentityCluster` produced by the
        identity assignment phase.

    Returns
    -------
    tuple[dict[str, str], dict[str, set[str]]]
        A pair ``(identity_hashes, path_to_clusters)`` where:
        - ``identity_hashes`` maps composite observation key → cluster UID.
        - ``path_to_clusters`` maps file path → set of cluster UIDs that
          contain that path in their observations.
    """
    identity_hashes: dict[str, str] = {}
    path_to_clusters: dict[str, set[str]] = {}

    for cluster_uid, cluster in identity_clusters.items():
        for snap_id, path in cluster.observations:
            hash_key = f"{snap_id}:{path}"
            identity_hashes[hash_key] = cluster_uid
            path_to_clusters.setdefault(path, set()).add(cluster_uid)

    return identity_hashes, path_to_clusters


@dataclass
class FeedbackAnalyzer:
    """Post-run quality analysis and KB update module.

    The analyzer consumes identity clusters and consistency reports, writes
    high-confidence results back to the Knowledge Base, resolves any detected
    contradictions, and reports quality metrics.

    Parameters
    ----------
    kb:
        The knowledge base instance to update.
    algorithm_version:
        Version identifier stamped into feedback reports (default ``"1.0.0"``).
    """

    kb: KnowledgeBase
    algorithm_version: str = "1.0.0"

    def analyze(
        self,
        identity_clusters: dict[str, IdentityCluster],
        consistency_report: ConsistencyReport | None = None,
    ) -> FeedbackReport:
        """Post-run analysis and KB update (identity-only path).

        Steps
        -----
        1. Build hash→cluster_uid mapping from *identity_clusters*.
        2. Update identities in KB (confidence ≥ 0.95).
        3. If *consistency_report* has conflicts, resolve each via
           :func:`~folderhistory.knowledge.update.resolve_contradiction`.
        4. Compute IGS via
           :func:`~folderhistory.knowledge.update.compute_igs`.
        5. Return :class:`FeedbackReport` with metrics.

        Parameters
        ----------
        identity_clusters:
            Mapping of cluster UID → :class:`IdentityCluster` from the
            identity assignment phase.
        consistency_report:
            Optional output of the consistency-check phase.  When provided
            and non-passing, conflicts are resolved as contradictions.

        Returns
        -------
        FeedbackReport
            Summary of everything that happened during this feedback pass.
        """
        identity_hashes, path_to_clusters = _build_identity_hashes(identity_clusters)

        # Step 2: Store high-confidence identities
        new_identities = update_identities(self.kb, identity_hashes)

        # Step 3: Resolve contradictions from the consistency report
        resolved_count = self._resolve_contradictions(
            consistency_report,
            path_to_clusters,
        )

        # Step 4: Identity Graph Stability
        igs = compute_igs(self.kb)

        # Step 5: Build report
        warnings_list: list[str] = []
        if igs < 0.8:
            warnings_list.append(f"IGS dropped to {igs:.3f}, below 0.8 threshold")

        iterations = (
            consistency_report.iterations_used if consistency_report is not None else 0
        )

        return FeedbackReport(
            run_id="",
            new_identities=new_identities,
            resolved_contradictions=resolved_count,
            quality_metrics={"igs": igs},
            iterations=iterations,
            warnings=warnings_list,
        )

    def analyze_with_project(
        self,
        identity_clusters: dict[str, IdentityCluster],
        project_fingerprints: dict[str, dict[str, float]],
        all_hashes: list[str],
        consistency_report: ConsistencyReport | None = None,
    ) -> FeedbackReport:
        """Full analysis including project fingerprints and hash frequencies.

        Steps
        -----
        1. Update hash observation frequencies.
        2. Update identities (confidence ≥ 0.95).
        3. Update projects (IDF-Jaccard > 0.7).
        4. Resolve contradictions from *consistency_report*.
        5. Compute IGS.
        6. Return :class:`FeedbackReport` with all metrics.

        Parameters
        ----------
        identity_clusters:
            Mapping of cluster UID → :class:`IdentityCluster`.
        project_fingerprints:
            Mapping of project UID → ``{hash: IDF_weight}`` fingerprint.
        all_hashes:
            All content hashes observed in this run, used to bump frequency
            counters.
        consistency_report:
            Optional consistency-check output.

        Returns
        -------
        FeedbackReport
            Summary of the full feedback pass.
        """
        identity_hashes, path_to_clusters = _build_identity_hashes(identity_clusters)

        # Step 1: Bump observation frequencies
        update_frequencies(self.kb, all_hashes)

        # Step 2: Store high-confidence identities
        new_identities = update_identities(self.kb, identity_hashes)

        # Step 3: Store confirmed project fingerprints
        updated_projects = update_projects(self.kb, project_fingerprints)

        # Step 4: Resolve contradictions
        resolved_count = self._resolve_contradictions(
            consistency_report,
            path_to_clusters,
        )

        # Step 5: IGS
        igs = compute_igs(self.kb)

        # Step 6: Build report
        warnings_list: list[str] = []
        if igs < 0.8:
            warnings_list.append(f"IGS dropped to {igs:.3f}, below 0.8 threshold")

        iterations = (
            consistency_report.iterations_used if consistency_report is not None else 0
        )

        return FeedbackReport(
            run_id="",
            new_identities=new_identities,
            resolved_contradictions=resolved_count,
            quality_metrics={
                "igs": igs,
                "projects_updated": float(updated_projects),
            },
            iterations=iterations,
            warnings=warnings_list,
        )

    def _resolve_contradictions(
        self,
        consistency_report: ConsistencyReport | None,
        path_to_clusters: dict[str, set[str]],
    ) -> int:
        """Resolve each conflict in *consistency_report* as a contradiction.

        For every :class:`ConflictRecord` with a non-empty ``hash``, the
        method collects candidate cluster UIDs by looking up the conflict's
        source and target paths in *path_to_clusters*, then delegates to
        :func:`~folderhistory.knowledge.update.resolve_contradiction`.

        Returns the number of conflicts that were resolved.
        """
        if consistency_report is None or consistency_report.passes:
            return 0

        resolved_count = 0
        for conflict in consistency_report.conflicts:
            if not conflict.hash:
                continue

            # Collect candidate identity UIDs from the path mapping
            candidates: set[str] = set()
            for path in conflict.source_paths:
                if path in path_to_clusters:
                    candidates.update(path_to_clusters[path])
            for path in conflict.target_paths:
                if path in path_to_clusters:
                    candidates.update(path_to_clusters[path])

            # Also consider the KB's current identity for this hash
            kb_identity = self.kb.get_identity(conflict.hash)
            if kb_identity is not None:
                candidates.add(kb_identity)

            # Pick two distinct identity UIDs, falling back to "unknown"
            identity_a: str = ""
            identity_b: str = ""

            if len(candidates) >= 2:
                sorted_candidates = sorted(candidates)
                identity_a = sorted_candidates[0]
                identity_b = sorted_candidates[1]
            elif len(candidates) == 1:
                identity_a = next(iter(candidates))
                identity_b = "unknown"
            else:
                identity_a = "unknown"
                identity_b = "unknown"

            _ = resolve_contradiction(
                self.kb,
                conflict.hash,
                identity_a,
                identity_b,
            )
            resolved_count += 1

        return resolved_count
