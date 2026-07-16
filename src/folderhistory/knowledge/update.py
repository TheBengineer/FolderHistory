"""Knowledge Base update rules and contradiction resolution.

This module provides the core update primitives for FolderHistory's iterative
feedback loop: storing identity clusters, project fingerprints, observation
frequencies, resolving contradictions, and measuring identity graph stability.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import cast

from folderhistory.knowledge import KnowledgeBase
from folderhistory.knowledge.types import ContradictionRecord


def _idf_jaccard(
    fp_a: dict[str, float],
    fp_b: dict[str, float],
) -> float:
    """Compute the IDF-weighted Jaccard similarity between two fingerprints.

    Parameters
    ----------
    fp_a:
        First fingerprint vector (hash → IDF weight).
    fp_b:
        Second fingerprint vector (hash → IDF weight).

    Returns
    -------
    float
        A value in [0, 1] where 1 means identical weighted sets.
        Returns 0.0 when both sets are empty.
    """
    keys_a = set(fp_a)
    keys_b = set(fp_b)
    intersection = keys_a & keys_b
    union = keys_a | keys_b

    if not union:
        return 0.0

    intersection_weight = sum(min(fp_a[k], fp_b[k]) for k in intersection)
    union_weight = sum(
        max(fp_a.get(k, 0.0), fp_b.get(k, 0.0)) for k in union
    )

    if union_weight == 0.0:
        return 0.0
    return intersection_weight / union_weight


def update_identities(
    kb: KnowledgeBase,
    identity_hashes: dict[str, str],
    confidence_threshold: float = 0.95,
) -> int:
    """Store high-confidence identity clusters into KB.

    Only stores entries where the hash's observation-derived confidence
    meets or exceeds *confidence_threshold*.

    Parameters
    ----------
    kb:
        The knowledge base instance.
    identity_hashes:
        Mapping of content hash → cluster UID.
    confidence_threshold:
        Minimum confidence required to store an identity (default 0.95).

    Returns
    -------
    int
        Number of identities actually stored.
    """
    stored = 0
    for hash_val, cluster_uid in identity_hashes.items():
        freq = kb.get_frequency(hash_val)
        confidence = min(1.0, freq * 0.1)
        if confidence >= confidence_threshold:
            kb.set_identity(hash_val, cluster_uid, confidence)
            stored += 1
    return stored


def update_projects(
    kb: KnowledgeBase,
    project_fingerprints: dict[str, dict[str, float]],
    jaccard_threshold: float = 0.7,
) -> int:
    """Store confirmed project fingerprints into ProjectDB.

    For each project UID with a fingerprint vector, the function checks
    whether a project with the same UID already exists in the KB.  If it
    does, the IDF-Jaccard similarity between the stored fingerprint and
    the incoming fingerprint must exceed *jaccard_threshold* for the
    update to proceed.

    Parameters
    ----------
    kb:
        The knowledge base instance.
    project_fingerprints:
        Mapping of project UID to ``{hash: IDF_weight}`` fingerprint.
    jaccard_threshold:
        Minimum IDF-Jaccard similarity to overwrite an existing project
        fingerprint (default 0.7).

    Returns
    -------
    int
        Number of projects stored.
    """
    stored = 0
    for project_uid, fingerprint in project_fingerprints.items():
        existing = kb.get_project(project_uid)
        if existing is not None and isinstance(existing, dict):
            # existing is dict[Unknown, Unknown] from the ABC — cast to access known keys
            existing_dict = cast("dict[str, object]", existing)
            existing_fp_raw = existing_dict.get("fingerprint")
            if isinstance(existing_fp_raw, dict):
                existing_fp = cast("dict[str, float]", existing_fp_raw)
                jaccard = _idf_jaccard(fingerprint, existing_fp)
                if jaccard < jaccard_threshold:
                    continue

        kb.set_project(project_uid, fingerprint)
        stored += 1

    return stored


def update_frequencies(
    kb: KnowledgeBase,
    hashes: list[str],
) -> None:
    """Increment the observation count for each hash in KB."""
    for h in hashes:
        _ = kb.increment_frequency(h)


def resolve_contradiction(
    kb: KnowledgeBase,
    hash: str,
    identity_a: str,
    identity_b: str,
) -> ContradictionRecord:
    """Resolve a contradiction between two identity clusters for *hash*.

    Strategy chain (tried in order):

    1. **CONFIDENCE_WEIGHT** — The identity currently stored in the KB
       (most recently set) is treated as having higher confidence.
    2. **EVIDENCE_WEIGHT** — If the hash has been observed at least once
       the current KB identity is backed by evidence; identity_a wins.
    3. **Default** — Neither strategy resolves, so CONFIDENCE_WEIGHT is
       applied with identity_a as the winner.

    Parameters
    ----------
    kb:
        The knowledge base instance.
    hash:
        The content hash that triggered the contradiction.
    identity_a:
        First conflicting cluster UID.
    identity_b:
        Second conflicting cluster UID.

    Returns
    -------
    ContradictionRecord
        The recorded contradiction with the applied strategy.
    """
    utc_now = datetime.now(timezone.utc).isoformat()

    # Trivial case: both identities are the same — no contradiction
    if identity_a == identity_b:
        record = ContradictionRecord(
            hash=hash,
            identity_a=identity_a,
            identity_b=identity_b,
            strategy_applied="CONFIDENCE_WEIGHT",
            resolved_at=utc_now,
            resolution_confidence=1.0,
        )
        kb.add_contradiction(record)
        return record

    freq = kb.get_frequency(hash)
    current = kb.get_identity(hash)

    # Strategy 1 — CONFIDENCE_WEIGHT: the currently stored identity wins
    if current == identity_a:
        strategy = "CONFIDENCE_WEIGHT"
        resolution_confidence = min(0.5 + freq * 0.1, 0.95)
    elif current == identity_b:
        strategy = "CONFIDENCE_WEIGHT"
        resolution_confidence = min(0.5 + freq * 0.1, 0.95)
    elif freq > 0 and current is not None:
        # Strategy 2 — EVIDENCE_WEIGHT: the hash has been observed,
        # giving weight to the current KB identity
        strategy = "EVIDENCE_WEIGHT"
        resolution_confidence = min(0.5 + freq * 0.05, 0.85)
    else:
        # Default: neither strategy can resolve
        strategy = "CONFIDENCE_WEIGHT"
        resolution_confidence = 0.5

    record = ContradictionRecord(
        hash=hash,
        identity_a=identity_a,
        identity_b=identity_b,
        strategy_applied=strategy,
        resolved_at=utc_now,
        resolution_confidence=resolution_confidence,
    )
    kb.add_contradiction(record)
    return record


def compute_igs(kb: KnowledgeBase) -> float:
    """Identity Graph Stability.

    IGS = 1 - (len(contradictions) / max(len(identities), 1))

    Unique identity UIDs are collected from contradiction records (the only
    identity enumeration available through the ABC).  Returns 1.0 when there
    are no identities.  A warning is emitted when IGS < 0.8 (a drop of 0.2
    from the baseline of 1.0).

    Parameters
    ----------
    kb:
        The knowledge base instance.

    Returns
    -------
    float
        The IGS score in [0, 1].
    """
    contradictions = kb.get_contradictions()
    num_contradictions = len(contradictions)

    # Collect unique cluster UIDs referenced across all contradictions
    identity_uids: set[str] = set()
    for c in contradictions:
        if isinstance(c, ContradictionRecord):
            identity_uids.add(c.identity_a)
            identity_uids.add(c.identity_b)

    num_identities = len(identity_uids)

    if num_identities == 0:
        return 1.0

    igs = 1.0 - (num_contradictions / max(num_identities, 1))

    if igs < 0.8:
        warnings.warn(
            f"IGS dropped to {igs:.3f}, below 0.8 threshold",
            UserWarning,
            stacklevel=2,
        )

    return igs
