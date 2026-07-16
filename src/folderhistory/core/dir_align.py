"""Directory-level pre-alignment for FolderHistory.

Matches project roots across snapshots before file-level identity assignment.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict
from typing import cast

from folderhistory.knowledge import KnowledgeBase
from folderhistory.types import FileRecord, Snapshot

logger = logging.getLogger(__name__)


# ── Component 1 ────────────────────────────────────────────────────────────────


def build_inverted_index(
    snapshots: list[Snapshot],
) -> dict[str, list[tuple[str, str]]]:
    """Build a content-hash → [(snapshot_id, relative_path), …] mapping.

    Uses ``normalized_blake3`` when available (non-``None``), falling back to
    ``raw_blake3``.  Files where *both* are effectively empty are skipped with a
    warning.

    Complexity: O(total_files) time and space.
    """
    idx: dict[str, list[tuple[str, str]]] = {}

    for snap in snapshots:
        for f in snap.files:
            h = f.normalized_blake3 if f.normalized_blake3 is not None else f.raw_blake3
            if h:
                idx.setdefault(h, []).append((snap.id, f.path))
            else:
                logger.warning(  # noqa: E501
                    "File %r in snapshot %s has no usable hash (normalized_blake3=%r, raw_blake3=%r)",
                    f.path,
                    snap.id,
                    f.normalized_blake3,
                    f.raw_blake3,
                )

    return idx


# ── Component 2 ────────────────────────────────────────────────────────────────


def _hash_frequency(
    inverted_index: dict[str, list[tuple[str, str]]],
) -> dict[str, int]:
    """Derive per-hash global frequency (unique snapshots a hash appears in)."""
    freq: dict[str, int] = {}
    for h, entries in inverted_index.items():
        unique: set[str] = {sid for sid, _ in entries}
        freq[h] = len(unique)
    return freq


def idf_weighted_jaccard(
    hashes_a: set[str],
    hashes_b: set[str],
    global_freq: dict[str, int],
    num_snapshots: int,
) -> float:
    """Jaccard index with IDF discount for high-frequency hashes.

    A hash is considered *common* when its global frequency ≥ sqrt(N).  Common
    hashes are excluded from both the intersection and the union before
    computing the Jaccard coefficient.

    Returns 0.0 when *both* ``hashes_a`` and ``hashes_b`` are empty, or when
    after discounting the union is empty.
    """
    threshold = math.sqrt(num_snapshots)

    specific_a = {h for h in hashes_a if global_freq.get(h, 0) < threshold}
    specific_b = {h for h in hashes_b if global_freq.get(h, 0) < threshold}

    intersection = specific_a & specific_b
    union = specific_a | specific_b

    if not union:
        return 0.0
    return len(intersection) / len(union)


# ── Component 3 ────────────────────────────────────────────────────────────────


def _get_usable_hash(f: FileRecord) -> str | None:
    """Return the best content hash for *f*, or ``None`` if unavailable."""
    h = f.normalized_blake3 if f.normalized_blake3 is not None else f.raw_blake3
    return h if h else None


# ── Fingerprint helpers (KB probe) ─────────────────────────────────────────────


_SIMILARITY_KNOWN = 0.7
_SIMILARITY_TENTATIVE = 0.4


def _compute_project_fingerprint(snapshot: Snapshot) -> str:
    """Return a deterministic fingerprint string for *snapshot*.

    The fingerprint is a SHA-256 hex digest of the sorted list of all
    usable content hashes in the snapshot.  An empty string is returned
    when the snapshot has no usable hashes.
    """
    hashes = sorted(
        h for f in snapshot.files if (h := _get_usable_hash(f)) is not None
    )
    if not hashes:
        return ""
    return hashlib.sha256("".join(hashes).encode()).hexdigest()


def _plain_jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def content_probe_subtree(
    large_snapshot: Snapshot,
    reference_snapshot: Snapshot,
    inverted_index: dict[str, list[tuple[str, str]]],  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
    threshold: float = 0.5,
) -> tuple[str | None, float]:
    """Locate the subdirectory within *large_snapshot* that best resembles
    *reference_snapshot*.

    For every ancestor directory of every file in *large_snapshot*, the
    function computes the fraction of files whose content hash also appears in
    the reference.  Among those scoring at or above *threshold*, the deepest
    (most-specific) directory wins.

    Returns
    -------
    ``(candidate_root_path, confidence_score)``, or ``(None, 0.0)`` when no
    directory meets the threshold or the reference is empty.
    """
    # ── Collect reference hashes ────────────────────────────────────────────
    ref_hashes: set[str] = set()
    for f in reference_snapshot.files:
        h = _get_usable_hash(f)
        if h is not None:
            ref_hashes.add(h)

    if not ref_hashes:
        return (None, 0.0)

    # ── Score every ancestor directory in the large snapshot ────────────────
    total: dict[str, int] = defaultdict(int)
    matched: dict[str, int] = defaultdict(int)

    for f in large_snapshot.files:
        h = _get_usable_hash(f)
        if h is None:
            continue

        parts = f.path.split("/")
        # Every prefix except the last component (file itself) is a directory
        for i in range(len(parts) - 1):
            prefix = "/".join(parts[: i + 1])
            total[prefix] += 1
            if h in ref_hashes:
                matched[prefix] += 1

    # ── Find the best directory ─────────────────────────────────────────────
    best_dir: str | None = None
    best_score = 0.0

    for prefix, t in total.items():
        if t == 0:
            continue
        score = matched[prefix] / t
        if score >= threshold:
            if best_dir is None:
                best_dir = prefix
                best_score = score
            elif score > best_score:
                best_dir = prefix
                best_score = score
            elif score == best_score and len(prefix.split("/")) > len(best_dir.split("/")):
                best_dir = prefix
                best_score = score

    if best_dir is None:
        return (None, 0.0)

    return (best_dir, best_score)


# ── Component 4 ────────────────────────────────────────────────────────────────


def normalize_paths_to_root(
    snapshots: list[Snapshot],
    root_candidates: dict[str, str],
) -> list[Snapshot]:
    """Return new ``Snapshot`` instances with paths rebased relative to their project
    root.

    *root_candidates* maps ``snapshot_id → root_path`` (a relative path within
    the snapshot).  Snapshots not present in the dictionary, or whose root path
    is empty or ``"."``, are returned unchanged.
    """
    result: list[Snapshot] = []

    for snap in snapshots:
        if snap.id not in root_candidates:
            result.append(snap)
            continue

        root = root_candidates[snap.id]
        if not root or root == ".":
            result.append(snap)
            continue

        root_prefix = root + "/"
        new_files = [
            FileRecord(
                path=f.path[len(root_prefix) :]
                if f.path.startswith(root_prefix)
                else f.path,
                size=f.size,
                mode=f.mode,
                mtime_ns=f.mtime_ns,
                ctime_ns=f.ctime_ns,
                raw_blake3=f.raw_blake3,
                normalized_blake3=f.normalized_blake3,
                line_ending=f.line_ending,
                xxhash64=f.xxhash64,
                is_symlink=f.is_symlink,
                target_path=f.target_path,
            )
            for f in snap.files
        ]

        new_source = snap.source_path / root
        result.append(
            Snapshot(
                id=snap.id,
                timestamp=snap.timestamp,
                source_path=new_source,
                files=new_files,
            )
        )

    return result


# ── Component 5 ────────────────────────────────────────────────────────────────


def resolve_project_identities(
    snapshots: list[Snapshot],
    global_inverted_index: dict[str, list[tuple[str, str]]],
    kb: KnowledgeBase | None = None,
    mode: str = "auto",
) -> dict[str, list[str]]:
    """Group snapshot IDs into project identities using IDF-weighted Jaccard.

    Two snapshots whose ``idf_weighted_jaccard`` exceeds 0.5 are considered
    the same project.  Transitive closure (union-find) is applied so that
    ``A ≈ B`` and ``B ≈ C`` places all three in one group.

    Parameters
    ----------
    kb:
        Optional :class:`KnowledgeBase` for delta-mode acceleration.  When
        provided and *mode* is not ``"full"``, each snapshot's content
        fingerprint is checked against known projects before running the
        full pairwise comparison.
    mode:
        One of ``"auto"``, ``"full"``, or ``"delta"``.

        - ``"auto"`` (default): Snapshots that match a known project
          (Jaccard :math:`\\ge 0.7`) are assigned directly; the remaining
          snapshots go through the full pipeline.
        - ``"full"``: Run the full pipeline on all snapshots, ignoring any
          KB.  Equivalent to the pre-KB behaviour.
        - ``"delta"``: Only return groups for snapshots that match known
          projects.  The full pipeline is skipped entirely.

    Returns
    -------
    ``{project_uid: [snapshot_id, …], …}`` where the *project_uid* is
    the ID of the first snapshot in each group (by input order), or a
    KB-assigned identifier for delta-matched groups.
    """
    num_snapshots = len(snapshots)
    if num_snapshots == 0:
        return {}

    if mode not in ("auto", "full", "delta"):
        msg = f"Unknown mode: {mode!r} (expected 'auto', 'full', or 'delta')"
        raise ValueError(msg)

    # ── Global hash frequency ──────────────────────────────────────────────
    global_freq = _hash_frequency(global_inverted_index)

    # ── Per-snapshot hash sets ──────────────────────────────────────────────
    snapshot_hashes: dict[str, set[str]] = {}
    for snap in snapshots:
        hashes: set[str] = set()
        for f in snap.files:
            h = _get_usable_hash(f)
            if h is not None:
                hashes.add(h)
        snapshot_hashes[snap.id] = hashes

    # ── KB probe (delta mode support) ──────────────────────────────────────
    kb_result: dict[str, list[str]] = {}
    skipped_ids: set[str] = set()

    if kb is not None and mode != "full":
        for snap in snapshots:
            fp = _compute_project_fingerprint(snap)
            if not fp:
                continue

            stored = kb.get_project(fp)
            if stored is None:
                continue

            stored_dict: dict[str, object] = (
                cast("dict[str, object]", stored) if isinstance(stored, dict) else {}
            )
            stored_fp_raw = stored_dict.get("fingerprint", {})
            if not isinstance(stored_fp_raw, dict):
                continue

            stored_fp_typed = cast("dict[str, float]", stored_fp_raw)
            similarity = _plain_jaccard(
                snapshot_hashes.get(snap.id, set()),
                set(stored_fp_typed.keys()),
            )

            if similarity >= _SIMILARITY_KNOWN:
                kb_result.setdefault(fp, []).append(snap.id)
                skipped_ids.add(snap.id)

        if mode == "delta":
            return kb_result

    # ── Full pipeline (for full mode or auto-mode unmatched snapshots) ────
    if mode == "auto" and kb is not None:
        remaining = [s for s in snapshots if s.id not in skipped_ids]
    else:
        remaining = snapshots

    if not remaining:
        return kb_result

    # ── Union-Find ──────────────────────────────────────────────────────────
    parent: dict[str, str] = {s.id: s.id for s in remaining}

    def find(x: str) -> str:
        # Path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    # ── Pairwise comparison (on remaining snapshots) ───────────────────────
    rlen = len(remaining)
    for i in range(rlen):
        si = remaining[i]
        for j in range(i + 1, rlen):
            sj = remaining[j]
            jaccard = idf_weighted_jaccard(
                snapshot_hashes[si.id],
                snapshot_hashes[sj.id],
                global_freq,
                num_snapshots,  # Use total count for consistent IDF
            )
            if jaccard > 0.5:
                union(si.id, sj.id)

    # ── Group by component ─────────────────────────────────────────────────
    components: dict[str, list[str]] = {}
    for s in remaining:
        root = find(s.id)
        components.setdefault(root, []).append(s.id)

    result: dict[str, list[str]] = {}
    for snap_ids in components.values():
        uid = snap_ids[0]
        result[uid] = snap_ids

    # ── Merge KB results ───────────────────────────────────────────────────
    for uid, sids in kb_result.items():
        result[uid] = sids

    return result
