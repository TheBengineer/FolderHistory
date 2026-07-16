"""Operation derivation engine for FolderHistory.

From identity assignments, derive edit operations (create, delete, modify,
rename, move, copy) between consecutive snapshots.
"""

from __future__ import annotations

from typing import Literal

from folderhistory.types import EditOperation, FileRecord, IdentityCluster, Snapshot

# ── Public API ─────────────────────────────────────────────────────────────────


def derive_operations(
    snapshots: list[Snapshot],
    identities: dict[str, IdentityCluster],
) -> list[list[EditOperation]]:
    """Derive edit operations between consecutive snapshot pairs.

    For each consecutive snapshot pair ``(S_i, S_{i+1})``, the function inspects
    every identity cluster to determine what filesystem operations best explain
    the transition.

    Parameters
    ----------
    snapshots:
        Ordered list of snapshots.
    identities:
        Identity clusters (from :func:`~folderhistory.core.identity.assign_identities_exact`
        or equivalent).

    Returns
    -------
    One :class:`list` of :class:`EditOperation` per consecutive pair.
    Returns an empty list when fewer than two snapshots are provided.
    """
    if len(snapshots) < 2:
        return []

    # Build a lookup: (snapshot_id, file_path) → FileRecord
    file_lookup: dict[tuple[str, str], FileRecord] = {}
    for snap in snapshots:
        for f in snap.files:
            file_lookup[(snap.id, f.path)] = f

    results: list[list[EditOperation]] = []
    for i in range(len(snapshots) - 1):
        ops = _derive_ops_between(snapshots[i], snapshots[i + 1], identities, file_lookup)
        results.append(ops)

    return results


def derive_operations_between(
    snapshot_a: Snapshot,
    snapshot_b: Snapshot,
    identities: dict[str, IdentityCluster],
) -> list[EditOperation]:
    """Derive edit operations between two specific snapshots.

    This is a convenience wrapper when only a single pair is of interest,
    avoiding the need to build a full file lookup from all snapshots.

    Parameters
    ----------
    snapshot_a:
        The earlier (or reference) snapshot.
    snapshot_b:
        The later snapshot.
    identities:
        Identity clusters that may span *snapshot_a* and *snapshot_b*.

    Returns
    -------
    List of :class:`EditOperation` instances describing the transition from
    *snapshot_a* to *snapshot_b*.
    """
    file_lookup: dict[tuple[str, str], FileRecord] = {}
    for f in snapshot_a.files:
        file_lookup[(snapshot_a.id, f.path)] = f
    for f in snapshot_b.files:
        file_lookup[(snapshot_b.id, f.path)] = f

    return _derive_ops_between(snapshot_a, snapshot_b, identities, file_lookup)


# ── Internal helpers ───────────────────────────────────────────────────────────


def _classify_path_change(old_path: str, new_path: str) -> Literal["move", "rename"]:
    """Classify a path change as ``"move"`` or ``"rename"``.

    A *move* changes the parent directory while keeping the same basename.
    A *rename* changes the basename itself.
    """
    old_basename = old_path.rsplit("/", 1)[-1] if "/" in old_path else old_path
    new_basename = new_path.rsplit("/", 1)[-1] if "/" in new_path else new_path
    old_dir = old_path.rsplit("/", 1)[0] if "/" in old_path else ""
    new_dir = new_path.rsplit("/", 1)[0] if "/" in new_path else ""

    if old_basename == new_basename and old_dir != new_dir:
        return "move"
    return "rename"


def _file_record(
    file_lookup: dict[tuple[str, str], FileRecord],
    snap_id: str,
    path: str,
) -> FileRecord | None:
    """Look up a :class:`FileRecord` by snapshot id and path."""
    return file_lookup.get((snap_id, path))


def _find_source_path(
    snap_a: Snapshot,
    hash_value: str,
    cluster_paths_in_a: set[str],
) -> str | None:
    """Find a path in *snap_a* whose file has the given raw_blake3.

    Prefers paths that belong to the identity cluster (``cluster_paths_in_a``)
    over files outside the cluster.  Returns ``None`` when no file in
    *snap_a* has *hash_value*.
    """
    for fr in snap_a.files:
        if fr.raw_blake3 == hash_value and fr.path in cluster_paths_in_a:
            return fr.path
    for fr in snap_a.files:
        if fr.raw_blake3 == hash_value:
            return fr.path
    return None


def _derive_ops_between(
    snap_a: Snapshot,
    snap_b: Snapshot,
    identities: dict[str, IdentityCluster],
    file_lookup: dict[tuple[str, str], FileRecord],
) -> list[EditOperation]:
    """Core derivation logic shared by both public APIs."""
    ops: list[EditOperation] = []

    for cluster in identities.values():
        # Collect observations belonging to each snapshot.
        obs_in_a: list[tuple[str, str]] = [
            (sid, p) for (sid, p) in cluster.observations if sid == snap_a.id
        ]
        obs_in_b: list[tuple[str, str]] = [
            (sid, p) for (sid, p) in cluster.observations if sid == snap_b.id
        ]

        in_a = bool(obs_in_a)
        in_b = bool(obs_in_b)

        # ── Neither snapshot has this identity → skip ───────────────────────
        if not in_a and not in_b:
            continue

        # ── Present only in S_i → DELETE ────────────────────────────────────
        if in_a and not in_b:
            path_a = obs_in_a[0][1]
            rec_a = _file_record(file_lookup, snap_a.id, path_a)
            ops.append(
                EditOperation(
                    op_type="delete",
                    file_id=path_a,
                    source_path=path_a,
                    old_hash=rec_a.raw_blake3 if rec_a else None,
                    confidence=1.0,
                ),
            )
            continue

        # ── Present only in S_{i+1} → CREATE ────────────────────────────────
        if not in_a and in_b:
            path_b = obs_in_b[0][1]
            rec_b = _file_record(file_lookup, snap_b.id, path_b)
            ops.append(
                EditOperation(
                    op_type="create",
                    file_id=path_b,
                    target_path=path_b,
                    new_hash=rec_b.raw_blake3 if rec_b else None,
                    confidence=1.0,
                ),
            )
            continue

        # ── Present in both snapshots ────────────────────────────────────────
        # Pick the best path candidate for each snapshot.
        path_a = _pick_obs_path(obs_in_a, obs_in_b)
        path_b = _pick_obs_path(obs_in_b, obs_in_a)

        rec_a = _file_record(file_lookup, snap_a.id, path_a)
        rec_b = _file_record(file_lookup, snap_b.id, path_b)

        # ── Copy detection ─────────────────────────────────────────────────
        # If the cluster has more observations in S_{i+1} than in S_i, some
        # of those extra paths may be copies (same content hash) or new files
        # (different content hash).
        paths_a_set: set[str] = {p for _, p in obs_in_a}
        paths_b_set: set[str] = {p for _, p in obs_in_b}
        extra_in_b = (paths_b_set - paths_a_set) - {path_b}

        if extra_in_b:
            hashes_in_a: set[str] = {fr.raw_blake3 for fr in snap_a.files}

            for new_path in sorted(extra_in_b):
                rec_new = _file_record(file_lookup, snap_b.id, new_path)
                if rec_new is None:
                    continue

                if rec_new.raw_blake3 in hashes_in_a:
                    source_path = _find_source_path(snap_a, rec_new.raw_blake3, paths_a_set)
                    ops.append(
                        EditOperation(
                            op_type="copy",
                            file_id=source_path or new_path,
                            source_path=source_path,
                            target_path=new_path,
                            old_hash=rec_new.raw_blake3,
                            new_hash=rec_new.raw_blake3,
                            confidence=1.0,
                        ),
                    )
                else:
                    ops.append(
                        EditOperation(
                            op_type="create",
                            file_id=new_path,
                            target_path=new_path,
                            new_hash=rec_new.raw_blake3,
                            confidence=1.0,
                        ),
                    )

        same_path = path_a == path_b

        if same_path:
            # ── Same path → unchanged or MODIFY ─────────────────────────────
            if rec_a is not None and rec_b is not None and rec_a.raw_blake3 == rec_b.raw_blake3:
                # Identical content at the same path → no operation.
                continue

            # Content changed in place.
            ops.append(
                EditOperation(
                    op_type="modify",
                    file_id=path_a,
                    source_path=path_a,
                    target_path=path_a,
                    old_hash=rec_a.raw_blake3 if rec_a else None,
                    new_hash=rec_b.raw_blake3 if rec_b else None,
                    confidence=1.0,
                ),
            )
        else:
            # ── Different path → check content similarity ───────────────────
            same_raw = (
                rec_a is not None
                and rec_b is not None
                and rec_a.raw_blake3 == rec_b.raw_blake3
            )
            same_normalized = (
                rec_a is not None
                and rec_b is not None
                and rec_a.normalized_blake3 is not None
                and rec_b.normalized_blake3 is not None
                and rec_a.normalized_blake3 == rec_b.normalized_blake3
            )

            if same_raw:
                # Identical content at a different path → rename or move.
                op_type = _classify_path_change(path_a, path_b)
                ops.append(
                    EditOperation(
                        op_type=op_type,
                        file_id=path_a,
                        source_path=path_a,
                        target_path=path_b,
                        old_hash=rec_a.raw_blake3 if rec_a else None,
                        new_hash=rec_b.raw_blake3 if rec_b else None,
                        confidence=1.0,
                    ),
                )
            elif same_normalized:
                # Cross-platform content match (line-ending difference only)
                # → rename/move with reduced confidence.
                op_type = _classify_path_change(path_a, path_b)
                ops.append(
                    EditOperation(
                        op_type=op_type,
                        file_id=path_a,
                        source_path=path_a,
                        target_path=path_b,
                        old_hash=rec_a.raw_blake3 if rec_a else None,
                        new_hash=rec_b.raw_blake3 if rec_b else None,
                        confidence=0.8,
                    ),
                )
            else:
                # Completely different content → delete + create.
                ops.append(
                    EditOperation(
                        op_type="delete",
                        file_id=path_a,
                        source_path=path_a,
                        old_hash=rec_a.raw_blake3 if rec_a else None,
                        confidence=1.0,
                    ),
                )
                ops.append(
                    EditOperation(
                        op_type="create",
                        file_id=path_b,
                        target_path=path_b,
                        new_hash=rec_b.raw_blake3 if rec_b else None,
                        confidence=1.0,
                    ),
                )

    return ops


def _pick_obs_path(
    obs_this: list[tuple[str, str]],
    obs_other: list[tuple[str, str]],
) -> str:
    """Pick the best-matching path for a snapshot from a cluster's observations.

    If one of the observations in *obs_this* shares the same path as an
    observation in *obs_other*, that path is preferred (it likely represents
    the same file rather than a copy).  Otherwise the first observation is
    used.
    """
    other_paths = {p for _, p in obs_other}
    for _, p in obs_this:
        if p in other_paths:
            return p
    return obs_this[0][1]
