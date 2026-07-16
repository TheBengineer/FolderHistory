"""Pairwise identity matching engine for FolderHistory.

Provides exact-match functions that find files sharing the same content hash
across snapshot pairs, producing evidence that ``identity.py`` consumes to
assign global identities via Union-Find.
"""

from __future__ import annotations

from folderhistory.types import Snapshot

# ── Public API ─────────────────────────────────────────────────────────────────


def match_snapshots_exact(
    snapshots: list[Snapshot],
    *,
    project_uid: str = "default",
) -> list[dict[str, str]]:
    """Return per-pair hash → identity-key mappings.

    For every *consecutive* pair of snapshots ``(i, i+1)``, this function
    identifies files whose **raw BLAKE3** content digest appears in **both**
    snapshots of the pair.  Each such shared hash is recorded together with a
    representative identity key ``"<project_uid>:<relative_path>"``.

    Parameters
    ----------
    snapshots:
        Ordered list of snapshots to compare pairwise.
    project_uid:
        Project-unique identifier (``"default"`` for single-root mode).

    Returns
    -------
    One :class:`dict` per consecutive pair.  Each dict maps a hex hash string
    to an identity key that participated in the match.
    """
    if len(snapshots) < 2:
        return []

    results: list[dict[str, str]] = []

    for i in range(len(snapshots) - 1):
        snap_a = snapshots[i]
        snap_b = snapshots[i + 1]

        # Build hash → identity-key indices for both snapshots.
        hash_keys_a: dict[str, set[str]] = {}
        for f in snap_a.files:
            if f.raw_blake3:
                ikey = _identity_key(f.path, project_uid)
                hash_keys_a.setdefault(f.raw_blake3, set()).add(ikey)

        hash_keys_b: dict[str, set[str]] = {}
        for f in snap_b.files:
            if f.raw_blake3:
                ikey = _identity_key(f.path, project_uid)
                hash_keys_b.setdefault(f.raw_blake3, set()).add(ikey)

        # Hashes that appear in *both* snapshots represent content-identity
        # links.
        shared_hashes = set(hash_keys_a) & set(hash_keys_b)

        pair_map: dict[str, str] = {}
        for h in shared_hashes:
            all_keys: set[str] = hash_keys_a[h] | hash_keys_b[h]
            if all_keys:
                # Pick the lexicographically smallest key as representative.
                pair_map[h] = min(all_keys)

        results.append(pair_map)

    return results


def match_snapshots_exact_all_pairs(
    snapshots: list[Snapshot],
    *,
    project_uid: str = "default",
) -> list[dict[str, str]]:
    """Like :func:`match_snapshots_exact` but compares **all** unordered pairs.

    This is useful when snapshot ordering is uncertain and every pair should
    be examined for content-identity links.

    Parameters
    ----------
    snapshots:
        List of snapshots (order does not matter for this variant).
    project_uid:
        Project-unique identifier (``"default"`` for single-root mode).

    Returns
    -------
    One :class:`dict` per unordered pair ``(i, j)`` with *i < j*.
    """
    if len(snapshots) < 2:
        return []

    results: list[dict[str, str]] = []

    for i in range(len(snapshots)):
        for j in range(i + 1, len(snapshots)):
            snap_a = snapshots[i]
            snap_b = snapshots[j]

            hash_keys_a: dict[str, set[str]] = {}
            for f in snap_a.files:
                if f.raw_blake3:
                    ikey = _identity_key(f.path, project_uid)
                    hash_keys_a.setdefault(f.raw_blake3, set()).add(ikey)

            hash_keys_b: dict[str, set[str]] = {}
            for f in snap_b.files:
                if f.raw_blake3:
                    ikey = _identity_key(f.path, project_uid)
                    hash_keys_b.setdefault(f.raw_blake3, set()).add(ikey)

            shared_hashes = set(hash_keys_a) & set(hash_keys_b)

            pair_map: dict[str, str] = {}
            for h in shared_hashes:
                all_keys = hash_keys_a[h] | hash_keys_b[h]
                if all_keys:
                    pair_map[h] = min(all_keys)

            results.append(pair_map)

    return results


# ── Internal helpers ───────────────────────────────────────────────────────────


def _identity_key(path: str, project_uid: str = "default") -> str:
    """Build a serialised identity key ``"<project_uid>:<path>"``."""
    return f"{project_uid}:{path}"
