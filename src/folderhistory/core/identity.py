"""Global identity assignment engine for FolderHistory.

Uses content hash + relative-path matching with Union-Find to assign
global identities to files across snapshots.  A *global identity* is a
:class:`~folderhistory.types.IdentityCluster` that groups every observation
(``(snapshot_id, file_path)``) believed to represent the same logical file.
"""

from __future__ import annotations

from collections import Counter

from folderhistory.types import IdentityCluster, Snapshot

# ── Union-Find (Disjoint Set) ─────────────────────────────────────────────────


class DisjointSet:
    """Union-Find data structure with path compression and union by rank.

    Elements are added implicitly on first access via :meth:`find`.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        """Return the root representative of *x*, applying path compression."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            return x
        # Path compression: make x point directly to its root.
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        """Merge the sets containing *x* and *y*."""
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        # Union by rank: attach the shorter tree under the taller one.
        if self._rank[rx] < self._rank[ry]:
            self._parent[rx] = ry
        elif self._rank[rx] > self._rank[ry]:
            self._parent[ry] = rx
        else:
            self._parent[ry] = rx
            self._rank[rx] += 1

    def groups(self) -> dict[str, set[str]]:
        """Return ``{root: {members}}`` for every non-trivial component.

        Only includes sets with **more than one** element.  Singleton
        elements are still implicitly part of the structure but are *not*
        returned here.  Call :meth:`all_groups` to include singletons.
        """
        result: dict[str, set[str]] = {}
        for x in self._parent:
            root = self.find(x)
            members = result.setdefault(root, set())
            members.add(x)
        # Keep only non-trivial groups.
        return {r: m for r, m in result.items() if len(m) > 1}

    def all_groups(self) -> dict[str, set[str]]:
        """Return ``{root: {members}}`` for **every** component, including
        singleton elements that were never unioned with another element."""
        result: dict[str, set[str]] = {}
        for x in self._parent:
            root = self.find(x)
            result.setdefault(root, set()).add(x)
        return result


# ── Global identity assignment ────────────────────────────────────────────────

_IDENTITY_PROJECT_UID: str = "default"
"""Project UID used in single-root (Phase 1) mode."""


def _identity_key(path: str) -> str:
    """Build a serialised identity key ``"default:<path>"``."""
    return f"{_IDENTITY_PROJECT_UID}:{path}"


def assign_identities_exact(
    snapshots: list[Snapshot],
) -> dict[str, IdentityCluster]:
    """Assign global identities using exact content-hash matching.

    Algorithm
    ---------
    1.  For every file across all snapshots, build an identity key
        ``"default:<relative_path>"``.
    2.  Build a hash index: ``raw_blake3`` → set of identity keys.
    3.  For every hash that maps to **multiple** identity keys (indicating
        identical content at different paths — i.e. a rename), union those
        keys in the :class:`DisjointSet`.
    4.  Extract connected components from the Union-Find structure.
    5.  Create an :class:`~folderhistory.types.IdentityCluster` for every
        component.

    Because the identity key **includes the path**, files at the same path
    in different snapshots automatically belong to the same cluster (they are
    the same Union-Find element).  Hash-based union adds **cross-path** links
    that handle renames.

    Parameters
    ----------
    snapshots:
        Ordered (or unordered) list of snapshots to assign identities to.

    Returns
    -------
    ``{cluster_uid: IdentityCluster}`` where *cluster_uid* is the Union-Find
    root of the cluster's identity keys.
    """
    if not snapshots:
        return {}

    ds = DisjointSet()

    # Per-identity-key observation list:  ikey → [(snap_id, path), ...]
    obs: dict[str, list[tuple[str, str]]] = {}
    # Hash index:  raw_blake3 → set of identity keys
    hash_to_keys: dict[str, set[str]] = {}

    for snap in snapshots:
        for f in snap.files:
            ikey = _identity_key(f.path)
            obs.setdefault(ikey, []).append((snap.id, f.path))

            if f.raw_blake3:
                hash_to_keys.setdefault(f.raw_blake3, set()).add(ikey)

    # Union identity keys that share the same content hash (rename detection).
    for keys in hash_to_keys.values():
        if len(keys) >= 2:
            keys_sorted = sorted(keys)
            first = keys_sorted[0]
            for ik in keys_sorted[1:]:
                ds.union(first, ik)

    # Build clusters from all identity keys that were ever touched.
    all_ikeys: set[str] = set()
    for snap in snapshots:
        for f in snap.files:
            all_ikeys.add(_identity_key(f.path))

    # Ensure every key is registered in the DS (even singletons).
    for ikey in all_ikeys:
        _ = ds.find(ikey)

    root_to_ikeys: dict[str, set[str]] = ds.all_groups()

    clusters: dict[str, IdentityCluster] = {}
    for root, ikeys in root_to_ikeys.items():
        if not ikeys:
            continue

        # Collect all observations from all identity keys in this component.
        all_obs: list[tuple[str, str]] = []
        for ik in ikeys:
            all_obs.extend(obs.get(ik, []))

        if not all_obs:
            continue

        # Deduplicate observations (same snap_id + path).
        seen: set[tuple[str, str]] = set()
        deduped: list[tuple[str, str]] = []
        for ob in all_obs:
            if ob not in seen:
                seen.add(ob)
                deduped.append(ob)

        # Sort by snapshot id for stable output.
        deduped.sort(key=lambda x: (x[0], x[1]))

        # Canonical path = most common path among observations.
        path_counts = Counter(p for _, p in deduped)
        canonical: str | None = path_counts.most_common(1)[0][0] if deduped else None

        clusters[root] = IdentityCluster(
            uid=root,
            observations=deduped,
            canonical_path=canonical,
            confidence=1.0,
        )

    return clusters


def assign_identities_with_blocking(
    snapshots: list[Snapshot],
) -> dict[str, IdentityCluster]:
    """Size-blocked variant of :func:`assign_identities_exact`.

    Only files with **identical size** are compared for content-hash matching.
    Files of different sizes cannot have identical content, so this variant
    is more efficient for large snapshot sets at the cost of being slightly
    more complex.

    The blocking cascade is:

    1.  Group all files by **size**.
    2.  Within each size group, group by **xxhash64** (fast pre-filter).
    3.  Within each xxhash64 group, compare **raw_blake3**.
    4.  Union identity keys sharing the same raw_blake3.

    Path-based identity (same path across snapshots) is handled **before**
    blocking and always applies regardless of size.

    Parameters
    ----------
    snapshots:
        List of snapshots to assign identities to.

    Returns
    -------
    ``{cluster_uid: IdentityCluster}`` — same structure as
    :func:`assign_identities_exact`.
    """
    if not snapshots:
        return {}

    ds = DisjointSet()

    # Per-identity-key observation list.
    obs: dict[str, list[tuple[str, str]]] = {}
    # Group files by (size, xxhash64) blocks.
    blocks: dict[tuple[int, str], dict[str, set[str]]] = {}
    # All identity keys ever seen.
    all_ikeys: set[str] = set()

    for snap in snapshots:
        for f in snap.files:
            ikey = _identity_key(f.path)
            all_ikeys.add(ikey)
            obs.setdefault(ikey, []).append((snap.id, f.path))

            if f.raw_blake3 and f.xxhash64:
                block_key = (f.size, f.xxhash64)
                blocks.setdefault(block_key, {}).setdefault(
                    f.raw_blake3, set(),
                ).add(ikey)

    # Union identity keys sharing the same raw_blake3 within each block.
    for blake3_map in blocks.values():
        for keys in blake3_map.values():
            if len(keys) >= 2:
                keys_sorted = sorted(keys)
                first = keys_sorted[0]
                for ik in keys_sorted[1:]:
                    ds.union(first, ik)

    # Register all keys in DS (singletons included).
    for ikey in all_ikeys:
        _ = ds.find(ikey)

    root_to_ikeys = ds.all_groups()

    clusters: dict[str, IdentityCluster] = {}
    for root, ikeys in root_to_ikeys.items():
        if not ikeys:
            continue

        all_obs: list[tuple[str, str]] = []
        for ik in ikeys:
            all_obs.extend(obs.get(ik, []))

        if not all_obs:
            continue

        seen: set[tuple[str, str]] = set()
        deduped: list[tuple[str, str]] = []
        for ob in all_obs:
            if ob not in seen:
                seen.add(ob)
                deduped.append(ob)

        deduped.sort(key=lambda x: (x[0], x[1]))

        path_counts = Counter(p for _, p in deduped)
        canonical = path_counts.most_common(1)[0][0] if deduped else None

        clusters[root] = IdentityCluster(
            uid=root,
            observations=deduped,
            canonical_path=canonical,
            confidence=1.0,
        )

    return clusters
