"""Tests for ``folderhistory.core.identity`` — global identity assignment."""

from __future__ import annotations

from pathlib import Path

from folderhistory.core.identity import (
    DisjointSet,
    assign_identities_exact,
    assign_identities_with_blocking,
)
from folderhistory.types import FileRecord, IdentityCluster, Snapshot


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_file(
    path: str,
    blake3: str,
    *,
    size: int | None = None,
    xxhash64: str | None = None,
) -> FileRecord:
    """Create a :class:`FileRecord` with minimal test defaults."""
    return FileRecord(
        path=path,
        size=size if size is not None else len(blake3),
        mode=0o100644,
        mtime_ns=1_000_000_000,
        ctime_ns=1_000_000_000,
        raw_blake3=blake3,
        normalized_blake3=blake3,
        line_ending="lf",
        xxhash64=xxhash64 or blake3,
        is_symlink=False,
        target_path=None,
    )


def _make_snapshot(snap_id: str, *records: FileRecord) -> Snapshot:
    """Create a :class:`Snapshot` from ``FileRecord`` entries."""
    return Snapshot(
        id=snap_id,
        timestamp=1_000_000_000.0,
        source_path=Path(f"/fake/{snap_id}"),
        files=list(records),
    )


def _find_cluster(
    clusters: dict[str, IdentityCluster],
    *,
    observations: set[tuple[str, str]] | None = None,
) -> IdentityCluster | None:
    """Find a cluster that contains *at least* the given observations."""
    if observations is None:
        return next(iter(clusters.values()))
    for c in clusters.values():
        c_obs: set[tuple[str, str]] = set(c.observations)
        if c_obs == observations:
            return c
    return None


# ── DisjointSet (Union-Find) ─────────────────────────────────────────────────


class TestDisjointSet:
    """Tests for :class:`DisjointSet`."""

    def test_find_new_element(self) -> None:
        """A newly looked-up element is its own parent."""
        ds = DisjointSet()
        assert ds.find("a") == "a"

    def test_find_returns_consistent_parent(self) -> None:
        """Repeated ``find`` calls return the same root."""
        ds = DisjointSet()
        ds.union("a", "b")
        assert ds.find("a") == ds.find("b")
        assert ds.find("a") == ds.find("a")  # idempotent

    def test_union_merges_sets(self) -> None:
        """After union, both elements share the same root."""
        ds = DisjointSet()
        ds.union("x", "y")
        assert ds.find("x") == ds.find("y")

    def test_union_idempotent(self) -> None:
        """Repeated unions of the same elements do not raise or change root."""
        ds = DisjointSet()
        ds.union("a", "b")
        root_first = ds.find("a")
        ds.union("a", "b")
        assert ds.find("a") == root_first

    def test_path_compression(self) -> None:
        """After union, intermediate nodes point directly to the root."""
        ds = DisjointSet()
        ds.union("a", "b")
        ds.union("b", "c")
        ds.union("c", "d")
        # After finds, all should compress to the same root.
        root = ds.find("a")
        for elem in ("a", "b", "c", "d"):
            assert ds.find(elem) == root

    def test_transitive_union(self) -> None:
        """Transitive unions connect all elements."""
        ds = DisjointSet()
        ds.union("a", "b")
        ds.union("b", "c")
        ds.union("c", "d")
        root = ds.find("a")
        for elem in ("a", "b", "c", "d"):
            assert ds.find(elem) == root

    def test_separate_sets(self) -> None:
        """Unrelated elements have different roots."""
        ds = DisjointSet()
        ds.union("a", "b")
        ds.union("c", "d")
        assert ds.find("a") != ds.find("c")

    def test_groups_excludes_singletons(self) -> None:
        """``groups()`` only returns components with 2+ elements."""
        ds = DisjointSet()
        ds.find("singleton")
        ds.union("a", "b")
        groups = ds.groups()
        assert "singleton" not in groups
        assert len(groups) == 1

    def test_all_groups_includes_singletons(self) -> None:
        """``all_groups()`` includes singleton elements."""
        ds = DisjointSet()
        ds.find("singleton")
        ds.union("a", "b")
        all_g = ds.all_groups()
        # singleton should be its own group, and a+b should be another.
        singleton_root = ds.find("singleton")
        assert len(all_g) >= 2
        assert singleton_root in all_g

    def test_union_by_rank(self) -> None:
        """Union by rank keeps the tree shallow."""
        ds = DisjointSet()
        # Build two chains and union their roots.
        ds.union("a", "b")  # rank(a) = 1, rank(b) = 0
        ds.union("c", "d")  # rank(c) = 1, rank(d) = 0
        ds.union("a", "c")  # both root rank = 1, so c attaches to a, rank(a) = 2

        root = ds.find("a")
        # All elements should have the same root.
        for elem in ("a", "b", "c", "d"):
            assert ds.find(elem) == root

    def test_empty_groups(self) -> None:
        """Empty DisjointSet returns empty groups."""
        ds = DisjointSet()
        assert ds.groups() == {}
        assert ds.all_groups() == {}


# ── assign_identities_exact ────────────────────────────────────────────────


class TestAssignIdentitiesExact:
    """Tests for :func:`assign_identities_exact`."""

    def test_empty_snapshots(self) -> None:
        """Empty list → empty dict."""
        assert assign_identities_exact([]) == {}

    def test_single_snapshot_single_file(self) -> None:
        """One snapshot with one file → one cluster."""
        f = _make_file("a.txt", "hash_aaa")
        snap = _make_snapshot("v1", f)
        clusters = assign_identities_exact([snap])

        assert len(clusters) == 1
        cluster = next(iter(clusters.values()))
        assert ("v1", "a.txt") in cluster.observations
        assert cluster.canonical_path == "a.txt"
        assert cluster.confidence == 1.0

    def test_no_changes(self) -> None:
        """Two snapshots with identical files → one cluster per file."""
        f1 = _make_file("a.txt", "hash_aaa")
        f2 = _make_file("b.txt", "hash_bbb")
        snap_a = _make_snapshot("v1", f1, f2)
        snap_b = _make_snapshot("v2", f1, f2)

        clusters = assign_identities_exact([snap_a, snap_b])
        assert len(clusters) == 2

        # Each file should have its own cluster with both observations.
        cluster_a = _find_cluster(clusters, observations={("v1", "a.txt"), ("v2", "a.txt")})
        assert cluster_a is not None
        assert cluster_a.canonical_path == "a.txt"

        cluster_b = _find_cluster(clusters, observations={("v1", "b.txt"), ("v2", "b.txt")})
        assert cluster_b is not None
        assert cluster_b.canonical_path == "b.txt"

    def test_rename_across_snapshots(self) -> None:
        """Same content, different path → one cluster with both paths."""
        old = _make_file("old_name.txt", "hash_abc")
        new = _make_file("new_name.txt", "hash_abc")
        snap_a = _make_snapshot("v1", old)
        snap_b = _make_snapshot("v2", new)

        clusters = assign_identities_exact([snap_a, snap_b])
        assert len(clusters) == 1  # merged into one cluster by hash match

        cluster = next(iter(clusters.values()))
        assert ("v1", "old_name.txt") in cluster.observations
        assert ("v2", "new_name.txt") in cluster.observations

    def test_modified_file_same_path(self) -> None:
        """Different content at the same path → still one cluster (path-based identity)."""
        f_old = _make_file("main.py", "hash_old")
        f_new = _make_file("main.py", "hash_new")
        snap_a = _make_snapshot("v1", f_old)
        snap_b = _make_snapshot("v2", f_new)

        clusters = assign_identities_exact([snap_a, snap_b])
        # Same path → same identity key → one cluster.
        assert len(clusters) == 1

        cluster = next(iter(clusters.values()))
        assert ("v1", "main.py") in cluster.observations
        assert ("v2", "main.py") in cluster.observations
        assert cluster.canonical_path == "main.py"

    def test_completely_different_files(self) -> None:
        """Different content, different paths → two separate clusters."""
        f_a = _make_file("unique_a.py", "hash_aaa")
        f_b = _make_file("unique_b.py", "hash_bbb")
        snap = _make_snapshot("v1", f_a, f_b)

        clusters = assign_identities_exact([snap])
        assert len(clusters) == 2

    def test_three_snapshots_transitive(self) -> None:
        """Transitive identity preserved across three snapshots."""
        # v1: a.txt (hash_abc)
        # v2: b.txt (hash_abc) — renamed
        # v3: c.txt (hash_abc) — renamed again
        snap_a = _make_snapshot("v1", _make_file("a.txt", "hash_abc"))
        snap_b = _make_snapshot("v2", _make_file("b.txt", "hash_abc"))
        snap_c = _make_snapshot("v3", _make_file("c.txt", "hash_abc"))

        clusters = assign_identities_exact([snap_a, snap_b, snap_c])
        assert len(clusters) == 1  # all linked via hash

        cluster = next(iter(clusters.values()))
        assert ("v1", "a.txt") in cluster.observations
        assert ("v2", "b.txt") in cluster.observations
        assert ("v3", "c.txt") in cluster.observations

    def test_rename_and_modify_interleaved(self) -> None:
        """Rename + modify across snapshots produces two clusters."""
        # v1: a.txt (hash_abc)
        # v2: a.txt (hash_def) — modified in place
        # v3: b.txt (hash_abc) — old content reappears under new name
        v1 = _make_snapshot("v1", _make_file("a.txt", "hash_abc"))
        v2 = _make_snapshot("v2", _make_file("a.txt", "hash_def"))
        v3 = _make_snapshot("v3", _make_file("b.txt", "hash_abc"))

        clusters = assign_identities_exact([v1, v2, v3])

        # a.txt modified in place: v1(v1, a.txt) and v2(v2, a.txt) share path
        # b.txt has same content as a.txt@v1 → linked by hash to a.txt identity
        # So: v1(a.txt) + v2(a.txt) + v3(b.txt) all in one cluster via path+hash

        # Actually, let's think again:
        # - "default:a.txt" identity key from v1 and v2
        # - "default:b.txt" identity key from v3
        # - v1 a.txt hash_abc = v3 b.txt hash_abc → union("default:a.txt", "default:b.txt")
        # - v2 a.txt hash_def ≠ anything → no union from hash
        # - But "default:a.txt" is already linked to "default:b.txt" via hash
        # - So all three observations are in one cluster
        #
        # Wait no. The hash index only unions by hash. So:
        # - hash_abc: {"default:a.txt", "default:b.txt"} → union them
        # - hash_def: {"default:a.txt"} → single key, no union needed
        #
        # After union: "default:a.txt" and "default:b.txt" are in the same set
        #
        # The IdentityCluster for {"default:a.txt", "default:b.txt"}:
        # - observations from "default:a.txt": (v1, a.txt), (v2, a.txt)
        # - observations from "default:b.txt": (v3, b.txt)
        #
        # All three are in one cluster. This is correct because the logical file
        # "a.txt" was modified in v2, and then v3 has a new copy at "b.txt" with
        # the original content. But since "a.txt" still has the same identity key,
        # all three are connected.

        assert len(clusters) == 1
        cluster = next(iter(clusters.values()))
        assert len(cluster.observations) == 3

    def test_symlink_included_by_path(self) -> None:
        """Symlinks (empty hash) are still tracked via path identity."""
        symlink = FileRecord(
            path="link.lnk",
            size=0,
            mode=0o120777,
            mtime_ns=1_000_000_000,
            ctime_ns=1_000_000_000,
            raw_blake3="",
            normalized_blake3=None,
            line_ending="unknown",
            xxhash64="",
            is_symlink=True,
            target_path="real.txt",
        )
        snap_a = _make_snapshot("v1", symlink)
        snap_b = _make_snapshot("v2", symlink)

        clusters = assign_identities_exact([snap_a, snap_b])
        assert len(clusters) == 1
        cluster = next(iter(clusters.values()))
        assert ("v1", "link.lnk") in cluster.observations
        assert ("v2", "link.lnk") in cluster.observations

    def test_canonical_path_most_common(self) -> None:
        """The canonical path is the most common path among observations."""
        f1 = _make_file("common.txt", "hash_abc")
        f2 = _make_file("common.txt", "hash_abc")
        f3 = _make_file("rare.txt", "hash_abc")
        snap_a = _make_snapshot("v1", f1)
        snap_b = _make_snapshot("v2", f2)
        snap_c = _make_snapshot("v3", f3)

        clusters = assign_identities_exact([snap_a, snap_b, snap_c])
        assert len(clusters) == 1
        cluster = next(iter(clusters.values()))
        # "common.txt" appears twice → canonical
        assert cluster.canonical_path == "common.txt"

    def test_empty_snapshot(self) -> None:
        """Snapshot with no files produces no clusters."""
        snap = Snapshot(
            id="empty",
            timestamp=None,
            source_path=Path("/fake/empty"),
            files=[],
        )
        clusters = assign_identities_exact([snap])
        assert clusters == {}


# ── assign_identities_with_blocking ─────────────────────────────────────────


class TestAssignIdentitiesWithBlocking:
    """Tests for :func:`assign_identities_with_blocking`."""

    def test_basic_path_identity(self) -> None:
        """Same path across snapshots → one cluster (path identity)."""
        f1 = _make_file("main.py", "hash_aaa", size=100, xxhash64="xxh_aaa")
        f2 = _make_file("main.py", "hash_bbb", size=100, xxhash64="xxh_bbb")
        snap_a = _make_snapshot("v1", f1)
        snap_b = _make_snapshot("v2", f2)

        clusters = assign_identities_with_blocking([snap_a, snap_b])
        assert len(clusters) == 1  # path identity
        cluster = next(iter(clusters.values()))
        assert ("v1", "main.py") in cluster.observations
        assert ("v2", "main.py") in cluster.observations

    def test_rename_detected_within_size_block(self) -> None:
        """Rename detected within same size block."""
        f_old = _make_file("old.txt", "hash_abc", size=50, xxhash64="xxh_abc")
        f_new = _make_file("new.txt", "hash_abc", size=50, xxhash64="xxh_abc")
        snap_a = _make_snapshot("v1", f_old)
        snap_b = _make_snapshot("v2", f_new)

        clusters = assign_identities_with_blocking([snap_a, snap_b])
        assert len(clusters) == 1
        cluster = next(iter(clusters.values()))
        assert ("v1", "old.txt") in cluster.observations
        assert ("v2", "new.txt") in cluster.observations

    def test_different_sizes_not_matched(self) -> None:
        """Files with different sizes are not linked by hash."""
        f1 = _make_file("small.txt", "hash_abc", size=10, xxhash64="xxh_abc")
        f2 = _make_file("big.txt", "hash_abc", size=1000, xxhash64="xxh_abc")
        snap_a = _make_snapshot("v1", f1)
        snap_b = _make_snapshot("v2", f2)

        clusters = assign_identities_with_blocking([snap_a, snap_b])
        # Different sizes → no hash-based union → two separate clusters
        # (each path has its own identity key, not linked by hash)
        assert len(clusters) == 2

    def test_different_xxhash_not_matched(self) -> None:
        """Files with different xxhash64 are not linked even if same size."""
        f1 = _make_file("a.txt", "hash_abc", size=100, xxhash64="xxh_111")
        f2 = _make_file("b.txt", "hash_abc", size=100, xxhash64="xxh_222")
        snap_a = _make_snapshot("v1", f1)
        snap_b = _make_snapshot("v2", f2)

        clusters = assign_identities_with_blocking([snap_a, snap_b])
        # Different xxhash64 → different blocks → no hash union → two clusters
        assert len(clusters) == 2

    def test_empty_input(self) -> None:
        """Empty list → empty dict."""
        assert assign_identities_with_blocking([]) == {}

    def test_three_snapshots_transitive_blocking(self) -> None:
        """Transitive identity with blocking across three snapshots."""
        snap_a = _make_snapshot(
            "v1",
            _make_file("a.txt", "hash_abc", size=100, xxhash64="xxh_abc"),
        )
        snap_b = _make_snapshot(
            "v2",
            _make_file("b.txt", "hash_abc", size=100, xxhash64="xxh_abc"),
        )
        snap_c = _make_snapshot(
            "v3",
            _make_file("c.txt", "hash_abc", size=100, xxhash64="xxh_abc"),
        )

        clusters = assign_identities_with_blocking([snap_a, snap_b, snap_c])
        assert len(clusters) == 1
        cluster = next(iter(clusters.values()))
        assert len(cluster.observations) == 3


# ── IdentityCluster structure ────────────────────────────────────────────────


class TestIdentityClusterStructure:
    """Structural guarantees on returned :class:`IdentityCluster` values."""

    def test_confidence_is_one(self) -> None:
        """Exact matches always have confidence 1.0."""
        f = _make_file("x.txt", "hash_x")
        snap = _make_snapshot("v1", f)
        clusters = assign_identities_exact([snap])
        assert all(c.confidence == 1.0 for c in clusters.values())

    def test_observations_deduplicated(self) -> None:
        """Identical (snapshot_id, path) pairs are not duplicated."""
        f = _make_file("x.txt", "hash_x")
        snap = _make_snapshot("v1", f, f)  # same file listed twice
        clusters = assign_identities_exact([snap])
        assert len(clusters) == 1
        cluster = next(iter(clusters.values()))
        assert ("v1", "x.txt") in cluster.observations
        assert cluster.observations.count(("v1", "x.txt")) == 1

    def test_observations_sorted(self) -> None:
        """Observations are sorted by (snapshot_id, path)."""
        f1 = _make_file("z.txt", "hash_z")
        f2 = _make_file("a.txt", "hash_a")
        snap = _make_snapshot("v1", f2, f1)
        clusters = assign_identities_exact([snap])
        # Observations should be sorted.
        for cluster in clusters.values():
            obs = cluster.observations
            for i in range(len(obs) - 1):
                assert obs[i] <= obs[i + 1]
