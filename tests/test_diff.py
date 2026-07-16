"""Tests for ``folderhistory.core.diff`` — operation derivation engine."""

from __future__ import annotations

from pathlib import Path

from folderhistory.core.diff import derive_operations, derive_operations_between
from folderhistory.types import EditOperation, FileRecord, IdentityCluster, Snapshot


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_file(
    path: str,
    blake3: str,
    *,
    normalized_blake3: str | None = None,
) -> FileRecord:
    """Create a :class:`FileRecord` with minimal test defaults."""
    return FileRecord(
        path=path,
        size=len(blake3),
        mode=0o100644,
        mtime_ns=1_000_000_000,
        ctime_ns=1_000_000_000,
        raw_blake3=blake3,
        normalized_blake3=normalized_blake3,
        line_ending="lf",
        xxhash64=blake3,
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


def _make_cluster(
    uid: str,
    observations: list[tuple[str, str]],
    *,
    confidence: float = 1.0,
) -> IdentityCluster:
    """Create an :class:`IdentityCluster` with the given observations."""
    return IdentityCluster(
        uid=uid,
        observations=observations,
        canonical_path=observations[0][1] if observations else None,
        confidence=confidence,
    )


# ── derive_operations_between ───────────────────────────────────────────────


class TestDeriveOperationsBetween:
    """Tests for :func:`derive_operations_between` — single pair."""

    def test_empty_snapshots(self) -> None:
        """Empty snapshots with no identities produce an empty list."""
        snap_a = _make_snapshot("v1")
        snap_b = _make_snapshot("v2")
        ops = derive_operations_between(snap_a, snap_b, {})
        assert ops == []

    def test_no_changes(self) -> None:
        """Identical files in both snapshots produce no operations."""
        f = _make_file("a.txt", "hash_aaa")
        snap_a = _make_snapshot("v1", f)
        snap_b = _make_snapshot("v2", f)

        identities = {
            "default:a.txt": _make_cluster(
                "default:a.txt",
                [("v1", "a.txt"), ("v2", "a.txt")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert ops == []

    def test_add_file(self) -> None:
        """File present only in the second snapshot → one CREATE."""
        snap_a = _make_snapshot("v1")
        f = _make_file("new.txt", "hash_new")
        snap_b = _make_snapshot("v2", f)

        identities = {
            "default:new.txt": _make_cluster(
                "default:new.txt",
                [("v2", "new.txt")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert len(ops) == 1
        assert ops[0].op_type == "create"
        assert ops[0].file_id == "new.txt"
        assert ops[0].target_path == "new.txt"
        assert ops[0].new_hash == "hash_new"
        assert ops[0].source_path is None
        assert ops[0].old_hash is None
        assert ops[0].confidence == 1.0

    def test_delete_file(self) -> None:
        """File present only in the first snapshot → one DELETE."""
        f = _make_file("old.txt", "hash_old")
        snap_a = _make_snapshot("v1", f)
        snap_b = _make_snapshot("v2")

        identities = {
            "default:old.txt": _make_cluster(
                "default:old.txt",
                [("v1", "old.txt")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert len(ops) == 1
        assert ops[0].op_type == "delete"
        assert ops[0].file_id == "old.txt"
        assert ops[0].source_path == "old.txt"
        assert ops[0].old_hash == "hash_old"
        assert ops[0].target_path is None
        assert ops[0].new_hash is None
        assert ops[0].confidence == 1.0

    def test_modify_file(self) -> None:
        """Same path, different hash → one MODIFY."""
        f_old = _make_file("main.py", "hash_old")
        f_new = _make_file("main.py", "hash_new")
        snap_a = _make_snapshot("v1", f_old)
        snap_b = _make_snapshot("v2", f_new)

        identities = {
            "default:main.py": _make_cluster(
                "default:main.py",
                [("v1", "main.py"), ("v2", "main.py")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert len(ops) == 1
        assert ops[0].op_type == "modify"
        assert ops[0].file_id == "main.py"
        assert ops[0].source_path == "main.py"
        assert ops[0].target_path == "main.py"
        assert ops[0].old_hash == "hash_old"
        assert ops[0].new_hash == "hash_new"
        assert ops[0].confidence == 1.0

    def test_rename_file(self) -> None:
        """Different path, same hash → one RENAME (not delete+create)."""
        f_old = _make_file("old_name.txt", "hash_abc")
        f_new = _make_file("new_name.txt", "hash_abc")
        snap_a = _make_snapshot("v1", f_old)
        snap_b = _make_snapshot("v2", f_new)

        identities = {
            "default:old_name.txt": _make_cluster(
                "cluster:abc",
                [("v1", "old_name.txt"), ("v2", "new_name.txt")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert len(ops) == 1
        assert ops[0].op_type == "rename"
        assert ops[0].file_id == "old_name.txt"
        assert ops[0].source_path == "old_name.txt"
        assert ops[0].target_path == "new_name.txt"
        assert ops[0].old_hash == "hash_abc"
        assert ops[0].new_hash == "hash_abc"
        assert ops[0].confidence == 1.0

    def test_move_to_subdirectory(self) -> None:
        """Same basename, different directory → one MOVE."""
        f_old = _make_file("data.txt", "hash_xyz")
        f_new = _make_file("archive/data.txt", "hash_xyz")
        snap_a = _make_snapshot("v1", f_old)
        snap_b = _make_snapshot("v2", f_new)

        identities = {
            "cluster:move": _make_cluster(
                "cluster:move",
                [("v1", "data.txt"), ("v2", "archive/data.txt")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert len(ops) == 1
        assert ops[0].op_type == "move"
        assert ops[0].file_id == "data.txt"
        assert ops[0].source_path == "data.txt"
        assert ops[0].target_path == "archive/data.txt"
        assert ops[0].confidence == 1.0

    def test_move_from_root_to_subdir(self) -> None:
        """Move from root to subdirectory → one MOVE."""
        f_old = _make_file("readme.md", "hash_rd")
        f_new = _make_file("docs/readme.md", "hash_rd")
        snap_a = _make_snapshot("v1", f_old)
        snap_b = _make_snapshot("v2", f_new)

        identities = {
            "cluster:root2dir": _make_cluster(
                "cluster:root2dir",
                [("v1", "readme.md"), ("v2", "docs/readme.md")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert len(ops) == 1
        assert ops[0].op_type == "move"
        assert ops[0].source_path == "readme.md"
        assert ops[0].target_path == "docs/readme.md"

    def test_move_to_parent(self) -> None:
        """Move from subdirectory up to root → one MOVE."""
        f_old = _make_file("sub/notes.txt", "hash_nt")
        f_new = _make_file("notes.txt", "hash_nt")
        snap_a = _make_snapshot("v1", f_old)
        snap_b = _make_snapshot("v2", f_new)

        identities = {
            "cluster:dir2root": _make_cluster(
                "cluster:dir2root",
                [("v1", "sub/notes.txt"), ("v2", "notes.txt")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert len(ops) == 1
        assert ops[0].op_type == "move"
        assert ops[0].source_path == "sub/notes.txt"
        assert ops[0].target_path == "notes.txt"

    def test_no_identities(self) -> None:
        """Empty identities dict produces no operations even with files."""
        f = _make_file("orphan.txt", "hash_xxx")
        snap_a = _make_snapshot("v1", f)
        snap_b = _make_snapshot("v2", f)
        ops = derive_operations_between(snap_a, snap_b, {})
        assert ops == []

    def test_multiple_operations(self) -> None:
        """Multiple files with different changes in one pair."""
        f1_old = _make_file("keep.txt", "hash_same")
        f1_new = _make_file("keep.txt", "hash_same")

        f2_old = _make_file("mod.py", "hash_old")
        f2_new = _make_file("mod.py", "hash_new")

        f3_old = _make_file("gone.txt", "hash_del")
        f4_new = _make_file("added.txt", "hash_add")

        f5_old = _make_file("old.md", "hash_ren")
        f5_new = _make_file("new.md", "hash_ren")

        snap_a = _make_snapshot("v1", f1_old, f2_old, f3_old, f5_old)
        snap_b = _make_snapshot("v2", f1_new, f2_new, f4_new, f5_new)

        identities = {
            "default:keep.txt": _make_cluster(
                "default:keep.txt",
                [("v1", "keep.txt"), ("v2", "keep.txt")],
            ),
            "default:mod.py": _make_cluster(
                "default:mod.py",
                [("v1", "mod.py"), ("v2", "mod.py")],
            ),
            "default:gone.txt": _make_cluster(
                "default:gone.txt",
                [("v1", "gone.txt")],
            ),
            "default:added.txt": _make_cluster(
                "default:added.txt",
                [("v2", "added.txt")],
            ),
            "cluster:ren": _make_cluster(
                "cluster:ren",
                [("v1", "old.md"), ("v2", "new.md")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        # Expected: modify (mod.py), delete (gone.txt), create (added.txt), rename (old.md→new.md)
        assert len(ops) == 4

        op_map: dict[str, EditOperation] = {}
        for op in ops:
            key = f"{op.op_type}:{op.file_id}"
            op_map[key] = op

        assert "modify:mod.py" in op_map
        assert "delete:gone.txt" in op_map
        assert "create:added.txt" in op_map
        assert "rename:old.md" in op_map

    def test_uses_cluster_observations(self) -> None:
        """Operations are derived from identity cluster observations, not direct
        file comparison."""
        # The identity cluster links path_a@v1 to path_b@v2.
        # The path mapping comes from the cluster, not from comparing files.
        f_at_v1 = _make_file("src_a.py", "hash_hello")
        f_at_v2 = _make_file("lib_b.py", "hash_hello")
        snap_a = _make_snapshot("v1", f_at_v1)
        snap_b = _make_snapshot("v2", f_at_v2)

        identities = {
            "cluster:cross": _make_cluster(
                "cluster:cross",
                [("v1", "src_a.py"), ("v2", "lib_b.py")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert len(ops) == 1
        assert ops[0].op_type == "rename"
        assert ops[0].source_path == "src_a.py"
        assert ops[0].target_path == "lib_b.py"

    def test_same_path_same_hash_no_op(self) -> None:
        """Same path with same hash across snapshots → no operation."""
        f = _make_file("stable.txt", "hash_stable")
        snap_a = _make_snapshot("v1", f)
        snap_b = _make_snapshot("v2", f)

        identities = {
            "default:stable.txt": _make_cluster(
                "default:stable.txt",
                [("v1", "stable.txt"), ("v2", "stable.txt")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert ops == []

    def test_diff_path_diff_hash_delete_create(self) -> None:
        """Different path with different hash → DELETE + CREATE."""
        f_old = _make_file("a_old.py", "hash_aaa")
        f_new = _make_file("b_new.py", "hash_bbb")
        snap_a = _make_snapshot("v1", f_old)
        snap_b = _make_snapshot("v2", f_new)

        identities = {
            "cluster:unrelated": _make_cluster(
                "cluster:unrelated",
                [("v1", "a_old.py"), ("v2", "b_new.py")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert len(ops) == 2

        delete_ops = [op for op in ops if op.op_type == "delete"]
        create_ops = [op for op in ops if op.op_type == "create"]
        assert len(delete_ops) == 1
        assert len(create_ops) == 1

        assert delete_ops[0].source_path == "a_old.py"
        assert delete_ops[0].file_id == "a_old.py"
        assert delete_ops[0].old_hash == "hash_aaa"

        assert create_ops[0].target_path == "b_new.py"
        assert create_ops[0].file_id == "b_new.py"
        assert create_ops[0].new_hash == "hash_bbb"

    def test_diff_path_normalized_hash_match(self) -> None:
        """Different path, different raw hash but same normalized hash
        → RENAME with confidence 0.8 (line-ending-only change)."""
        f_old = _make_file("main.py", "raw_crcr", normalized_blake3="norm_abc")
        f_new = _make_file("renamed.py", "raw_lf", normalized_blake3="norm_abc")
        snap_a = _make_snapshot("v1", f_old)
        snap_b = _make_snapshot("v2", f_new)

        identities = {
            "cluster:nrm": _make_cluster(
                "cluster:nrm",
                [("v1", "main.py"), ("v2", "renamed.py")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert len(ops) == 1
        assert ops[0].op_type == "rename"
        assert ops[0].old_hash == "raw_crcr"
        assert ops[0].new_hash == "raw_lf"
        assert ops[0].confidence == 0.8
        assert ops[0].source_path == "main.py"
        assert ops[0].target_path == "renamed.py"


# ── derive_operations (multi-pair) ──────────────────────────────────────────


class TestDeriveOperations:
    """Tests for :func:`derive_operations` — multi-pair."""

    def test_fewer_than_two_snapshots(self) -> None:
        """Zero or one snapshot returns empty list."""
        assert derive_operations([], {}) == []
        snap = _make_snapshot("v1")
        assert derive_operations([snap], {}) == []

    def test_two_snapshots_no_changes(self) -> None:
        """Two identical snapshots → one empty inner list."""
        f = _make_file("a.txt", "hash_aaa")
        snap_a = _make_snapshot("v1", f)
        snap_b = _make_snapshot("v2", f)

        identities = {
            "default:a.txt": _make_cluster(
                "default:a.txt",
                [("v1", "a.txt"), ("v2", "a.txt")],
            ),
        }

        ops_list = derive_operations([snap_a, snap_b], identities)
        assert len(ops_list) == 1
        assert ops_list[0] == []

    def test_three_snapshots(self) -> None:
        """Three snapshots: one change in each pair."""
        # v1: a.txt (hash_abc)
        # v2: a.txt (hash_def)  → modify
        # v3: b.txt (hash_def)  → rename
        f1 = _make_file("a.txt", "hash_abc")
        f2 = _make_file("a.txt", "hash_def")
        f3 = _make_file("b.txt", "hash_def")

        snap_a = _make_snapshot("v1", f1)
        snap_b = _make_snapshot("v2", f2)
        snap_c = _make_snapshot("v3", f3)

        # A single identity cluster spans all three snapshots (as assign_identities_exact would produce).
        identities = {
            "cluster:abc": _make_cluster(
                "cluster:abc",
                [("v1", "a.txt"), ("v2", "a.txt"), ("v3", "b.txt")],
            ),
        }

        ops_list = derive_operations([snap_a, snap_b, snap_c], identities)
        assert len(ops_list) == 2

        # Pair 0: v1 → v2
        assert len(ops_list[0]) == 1
        assert ops_list[0][0].op_type == "modify"
        assert ops_list[0][0].file_id == "a.txt"

        # Pair 1: v2 → v3
        assert len(ops_list[1]) == 1
        assert ops_list[1][0].op_type == "rename"
        assert ops_list[1][0].source_path == "a.txt"
        assert ops_list[1][0].target_path == "b.txt"

    def test_cluster_observations_across_pairs(self) -> None:
        """A cluster spanning multiple pairs contributes differently per pair."""
        f = _make_file("log.txt", "hash_log")
        snap_a = _make_snapshot("v1", f)
        snap_b = _make_snapshot("v2", f)
        snap_c = _make_snapshot("v3")

        identities = {
            "default:log.txt": _make_cluster(
                "default:log.txt",
                [("v1", "log.txt"), ("v2", "log.txt")],
            ),
        }

        ops_list = derive_operations([snap_a, snap_b, snap_c], identities)
        assert len(ops_list) == 2

        # Pair 0: v1 → v2, log.txt unchanged → no op
        assert ops_list[0] == []

        # Pair 1: v2 → v3, log.txt is in v2 but not in v3 → delete
        assert len(ops_list[1]) == 1
        assert ops_list[1][0].op_type == "delete"
        assert ops_list[1][0].source_path == "log.txt"

    def test_no_identities_multi_pair(self) -> None:
        """No identities → empty operations for all pairs."""
        f = _make_file("x.txt", "hash_x")
        snap_a = _make_snapshot("v1", f)
        snap_b = _make_snapshot("v2", f)
        snap_c = _make_snapshot("v3", f)

        ops_list = derive_operations([snap_a, snap_b, snap_c], {})
        assert len(ops_list) == 2
        assert ops_list[0] == []
        assert ops_list[1] == []

    def test_add_then_delete_chain(self) -> None:
        """Create in one pair, then delete in the next."""
        snap_a = _make_snapshot("v1")
        f = _make_file("temp.txt", "hash_temp")
        snap_b = _make_snapshot("v2", f)
        snap_c = _make_snapshot("v3")

        identities = {
            "default:temp.txt": _make_cluster(
                "default:temp.txt",
                [("v2", "temp.txt")],
            ),
        }

        ops_list = derive_operations([snap_a, snap_b, snap_c], identities)
        assert len(ops_list) == 2

        assert len(ops_list[0]) == 1
        assert ops_list[0][0].op_type == "create"
        assert ops_list[0][0].target_path == "temp.txt"

        # Pair 1: v2→v3: file in v2, not in v3 → delete
        assert len(ops_list[1]) == 1
        assert ops_list[1][0].op_type == "delete"
        assert ops_list[1][0].source_path == "temp.txt"


# ── Copy operations ─────────────────────────────────────────────────────────


class TestCopyOperations:
    """Tests for copy operation detection in :func:`derive_operations_between`."""

    def test_copy_file(self) -> None:
        """Identity has 1 obs in S_i, 2 obs in S_{i+1} (one shared + one new
        with same hash) → 1 copy op.
        """
        f_orig = _make_file("original.txt", "hash_abc")
        f_copy = _make_file("copy.txt", "hash_abc")
        snap_a = _make_snapshot("v1", f_orig)
        snap_b = _make_snapshot("v2", f_orig, f_copy)

        identities = {
            "cluster:copy": _make_cluster(
                "cluster:copy",
                [("v1", "original.txt"), ("v2", "original.txt"), ("v2", "copy.txt")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert len(ops) == 1
        assert ops[0].op_type == "copy"
        assert ops[0].file_id == "original.txt"
        assert ops[0].source_path == "original.txt"
        assert ops[0].target_path == "copy.txt"
        assert ops[0].old_hash == "hash_abc"
        assert ops[0].new_hash == "hash_abc"
        assert ops[0].confidence == 1.0

    def test_copy_multiple_files(self) -> None:
        """Several copies in one transition."""
        f_orig = _make_file("source.txt", "hash_xyz")
        f_copy1 = _make_file("backup/copy1.txt", "hash_xyz")
        f_copy2 = _make_file("backup/copy2.txt", "hash_xyz")
        snap_a = _make_snapshot("v1", f_orig)
        snap_b = _make_snapshot("v2", f_orig, f_copy1, f_copy2)

        identities = {
            "cluster:multi": _make_cluster(
                "cluster:multi",
                [
                    ("v1", "source.txt"),
                    ("v2", "source.txt"),
                    ("v2", "backup/copy1.txt"),
                    ("v2", "backup/copy2.txt"),
                ],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert len(ops) == 2
        copy_ops = sorted(ops, key=lambda o: o.target_path or "")
        assert copy_ops[0].op_type == "copy"
        assert copy_ops[0].source_path == "source.txt"
        assert copy_ops[0].target_path == "backup/copy1.txt"
        assert copy_ops[1].op_type == "copy"
        assert copy_ops[1].source_path == "source.txt"
        assert copy_ops[1].target_path == "backup/copy2.txt"

    def test_copy_does_not_create(self) -> None:
        """New path with different hash → create op, not copy."""
        f_orig = _make_file("f.txt", "hash_aaa")
        f_new = _make_file("new.txt", "hash_bbb")
        snap_a = _make_snapshot("v1", f_orig)
        snap_b = _make_snapshot("v2", f_orig, f_new)

        identities = {
            "cluster:new": _make_cluster(
                "cluster:new",
                [("v1", "f.txt"), ("v2", "f.txt"), ("v2", "new.txt")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert len(ops) == 1
        assert ops[0].op_type == "create"
        assert ops[0].file_id == "new.txt"
        assert ops[0].target_path == "new.txt"
        assert ops[0].new_hash == "hash_bbb"
        assert ops[0].source_path is None

    def test_copy_renames_unaffected(self) -> None:
        """Rename still works alongside copies."""
        f_old = _make_file("old.py", "hash_ren")
        f_new = _make_file("new.py", "hash_ren")
        f_orig = _make_file("data.txt", "hash_cp")
        f_copy = _make_file("data_backup.txt", "hash_cp")
        snap_a = _make_snapshot("v1", f_old, f_orig)
        snap_b = _make_snapshot("v2", f_new, f_orig, f_copy)

        identities = {
            "cluster:rename": _make_cluster(
                "cluster:rename",
                [("v1", "old.py"), ("v2", "new.py")],
            ),
            "cluster:copy": _make_cluster(
                "cluster:copy",
                [("v1", "data.txt"), ("v2", "data.txt"), ("v2", "data_backup.txt")],
            ),
        }

        ops = derive_operations_between(snap_a, snap_b, identities)
        assert len(ops) == 2

        op_map: dict[str, EditOperation] = {}
        for op in ops:
            key = f"{op.op_type}:{op.file_id}"
            op_map[key] = op

        assert "rename:old.py" in op_map
        rename_op = op_map["rename:old.py"]
        assert rename_op.source_path == "old.py"
        assert rename_op.target_path == "new.py"

        assert "copy:data.txt" in op_map
        copy_op = op_map["copy:data.txt"]
        assert copy_op.source_path == "data.txt"
        assert copy_op.target_path == "data_backup.txt"
