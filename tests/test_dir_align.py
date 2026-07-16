"""Tests for :mod:`folderhistory.core.dir_align`."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from folderhistory.core.dir_align import (
    build_inverted_index,
    content_probe_subtree,
    idf_weighted_jaccard,
    normalize_paths_to_root,
    resolve_project_identities,
)
from folderhistory.types import FileRecord, Snapshot


# ── Test helpers ────────────────────────────────────────────────────────────────


def _file(
    path: str,
    raw_b3: str = "a" * 64,
    norm_b3: str | None = "a" * 64,
) -> FileRecord:
    """Create a minimal ``FileRecord`` with controllable hashes."""
    return FileRecord(
        path=path,
        size=100,
        mode=0o644,
        mtime_ns=1000,
        ctime_ns=1000,
        raw_blake3=raw_b3,
        normalized_blake3=norm_b3,
        line_ending="lf",
        xxhash64="xx64",
        is_symlink=False,
    )


def _snap(
    sid: str,
    files: list[FileRecord] | None = None,
) -> Snapshot:
    """Create a minimal ``Snapshot`` with a synthetic source path."""
    return Snapshot(
        id=sid,
        timestamp=1000.0,
        source_path=Path(f"/tmp/{sid}"),
        files=files or [],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  build_inverted_index
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildInvertedIndex:
    def test_basic(self) -> None:
        """Single snapshot with one file maps hash → [(sid, path)]."""
        s = _snap("s1", [_file("a.txt", raw_b3="h1", norm_b3="h1")])
        idx = build_inverted_index([s])
        assert idx == {"h1": [("s1", "a.txt")]}

    def test_empty_snapshots(self) -> None:
        """Empty list produces empty dict."""
        assert build_inverted_index([]) == {}

    def test_multi_snapshot(self) -> None:
        """Hash shared across snapshots accumulates all (sid, path) pairs."""
        s1 = _snap("s1", [_file("a.txt", raw_b3="h1", norm_b3="h1")])
        s2 = _snap("s2", [_file("b.txt", raw_b3="h1", norm_b3="h1")])
        idx = build_inverted_index([s1, s2])
        assert idx == {"h1": [("s1", "a.txt"), ("s2", "b.txt")]}

    def test_normalized_blake3_fallback(self) -> None:
        """When ``normalized_blake3`` is ``None``, ``raw_blake3`` is used."""
        s = _snap(
            "s1",
            [
                _file("bin.dat", raw_b3="r1", norm_b3=None),  # binary file
                _file("text.py", raw_b3="r2", norm_b3="n2"),  # normalised
            ],
        )
        idx = build_inverted_index([s])
        # bin.dat uses raw (r1), text.py uses normalised (n2)
        assert idx["r1"] == [("s1", "bin.dat")]
        assert "r2" not in idx
        assert idx["n2"] == [("s1", "text.py")]

    def test_warns_on_no_hash(self, caplog: pytest.LogCaptureFixture) -> None:
        """Both hashes empty/None → warning logged, no index entry."""
        snap = _snap("s1", [_file("symlink", raw_b3="", norm_b3=None)])
        idx = build_inverted_index([snap])
        assert idx == {}
        assert any(
            "no usable hash" in record.message
            and "symlink" in record.message
            and "s1" in record.message
            for record in caplog.records
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  idf_weighted_jaccard
# ═══════════════════════════════════════════════════════════════════════════════


class TestIdfWeightedJaccard:
    def test_identical_sets(self) -> None:
        """Two identical hash sets → 1.0."""
        hashes = {"h1", "h2", "h3"}
        freq = {"h1": 1, "h2": 2, "h3": 3}
        # N=9 → sqrt(9)=3 → hashes with freq<3 are specific
        # h1(freq=1) and h2(freq=2) are specific, h3(freq=3) is common
        # specific(A)={h1,h2}, specific(B)={h1,h2} → J=2/2=1.0
        assert idf_weighted_jaccard(hashes, hashes, freq, 9) == 1.0

    def test_disjoint_sets(self) -> None:
        """No overlap → 0.0."""
        a = {"h1"}
        b = {"h2"}
        freq = {"h1": 1, "h2": 1}
        assert idf_weighted_jaccard(a, b, freq, 9) == 0.0

    def test_partial_overlap(self) -> None:
        """Partial overlap produces intermediate value."""
        a = {"h1", "h2"}
        b = {"h2", "h3"}
        freq = {"h1": 1, "h2": 1, "h3": 1}
        # N=9 → sqrt=3 → all freq<3, all specific
        # intersection={h2}, union={h1,h2,h3} → J=1/3≈0.333
        j = idf_weighted_jaccard(a, b, freq, 9)
        assert j == pytest.approx(1 / 3)

    def test_both_empty(self) -> None:
        """Both sets empty → 0.0 (union is empty)."""
        assert idf_weighted_jaccard(set(), set(), {}, 9) == 0.0

    def test_one_set_empty(self) -> None:
        """One set empty → 0.0."""
        a = {"h1"}
        freq = {"h1": 1}
        assert idf_weighted_jaccard(a, set(), freq, 9) == 0.0

    def test_common_hash_discounting(self) -> None:
        """High-frequency hashes are excluded from both sides."""
        # N=9, sqrt(9)=3.  h3 has freq=9 ≥ 3 → common, excluded.
        a = {"h1", "h2", "h3"}
        b = {"h1", "h2"}
        freq = {"h1": 1, "h2": 2, "h3": 9}
        # specific(A)={h1,h2}, specific(B)={h1,h2}
        # intersection={h1,h2}, union={h1,h2} → J=2/2=1.0
        assert idf_weighted_jaccard(a, b, freq, 9) == 1.0

    def test_all_hashes_common(self) -> None:
        """When every hash is common, specific sets are empty → 0.0."""
        a = {"h1", "h2"}
        b = {"h1", "h2"}
        freq = {"h1": 9, "h2": 9}
        # N=9, sqrt=3 → all freq≥3 → common
        assert idf_weighted_jaccard(a, b, freq, 9) == 0.0

    def test_threshold_boundary(self) -> None:
        """Hash at exactly sqrt(N) is NOT specific (excluded)."""
        a = {"h1"}
        b = {"h1"}
        # N=4, sqrt(4)=2.  freq=2 → not < 2 → common → excluded
        freq = {"h1": 2}
        j = idf_weighted_jaccard(a, b, freq, 4)
        assert j == 0.0

    def test_hash_missing_from_global_freq(self) -> None:
        """Hash not in global_freq defaults to freq=0 (< threshold)."""
        a = {"unknown"}
        b = {"unknown"}
        j = idf_weighted_jaccard(a, b, {}, 9)
        assert j == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  content_probe_subtree
# ═══════════════════════════════════════════════════════════════════════════════


class TestContentProbeSubtree:
    """Helper to reduce repetition."""

    def _run(
        self,
        large_files: list[FileRecord],
        ref_files: list[FileRecord],
        threshold: float = 0.5,
    ) -> tuple[str | None, float]:
        large = _snap("large", large_files)
        ref = _snap("ref", ref_files)
        # Build a minimal inverted index covering all hashes used
        # (not strictly required by the current impl but expected by the
        # signature).
        idx = build_inverted_index([large, ref])
        return content_probe_subtree(large, ref, idx, threshold=threshold)

    def test_match_found(self) -> None:
        """Reference lives under ``backup/project/`` in the large snapshot."""
        large = [
            _file("backup/notes.txt", raw_b3="x", norm_b3="x"),
            _file("backup/project/main.py", raw_b3="a", norm_b3="a"),
            _file("backup/project/utils.py", raw_b3="b", norm_b3="b"),
        ]
        ref = [
            _file("main.py", raw_b3="a", norm_b3="a"),
            _file("utils.py", raw_b3="b", norm_b3="b"),
        ]
        root, score = self._run(large, ref)
        assert root == "backup/project"
        assert score == 1.0

    def test_empty_reference(self) -> None:
        """Empty reference → ``(None, 0.0)``."""
        large = [_file("a.txt", raw_b3="a", norm_b3="a")]
        root, score = self._run(large, [])
        assert root is None
        assert score == 0.0

    def test_threshold_too_high(self) -> None:
        """When the best score is below threshold → ``(None, 0.0)``."""
        large = [
            _file("backup/project/main.py", raw_b3="a", norm_b3="a"),
            _file("backup/project/other.py", raw_b3="x", norm_b3="x"),
        ]
        ref = [
            _file("main.py", raw_b3="a", norm_b3="a"),
            _file("utils.py", raw_b3="b", norm_b3="b"),
        ]
        # backup/project: total=2, matched=1 → score=0.5
        # threshold=0.8 → none qualify
        root, score = self._run(large, ref, threshold=0.8)
        assert root is None
        assert score == 0.0

    def test_prefer_deepest(self) -> None:
        """Among equal-scoring directories, pick the deepest one."""
        large = [
            _file("a/b/c/x.txt", raw_b3="h1", norm_b3="h1"),
            _file("a/b/y.txt", raw_b3="h1", norm_b3="h1"),
            _file("a/z.txt", raw_b3="h1", norm_b3="h1"),
        ]
        ref = [_file("x.txt", raw_b3="h1", norm_b3="h1")]
        # All three directories score 1.0 (their files all match)
        # Deepest is a/b/c → 3 parts
        root, score = self._run(large, ref)
        assert root == "a/b/c"
        assert score == 1.0

    def test_no_ref_hash_overlap(self) -> None:
        """No files match → best score 0, still returns deepest above threshold
        (which there isn't one)."""
        large = [
            _file("stuff/data.bin", raw_b3="x", norm_b3="x"),
        ]
        ref = [
            _file("main.py", raw_b3="a", norm_b3="a"),
        ]
        root, score = self._run(large, ref)
        # stuff: total=1, matched=0 → score=0.0 < 0.5
        assert root is None
        assert score == 0.0

    def test_fallback_to_raw_blake3(self) -> None:
        """Large-snapshot file with ``normalized_blake3=None`` still matches
        via its ``raw_blake3``."""
        large = [
            _file("backup/project/binary.dat", raw_b3="r1", norm_b3=None),
        ]
        ref = [
            _file("binary.dat", raw_b3="r1", norm_b3=None),
        ]
        root, score = self._run(large, ref)
        assert root == "backup/project"
        assert score == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  normalize_paths_to_root
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizePathsToRoot:
    def test_basic_normalization(self) -> None:
        """Strip the root prefix and update ``source_path``."""
        s = _snap(
            "s1",
            [
                _file("backup/project/main.py", raw_b3="a", norm_b3="a"),
                _file("backup/project/utils.py", raw_b3="b", norm_b3="b"),
            ],
        )
        result = normalize_paths_to_root([s], {"s1": "backup/project"})
        assert len(result) == 1
        out = result[0]
        assert out.id == "s1"
        assert out.files[0].path == "main.py"
        assert out.files[1].path == "utils.py"
        assert out.source_path == Path("/tmp/s1/backup/project")

    def test_noop_for_unmatched(self) -> None:
        """Snapshot not in ``root_candidates`` → returned unchanged."""
        s = _snap(
            "s1",
            [_file("some/file.txt", raw_b3="a", norm_b3="a")],
        )
        result = normalize_paths_to_root([s], {})
        assert len(result) == 1
        assert result[0] is s  # same object

    def test_root_is_dot(self) -> None:
        """``.`` root means no normalization (same object)."""
        s = _snap(
            "s1",
            [_file("file.txt", raw_b3="a", norm_b3="a")],
        )
        result = normalize_paths_to_root([s], {"s1": "."})
        assert len(result) == 1
        assert result[0] is s

    def test_root_is_empty(self) -> None:
        """Empty-string root means no normalization (same object)."""
        s = _snap(
            "s1",
            [_file("file.txt", raw_b3="a", norm_b3="a")],
        )
        result = normalize_paths_to_root([s], {"s1": ""})
        assert len(result) == 1
        assert result[0] is s

    def test_files_outside_root_unchanged(self) -> None:
        """Files with path not starting with root prefix stay as-is."""
        s = _snap(
            "s1",
            [
                _file("project/main.py", raw_b3="a", norm_b3="a"),
                _file("other/notes.txt", raw_b3="b", norm_b3="b"),
            ],
        )
        result = normalize_paths_to_root([s], {"s1": "project"})
        out = result[0]
        assert out.files[0].path == "main.py"  # stripped
        assert out.files[1].path == "other/notes.txt"  # unchanged

    def test_immutability(self) -> None:
        """Original snapshot is not mutated."""
        orig = _snap(
            "s1",
            [_file("root/file.txt", raw_b3="a", norm_b3="a")],
        )
        _ = normalize_paths_to_root([orig], {"s1": "root"})
        assert orig.files[0].path == "root/file.txt"


# ═══════════════════════════════════════════════════════════════════════════════
#  resolve_project_identities
# ═══════════════════════════════════════════════════════════════════════════════


class TestResolveProjectIdentities:
    def _similar_pair(self) -> tuple[list[Snapshot], dict[str, list[tuple[str, str]]]]:
        """Two snapshots sharing enough specific hashes to exceed 0.5,
        plus 7 empty filler snapshots so that sqrt(N) > the shared frequency
        (preventing those hashes from being discounted as common).

        With N=9: sqrt(9)=3.  Shared hashes have freq=2 < 3 → not discounted.
        specific(s1) = {h1,h2,h3,h4} ∩ {freq<3} = {h1,h2,h3,h4}
        specific(s2) = {h1,h2,h3,h5}
        intersection = {h1,h2,h3}, union = {h1,h2,h3,h4,h5}
        Jaccard = 3/5 = 0.6 > 0.5
        """
        snapshots = [
            _snap(
                "s1",
                [
                    _file("a.py", raw_b3="h1", norm_b3="h1"),
                    _file("b.py", raw_b3="h2", norm_b3="h2"),
                    _file("c.py", raw_b3="h3", norm_b3="h3"),
                    _file("d.py", raw_b3="h4", norm_b3="h4"),
                ],
            ),
            _snap(
                "s2",
                [
                    _file("a.py", raw_b3="h1", norm_b3="h1"),
                    _file("b.py", raw_b3="h2", norm_b3="h2"),
                    _file("c.py", raw_b3="h3", norm_b3="h3"),
                    _file("e.py", raw_b3="h5", norm_b3="h5"),
                ],
            ),
        ]
        # 7 empty filler snapshots to reach N=9
        for i in range(3, 10):
            snapshots.append(_snap(f"s{i}"))

        idx = build_inverted_index(snapshots)
        return snapshots, idx

    def _separate_projects(self) -> tuple[list[Snapshot], dict[str, list[tuple[str, str]]]]:
        """Two snapshots with no shared hashes → separate groups."""
        snapshots = [
            _snap("s1", [_file("a.py", raw_b3="h_a", norm_b3="h_a")]),
            _snap("s2", [_file("b.py", raw_b3="h_b", norm_b3="h_b")]),
        ]
        idx = build_inverted_index(snapshots)
        return snapshots, idx

    def test_two_snapshots_same_project(self) -> None:
        """Two snapshots with IDF-weighted Jaccard > 0.5 → same group."""
        snapshots, idx = self._similar_pair()
        groups = resolve_project_identities(snapshots, idx)

        assert len(groups) >= 1
        # s1 and s2 must be together
        s_group = None
        for uid, members in groups.items():
            if "s1" in members or "s2" in members:
                s_group = members
                break
        assert s_group is not None
        assert "s1" in s_group
        assert "s2" in s_group

    def test_different_projects_stay_separate(self) -> None:
        """Snapshots with no common hashes → separate groups."""
        snapshots, idx = self._separate_projects()
        groups = resolve_project_identities(snapshots, idx)

        assert len(groups) == 2
        # Each snapshot in its own group
        for uid, members in groups.items():
            assert len(members) == 1

    def test_project_uid_is_first_snapshot(self) -> None:
        """The group key is the first snapshot ID in that group (by input
        order)."""
        snapshots, idx = self._separate_projects()
        groups = resolve_project_identities(snapshots, idx)
        # s1 is first in input order → its uid should be "s1"
        assert "s1" in groups
        assert groups["s1"] == ["s1"]

    def test_three_snapshots_transitive(self) -> None:
        """A≈B and B≈C → all three in one group (transitive closure)."""
        # With N=10, sqrt(10)≈3.16.
        # s1 hashes: h1,h2,h3,h4   (all freq=2 shared with s2)
        # s2 hashes: h1,h2,h3,h5   (all freq=2 shared with s1 and s3)
        # s3 hashes: h1,h2,h3,h6   (all freq=2 or 3 shared with s1,s2)
        # Fillers: 7 more snapshots to reach N=10
        snapshots = [
            _snap("s1", [_file(f"f{x}.py", raw_b3=x, norm_b3=x) for x in ("h1", "h2", "h3", "h4")]),
            _snap("s2", [_file(f"f{x}.py", raw_b3=x, norm_b3=x) for x in ("h1", "h2", "h3", "h5")]),
            _snap("s3", [_file(f"f{x}.py", raw_b3=x, norm_b3=x) for x in ("h1", "h2", "h3", "h6")]),
        ]
        for i in range(4, 11):
            snapshots.append(_snap(f"s{i}"))

        idx = build_inverted_index(snapshots)
        groups = resolve_project_identities(snapshots, idx)

        # With N=10, sqrt=3.16; h1,h2,h3 have freq=3 < 3.16 → specific
        # specific(s1)={h1,h2,h3,h4}, specific(s2)={h1,h2,h3,h5}
        # J(s1,s2)=3/5=0.6 > 0.5 → union
        # specific(s2)={h1,h2,h3,h5}, specific(s3)={h1,h2,h3,h6}
        # J(s2,s3)=3/5=0.6 > 0.5 → union
        # Transitive closure → all three together
        s_group = None
        for uid, members in groups.items():
            if "s1" in members or "s2" in members or "s3" in members:
                s_group = members
                break
        assert s_group is not None
        assert "s1" in s_group
        assert "s2" in s_group
        assert "s3" in s_group

    def test_empty_input(self) -> None:
        """Empty snapshot list → empty dict."""
        assert resolve_project_identities([], {}) == {}

    def test_single_snapshot(self) -> None:
        """Single snapshot → one group with just that snapshot."""
        s = _snap("solo", [_file("a.txt", raw_b3="h1", norm_b3="h1")])
        idx = build_inverted_index([s])
        groups = resolve_project_identities([s], idx)
        assert groups == {"solo": ["solo"]}
