"""Tests for ``folderhistory.feedback`` — the iterative feedback loop analyzer."""

from __future__ import annotations

from pathlib import Path

import pytest

from folderhistory.core.consistency import ConflictRecord, ConsistencyReport
from folderhistory.feedback.analyzer import FeedbackAnalyzer
from folderhistory.knowledge.json_kb import JSONKnowledgeBase
from folderhistory.knowledge.types import ContradictionRecord, FeedbackReport
from folderhistory.types import IdentityCluster


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_kb(path: Path) -> JSONKnowledgeBase:
    kb = JSONKnowledgeBase(path)
    kb.open()
    return kb


def _make_cluster(
    uid: str,
    observations: list[tuple[str, str]],
    confidence: float = 1.0,
) -> IdentityCluster:
    return IdentityCluster(
        uid=uid,
        observations=observations,
        canonical_path=observations[0][1] if observations else None,
        confidence=confidence,
    )


# ── FeedbackAnalyzer construction ─────────────────────────────────────────────


class TestFeedbackAnalyzerConstruction:
    """Tests for :class:`FeedbackAnalyzer` construction."""

    def test_default_construction(self, tmp_path: Path) -> None:
        """Default construction succeeds with algorithm_version=1.0.0."""
        kb = _make_kb(tmp_path / "kb.json")
        analyzer = FeedbackAnalyzer(kb=kb)
        assert analyzer.algorithm_version == "1.0.0"
        assert analyzer.kb is kb

    def test_custom_algorithm_version(self, tmp_path: Path) -> None:
        """Custom algorithm_version is accepted."""
        kb = _make_kb(tmp_path / "kb.json")
        analyzer = FeedbackAnalyzer(kb=kb, algorithm_version="2.1.0")
        assert analyzer.algorithm_version == "2.1.0"


# ── analyze (identity-only path) ──────────────────────────────────────────────


class TestAnalyze:
    """Tests for :meth:`FeedbackAnalyzer.analyze`."""

    def test_empty_inputs(self, tmp_path: Path) -> None:
        """Empty identity_clusters yields a FeedbackReport with all zeros."""
        kb = _make_kb(tmp_path / "kb.json")
        analyzer = FeedbackAnalyzer(kb=kb)

        report = analyzer.analyze(identity_clusters={})

        assert isinstance(report, FeedbackReport)
        assert report.new_identities == 0
        assert report.resolved_contradictions == 0
        assert report.iterations == 0
        assert report.quality_metrics["igs"] == pytest.approx(1.0)
        assert report.warnings == []

    def test_empty_inputs_with_pass_consistency(self, tmp_path: Path) -> None:
        """Empty identity_clusters with a passing consistency report."""
        kb = _make_kb(tmp_path / "kb.json")
        analyzer = FeedbackAnalyzer(kb=kb)

        report = analyzer.analyze(
            identity_clusters={},
            consistency_report=ConsistencyReport(passes=True),
        )

        assert report.new_identities == 0
        assert report.resolved_contradictions == 0
        assert report.iterations == 0

    def test_single_cluster_stored(self, tmp_path: Path) -> None:
        """Single high-confidence cluster → identity stored in KB."""
        kb = _make_kb(tmp_path / "kb.json")
        # Pre-seed frequency so that 0.1 * freq >= 0.95
        composite_key = "v1:a.txt"
        for _ in range(10):
            _ = kb.increment_frequency(composite_key)

        cluster = _make_cluster("c1", [("v1", "a.txt")], confidence=1.0)
        analyzer = FeedbackAnalyzer(kb=kb)

        report = analyzer.analyze(identity_clusters={"c1": cluster})

        assert report.new_identities == 1
        assert kb.get_identity(composite_key) == "c1"

    def test_below_threshold_not_stored(self, tmp_path: Path) -> None:
        """Cluster below confidence threshold → NOT stored in KB."""
        kb = _make_kb(tmp_path / "kb.json")
        # No frequency → confidence = 0 < 0.95
        cluster = _make_cluster("c-low", [("v1", "low.txt")], confidence=0.3)
        analyzer = FeedbackAnalyzer(kb=kb)

        report = analyzer.analyze(identity_clusters={"c-low": cluster})

        assert report.new_identities == 0
        assert kb.get_identity("v1:low.txt") is None

    def test_multiple_clusters_mixed_confidence(self, tmp_path: Path) -> None:
        """Mixed-confidence clusters — only high-confidence ones stored."""
        kb = _make_kb(tmp_path / "kb.json")

        high_key = "v1:high.txt"
        for _ in range(10):
            _ = kb.increment_frequency(high_key)

        clusters = {
            "c-high": _make_cluster("c-high", [("v1", "high.txt")], confidence=1.0),
            "c-low": _make_cluster("c-low", [("v1", "low.txt")], confidence=0.2),
        }
        analyzer = FeedbackAnalyzer(kb=kb)

        report = analyzer.analyze(identity_clusters=clusters)

        assert report.new_identities == 1
        assert kb.get_identity("v1:high.txt") == "c-high"
        assert kb.get_identity("v1:low.txt") is None

    def test_consistency_report_iterations_propagated(self, tmp_path: Path) -> None:
        """iterations_used from consistency report appears in FeedbackReport."""
        kb = _make_kb(tmp_path / "kb.json")
        analyzer = FeedbackAnalyzer(kb=kb)

        report = analyzer.analyze(
            identity_clusters={},
            consistency_report=ConsistencyReport(
                passes=True,
                iterations_used=3,
            ),
        )

        assert report.iterations == 3


# ── analyze_with_project (full pipeline) ─────────────────────────────────────


class TestAnalyzeWithProject:
    """Tests for :meth:`FeedbackAnalyzer.analyze_with_project`."""

    def test_empty_inputs(self, tmp_path: Path) -> None:
        """Empty inputs produce a valid FeedbackReport."""
        kb = _make_kb(tmp_path / "kb.json")
        analyzer = FeedbackAnalyzer(kb=kb)

        report = analyzer.analyze_with_project(
            identity_clusters={},
            project_fingerprints={},
            all_hashes=[],
        )

        assert isinstance(report, FeedbackReport)
        assert report.new_identities == 0
        assert report.resolved_contradictions == 0
        assert report.quality_metrics["igs"] == pytest.approx(1.0)

    def test_hash_frequencies_updated(self, tmp_path: Path) -> None:
        """All content hashes get their observation count bumped."""
        kb = _make_kb(tmp_path / "kb.json")
        analyzer = FeedbackAnalyzer(kb=kb)

        _ = analyzer.analyze_with_project(
            identity_clusters={},
            project_fingerprints={},
            all_hashes=["hash_a", "hash_b", "hash_a"],
        )

        assert kb.get_frequency("hash_a") == 2
        assert kb.get_frequency("hash_b") == 1
        assert kb.get_frequency("nonexistent") == 0

    def test_identities_stored_with_frequencies(self, tmp_path: Path) -> None:
        """High-frequency identities stored even in project path."""
        kb = _make_kb(tmp_path / "kb.json")
        composite_key = "v1:main.py"
        for _ in range(10):
            _ = kb.increment_frequency(composite_key)

        cluster = _make_cluster("c-main", [("v1", "main.py")], confidence=1.0)
        analyzer = FeedbackAnalyzer(kb=kb)

        report = analyzer.analyze_with_project(
            identity_clusters={"c-main": cluster},
            project_fingerprints={},
            all_hashes=["main_hash"],
        )

        assert report.new_identities == 1
        assert kb.get_identity(composite_key) == "c-main"
        # all_hashes frequencies also bumped
        assert kb.get_frequency("main_hash") == 1

    def test_projects_updated(self, tmp_path: Path) -> None:
        """New project fingerprints are stored."""
        kb = _make_kb(tmp_path / "kb.json")
        analyzer = FeedbackAnalyzer(kb=kb)

        fp = {"hash_x": 0.8, "hash_y": 0.4}
        report = analyzer.analyze_with_project(
            identity_clusters={},
            project_fingerprints={"project-alpha": fp},
            all_hashes=[],
        )

        assert report.quality_metrics["projects_updated"] == pytest.approx(1.0)

    def test_projects_with_jaccard_filter(self, tmp_path: Path) -> None:
        """Dissimilar project fingerprints are rejected by Jaccard threshold."""
        kb = _make_kb(tmp_path / "kb.json")
        analyzer = FeedbackAnalyzer(kb=kb)

        # Pre-seed a project with fingerprint A
        kb.set_project("project-beta", {"hash_a": 0.8, "hash_b": 0.4})

        # Dissimilar fingerprint — Jaccard = 0.0 < 0.7 → rejected
        report = analyzer.analyze_with_project(
            identity_clusters={},
            project_fingerprints={"project-beta": {"hash_c": 0.9}},
            all_hashes=[],
        )

        # Should be rejected; quality_metrics shows 0 new
        assert report.quality_metrics["projects_updated"] == pytest.approx(0.0)

    def test_full_pipeline_no_error(self, tmp_path: Path) -> None:
        """Full pipeline with real data runs without error."""
        kb = _make_kb(tmp_path / "kb.json")

        # Seed frequencies for identity storage
        key_a = "v1:file_a.py"
        key_b = "v2:file_b.py"
        for _ in range(10):
            _ = kb.increment_frequency(key_a)
        for _ in range(10):
            _ = kb.increment_frequency(key_b)

        clusters = {
            "cluster-a": _make_cluster("cluster-a", [("v1", "file_a.py")]),
            "cluster-b": _make_cluster("cluster-b", [("v2", "file_b.py")]),
        }
        fingerprints = {"proj-1": {"h1": 0.7, "h2": 0.3}}
        all_hashes = ["h1", "h2", "h3"]

        analyzer = FeedbackAnalyzer(kb=kb)
        report = analyzer.analyze_with_project(
            identity_clusters=clusters,
            project_fingerprints=fingerprints,
            all_hashes=all_hashes,
        )

        assert report.new_identities == 2
        assert kb.get_identity(key_a) == "cluster-a"
        assert kb.get_frequency("h1") == 1
        assert report.quality_metrics["projects_updated"] == pytest.approx(1.0)
        assert report.quality_metrics["igs"] == pytest.approx(1.0)


# ── Contradiction resolution ──────────────────────────────────────────────────


class TestAnalyzeContradictions:
    """Tests for contradiction resolution during feedback analysis."""

    def test_no_consistency_report_returns_zero(self, tmp_path: Path) -> None:
        """No consistency report → zero contradictions resolved."""
        kb = _make_kb(tmp_path / "kb.json")
        analyzer = FeedbackAnalyzer(kb=kb)

        report = analyzer.analyze(identity_clusters={})
        assert report.resolved_contradictions == 0

    def test_passing_consistency_no_resolution(self, tmp_path: Path) -> None:
        """Passing consistency report → no resolution needed."""
        kb = _make_kb(tmp_path / "kb.json")
        analyzer = FeedbackAnalyzer(kb=kb)

        report = analyzer.analyze(
            identity_clusters={},
            consistency_report=ConsistencyReport(passes=True),
        )
        assert report.resolved_contradictions == 0

    def test_conflict_without_hash_skipped(self, tmp_path: Path) -> None:
        """ConflictRecord with empty hash is skipped."""
        kb = _make_kb(tmp_path / "kb.json")
        analyzer = FeedbackAnalyzer(kb=kb)

        conflict = ConflictRecord(hash="")  # empty hash
        report = ConsistencyReport(passes=False, conflicts=[conflict])

        feedback = analyzer.analyze(
            identity_clusters={},
            consistency_report=report,
        )
        assert feedback.resolved_contradictions == 0
        assert len(kb.get_contradictions()) == 0

    def test_conflict_resolved_via_path_mapping(self, tmp_path: Path) -> None:
        """Conflict with hash and matching paths → contradiction recorded."""
        kb = _make_kb(tmp_path / "kb.json")
        clusters = {
            "cluster-a": _make_cluster("cluster-a", [("v1", "path_a.txt")]),
            "cluster-b": _make_cluster("cluster-b", [("v1", "path_b.txt")]),
        }

        conflict = ConflictRecord(
            hash="content_hash_123",
            source_paths=["path_a.txt"],
            target_paths=["path_b.txt"],
            conflict_type="identity_mismatch",
            confidence_a=0.8,
            confidence_b=0.4,
        )
        report = ConsistencyReport(passes=False, conflicts=[conflict])

        analyzer = FeedbackAnalyzer(kb=kb)
        with pytest.warns(UserWarning, match="IGS dropped"):
            feedback = analyzer.analyze(
                identity_clusters=clusters,
                consistency_report=report,
            )

        assert feedback.resolved_contradictions == 1

        # Check the contradiction was stored in KB
        contradictions = kb.get_contradictions()
        assert len(contradictions) == 1
        recorded = contradictions[0]
        assert isinstance(recorded, ContradictionRecord)
        assert recorded.hash == "content_hash_123"
        assert recorded.identity_a in ("cluster-a", "cluster-b")
        assert recorded.identity_b in ("cluster-a", "cluster-b")

    def test_conflict_resolved_via_kb_identity(self, tmp_path: Path) -> None:
        """When path mapping yields no clusters, KB identity is used."""
        kb = _make_kb(tmp_path / "kb.json")
        # Set a KB identity for the hash
        hash_key = "obs_hash"
        for _ in range(10):
            _ = kb.increment_frequency(hash_key)
        # Store identity via update_identities
        from folderhistory.knowledge.update import update_identities

        _ = update_identities(kb, {hash_key: "kb-cluster"})

        conflict = ConflictRecord(
            hash=hash_key,
            source_paths=["unknown_path.txt"],
            target_paths=[],
            conflict_type="identity_mismatch",
            confidence_a=0.5,
            confidence_b=0.5,
        )
        report = ConsistencyReport(passes=False, conflicts=[conflict])

        analyzer = FeedbackAnalyzer(kb=kb)
        clusters = {
            "other-cluster": _make_cluster("other-cluster", [("v1", "other.txt")]),
        }
        with pytest.warns(UserWarning, match="IGS dropped"):
            feedback = analyzer.analyze(
                identity_clusters=clusters,
                consistency_report=report,
            )

        assert feedback.resolved_contradictions == 1
        contradictions = kb.get_contradictions()
        assert len(contradictions) == 1
        recorded = contradictions[0]
        assert recorded.identity_a == "kb-cluster"

    def test_multiple_conflicts_resolved(self, tmp_path: Path) -> None:
        """Multiple conflicts all get individually resolved."""
        kb = _make_kb(tmp_path / "kb.json")
        clusters = {
            "c-a": _make_cluster("c-a", [("v1", "a.txt")]),
            "c-b": _make_cluster("c-b", [("v1", "b.txt")]),
            "c-c": _make_cluster("c-c", [("v1", "c.txt")]),
        }

        conflicts = [
            ConflictRecord(
                hash="h1",
                source_paths=["a.txt"],
                target_paths=["b.txt"],
                conflict_type="identity_mismatch",
            ),
            ConflictRecord(
                hash="h2",
                source_paths=["b.txt"],
                target_paths=["c.txt"],
                conflict_type="path_mismatch",
            ),
            ConflictRecord(
                hash="h3",
                source_paths=["c.txt"],
                target_paths=["a.txt"],
                conflict_type="temporal_overlap",
            ),
        ]
        report = ConsistencyReport(passes=False, conflicts=conflicts)

        analyzer = FeedbackAnalyzer(kb=kb)
        with pytest.warns(UserWarning, match="IGS dropped"):
            feedback = analyzer.analyze(identity_clusters=clusters, consistency_report=report)

        assert feedback.resolved_contradictions == 3
        assert len(kb.get_contradictions()) == 3

    def test_conflict_in_analyze_with_project(self, tmp_path: Path) -> None:
        """Contradictions are also resolved in analyze_with_project."""
        kb = _make_kb(tmp_path / "kb.json")
        clusters = {
            "c1": _make_cluster("c1", [("v1", "src/a.py")]),
            "c2": _make_cluster("c2", [("v1", "src/b.py")]),
        }

        conflict = ConflictRecord(
            hash="conflict_hash",
            source_paths=["src/a.py"],
            target_paths=["src/b.py"],
            conflict_type="identity_mismatch",
        )
        report = ConsistencyReport(passes=False, conflicts=[conflict])

        analyzer = FeedbackAnalyzer(kb=kb)
        with pytest.warns(UserWarning, match="IGS dropped"):
            feedback = analyzer.analyze_with_project(
                identity_clusters=clusters,
                project_fingerprints={},
                all_hashes=["h1"],
                consistency_report=report,
            )

        assert feedback.resolved_contradictions == 1
        assert len(kb.get_contradictions()) == 1


# ── IGS computation ───────────────────────────────────────────────────────────


class TestAnalyzeIgs:
    """Tests for IGS computation during feedback analysis."""

    def test_igs_stable_kb(self, tmp_path: Path) -> None:
        """Stable KB (no contradictions) → IGS = 1.0."""
        kb = _make_kb(tmp_path / "kb.json")
        analyzer = FeedbackAnalyzer(kb=kb)

        report = analyzer.analyze(identity_clusters={})

        assert report.quality_metrics["igs"] == pytest.approx(1.0)

    def test_igs_drops_with_contradictions(self, tmp_path: Path) -> None:
        """With 1 contradiction across 2 identities → IGS = 0.5."""
        kb = _make_kb(tmp_path / "kb.json")
        clusters = {
            "cluster-x": _make_cluster("cluster-x", [("v1", "x.txt")]),
            "cluster-y": _make_cluster("cluster-y", [("v1", "y.txt")]),
        }

        conflict = ConflictRecord(
            hash="hash_xy",
            source_paths=["x.txt"],
            target_paths=["y.txt"],
            conflict_type="identity_mismatch",
        )
        report = ConsistencyReport(passes=False, conflicts=[conflict])

        analyzer = FeedbackAnalyzer(kb=kb)
        with pytest.warns(UserWarning, match="IGS dropped"):
            feedback = analyzer.analyze(
                identity_clusters=clusters,
                consistency_report=report,
            )

        assert feedback.quality_metrics["igs"] == pytest.approx(0.5)

    def test_igs_with_multiple_contradictions(self, tmp_path: Path) -> None:
        """Multiple contradictions → IGS reflects ratio."""
        kb = _make_kb(tmp_path / "kb.json")
        clusters = {
            "c-1": _make_cluster("c-1", [("v1", "f1.txt")]),
            "c-2": _make_cluster("c-2", [("v1", "f2.txt")]),
            "c-3": _make_cluster("c-3", [("v1", "f3.txt")]),
        }

        conflicts = [
            ConflictRecord(hash="h1", source_paths=["f1.txt"], target_paths=["f2.txt"]),
            ConflictRecord(hash="h2", source_paths=["f2.txt"], target_paths=["f3.txt"]),
        ]
        report = ConsistencyReport(passes=False, conflicts=conflicts)

        analyzer = FeedbackAnalyzer(kb=kb)
        with pytest.warns(UserWarning, match="IGS dropped"):
            feedback = analyzer.analyze(
                identity_clusters=clusters,
                consistency_report=report,
            )

        # 2 contradictions / 3 identities ≈ 0.333 -> IGS ≈ 0.667
        expected_igs = 1.0 - 2.0 / 3.0
        assert feedback.quality_metrics["igs"] == pytest.approx(expected_igs)

    def test_igs_warning_in_warnings_list(self, tmp_path: Path) -> None:
        """Low IGS includes a warning in the report."""
        kb = _make_kb(tmp_path / "kb.json")
        clusters = {
            "a": _make_cluster("a", [("v1", "a.txt")]),
            "b": _make_cluster("b", [("v1", "b.txt")]),
        }

        conflict = ConflictRecord(hash="h", source_paths=["a.txt"], target_paths=["b.txt"])
        report = ConsistencyReport(passes=False, conflicts=[conflict])

        analyzer = FeedbackAnalyzer(kb=kb)
        with pytest.warns(UserWarning):
            feedback = analyzer.analyze(
                identity_clusters=clusters,
                consistency_report=report,
            )

        assert any("IGS dropped" in w for w in feedback.warnings)


# ── Integration ────────────────────────────────────────────────────────────────


class TestIntegration:
    """End-to-end integration tests for the feedback analyzer."""

    def test_full_feedback_cycle(self, tmp_path: Path) -> None:
        """Complete feedback cycle with all stages.

        Runs through: frequency bump → identity store → project store →
        contradiction resolution → IGS.
        """
        path = tmp_path / "kb.json"
        kb = _make_kb(path)

        # Pre-seed frequencies for identity storage
        key_main = "v1:main.py"
        key_util = "v1:util.py"
        for _ in range(10):
            _ = kb.increment_frequency(key_main)
        for _ in range(10):
            _ = kb.increment_frequency(key_util)

        clusters = {
            "cluster-main": _make_cluster("cluster-main", [("v1", "main.py")]),
            "cluster-util": _make_cluster("cluster-util", [("v1", "util.py")]),
        }
        fingerprints = {"project-app": {"main.py": 0.8, "util.py": 0.3}}
        all_hashes = ["h1", "h2"]

        conflict = ConflictRecord(
            hash="h1",
            source_paths=["main.py"],
            target_paths=["util.py"],
            conflict_type="identity_mismatch",
        )
        consistency = ConsistencyReport(passes=False, conflicts=[conflict])

        analyzer = FeedbackAnalyzer(kb=kb)
        with pytest.warns(UserWarning, match="IGS dropped"):
            report = analyzer.analyze_with_project(
                identity_clusters=clusters,
                project_fingerprints=fingerprints,
                all_hashes=all_hashes,
                consistency_report=consistency,
            )

        # Frequency check
        assert kb.get_frequency("h1") == 1
        assert kb.get_frequency("h2") == 1

        # Identity check
        assert kb.get_identity(key_main) == "cluster-main"
        assert kb.get_identity(key_util) == "cluster-util"

        # Contradiction check
        assert len(kb.get_contradictions()) == 1

        # Report metrics
        assert report.new_identities == 2
        assert report.resolved_contradictions == 1
        assert report.quality_metrics["projects_updated"] == pytest.approx(1.0)
        assert report.quality_metrics["igs"] == pytest.approx(0.5)

        # Close and reopen → data persists
        kb.close()
        kb2 = _make_kb(path)
        assert kb2.get_identity(key_main) == "cluster-main"
        assert kb2.get_frequency("h1") == 1
        assert len(kb2.get_contradictions()) == 1
        kb2.close()

    def test_empty_run_then_full(self, tmp_path: Path) -> None:
        """An empty run followed by a full run accumulates knowledge correctly."""
        path = tmp_path / "kb.json"
        kb = _make_kb(path)
        analyzer = FeedbackAnalyzer(kb=kb)

        # First: empty run
        r1 = analyzer.analyze(identity_clusters={})
        assert r1.new_identities == 0
        assert r1.quality_metrics["igs"] == pytest.approx(1.0)

        # Second: run with data
        key = "v1:data.txt"
        for _ in range(10):
            _ = kb.increment_frequency(key)

        cluster = _make_cluster("c-data", [("v1", "data.txt")])
        r2 = analyzer.analyze(identity_clusters={"c-data": cluster})
        assert r2.new_identities == 1
        assert kb.get_identity(key) == "c-data"
        assert r2.quality_metrics["igs"] == pytest.approx(1.0)

        kb.close()

    def test_multiple_analyze_calls_accumulate(self, tmp_path: Path) -> None:
        """Multiple analyze calls build up identities incrementally."""
        path = tmp_path / "kb.json"
        kb = _make_kb(path)
        analyzer = FeedbackAnalyzer(kb=kb)

        # First call: cluster-a
        key_a = "v1:a.txt"
        for _ in range(10):
            _ = kb.increment_frequency(key_a)
        r1 = analyzer.analyze(
            identity_clusters={
                "c-a": _make_cluster("c-a", [("v1", "a.txt")]),
            },
        )
        assert r1.new_identities == 1

        # Second call: cluster-b
        key_b = "v1:b.txt"
        for _ in range(10):
            _ = kb.increment_frequency(key_b)
        r2 = analyzer.analyze(
            identity_clusters={
                "c-b": _make_cluster("c-b", [("v1", "b.txt")]),
            },
        )
        assert r2.new_identities == 1

        # Both identities present in KB
        assert kb.get_identity(key_a) == "c-a"
        assert kb.get_identity(key_b) == "c-b"

        # IGS should be stable (no contradictions)
        assert r2.quality_metrics["igs"] == pytest.approx(1.0)

        kb.close()
