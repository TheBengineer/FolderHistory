"""Tests for the KB update rules and contradiction resolution module."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from folderhistory.knowledge.update import (
    compute_igs,
    resolve_contradiction,
    update_frequencies,
    update_identities,
    update_projects,
)
from folderhistory.knowledge.types import ContradictionRecord
from folderhistory.knowledge.json_kb import JSONKnowledgeBase


# ── helpers ─────────────────────────────────────────────────────────────────


def _make_kb(path: Path) -> JSONKnowledgeBase:
    kb = JSONKnowledgeBase(path)
    kb.open()
    return kb


def _get_fingerprint(kb: JSONKnowledgeBase, uid: str) -> dict[str, float] | None:
    raw = kb.get_project(uid)
    if raw is not None and isinstance(raw, dict):
        raw_dict = cast("dict[str, object]", raw)
        fp_val = raw_dict.get("fingerprint")
        if isinstance(fp_val, dict):
            return cast("dict[str, float]", fp_val)
    return None


# We test _idf_jaccard indirectly via update_projects behavior, which is the
# intended public interface.  The function is private by design.


# ── update_identities ───────────────────────────────────────────────────────


class TestUpdateIdentities:
    def test_empty_dict_returns_zero(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        assert update_identities(kb, {}) == 0

    def test_stores_high_confidence_identities(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        for _ in range(10):
            _ = kb.increment_frequency("hash-a")
        for _ in range(10):
            _ = kb.increment_frequency("hash-b")
        assert update_identities(kb, {"hash-a": "c1", "hash-b": "c2"}) == 2
        assert kb.get_identity("hash-a") == "c1"
        assert kb.get_identity("hash-b") == "c2"

    def test_skips_low_confidence(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        assert update_identities(kb, {"hash-low": "c-x"}) == 0
        assert kb.get_identity("hash-low") is None

    def test_skips_below_threshold(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        for _ in range(5):
            _ = kb.increment_frequency("hash-mid")
        assert update_identities(kb, {"hash-mid": "c-y"}) == 0

    def test_mixed_confidence(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        for _ in range(10):
            _ = kb.increment_frequency("high")
        assert update_identities(kb, {"high": "c1", "low": "c2"}) == 1
        assert kb.get_identity("high") == "c1"
        assert kb.get_identity("low") is None

    def test_custom_threshold(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        for _ in range(3):
            _ = kb.increment_frequency("hash-c")
        assert update_identities(kb, {"hash-c": "c-z"}, confidence_threshold=0.25) == 1
        assert kb.get_identity("hash-c") == "c-z"

    def test_persists_across_reopen(self, tmp_path: Path) -> None:
        path = tmp_path / "kb.json"
        kb = _make_kb(path)
        for _ in range(10):
            _ = kb.increment_frequency("persist")
        _ = update_identities(kb, {"persist": "c-p"})
        kb.close()
        kb2 = _make_kb(path)
        assert kb2.get_identity("persist") == "c-p"


# ── update_projects ─────────────────────────────────────────────────────────


class TestUpdateProjects:
    def test_empty_dict_returns_zero(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        assert update_projects(kb, {}) == 0

    def test_stores_new_project(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        fp = {"hash1": 0.8, "hash2": 0.3}
        assert update_projects(kb, {"proj-1": fp}) == 1
        assert _get_fingerprint(kb, "proj-1") == fp

    def test_updates_similar_project(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        kb.set_project("proj-alpha", {"hash_a": 0.8, "hash_b": 0.4})
        # Jaccard ≈ 0.94 > 0.7 → update succeeds
        fp_sim = {"hash_a": 0.75, "hash_b": 0.38}
        assert update_projects(kb, {"proj-alpha": fp_sim}) == 1

    def test_skips_dissimilar_project(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        fp_orig = {"hash_a": 0.8, "hash_b": 0.4}
        kb.set_project("proj-beta", fp_orig)
        # Jaccard = 0.0 < 0.7 → skip
        assert update_projects(kb, {"proj-beta": {"hash_c": 0.9}}) == 0
        assert _get_fingerprint(kb, "proj-beta") == fp_orig

    def test_custom_jaccard_threshold(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        fp_a = {"hash_x": 0.9}
        fp_b = {"hash_x": 0.5, "hash_y": 0.4}
        fp_c = {"hash_z": 0.8}
        kb.set_project("proj-gamma", fp_a)
        # J(a,b) ≈ 0.56 > 0.2 → passes
        assert update_projects(kb, {"proj-gamma": fp_b}, jaccard_threshold=0.2) == 1
        # J(b,c) = 0.0 < 0.7 → rejected (stored fp is now fp_b)
        assert update_projects(kb, {"proj-gamma": fp_c}, jaccard_threshold=0.7) == 0

    def test_multiple_projects(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        fps = {"p1": {"h1": 0.7}, "p2": {"h2": 0.8}, "p3": {"h3": 0.9}}
        assert update_projects(kb, fps) == 3
        for uid in fps:
            assert _get_fingerprint(kb, uid) == fps[uid]


# ── update_frequencies ──────────────────────────────────────────────────────


class TestUpdateFrequencies:
    def test_empty_list_is_noop(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        update_frequencies(kb, [])
        assert kb.get_frequency("x") == 0

    def test_increments_single_hash(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        update_frequencies(kb, ["h1"])
        assert kb.get_frequency("h1") == 1

    def test_increments_multiple(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        update_frequencies(kb, ["a", "b", "c"])
        assert kb.get_frequency("a") == 1
        assert kb.get_frequency("b") == 1
        assert kb.get_frequency("c") == 1

    def test_repeated_hash(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        update_frequencies(kb, ["d", "d", "d"])
        assert kb.get_frequency("d") == 3

    def test_increments_existing(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        for _ in range(5):
            _ = kb.increment_frequency("seen")
        update_frequencies(kb, ["seen"])
        assert kb.get_frequency("seen") == 6

    def test_persists_across_reopen(self, tmp_path: Path) -> None:
        path = tmp_path / "kb.json"
        kb = _make_kb(path)
        update_frequencies(kb, ["f"])
        kb.close()
        kb2 = _make_kb(path)
        assert kb2.get_frequency("f") == 1


# ── resolve_contradiction ───────────────────────────────────────────────────


class TestResolveContradiction:
    def test_same_identity_returns_confidence_one(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        r = resolve_contradiction(kb, "h", "A", "A")
        assert r.strategy_applied == "CONFIDENCE_WEIGHT"
        assert r.resolution_confidence == pytest.approx(1.0)
        assert r.hash == "h"
        assert r.identity_a == "A"
        assert r.identity_b == "A"

    def test_current_matches_a(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        kb.set_identity("h", "A", 0.9)
        r = resolve_contradiction(kb, "h", "A", "B")
        assert r.strategy_applied == "CONFIDENCE_WEIGHT"

    def test_current_matches_b(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        kb.set_identity("h", "B", 0.9)
        r = resolve_contradiction(kb, "h", "A", "B")
        assert r.strategy_applied == "CONFIDENCE_WEIGHT"

    def test_current_neither_with_freq_uses_evidence(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        kb.set_identity("h", "OTHER", 0.9)
        for _ in range(3):
            _ = kb.increment_frequency("h")
        r = resolve_contradiction(kb, "h", "A", "B")
        assert r.strategy_applied == "EVIDENCE_WEIGHT"

    def test_current_neither_no_freq_defaults(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        r = resolve_contradiction(kb, "unknown", "A", "B")
        assert r.strategy_applied == "CONFIDENCE_WEIGHT"

    def test_adds_to_kb(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        n = len(kb.get_contradictions())
        _ = resolve_contradiction(kb, "h", "A", "B")
        assert len(kb.get_contradictions()) == n + 1

    def test_record_fields(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        r = resolve_contradiction(kb, "h-r", "id-A", "id-B")
        assert r.hash == "h-r"
        assert r.identity_a == "id-A"
        assert r.identity_b == "id-B"
        assert r.strategy_applied in ("CONFIDENCE_WEIGHT", "EVIDENCE_WEIGHT")
        assert r.resolved_at != ""
        assert isinstance(r.resolution_confidence, float)

    def test_confidence_with_frequency(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        kb.set_identity("h", "A", 0.9)
        for _ in range(5):
            _ = kb.increment_frequency("h")
        r = resolve_contradiction(kb, "h", "A", "B")
        assert r.resolution_confidence == pytest.approx(0.95)

    def test_confidence_default(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        r = resolve_contradiction(kb, "unknown", "A", "B")
        assert r.resolution_confidence == pytest.approx(0.5)


# ── compute_igs ─────────────────────────────────────────────────────────────


class TestComputeIgs:
    def test_empty_kb_returns_one(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        assert compute_igs(kb) == pytest.approx(1.0)

    def test_stable_graph_no_warning(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        assert compute_igs(kb) == pytest.approx(1.0)

    def test_zero_with_self_contradiction(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        kb.add_contradiction(ContradictionRecord("h", "X", "X", "CONFIDENCE_WEIGHT"))
        with pytest.warns(UserWarning, match="IGS dropped"):
            igs = compute_igs(kb)
        assert igs == pytest.approx(0.0)

    def test_one_contradiction(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        kb.add_contradiction(ContradictionRecord("h", "A", "B", "CONFIDENCE_WEIGHT"))
        with pytest.warns(UserWarning, match="IGS dropped"):
            igs = compute_igs(kb)
        assert igs == pytest.approx(0.5)

    def test_multiple_contradictions(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        kb.add_contradiction(ContradictionRecord("h1", "A", "B", "CONFIDENCE_WEIGHT"))
        kb.add_contradiction(ContradictionRecord("h2", "B", "C", "EVIDENCE_WEIGHT"))
        with pytest.warns(UserWarning, match="IGS dropped"):
            igs = compute_igs(kb)
        assert igs == pytest.approx(1.0 - 2.0 / 3.0)

    def test_warns_when_below_threshold(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        kb.add_contradiction(ContradictionRecord("h", "A", "B", "CONFIDENCE_WEIGHT"))
        with pytest.warns(UserWarning, match="IGS dropped"):
            _ = compute_igs(kb)

    def test_correct_formula(self, tmp_path: Path) -> None:
        kb = _make_kb(tmp_path / "kb.json")
        kb.add_contradiction(ContradictionRecord("h1", "A", "B", "CONFIDENCE_WEIGHT"))
        kb.add_contradiction(ContradictionRecord("h2", "C", "D", "EVIDENCE_WEIGHT"))
        with pytest.warns(UserWarning, match="IGS dropped"):
            igs = compute_igs(kb)
        assert igs == pytest.approx(0.5)


# ── integration ──────────────────────────────────────────────────────────────


class TestIntegration:
    def test_full_update_cycle(self, tmp_path: Path) -> None:
        path = tmp_path / "kb.json"
        kb = _make_kb(path)

        # 1. Update frequencies
        update_frequencies(kb, ["h1", "h2", "h1", "h3"])
        assert kb.get_frequency("h1") == 2
        assert kb.get_frequency("h2") == 1
        assert kb.get_frequency("h3") == 1

        # 2. Update identities (need freq >= 10 for default 0.95 threshold)
        for _ in range(8):
            _ = kb.increment_frequency("h1")
        stored = update_identities(kb, {"h1": "cluster-main", "h2": "cluster-other"})
        assert stored == 1
        assert kb.get_identity("h1") == "cluster-main"

        # 3. Update projects
        assert update_projects(kb, {"project-alpha": {"h1": 0.8, "h2": 0.3}}) == 1

        # 4. Resolve contradiction
        r = resolve_contradiction(kb, "h1", "cluster-main", "cluster-old")
        assert r.strategy_applied == "CONFIDENCE_WEIGHT"

        # 5. Compute IGS
        with pytest.warns(UserWarning, match="IGS dropped"):
            igs = compute_igs(kb)
        assert igs == pytest.approx(0.5)

        kb.close()
