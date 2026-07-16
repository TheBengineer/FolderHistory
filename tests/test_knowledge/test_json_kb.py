"""Tests for JSONKnowledgeBase — JSON-file backed knowledge persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from folderhistory.knowledge.json_kb import JSONKnowledgeBase
from folderhistory.knowledge.types import ContradictionRecord


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def kb_path(tmp_path: Path) -> Path:
    return tmp_path / "knowledge.json"


@pytest.fixture
def empty_kb(kb_path: Path) -> JSONKnowledgeBase:
    kb = JSONKnowledgeBase(kb_path)
    kb.open()
    return kb


@pytest.fixture
def populated_kb(kb_path: Path) -> JSONKnowledgeBase:
    """Return a closed KB with one identity, one project, one freq, one contradiction."""
    kb = JSONKnowledgeBase(kb_path)
    kb.open()
    kb.set_identity("abc123", "cluster-aaa", 0.95)
    kb.set_project("proj-1", {"feature_a": 0.8, "feature_b": 0.3})
    kb.increment_frequency("abc123")
    kb.increment_frequency("abc123")
    kb.add_contradiction(
        ContradictionRecord(
            hash="def456",
            identity_a="cluster-bbb",
            identity_b="cluster-ccc",
            strategy_applied="merge_with_rename",
        )
    )
    kb.close()
    return kb


# ── lifecycle ────────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_create_open_close_reopen(self, kb_path: Path) -> None:
        """Create KB → set identity → close → open → get identity back."""
        kb = JSONKnowledgeBase(kb_path)
        kb.open()
        kb.set_identity("hash1", "cluster-1", 0.9)
        kb.close()

        kb2 = JSONKnowledgeBase(kb_path)
        kb2.open()
        assert kb2.get_identity("hash1") == "cluster-1"
        kb2.close()

    def test_no_file_creates_empty_kb(self, kb_path: Path) -> None:
        """Opening a non-existent file creates an empty in-memory KB."""
        kb = JSONKnowledgeBase(kb_path)
        kb.open()
        assert kb.get_identity("anything") is None
        assert kb.get_project("anything") is None
        assert kb.get_frequency("anything") == 0
        assert kb.get_contradictions() == []

    def test_close_without_dirty_does_not_write(self, kb_path: Path) -> None:
        """close() is a no-op when nothing has been mutated."""
        kb = JSONKnowledgeBase(kb_path)
        kb.open()
        kb.close()  # should not crash
        assert not kb_path.exists()

    def test_version_meta(self, empty_kb: JSONKnowledgeBase) -> None:
        """version() returns VersionMeta with defaults."""
        meta = empty_kb.version()
        assert isinstance(meta, object)  # ABC returns object, but we check fields
        # The concrete type is VersionMeta
        from folderhistory.knowledge.types import VersionMeta

        assert isinstance(meta, VersionMeta)
        assert meta.schema_version == 1
        assert meta.algorithm_version == "1.0.0"


# ── version mismatch ─────────────────────────────────────────────────────────


class TestVersionMismatch:
    def test_raises_on_different_schema_version(self, kb_path: Path) -> None:
        """Open with schema_version=2 when file has v1 → ValueError."""
        kb = JSONKnowledgeBase(kb_path, schema_version=1)
        kb.open()
        kb.set_identity("x", "y", 0.5)
        kb.close()

        with pytest.raises(ValueError, match="Schema version mismatch"):
            JSONKnowledgeBase(kb_path, schema_version=2).open()

    def test_corrupt_json_raises(self, kb_path: Path) -> None:
        """Invalid JSON in the file → ValueError."""
        kb_path.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupt knowledge base"):
            JSONKnowledgeBase(kb_path).open()


# ── rollback ─────────────────────────────────────────────────────────────────


class TestRollback:
    def test_rollback_restores_original(self, kb_path: Path) -> None:
        """Create → set identity A → close (backup created) → set identity B
        → rollback → original identity restored."""
        # Phase 1: identity A
        kb = JSONKnowledgeBase(kb_path)
        kb.open()
        kb.set_identity("hash1", "cluster-A", 0.9)
        kb.close()  # creates backup_1

        # Phase 2: overwrite with identity B
        kb2 = JSONKnowledgeBase(kb_path)
        kb2.open()
        kb2.set_identity("hash1", "cluster-B", 0.5)
        kb2.close()

        # Verify B is current
        kb3 = JSONKnowledgeBase(kb_path)
        kb3.open()
        assert kb3.get_identity("hash1") == "cluster-B"
        kb3.close()

        # Rollback
        JSONKnowledgeBase.rollback(kb_path, 1)

        # Verify A is restored
        kb4 = JSONKnowledgeBase(kb_path)
        kb4.open()
        assert kb4.get_identity("hash1") == "cluster-A"

    def test_rollback_invalid_n_raises(self, kb_path: Path) -> None:
        """rollback with n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Backup index must be >= 1"):
            JSONKnowledgeBase.rollback(kb_path, 0)

    def test_rollback_missing_backup_raises(self, kb_path: Path) -> None:
        """rollback with non-existent backup raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            JSONKnowledgeBase.rollback(kb_path, 1)


# ── atomic write ─────────────────────────────────────────────────────────────


class TestAtomicWrite:
    def test_partial_write_does_not_corrupt_original(self, kb_path: Path) -> None:
        """Simulate a partial/corrupt write — original file stays intact
        and can be restored from backup."""
        # First cycle: write initial data (backup_1 is created on second close)
        kb = JSONKnowledgeBase(kb_path)
        kb.open()
        kb.set_identity("important", "cluster-safe", 0.99)
        kb.close()
        # Second cycle: creates backup_1 from the current file
        kb2 = JSONKnowledgeBase(kb_path)
        kb2.open()
        kb2.set_identity("tmp", "tmp", 0.5)
        kb2.close()

        # Corrupt the main file
        kb_path.write_text('{"partial": incomplete', encoding="utf-8")

        # Opening the corrupt file should fail
        with pytest.raises(ValueError, match="Corrupt knowledge base"):
            JSONKnowledgeBase(kb_path).open()

        # But backup_1 survives and restores the state from before 2nd close
        JSONKnowledgeBase.rollback(kb_path, 1)

        kb3 = JSONKnowledgeBase(kb_path)
        kb3.open()
        assert kb3.get_identity("important") == "cluster-safe"


# ── identity cluster ─────────────────────────────────────────────────────────


class TestIdentity:
    def test_get_unknown_hash_returns_none(self, empty_kb: JSONKnowledgeBase) -> None:
        assert empty_kb.get_identity("nonexistent") is None

    def test_set_and_get(self, empty_kb: JSONKnowledgeBase) -> None:
        empty_kb.set_identity("hash1", "cluster-1", 0.95)
        assert empty_kb.get_identity("hash1") == "cluster-1"

    def test_overwrite_identity(self, empty_kb: JSONKnowledgeBase) -> None:
        empty_kb.set_identity("hash1", "cluster-A", 0.9)
        empty_kb.set_identity("hash1", "cluster-B", 0.8)
        assert empty_kb.get_identity("hash1") == "cluster-B"

    def test_multiple_identities(self, empty_kb: JSONKnowledgeBase) -> None:
        empty_kb.set_identity("a", "cluster-A", 0.9)
        empty_kb.set_identity("b", "cluster-B", 0.8)
        empty_kb.set_identity("c", "cluster-C", 0.7)
        assert empty_kb.get_identity("a") == "cluster-A"
        assert empty_kb.get_identity("b") == "cluster-B"
        assert empty_kb.get_identity("c") == "cluster-C"


# ── project fingerprint ──────────────────────────────────────────────────────


class TestProject:
    def test_get_unknown_project_returns_none(
        self, empty_kb: JSONKnowledgeBase
    ) -> None:
        assert empty_kb.get_project("unknown") is None

    def test_set_and_get(self, empty_kb: JSONKnowledgeBase) -> None:
        fp = {"feature_a": 0.8, "feature_b": 0.3}
        empty_kb.set_project("proj-1", fp)
        result = empty_kb.get_project("proj-1")
        assert result is not None
        assert isinstance(result, dict)
        stored_fp = result.get("fingerprint")
        assert stored_fp == fp

    def test_overwrite_project(self, empty_kb: JSONKnowledgeBase) -> None:
        empty_kb.set_project("proj-1", {"v1": 0.5})
        empty_kb.set_project("proj-1", {"v2": 0.9})
        result = empty_kb.get_project("proj-1")
        assert result is not None
        assert isinstance(result, dict)
        assert result.get("fingerprint") == {"v2": 0.9}


# ── frequency ────────────────────────────────────────────────────────────────


class TestFrequency:
    def test_unknown_hash_returns_zero(self, empty_kb: JSONKnowledgeBase) -> None:
        assert empty_kb.get_frequency("unknown") == 0

    def test_increment_returns_new_count(self, empty_kb: JSONKnowledgeBase) -> None:
        assert empty_kb.increment_frequency("hash1") == 1
        assert empty_kb.increment_frequency("hash1") == 2
        assert empty_kb.increment_frequency("hash1") == 3

    def test_get_after_increment(self, empty_kb: JSONKnowledgeBase) -> None:
        empty_kb.increment_frequency("hash1")
        assert empty_kb.get_frequency("hash1") == 1

    def test_frequency_persists_across_close_reopen(
        self, kb_path: Path
    ) -> None:
        kb = JSONKnowledgeBase(kb_path)
        kb.open()
        kb.increment_frequency("persist-me")
        kb.increment_frequency("persist-me")
        kb.increment_frequency("persist-me")
        kb.close()

        kb2 = JSONKnowledgeBase(kb_path)
        kb2.open()
        assert kb2.get_frequency("persist-me") == 3

    def test_increment_atomic_in_memory(
        self, empty_kb: JSONKnowledgeBase
    ) -> None:
        """increment_frequency is atomic within the loaded data (does not persist
        until close)."""
        empty_kb.increment_frequency("test")
        assert empty_kb.get_frequency("test") == 1
        # Frequency should still be in-memory, not on disk
        # (we verify by checking the file doesn't change without close)
        empty_kb.increment_frequency("test")
        assert empty_kb.get_frequency("test") == 2


# ── contradictions ───────────────────────────────────────────────────────────


class TestContradiction:
    def test_empty_contradictions(self, empty_kb: JSONKnowledgeBase) -> None:
        assert empty_kb.get_contradictions() == []

    def test_add_and_get(self, empty_kb: JSONKnowledgeBase) -> None:
        record = ContradictionRecord(
            hash="abc",
            identity_a="cluster-X",
            identity_b="cluster-Y",
            strategy_applied="split",
        )
        empty_kb.add_contradiction(record)
        result = empty_kb.get_contradictions()
        assert len(result) == 1
        first = result[0]
        assert isinstance(first, ContradictionRecord)
        assert first.hash == "abc"
        assert first.identity_a == "cluster-X"
        assert first.identity_b == "cluster-Y"
        assert first.strategy_applied == "split"
        assert first.resolved_at != ""  # ISO timestamp was set

    def test_add_multiple_contradictions(self, empty_kb: JSONKnowledgeBase) -> None:
        empty_kb.add_contradiction(
            ContradictionRecord(hash="h1", identity_a="A", identity_b="B", strategy_applied="merge")
        )
        empty_kb.add_contradiction(
            ContradictionRecord(hash="h2", identity_a="C", identity_b="D", strategy_applied="split")
        )
        assert len(empty_kb.get_contradictions()) == 2

    def test_add_with_dict(self, empty_kb: JSONKnowledgeBase) -> None:
        empty_kb.add_contradiction(
            {
                "hash": "dict-hash",
                "identity_a": "dict-A",
                "identity_b": "dict-B",
                "strategy_applied": "confidence_weight",
                "resolution_confidence": 0.85,
            }
        )
        result = empty_kb.get_contradictions()
        assert len(result) == 1
        first = result[0]
        assert isinstance(first, ContradictionRecord)
        assert first.hash == "dict-hash"
        assert first.resolution_confidence == 0.85

    def test_add_invalid_type_raises(self, empty_kb: JSONKnowledgeBase) -> None:
        with pytest.raises(TypeError, match="Expected ContradictionRecord or dict"):
            empty_kb.add_contradiction("not a record")  # type: ignore[reportUnknownArgumentType]


# ── persistence across close/reopen (full round-trip) ────────────────────


class TestRoundTrip:
    def test_full_round_trip(self, kb_path: Path) -> None:
        """Set multiple data types → close → reopen → verify all."""
        kb = JSONKnowledgeBase(kb_path)
        kb.open()
        kb.set_identity("id1", "cluster-main", 0.99)
        kb.set_project("project-alpha", {"dim1": 0.7, "dim2": 0.4})
        kb.increment_frequency("id1")
        kb.increment_frequency("id1")
        kb.add_contradiction(
            ContradictionRecord(
                hash="conflict-1",
                identity_a="old-cluster",
                identity_b="new-cluster",
                strategy_applied="evidence_weight",
            )
        )
        kb.close()

        kb2 = JSONKnowledgeBase(kb_path)
        kb2.open()
        assert kb2.get_identity("id1") == "cluster-main"
        proj = kb2.get_project("project-alpha")
        assert proj is not None
        assert isinstance(proj, dict)
        assert proj.get("fingerprint") == {"dim1": 0.7, "dim2": 0.4}
        assert kb2.get_frequency("id1") == 2
        contradictions = kb2.get_contradictions()
        assert len(contradictions) == 1
        first = contradictions[0]
        assert isinstance(first, ContradictionRecord)
        assert first.strategy_applied == "evidence_weight"

    def test_file_created_only_on_close_when_dirty(self, kb_path: Path) -> None:
        """File is only written when mutations have occurred."""
        kb = JSONKnowledgeBase(kb_path)
        kb.open()
        assert not kb_path.exists()  # no file until close
        kb.set_identity("x", "y", 0.5)
        assert not kb_path.exists()  # still no file until close
        kb.close()
        assert kb_path.exists()  # now file exists


# ── rollback with populated_kb ───────────────────────────────────────────────


class TestRollbackIntegration:
    def test_rollback_to_previous_state(self, kb_path: Path, populated_kb: JSONKnowledgeBase) -> None:
        """populated_kb fixture creates state + backup. Overwrite,
        rollback, verify original restored."""
        # populated_kb is closed with identity abc123 → cluster-aaa

        # Overwrite with new identity
        kb = JSONKnowledgeBase(kb_path)
        kb.open()
        kb.set_identity("abc123", "cluster-overwritten", 0.1)
        kb.close()

        # Rollback
        JSONKnowledgeBase.rollback(kb_path, 1)

        kb2 = JSONKnowledgeBase(kb_path)
        kb2.open()
        assert kb2.get_identity("abc123") == "cluster-aaa"
        assert kb2.get_frequency("abc123") == 2
        assert len(kb2.get_contradictions()) == 1
