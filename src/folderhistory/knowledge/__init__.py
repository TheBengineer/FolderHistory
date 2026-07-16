"""Knowledge Base protocol and data types for FolderHistory's iterative feedback loop.

This module defines the abstract contract (KnowledgeBase ABC) and a read-only
protocol (ReadOnlyKB) that all persistence backends (JSON, SQLite, …) must
implement.  Concrete types are in ``types.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


class KnowledgeBase(ABC):
    """Persistent store for identity clusters and project fingerprints.

    Every read–write method on this ABC is part of the cross-run feedback
    loop: observations are accumulated, contradictions are detected and
    resolved, and the resulting knowledge carries forward into the next run.
    """

    # ── lifecycle ──────────────────────────────────────────────────────

    @abstractmethod
    def open(self) -> None:
        """Open or initialise the backing store."""

    @abstractmethod
    def close(self) -> None:
        """Flush and release resources."""

    # ── identity cluster (file-hash → cluster-UID) ─────────────────────

    @abstractmethod
    def get_identity(self, hash: str) -> str | None:
        """Return the cluster UID for *hash*, or ``None``."""

    @abstractmethod
    def set_identity(self, hash: str, cluster_uid: str, confidence: float) -> None:
        """Associate *hash* with a cluster at the given confidence."""

    # ── project fingerprint (project‑UID → feature‑vector) ─────────────

    @abstractmethod
    def get_project(self, fingerprint: str) -> object | None:
        """Return stored project data for *fingerprint*, or ``None``."""

    @abstractmethod
    def set_project(self, project_uid: str, fingerprint: dict[str, float]) -> None:
        """Persist a project's fingerprint vector."""

    # ── observation frequency ──────────────────────────────────────────

    @abstractmethod
    def get_frequency(self, hash: str) -> int:
        """Return how many times *hash* has been observed."""

    @abstractmethod
    def increment_frequency(self, hash: str) -> int:
        """Increment the observation counter for *hash* and return the new count."""

    # ── versioning ─────────────────────────────────────────────────────

    @abstractmethod
    def version(self) -> object:
        """Return schema / algorithm version metadata (``VersionMeta``)."""

    # ── contradiction management ───────────────────────────────────────

    @abstractmethod
    def get_contradictions(self) -> list[object]:
        """Return all unresolved contradictions."""

    @abstractmethod
    def add_contradiction(self, record: object) -> None:
        """Persist a contradiction record for later resolution."""


class ReadOnlyKB(Protocol):
    """Read-only view over a knowledge base.

    Satisfied by any object that exposes the following three methods with
    compatible signatures — no inheritance required (structural subtyping).
    """

    def get_identity(self, hash: str) -> str | None: ...
    def get_project(self, fingerprint: str) -> object | None: ...
    def get_frequency(self, hash: str) -> int: ...


# ── factory ─────────────────────────────────────────────────────────────


def create_knowledge_base(kind: str = "json") -> KnowledgeBase:
    """Return a :class:`KnowledgeBase` implementation identified by *kind*.

    Raises
    ------
    NotImplementedError
        No concrete implementations are available yet.  Subclasses of
        ``KnowledgeBase`` are registered in :mod:`folderhistory.knowledge.*`.
    """
    msg = f"No KnowledgeBase implementation for {kind!r}"
    raise NotImplementedError(msg)
