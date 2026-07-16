"""JSON-file backed KnowledgeBase.

Persists identity clusters, project fingerprints, observation frequencies,
contradictions, and version metadata to a single JSON file.

Atomic writes via temp file + rename.  The last N versions are kept as
backups in the same directory.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import cast, override

from folderhistory.knowledge import KnowledgeBase
from folderhistory.knowledge.types import ContradictionRecord, VersionMeta

_BACKUP_COUNT = 5


def _as_dict(val: object) -> dict[str, object]:
    """Safely coerce *val* to ``dict[str, object]``."""
    return cast("dict[str, object]", val) if isinstance(val, dict) else {}


def _as_list(val: object) -> list[object]:
    """Safely coerce *val* to ``list[object]``."""
    return cast("list[object]", val) if isinstance(val, list) else []


def _get_str(d: dict[str, object] | None, key: str, default: str = "") -> str:
    """Extract a string value from a dict with a fallback."""
    if d is None:
        return default
    val = d.get(key, default)
    return str(val) if not isinstance(val, str) else val


def _get_int(d: dict[str, object] | None, key: str, default: int = 0) -> int:
    """Extract an integer value from a dict with a fallback."""
    if d is None:
        return default
    val = d.get(key, default)
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    return default


def _load_json(raw: str) -> object:
    """Deserialize *raw* JSON to a Python object."""
    return cast("object", json.loads(raw))


def _get_float(d: dict[str, object] | None, key: str, default: float = 0.0) -> float:
    """Extract a float value from a dict with a fallback."""
    if d is None:
        return default
    val = d.get(key, default)
    if isinstance(val, (int, float)):
        return float(val)
    return default


class JSONKnowledgeBase(KnowledgeBase):
    """KnowledgeBase backed by a JSON file on disk.

    Parameters
    ----------
    path:
        Path to the JSON file.
    schema_version:
        Expected schema version.  If the file on disk holds a different
        version, :meth:`open` raises :class:`ValueError`.
    """

    _path: Path
    _schema_version: int
    _data: dict[str, object]
    _dirty: bool

    def __init__(self, path: Path, schema_version: int = 1) -> None:
        self._path = path
        self._schema_version = schema_version
        self._data = {}
        self._dirty = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    @override
    def open(self) -> None:
        if self._path.exists():
            raw = self._path.read_text(encoding="utf-8")
            try:
                parsed_raw = _load_json(raw)
            except json.JSONDecodeError as exc:
                msg = f"Corrupt knowledge base at {self._path}: {exc}"
                raise ValueError(msg) from exc

            if not isinstance(parsed_raw, dict):
                raise ValueError(
                    "Expected JSON object at "
                    + str(self._path)
                    + ", got "
                    + type(parsed_raw).__name__
                )

            parsed: dict[str, object] = cast("dict[str, object]", parsed_raw)
            stored_version = _as_dict(parsed.get("version"))
            ver = _get_int(stored_version, "schema_version", 0)
            if ver != self._schema_version:
                raise ValueError(
                    "Schema version mismatch: file has v"
                    + str(ver)
                    + ", expected v"
                    + str(self._schema_version)
                    + ".  "
                    + "Hint: re-run with --mode full to reindex."
                )
            self._data = parsed
        else:
            self._data = {
                "version": {
                    "schema_version": self._schema_version,
                    "algorithm_version": "1.0.0",
                },
                "identities": {},
                "projects": {},
                "frequencies": {},
                "contradictions": [],
                "history": [],
            }
        self._dirty = False

    @override
    def close(self) -> None:
        if not self._dirty:
            return
        self._rotate_backups()
        self._atomic_write()

    # ── backup rotation ────────────────────────────────────────────────────

    def _rotate_backups(self) -> None:
        parent = self._path.parent
        stem = self._path.stem
        suffix = self._path.suffix

        # Shift existing backups: N-1 → N
        for i in range(_BACKUP_COUNT - 1, 0, -1):
            src = parent / f"{stem}_backup_{i}{suffix}"
            dst = parent / f"{stem}_backup_{i + 1}{suffix}"
            if src.exists():
                _ = shutil.move(str(src), str(dst))

        # Copy current → backup_1
        if self._path.exists():
            _ = shutil.copy2(str(self._path), str(parent / f"{stem}_backup_1{suffix}"))

    def _atomic_write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self._path.parent),
            prefix=f".{self._path.stem}_",
            suffix=".tmp",
            delete=False,
        )
        try:
            json.dump(self._data, tmp, ensure_ascii=False, indent=2)
            tmp.close()
            _ = shutil.move(tmp.name, str(self._path))
        except BaseException:
            Path(tmp.name).unlink(missing_ok=True)
            raise
        self._dirty = False

    # ── identity cluster (file-hash → cluster-UID) ─────────────────────────

    @override
    def get_identity(self, hash: str) -> str | None:
        identities_raw = _as_dict(self._data.get("identities"))
        entry = identities_raw.get(hash)
        if entry is None:
            return None
        entry_dict = _as_dict(entry)
        uid_val = entry_dict.get("cluster_uid", "")
        return str(uid_val) if uid_val else None

    @override
    def set_identity(self, hash: str, cluster_uid: str, confidence: float) -> None:
        raw = self._data.get("identities")
        identities: dict[str, object] = _as_dict(raw)
        if not isinstance(raw, dict):
            self._data["identities"] = identities
        identities[hash] = {"cluster_uid": cluster_uid, "confidence": confidence}
        self._dirty = True

    # ── project fingerprint (project‑UID → feature‑vector) ─────────────────

    @override
    def get_project(self, fingerprint: str) -> object | None:
        projects_raw = _as_dict(self._data.get("projects"))
        return projects_raw.get(fingerprint)

    @override
    def set_project(self, project_uid: str, fingerprint: dict[str, float]) -> None:
        raw = self._data.get("projects")
        projects: dict[str, object] = _as_dict(raw)
        if not isinstance(raw, dict):
            self._data["projects"] = projects
        projects[project_uid] = {
            "fingerprint": fingerprint,
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }
        self._dirty = True

    # ── observation frequency ──────────────────────────────────────────────

    @override
    def get_frequency(self, hash: str) -> int:
        freqs_raw = _as_dict(self._data.get("frequencies"))
        raw_val: object = freqs_raw.get(hash, 0)
        if isinstance(raw_val, int):
            return raw_val
        if isinstance(raw_val, float):
            return int(raw_val)
        return 0

    @override
    def increment_frequency(self, hash: str) -> int:
        current = self.get_frequency(hash) + 1
        raw = self._data.get("frequencies")
        freqs: dict[str, object] = _as_dict(raw)
        if not isinstance(raw, dict):
            self._data["frequencies"] = freqs
        freqs[hash] = current
        self._dirty = True
        return current

    # ── versioning ─────────────────────────────────────────────────────────

    @override
    def version(self) -> object:
        version_raw = _as_dict(self._data.get("version"))
        return VersionMeta(
            schema_version=_get_int(version_raw, "schema_version", self._schema_version),
            algorithm_version=_get_str(version_raw, "algorithm_version", "1.0.0"),
            last_run_id=_get_str(version_raw, "last_run_id"),
            run_count=_get_int(version_raw, "run_count"),
        )

    # ── contradiction management ───────────────────────────────────────────

    @override
    def get_contradictions(self) -> list[object]:
        raw_list = _as_list(self._data.get("contradictions"))
        result: list[object] = []
        for c in raw_list:
            d = _as_dict(c)
            result.append(
                ContradictionRecord(
                    hash=_get_str(d, "hash"),
                    identity_a=_get_str(d, "identity_a"),
                    identity_b=_get_str(d, "identity_b"),
                    strategy_applied=_get_str(d, "strategy_applied"),
                    resolved_at=_get_str(d, "resolved_at"),
                    resolution_confidence=_get_float(d, "resolution_confidence"),
                )
            )
        return result

    @override
    def add_contradiction(self, record: object) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if isinstance(record, ContradictionRecord):
            entry: dict[str, object] = {
                "hash": record.hash,
                "identity_a": record.identity_a,
                "identity_b": record.identity_b,
                "strategy_applied": record.strategy_applied,
                "resolved_at": record.resolved_at if record.resolved_at else now,
                "resolution_confidence": record.resolution_confidence,
            }
        elif isinstance(record, dict):
            record_typed: dict[str, object] = cast("dict[str, object]", record)
            entry = {}
            for k_rec, v_rec in record_typed.items():
                entry[k_rec] = v_rec
            if "resolved_at" not in entry:
                entry["resolved_at"] = now
        else:
            raise TypeError(
                f"Expected ContradictionRecord or dict, got {type(record).__name__}"
            )
        raw = self._data.get("contradictions")
        contradictions: list[object] = _as_list(raw)
        if not isinstance(raw, list):
            self._data["contradictions"] = contradictions
        contradictions.append(entry)
        self._dirty = True

    # ── class methods ──────────────────────────────────────────────────────

    @classmethod
    def rollback(cls, path: Path, n: int = 1) -> None:
        """Restore backup *n* by copying it over the main file.

        Parameters
        ----------
        path:
            Path to the main KB file.
        n:
            Backup index (1-based).  ``n=1`` is the most recent backup.

        Raises
        ------
        ValueError
            If *n* < 1.
        FileNotFoundError
            If the requested backup file does not exist.
        """
        if n < 1:
            raise ValueError(f"Backup index must be >= 1, got {n}")
        stem = path.stem
        suffix = path.suffix
        backup_path = path.parent / f"{stem}_backup_{n}{suffix}"
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup {n} not found at {backup_path}")
        _ = shutil.copy2(str(backup_path), str(path))
