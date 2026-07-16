#!/usr/bin/env python3
"""Synthetic FolderHistory dummy data generator.

Generates snapshot directories with ground-truth metadata for testing
FolderHistory's identity resolution and operation derivation.

Usage
-----
    python tests/fixtures/gen_dummy_data.py --list-scenarios
    python tests/fixtures/gen_dummy_data.py -s ordered-evolution -n 5 -d 42 -o /tmp/test_dd
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import blake3
import orjson
import xxhash

if TYPE_CHECKING:
    from collections.abc import Callable

# ── Constants ────────────────────────────────────────────────────────────────

SCENARIO_DEFAULTS: dict[str, int] = {
    "ordered-evolution": 5,
    "unordered-timestamps": 4,
    "branching": 4,
    "flash-drive-chain": 4,
    "corruption-mix": 6,
    "photo-curation": 7,
}

_NS_IN_SEC = 1_000_000_000


# ── State tracking ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _TrackedFile:
    """A tracked file with identity metadata spanning snapshots.

    Attributes:
        file_id: UUID identifying this logical file across snapshots.
        path: Relative path within the snapshot.
        content: Raw file content as bytes.
        mtime_ns: Modification timestamp in nanoseconds since epoch.
        ctime_ns: Creation/metadata-change timestamp in nanoseconds.
        mode: Unix file mode (e.g. 0o644).
        is_symlink: Whether this entry is a symbolic link.
        target_path: Symlink target, if applicable.

    """

    file_id: str
    path: str
    content: bytes
    mtime_ns: int
    ctime_ns: int
    mode: int
    is_symlink: bool = False
    target_path: str | None = None
    project_uid: str = ""  # empty = single-project mode
    curation_status: str | None = None  # "curated" | "original" | "duplicate" | None


class _SnapshotState:
    """Files present in a single snapshot."""

    def __init__(self) -> None:
        self.files: dict[str, _TrackedFile] = {}

    def add_file(self, file: _TrackedFile) -> None:
        """Register a tracked file in this snapshot."""
        self.files[file.path] = file


class _GeneratorState:
    """Complete generator state for a scenario.

    Attributes:
        snapshots: Ordered list of snapshot states.
        timestamps: One float per snapshot (seconds since epoch).
        seed: Random seed for reproducibility.
        scenario: Scenario name string.
        snapshot_names: Display names (e.g. ["S0", "S1", ...]).

    """

    def __init__(self) -> None:
        self.snapshots: list[_SnapshotState] = []
        self.timestamps: list[float] = []
        self.seed: int = 42
        self.scenario: str = ""
        self.snapshot_names: list[str] = []
        self.source_paths: list[str] | None = None
        self.project_roots: dict[str, str] = {}


# ── Content utilities ────────────────────────────────────────────────────────


def _make_content(path: str, version: int, seed: int) -> bytes:
    """Produce deterministic content via SHA256 hex digest."""
    h = hashlib.sha256(f"{path}:v{version}:{seed}".encode()).hexdigest()
    return h.encode()


def _make_text(path: str, version: int, seed: int) -> bytes:
    """Multi-line text content (useful for line-ending testing)."""
    h = hashlib.sha256(f"{path}:v{version}:{seed}".encode()).hexdigest()
    return (
        f"# {path} v{version} {h[:16]}\n"
        f"line1\n"
        f"line2\n"
        f"line3\n"
    ).encode()


def _make_fid(name: str, seed: int) -> str:
    """Deterministic UUID v4-compatible file_id for a logical file."""
    digest = hashlib.sha256(f"fh:fid:{name}:{seed}".encode()).digest()[:16]
    return str(uuid.UUID(int=int.from_bytes(digest, "big")))


def _compute_timestamp(
    baseline: float,
    idx: int,
    seed: int,
) -> float:
    """Compute deterministic timestamp for snapshot N.

    Baseline + idx * 86400 + small jitter derived from seed+idx.
    """
    rng = random.Random(seed + idx)  # noqa: S311
    jitter = rng.uniform(-3600, 3600)  # ±1 hour
    return baseline + idx * 86400 + jitter


def _write_file(
    base_dir: Path,
    path: str,
    content: bytes,
    mtime_ns: int,
    _ctime_ns: int,  # accepted for ground truth metadata
    mode: int = 0o644,
) -> Path:
    """Write file with explicit timestamps. Creates parent dirs.

    Sets mtime via os.utime and permissions via chmod.
    Note: ctime cannot be directly set in portable Python; the
    value is accepted for ground truth metadata but the actual
    filesystem ctime reflects the chmod/utime operations.
    """
    full_path = base_dir / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    _ = full_path.write_bytes(content)
    _ = os.utime(full_path, ns=(mtime_ns, mtime_ns))
    _ = full_path.chmod(mode)
    return full_path


# ── Manifest generation helpers ────────────────────────────────────────────────


def _detect_line_ending(content: bytes) -> str:
    """Detect line endings: crlf, lf, mixed, binary, unknown."""
    if b"\x00" in content[:512]:
        return "binary"
    crlf_count = content.count(b"\r\n")
    lf_count = content.count(b"\n") - crlf_count
    if crlf_count > 0 and lf_count > 0:
        return "mixed"
    if crlf_count > 0:
        return "crlf"
    if lf_count > 0:
        return "lf"
    return "unknown"


def _write_manifests(state: _GeneratorState, output_dir: Path) -> None:
    """Write JSON manifests for all snapshots.

    Produces ``manifests/S{name}.json`` files consumable by
    ``ingest_manifest()``.  Each manifest mirrors the ``Snapshot``
    structure with real BLAKE3, xxHash64 hashes and line-ending detection.
    """
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    for i, snap_state in enumerate(state.snapshots):
        name = state.snapshot_names[i]
        manifest = {
            "id": name,
            "timestamp": state.timestamps[i],
            "source_path": str(output_dir / name),
            "files": [
                {
                    "path": tf.path,
                    "size": len(tf.content),
                    "mode": tf.mode,
                    "mtime_ns": tf.mtime_ns,
                    "ctime_ns": tf.ctime_ns,
                    "raw_blake3": blake3.blake3(tf.content).hexdigest(),
                    "normalized_blake3": (
                        blake3.blake3(tf.content.replace(b"\r\n", b"\n")).hexdigest()
                        if b"\x00" not in tf.content[:512]
                        else None
                    ),
                    "line_ending": _detect_line_ending(tf.content),
                    "xxhash64": xxhash.xxh64(tf.content).hexdigest(),
                    "is_symlink": tf.is_symlink,
                    "target_path": tf.target_path,
                }
                for tf in snap_state.files.values()
            ],
        }
        manifest_path = manifests_dir / f"{name}.json"
        _ = manifest_path.write_bytes(
            orjson.dumps(manifest, option=orjson.OPT_INDENT_2),
        )


# ── Ground truth derivation (private helpers) ────────────────────────────────


def _build_snapshot_info(state: _GeneratorState) -> list[dict[str, str | float]]:
    """Build snapshot metadata list from generator state."""
    info: list[dict[str, str | float]] = []
    for i, name in enumerate(state.snapshot_names):
        source: str = (
            state.source_paths[i] if state.source_paths else name
        )
        info.append({
            "id": name,
            "timestamp": state.timestamps[i],
            "source_path": source,
        })
    return info


def _build_identity_clusters(state: _GeneratorState) -> list[dict[str, object]]:
    """Build identity clusters from file_id observations across snapshots."""
    file_id_observations: dict[str, list[tuple[str, str]]] = {}
    curated_observations: dict[str, list[tuple[str, str]]] = {}
    for i, snap in enumerate(state.snapshots):
        snap_id = state.snapshot_names[i]
        for path, tf in snap.files.items():
            file_id_observations.setdefault(tf.file_id, []).append(
                (snap_id, path),
            )
            if tf.curation_status == "curated":
                curated_observations.setdefault(tf.file_id, []).append(
                    (snap_id, path),
                )

    clusters: list[dict[str, object]] = []
    for fid, observations in file_id_observations.items():
        path_counts: dict[str, int] = {}
        for _, p in observations:
            path_counts[p] = path_counts.get(p, 0) + 1
        canonical: str | None = (
            max(path_counts, key=lambda k: path_counts[k])
            if path_counts
            else None
        )
        cluster: dict[str, object] = {  # type: ignore[reportUnknownMemberType]
            "uid": fid,
            "observations": observations,
            "canonical_path": canonical,
            "confidence": 1.0,
        }
        if fid in curated_observations:
            cluster["curated_observations"] = curated_observations[fid]
        clusters.append(cluster)
    return clusters


def _find_source_content(
    files_i: dict[str, list[_TrackedFile]],
    content: bytes,
) -> str | None:
    """Find the source path in S_i whose content matches. Returns path or None."""
    for tf_list in files_i.values():
        for tf in tf_list:
            if tf.content == content:
                return tf.path
    return None


def _build_operations(state: _GeneratorState) -> dict[str, list[dict[str, object]]]:
    """Derive edit operations between consecutive snapshot pairs."""
    operations: dict[str, list[dict[str, object]]] = {}
    for i in range(len(state.snapshots) - 1):
        snap_i = state.snapshot_names[i]
        snap_j = state.snapshot_names[i + 1]
        key = f"{snap_i}->{snap_j}"

        ops: list[dict[str, object]] = []

        # Build multi-path dicts: file_id -> list[_TrackedFile]
        # Supports copies (same file_id at multiple paths in one snapshot)
        files_i: dict[str, list[_TrackedFile]] = {}
        for tf in state.snapshots[i].files.values():
            files_i.setdefault(tf.file_id, []).append(tf)
        files_j: dict[str, list[_TrackedFile]] = {}
        for tf in state.snapshots[i + 1].files.values():
            files_j.setdefault(tf.file_id, []).append(tf)

        ids_i = set(files_i.keys())
        ids_j = set(files_j.keys())

        # Same file_id appears in both snapshots
        for fid in ids_i & ids_j:
            paths_i = {tf.path for tf in files_i[fid]}
            paths_j = {tf.path for tf in files_j[fid]}

            shared = paths_i & paths_j
            removed = paths_i - paths_j
            added = paths_j - paths_i

            # Shared paths: modify if content changed, no-op otherwise
            for path in shared:
                tf_i = next(t for t in files_i[fid] if t.path == path)
                tf_j = next(t for t in files_j[fid] if t.path == path)
                if tf_i.content != tf_j.content:
                    ops.append({  # type: ignore[reportUnknownMemberType]
                        "op_type": "modify",
                        "file_id": fid,
                        "source_path": tf_i.path,
                        "target_path": tf_j.path,
                        "confidence": 1.0,
                    })

            # Track rename sources to avoid double-emitting deletes
            rename_sources: set[str] = set()

            # Added paths: determine if rename, copy, or create
            for path in added:
                tf_j = next(t for t in files_j[fid] if t.path == path)

                # Classic rename: file_id moved to a single new path
                if not shared and len(paths_i) == 1 and len(paths_j) == 1:
                    source_path = next(iter(removed))
                    ops.append({  # type: ignore[reportUnknownMemberType]
                        "op_type": "rename",
                        "file_id": fid,
                        "source_path": source_path,
                        "target_path": path,
                        "confidence": 1.0,
                    })
                    rename_sources.add(source_path)
                else:
                    source_path = _find_source_content(files_i, tf_j.content)
                    if source_path:
                        ops.append({  # type: ignore[reportUnknownMemberType]
                            "op_type": "copy",
                            "file_id": fid,
                            "source_path": source_path,
                            "target_path": path,
                            "confidence": 1.0,
                        })
                    else:
                        ops.append({  # type: ignore[reportUnknownMemberType]
                            "op_type": "create",
                            "file_id": fid,
                            "source_path": None,
                            "target_path": path,
                            "confidence": 1.0,
                        })

            # Removed paths: delete if not already handled as rename
            for path in removed:
                if path not in rename_sources:
                    tf_i = next(t for t in files_i[fid] if t.path == path)
                    ops.append({  # type: ignore[reportUnknownMemberType]
                        "op_type": "delete",
                        "file_id": fid,
                        "source_path": tf_i.path,
                        "target_path": None,
                        "confidence": 1.0,
                    })

        # Deleted files: file_id entirely absent in S_j
        for fid in ids_i - ids_j:
            for tf in files_i[fid]:
                ops.append({  # type: ignore[reportUnknownMemberType]
                    "op_type": "delete",
                    "file_id": fid,
                    "source_path": tf.path,
                    "target_path": None,
                    "confidence": 1.0,
                })

        # Created files: file_id entirely new in S_j
        for fid in ids_j - ids_i:
            for tf in files_j[fid]:
                ops.append({  # type: ignore[reportUnknownMemberType]
                    "op_type": "create",
                    "file_id": fid,
                    "source_path": None,
                    "target_path": tf.path,
                    "confidence": 1.0,
                })

        operations[key] = ops
    return operations


def _build_files_metadata(state: _GeneratorState) -> dict[str, dict[str, object]]:
    """Build per-snapshot file metadata for ground truth."""
    meta: dict[str, dict[str, object]] = {}
    for i, snap in enumerate(state.snapshots):
        snap_id = state.snapshot_names[i]
        snap_files: dict[str, object] = {}
        for path, tf in snap.files.items():
            content_hash = hashlib.sha256(tf.content).hexdigest()
            le = "lf" if b"\x00" not in tf.content else "binary"
            snap_files[path] = {
                "raw_blake3": content_hash,
                "size": len(tf.content),
                "mtime_ns": tf.mtime_ns,
                "ctime_ns": tf.ctime_ns,
                "mode": tf.mode,
                "normalized_blake3": (
                    content_hash if le != "binary" else None
                ),
                "line_ending": le,
                "is_symlink": tf.is_symlink,
                "project_uid": tf.project_uid,
                "curation_status": tf.curation_status,
            }
        meta[snap_id] = snap_files
    return meta


def _derive_ground_truth(state: _GeneratorState) -> dict[str, object]:
    """Derive ground truth metadata from generator state.

    Delegates to specialised helpers for each section.
    """
    return {
        "scenario": state.scenario,
        "seed": state.seed,
        "snapshots": _build_snapshot_info(state),
        "expected_operations": _build_operations(state),
        "identity_clusters": _build_identity_clusters(state),
        "files": _build_files_metadata(state),
        "project_roots": state.project_roots,
    }


# ── Scenario generators ──────────────────────────────────────────────────────


def _generate_ordered_evolution(
    num_snapshots: int,
    seed: int,
) -> _GeneratorState:
    """Clean sequential evolution with create/modify/rename/delete.

    S0: README.md, src/main.py, src/utils.py, .gitignore
    S1: modify src/main.py (feature A), modify README.md (update docs)
    S2: create tests/test_main.py, modify src/utils.py (helper B)
    S3: rename src/utils.py -> src/helpers.py, modify README.md
    S4: create src/cli.py, delete .gitignore
    """
    state = _GeneratorState()
    state.seed = seed
    state.scenario = "ordered-evolution"
    state.snapshot_names = [f"S{i}" for i in range(num_snapshots)]
    baseline = 1700000000.0
    state.timestamps = [
        _compute_timestamp(baseline, i, seed) for i in range(num_snapshots)
    ]

    # Logical file identities (stable UUIDs across renames)
    fid_readme = _make_fid("ordered_readme", seed)
    fid_main = _make_fid("ordered_main", seed)
    fid_utils = _make_fid("ordered_utils", seed)  # utils.py -> helpers.py
    fid_gitignore = _make_fid("ordered_gitignore", seed)
    fid_test = _make_fid("ordered_test", seed)
    fid_cli = _make_fid("ordered_cli", seed)

    # Per-snapshot file definitions: (file_id, path, version)
    # Content version uses canonical content_path for identity stability
    #   - fid_utils uses "src/utils.py" as content_path across rename
    snapshot_plan: list[list[tuple[str, str, int, str]]] = [
        # S0: initial files
        [
            (fid_readme, "README.md", 0, "README.md"),
            (fid_main, "src/main.py", 0, "src/main.py"),
            (fid_utils, "src/utils.py", 0, "src/utils.py"),
            (fid_gitignore, ".gitignore", 0, ".gitignore"),
        ],
        # S1: modify main.py and README.md
        [
            (fid_readme, "README.md", 1, "README.md"),
            (fid_main, "src/main.py", 1, "src/main.py"),
            (fid_utils, "src/utils.py", 0, "src/utils.py"),
            (fid_gitignore, ".gitignore", 0, ".gitignore"),
        ],
        # S2: create tests, modify utils.py
        [
            (fid_readme, "README.md", 1, "README.md"),
            (fid_main, "src/main.py", 1, "src/main.py"),
            (fid_utils, "src/utils.py", 1, "src/utils.py"),
            (fid_gitignore, ".gitignore", 0, ".gitignore"),
            (fid_test, "tests/test_main.py", 0, "tests/test_main.py"),
        ],
        # S3: rename utils.py -> helpers.py, modify README.md
        [
            (fid_readme, "README.md", 2, "README.md"),
            (fid_main, "src/main.py", 1, "src/main.py"),
            (fid_utils, "src/helpers.py", 1, "src/utils.py"),
            (fid_gitignore, ".gitignore", 0, ".gitignore"),
            (fid_test, "tests/test_main.py", 0, "tests/test_main.py"),
        ],
        # S4: create cli.py, delete .gitignore
        [
            (fid_readme, "README.md", 2, "README.md"),
            (fid_main, "src/main.py", 1, "src/main.py"),
            (fid_utils, "src/helpers.py", 1, "src/utils.py"),
            (fid_test, "tests/test_main.py", 0, "tests/test_main.py"),
            (fid_cli, "src/cli.py", 0, "src/cli.py"),
        ],
    ]

    for snap_idx in range(min(num_snapshots, len(snapshot_plan))):
        snap = _SnapshotState()
        ts_ns = int(state.timestamps[snap_idx] * _NS_IN_SEC)
        for fid, path, version, content_path in snapshot_plan[snap_idx]:
            content = _make_content(content_path, version, seed)
            snap.add_file(
                _TrackedFile(
                    file_id=fid,
                    path=path,
                    content=content,
                    mtime_ns=ts_ns,
                    ctime_ns=ts_ns,
                    mode=0o644,
                ),
            )
        state.snapshots.append(snap)
    return state


def _generate_unordered_timestamps(
    num_snapshots: int,
    seed: int,
) -> _GeneratorState:
    """Snapshots whose timestamps don't match content evolution order.

    Content evolves S0->S1->S2->S3 naturally but timestamps use
    shuffled indices [3, 1, 4, 2] to create temporal mismatch.

    S0 (timestamp=3): README.md, src/main.py, src/utils.py
    S1 (timestamp=1): modify src/main.py, create src/cli.py
    S2 (timestamp=4): modify README.md, create tests/test_main.py
    S3 (timestamp=2): modify src/utils.py, create data/config.json
    """
    state = _GeneratorState()
    state.seed = seed
    state.scenario = "unordered-timestamps"
    state.snapshot_names = [f"S{i}" for i in range(num_snapshots)]
    baseline = 1700000000.0

    # Timestamp indices are shuffled: content order != temporal order
    timestamp_indices = [3, 1, 4, 2]
    state.timestamps = [
        _compute_timestamp(baseline, ts_idx, seed)
        for ts_idx in timestamp_indices[:num_snapshots]
    ]

    fid_readme = _make_fid("unordered_readme", seed)
    fid_main = _make_fid("unordered_main", seed)
    fid_utils = _make_fid("unordered_utils", seed)
    fid_cli = _make_fid("unordered_cli", seed)
    fid_test = _make_fid("unordered_test", seed)
    fid_config = _make_fid("unordered_config", seed)

    snapshot_plan: list[list[tuple[str, str, int]]] = [
        # S0: initial files
        [
            (fid_readme, "README.md", 0),
            (fid_main, "src/main.py", 0),
            (fid_utils, "src/utils.py", 0),
        ],
        # S1: modify main.py, create cli.py
        [
            (fid_readme, "README.md", 0),
            (fid_main, "src/main.py", 1),
            (fid_utils, "src/utils.py", 0),
            (fid_cli, "src/cli.py", 0),
        ],
        # S2: modify README.md, create tests
        [
            (fid_readme, "README.md", 1),
            (fid_main, "src/main.py", 1),
            (fid_utils, "src/utils.py", 0),
            (fid_cli, "src/cli.py", 0),
            (fid_test, "tests/test_main.py", 0),
        ],
        # S3: modify utils.py, create config.json
        [
            (fid_readme, "README.md", 1),
            (fid_main, "src/main.py", 1),
            (fid_utils, "src/utils.py", 1),
            (fid_cli, "src/cli.py", 0),
            (fid_test, "tests/test_main.py", 0),
            (fid_config, "data/config.json", 0),
        ],
    ]

    for snap_idx in range(min(num_snapshots, len(snapshot_plan))):
        snap = _SnapshotState()
        ts_ns = int(state.timestamps[snap_idx] * _NS_IN_SEC)
        for fid, path, version in snapshot_plan[snap_idx]:
            content = _make_content(path, version, seed)
            snap.add_file(
                _TrackedFile(
                    file_id=fid,
                    path=path,
                    content=content,
                    mtime_ns=ts_ns,
                    ctime_ns=ts_ns,
                    mode=0o644,
                ),
            )
        state.snapshots.append(snap)
    return state


def _generate_branching(
    num_snapshots: int,
    seed: int,
) -> _GeneratorState:
    """Divergent branches from common ancestor, then a merge state.

    S0 (ancestor): README.md, src/core.py, src/utils.py, data/config.json
    S1 (branch-a): modify src/core.py, create src/feature_a.py, del utils.py
    S2 (branch-b): modify src/utils.py, create src/feature_b.py, docs/guide.md
    S3 (merged): all files from both branches present (merge state)

    Identity clusters:
      - README.md, data/config.json span all 4
      - src/core.py spans S0, S1, S3 (not S2)
      - src/utils.py spans S0, S2, S3 (not S1)
      - Branch-specific files appear only in their lineage
    """
    state = _GeneratorState()
    state.seed = seed
    state.scenario = "branching"
    state.snapshot_names = [f"S{i}" for i in range(num_snapshots)]
    baseline = 1700000000.0
    state.timestamps = [
        _compute_timestamp(baseline, i, seed) for i in range(num_snapshots)
    ]

    fid_readme = _make_fid("branch_readme", seed)
    fid_core = _make_fid("branch_core", seed)  # S0, S1, S3
    fid_utils = _make_fid("branch_utils", seed)  # S0, S2, S3
    fid_config = _make_fid("branch_config", seed)  # all
    fid_feature_a = _make_fid("branch_feature_a", seed)  # S1, S3
    fid_feature_b = _make_fid("branch_feature_b", seed)  # S2, S3
    fid_guide = _make_fid("branch_guide", seed)  # S2, S3

    snapshot_plan: list[list[tuple[str, str, int]]] = [
        # S0 (ancestor)
        [
            (fid_readme, "README.md", 0),
            (fid_core, "src/core.py", 0),
            (fid_utils, "src/utils.py", 0),
            (fid_config, "data/config.json", 0),
        ],
        # S1 (branch-a): core.py modified, feature_a.py created, utils deleted
        [
            (fid_readme, "README.md", 0),
            (fid_core, "src/core.py", 1),
            (fid_config, "data/config.json", 0),
            (fid_feature_a, "src/feature_a.py", 0),
        ],
        # S2 (branch-b): utils.py modified, feature_b.py + guide created
        [
            (fid_readme, "README.md", 0),
            (fid_utils, "src/utils.py", 1),
            (fid_config, "data/config.json", 0),
            (fid_feature_b, "src/feature_b.py", 0),
            (fid_guide, "docs/guide.md", 0),
        ],
        # S3 (merged): all files from both branches
        [
            (fid_readme, "README.md", 0),
            (fid_core, "src/core.py", 1),
            (fid_utils, "src/utils.py", 1),
            (fid_config, "data/config.json", 0),
            (fid_feature_a, "src/feature_a.py", 0),
            (fid_feature_b, "src/feature_b.py", 0),
            (fid_guide, "docs/guide.md", 0),
        ],
    ]

    for snap_idx in range(min(num_snapshots, len(snapshot_plan))):
        snap = _SnapshotState()
        ts_ns = int(state.timestamps[snap_idx] * _NS_IN_SEC)
        for fid, path, version in snapshot_plan[snap_idx]:
            content = _make_content(path, version, seed)
            snap.add_file(
                _TrackedFile(
                    file_id=fid,
                    path=path,
                    content=content,
                    mtime_ns=ts_ns,
                    ctime_ns=ts_ns,
                    mode=0o644,
                ),
            )
        state.snapshots.append(snap)
    return state


def _generate_flash_drive_chain(
    num_snapshots: int,
    seed: int,
) -> _GeneratorState:
    """USB backup chain with ctime resets, CRLF injection, device copies.

    S0 (laptop): LF content, original ctimes
    S1 (USB): ctime reset to current, main.py has CRLF line endings
    S2 (USB week2): modify main.py, create new_feature.py
    S3 (new laptop): all ctime reset again, content identical to S2

    Each "device" has a distinct source_path. CRLF files contain
    actual \\r\\n byte sequences.
    """
    state = _GeneratorState()
    state.seed = seed
    state.scenario = "flash-drive-chain"
    state.snapshot_names = [f"S{i}" for i in range(num_snapshots)]
    baseline = 1700000000.0
    state.timestamps = [
        _compute_timestamp(baseline, i, seed) for i in range(num_snapshots)
    ]
    state.source_paths = ["laptop", "usb_drive", "usb_drive", "new_laptop"]

    fid_readme = _make_fid("flash_readme", seed)
    fid_main = _make_fid("flash_main", seed)
    fid_utils = _make_fid("flash_utils", seed)
    fid_config = _make_fid("flash_config", seed)
    fid_new_feature = _make_fid("flash_new_feature", seed)

    # Helper: content with newlines (for CRLF conversion)
    def lf(path: str, v: int) -> bytes:
        return _make_text(path, v, seed)

    def crlf(path: str, v: int) -> bytes:
        return _make_text(path, v, seed).replace(b"\n", b"\r\n")

    ts = [int(state.timestamps[i] * _NS_IN_SEC) for i in range(num_snapshots)]

    # S0: original laptop — all LF, original ctimes
    snap0 = _SnapshotState()
    snap0.add_file(
        _TrackedFile(fid_readme, "README.md", lf("README.md", 0), ts[0], ts[0], 0o644),
    )
    snap0.add_file(
        _TrackedFile(fid_main, "src/main.py", lf("src/main.py", 0), ts[0], ts[0], 0o644),
    )
    snap0.add_file(
        _TrackedFile(fid_utils, "src/utils.py", lf("src/utils.py", 0), ts[0], ts[0], 0o644),
    )
    snap0.add_file(
        _TrackedFile(fid_config, "data/config.json", lf("data/config.json", 0), ts[0], ts[0], 0o644),
    )
    state.snapshots.append(snap0)
    if num_snapshots < 2:
        return state

    # S1: USB copy — ctime reset to ts[1], main.py becomes CRLF, mtime preserved
    snap1 = _SnapshotState()
    snap1.add_file(
        _TrackedFile(fid_readme, "README.md", lf("README.md", 0), ts[0], ts[1], 0o644),
    )
    snap1.add_file(
        _TrackedFile(fid_main, "src/main.py", crlf("src/main.py", 0), ts[0], ts[1], 0o644),
    )
    snap1.add_file(
        _TrackedFile(fid_utils, "src/utils.py", lf("src/utils.py", 0), ts[0], ts[1], 0o644),
    )
    snap1.add_file(
        _TrackedFile(fid_config, "data/config.json", lf("data/config.json", 0), ts[0], ts[1], 0o644),
    )
    state.snapshots.append(snap1)
    if num_snapshots < 3:
        return state

    # S2: USB week2 — modify main.py (CRLF stays), create new_feature.py
    snap2 = _SnapshotState()
    snap2.add_file(
        _TrackedFile(fid_readme, "README.md", lf("README.md", 0), ts[2], ts[2], 0o644),
    )
    snap2.add_file(
        _TrackedFile(fid_main, "src/main.py", crlf("src/main.py", 1), ts[2], ts[2], 0o644),
    )
    snap2.add_file(
        _TrackedFile(fid_utils, "src/utils.py", lf("src/utils.py", 0), ts[2], ts[2], 0o644),
    )
    snap2.add_file(
        _TrackedFile(fid_config, "data/config.json", lf("data/config.json", 0), ts[2], ts[2], 0o644),
    )
    snap2.add_file(
        _TrackedFile(fid_new_feature, "src/new_feature.py", lf("src/new_feature.py", 0), ts[2], ts[2], 0o644),
    )
    state.snapshots.append(snap2)
    if num_snapshots < 4:
        return state

    # S3: new laptop — all ctime reset, content identical to S2
    snap3 = _SnapshotState()
    snap3.add_file(
        _TrackedFile(fid_readme, "README.md", lf("README.md", 0), ts[2], ts[3], 0o644),
    )
    snap3.add_file(
        _TrackedFile(fid_main, "src/main.py", crlf("src/main.py", 1), ts[2], ts[3], 0o644),
    )
    snap3.add_file(
        _TrackedFile(fid_utils, "src/utils.py", lf("src/utils.py", 0), ts[2], ts[3], 0o644),
    )
    snap3.add_file(
        _TrackedFile(fid_config, "data/config.json", lf("data/config.json", 0), ts[2], ts[3], 0o644),
    )
    snap3.add_file(
        _TrackedFile(fid_new_feature, "src/new_feature.py", lf("src/new_feature.py", 0), ts[2], ts[3], 0o644),
    )
    state.snapshots.append(snap3)
    return state


def _generate_corruption_mix(
    num_snapshots: int,
    seed: int,
) -> _GeneratorState:
    """Multi-corruption scenario with metadata and content edge cases.

    S0 (clean): standard files + assets/logo.png (binary)
    S1 (time): README.md mtime=0, create src/epoch_test.py mtime=0
    S2 (nested): deeply nested a/b/c/d/e/f/g/deep.txt, modify main.py
    S3 (binary): assets/corrupted.bin with null bytes, modify config.json
    S4 (perms): restricted/secret.txt mode=0o000, modify utils.py
    S5 (mixed): cross_platform.py (LF) -> cross_plat.py (CRLF) rename
    """
    state = _GeneratorState()
    state.seed = seed
    state.scenario = "corruption-mix"
    state.snapshot_names = [f"S{i}" for i in range(num_snapshots)]
    baseline = 1700000000.0
    state.timestamps = [
        _compute_timestamp(baseline, i, seed) for i in range(num_snapshots)
    ]

    fid_readme = _make_fid("corrupt_readme", seed)
    fid_main = _make_fid("corrupt_main", seed)
    fid_utils = _make_fid("corrupt_utils", seed)
    fid_config = _make_fid("corrupt_config", seed)
    fid_logo = _make_fid("corrupt_logo", seed)
    fid_epoch = _make_fid("corrupt_epoch", seed)
    fid_deep = _make_fid("corrupt_deep", seed)
    fid_corrupted_bin = _make_fid("corrupt_bin", seed)
    fid_secret = _make_fid("corrupt_secret", seed)
    fid_cross_plat = _make_fid("corrupt_cross_plat", seed)

    ts = [int(state.timestamps[i] * _NS_IN_SEC) for i in range(num_snapshots)]

    # S0: clean baseline
    snap0 = _SnapshotState()
    for fid, path, version in [
        (fid_readme, "README.md", 0),
        (fid_main, "src/main.py", 0),
        (fid_utils, "src/utils.py", 0),
        (fid_config, "data/config.json", 0),
    ]:
        content = _make_content(path, version, seed)
        snap0.add_file(
            _TrackedFile(fid, path, content, ts[0], ts[0], 0o644),
        )
    # Binary logo (no null bytes in hash content, so use random bytes)
    snap0.add_file(
        _TrackedFile(
            fid_logo,
            "assets/logo.png",
            bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + b"\x00\x00\x00\x00IEND",
            ts[0], ts[0], 0o644,
        ),
    )
    state.snapshots.append(snap0)
    if num_snapshots < 2:
        return state

    # S1: time corruption — mtime_ns = 0 (epoch)
    snap1 = _SnapshotState()
    snap1.add_file(
        _TrackedFile(fid_readme, "README.md", _make_content("README.md", 1, seed), 0, ts[1], 0o644),
    )
    snap1.add_file(
        _TrackedFile(fid_main, "src/main.py", _make_content("src/main.py", 0, seed), ts[1], ts[1], 0o644),
    )
    snap1.add_file(
        _TrackedFile(fid_utils, "src/utils.py", _make_content("src/utils.py", 0, seed), ts[1], ts[1], 0o644),
    )
    snap1.add_file(
        _TrackedFile(fid_config, "data/config.json", _make_content("data/config.json", 0, seed), ts[1], ts[1], 0o644),
    )
    snap1.add_file(
        _TrackedFile(fid_logo, "assets/logo.png", snap0.files["assets/logo.png"].content, ts[1], ts[1], 0o644),
    )
    snap1.add_file(
        _TrackedFile(fid_epoch, "src/epoch_test.py", _make_content("src/epoch_test.py", 0, seed), 0, ts[1], 0o644),
    )
    state.snapshots.append(snap1)
    if num_snapshots < 3:
        return state

    # S2: deep nesting + modify main.py
    snap2 = _SnapshotState()
    snap2.add_file(
        _TrackedFile(fid_readme, "README.md", _make_content("README.md", 1, seed), ts[2], ts[2], 0o644),
    )
    snap2.add_file(
        _TrackedFile(fid_main, "src/main.py", _make_content("src/main.py", 1, seed), ts[2], ts[2], 0o644),
    )
    snap2.add_file(
        _TrackedFile(fid_utils, "src/utils.py", _make_content("src/utils.py", 0, seed), ts[2], ts[2], 0o644),
    )
    snap2.add_file(
        _TrackedFile(fid_config, "data/config.json", _make_content("data/config.json", 0, seed), ts[2], ts[2], 0o644),
    )
    snap2.add_file(
        _TrackedFile(fid_logo, "assets/logo.png", snap0.files["assets/logo.png"].content, ts[2], ts[2], 0o644),
    )
    snap2.add_file(
        _TrackedFile(fid_epoch, "src/epoch_test.py", _make_content("src/epoch_test.py", 0, seed), ts[2], ts[2], 0o644),
    )
    snap2.add_file(
        _TrackedFile(
            fid_deep,
            "a/b/c/d/e/f/g/deep.txt",
            _make_content("a/b/c/d/e/f/g/deep.txt", 0, seed),
            ts[2], ts[2], 0o644,
        ),
    )
    state.snapshots.append(snap2)
    if num_snapshots < 4:
        return state

    # S3: binary corruption — corrupted.bin with null bytes, modify config.json
    snap3 = _SnapshotState()
    snap3.add_file(
        _TrackedFile(fid_readme, "README.md", _make_content("README.md", 1, seed), ts[3], ts[3], 0o644),
    )
    snap3.add_file(
        _TrackedFile(fid_main, "src/main.py", _make_content("src/main.py", 1, seed), ts[3], ts[3], 0o644),
    )
    snap3.add_file(
        _TrackedFile(fid_utils, "src/utils.py", _make_content("src/utils.py", 0, seed), ts[3], ts[3], 0o644),
    )
    snap3.add_file(
        _TrackedFile(fid_config, "data/config.json", _make_content("data/config.json", 1, seed), ts[3], ts[3], 0o644),
    )
    snap3.add_file(
        _TrackedFile(fid_logo, "assets/logo.png", snap0.files["assets/logo.png"].content, ts[3], ts[3], 0o644),
    )
    snap3.add_file(
        _TrackedFile(fid_epoch, "src/epoch_test.py", _make_content("src/epoch_test.py", 0, seed), ts[3], ts[3], 0o644),
    )
    snap3.add_file(
        _TrackedFile(fid_deep, "a/b/c/d/e/f/g/deep.txt", _make_content("a/b/c/d/e/f/g/deep.txt", 0, seed), ts[3], ts[3], 0o644),
    )
    # Binary file with null bytes
    null_content = b"\x00" * 64
    snap3.add_file(
        _TrackedFile(fid_corrupted_bin, "assets/corrupted.bin", null_content, ts[3], ts[3], 0o644),
    )
    state.snapshots.append(snap3)
    if num_snapshots < 5:
        return state

    # S4: permissions — restricted/secret.txt mode=0o000, modify utils.py
    snap4 = _SnapshotState()
    snap4.add_file(
        _TrackedFile(fid_readme, "README.md", _make_content("README.md", 1, seed), ts[4], ts[4], 0o644),
    )
    snap4.add_file(
        _TrackedFile(fid_main, "src/main.py", _make_content("src/main.py", 1, seed), ts[4], ts[4], 0o644),
    )
    snap4.add_file(
        _TrackedFile(fid_utils, "src/utils.py", _make_content("src/utils.py", 1, seed), ts[4], ts[4], 0o644),
    )
    snap4.add_file(
        _TrackedFile(fid_config, "data/config.json", _make_content("data/config.json", 1, seed), ts[4], ts[4], 0o644),
    )
    snap4.add_file(
        _TrackedFile(fid_logo, "assets/logo.png", snap0.files["assets/logo.png"].content, ts[4], ts[4], 0o644),
    )
    snap4.add_file(
        _TrackedFile(fid_epoch, "src/epoch_test.py", _make_content("src/epoch_test.py", 0, seed), ts[4], ts[4], 0o644),
    )
    snap4.add_file(
        _TrackedFile(fid_deep, "a/b/c/d/e/f/g/deep.txt", _make_content("a/b/c/d/e/f/g/deep.txt", 0, seed), ts[4], ts[4], 0o644),
    )
    snap4.add_file(
        _TrackedFile(fid_corrupted_bin, "assets/corrupted.bin", null_content, ts[4], ts[4], 0o644),
    )
    # Restricted permissions file
    snap4.add_file(
        _TrackedFile(
            fid_secret,
            "restricted/secret.txt",
            _make_content("restricted/secret.txt", 0, seed),
            ts[4], ts[4], 0o000,
        ),
    )
    # cross_platform.py (LF) — will be renamed in S5
    snap4.add_file(
        _TrackedFile(
            fid_cross_plat,
            "src/cross_platform.py",
            _make_text("src/cross_platform.py", 0, seed),
            ts[4], ts[4], 0o644,
        ),
    )
    state.snapshots.append(snap4)
    if num_snapshots < 6:
        return state

    # S5: mixed endings — rename cross_platform.py -> cross_plat.py with CRLF
    snap5 = _SnapshotState()
    snap5.add_file(
        _TrackedFile(fid_readme, "README.md", _make_content("README.md", 1, seed), ts[5], ts[5], 0o644),
    )
    snap5.add_file(
        _TrackedFile(fid_main, "src/main.py", _make_content("src/main.py", 1, seed), ts[5], ts[5], 0o644),
    )
    snap5.add_file(
        _TrackedFile(fid_utils, "src/utils.py", _make_content("src/utils.py", 1, seed), ts[5], ts[5], 0o644),
    )
    snap5.add_file(
        _TrackedFile(fid_config, "data/config.json", _make_content("data/config.json", 1, seed), ts[5], ts[5], 0o644),
    )
    snap5.add_file(
        _TrackedFile(fid_logo, "assets/logo.png", snap0.files["assets/logo.png"].content, ts[5], ts[5], 0o644),
    )
    snap5.add_file(
        _TrackedFile(fid_epoch, "src/epoch_test.py", _make_content("src/epoch_test.py", 0, seed), ts[5], ts[5], 0o644),
    )
    snap5.add_file(
        _TrackedFile(fid_deep, "a/b/c/d/e/f/g/deep.txt", _make_content("a/b/c/d/e/f/g/deep.txt", 0, seed), ts[5], ts[5], 0o644),
    )
    snap5.add_file(
        _TrackedFile(fid_corrupted_bin, "assets/corrupted.bin", null_content, ts[5], ts[5], 0o644),
    )
    snap5.add_file(
        _TrackedFile(fid_secret, "restricted/secret.txt", _make_content("restricted/secret.txt", 0, seed), ts[5], ts[5], 0o000),
    )
    # Renamed + CRLF: same file_id, different path, CRLF content
    snap5.add_file(
        _TrackedFile(
            fid_cross_plat,
            "src/cross_plat.py",
            _make_text("src/cross_platform.py", 0, seed).replace(b"\n", b"\r\n"),
            ts[5], ts[5], 0o644,
        ),
    )
    state.snapshots.append(snap5)
    return state


# ── Photo curation generator ─────────────────────────────────────────────────


def _generate_photo_curation(
    num_snapshots: int,
    seed: int,
) -> _GeneratorState:
    """Photo backup workflow with import, curation, and album creation.

    Simulates a photo backup where files are imported into date-named
    folders, then cherry-picked into category albums with curated copies
    and original deletions.

    Snapshot timeline (7 snapshots, 14 logical files A-N):
      S0: Import 4 photos into 2024-03-15/          [A, B, C, D]
      S1: Import 2 more into 2024-04-20/             [+E, F]
      S2: Import 2 more into 2024-05-10/             [+G, H]
      S3: Create hotrod pics/, copy A,C,E,G into it  [+curated copies]
      S4: Import 4 more into 2024-06-01/             [+I, J, K, L]
      S5: Delete originals A,C,E,G from dated        [curated copies remain]
      S6: Create track day pics/, copy J,L, del ori  [+M, N (new)]
    """
    state = _GeneratorState()
    state.seed = seed
    state.scenario = "photo-curation"
    state.snapshot_names = [f"S{i}" for i in range(num_snapshots)]
    baseline = 1700000000.0
    state.timestamps = [
        _compute_timestamp(baseline, i, seed) for i in range(num_snapshots)
    ]
    state.source_paths = None  # single location
    state.project_roots = {"photo-library": ""}

    # ── Logical file IDs (14 files: A-N) ──────────────────────────────────
    fid_a = _make_fid("photo_A", seed)
    fid_b = _make_fid("photo_B", seed)
    fid_c = _make_fid("photo_C", seed)
    fid_d = _make_fid("photo_D", seed)
    fid_e = _make_fid("photo_E", seed)
    fid_f = _make_fid("photo_F", seed)
    fid_g = _make_fid("photo_G", seed)
    fid_h = _make_fid("photo_H", seed)
    fid_i = _make_fid("photo_I", seed)
    fid_j = _make_fid("photo_J", seed)
    fid_k = _make_fid("photo_K", seed)
    fid_l = _make_fid("photo_L", seed)
    fid_m = _make_fid("photo_M", seed)
    fid_n = _make_fid("photo_N", seed)

    # ── File paths (date-named folders, HHMMSS filenames) ─────────────────
    paths: dict[str, str] = {
        "A": "2024-03-15/143022.jpg",
        "B": "2024-03-15/153045.jpg",
        "C": "2024-03-15/161508.jpg",
        "D": "2024-03-15/173531.jpg",
        "E": "2024-04-20/092315.jpg",
        "F": "2024-04-20/103338.jpg",
        "G": "2024-05-10/114401.jpg",
        "H": "2024-05-10/130424.jpg",
        "I": "2024-06-01/140447.jpg",
        "J": "2024-06-01/150510.jpg",
        "K": "2024-06-01/160533.jpg",
        "L": "2024-06-01/170556.jpg",
        "M": "2024-07-04/080101.jpg",
        "N": "2024-07-04/091515.jpg",
    }

    fids: dict[str, str] = {
        "A": fid_a, "B": fid_b, "C": fid_c, "D": fid_d,
        "E": fid_e, "F": fid_f, "G": fid_g, "H": fid_h,
        "I": fid_i, "J": fid_j, "K": fid_k, "L": fid_l,
        "M": fid_m, "N": fid_n,
    }

    hotrod_prefix = "hotrod pics/"
    track_prefix = "track day pics/"

    # Pre-compute content for each logical file (version 0 — never modified)
    content: dict[str, bytes] = {}
    for label, path in paths.items():
        content[label] = _make_content(path, 0, seed)

    def _tf(
        label: str,
        path: str,
        ts_ns: int,
        *,
        curation_status: str | None = None,
    ) -> _TrackedFile:
        """Build a TrackedFile for a photo label."""
        return _TrackedFile(
            file_id=fids[label],
            path=path,
            content=content[label],
            mtime_ns=ts_ns,
            ctime_ns=ts_ns,
            mode=0o644,
            project_uid="photo-library",
            curation_status=curation_status,
        )

    def _curated_path(label: str, prefix: str) -> str:
        """Album path for a photo (basename only, in album folder)."""
        fname = paths[label].split("/")[1]
        return f"{prefix}{fname}"

    ts = [int(state.timestamps[i] * _NS_IN_SEC) for i in range(num_snapshots)]

    # ── S0: Import 4 photos into 2024-03-15/ ──────────────────────────────
    snap0 = _SnapshotState()
    for label in ("A", "B", "C", "D"):
        snap0.add_file(_tf(label, paths[label], ts[0]))
    state.snapshots.append(snap0)
    if num_snapshots < 2:
        return state

    # ── S1: Import 2 more into 2024-04-20/ ────────────────────────────────
    snap1 = _SnapshotState()
    for label in ("A", "B", "C", "D"):
        snap1.add_file(_tf(label, paths[label], ts[1]))
    for label in ("E", "F"):
        snap1.add_file(_tf(label, paths[label], ts[1]))
    state.snapshots.append(snap1)
    if num_snapshots < 3:
        return state

    # ── S2: Import 2 more into 2024-05-10/ ────────────────────────────────
    snap2 = _SnapshotState()
    for label in ("A", "B", "C", "D", "E", "F"):
        snap2.add_file(_tf(label, paths[label], ts[2]))
    for label in ("G", "H"):
        snap2.add_file(_tf(label, paths[label], ts[2]))
    state.snapshots.append(snap2)
    if num_snapshots < 4:
        return state

    # ── S3: Create hotrod pics/, copy A, C, E, G into it ──────────────────
    snap3 = _SnapshotState()
    for label in ("A", "B", "C", "D", "E", "F", "G", "H"):
        curation = "original" if label in ("A", "C", "E", "G") else None
        snap3.add_file(
            _tf(label, paths[label], ts[3], curation_status=curation),
        )
    for label in ("A", "C", "E", "G"):
        snap3.add_file(
            _tf(
                label, _curated_path(label, hotrod_prefix), ts[3],
                curation_status="curated",
            ),
        )
    state.snapshots.append(snap3)
    if num_snapshots < 5:
        return state

    # ── S4: Import 4 more into 2024-06-01/ ────────────────────────────────
    snap4 = _SnapshotState()
    for label in ("A", "B", "C", "D", "E", "F", "G", "H"):
        curation = "original" if label in ("A", "C", "E", "G") else None
        snap4.add_file(
            _tf(label, paths[label], ts[4], curation_status=curation),
        )
    for label in ("A", "C", "E", "G"):
        snap4.add_file(
            _tf(
                label, _curated_path(label, hotrod_prefix), ts[4],
                curation_status="curated",
            ),
        )
    for label in ("I", "J", "K", "L"):
        snap4.add_file(_tf(label, paths[label], ts[4]))
    state.snapshots.append(snap4)
    if num_snapshots < 6:
        return state

    # ── S5: Delete originals A, C, E, G from dated folders ────────────────
    snap5 = _SnapshotState()
    for label in ("A", "C", "E", "G"):
        snap5.add_file(
            _tf(
                label, _curated_path(label, hotrod_prefix), ts[5],
                curation_status="curated",
            ),
        )
    for label in ("B", "D", "F", "H"):
        snap5.add_file(_tf(label, paths[label], ts[5]))
    for label in ("I", "J", "K", "L"):
        snap5.add_file(_tf(label, paths[label], ts[5]))
    state.snapshots.append(snap5)
    if num_snapshots < 7:
        return state

    # ── S6: Create track day pics/, copy J,L. Delete J,L from dated +MM,NN─
    snap6 = _SnapshotState()
    for label in ("A", "C", "E", "G"):
        snap6.add_file(
            _tf(
                label, _curated_path(label, hotrod_prefix), ts[6],
                curation_status="curated",
            ),
        )
    for label in ("B", "D", "F", "H"):
        snap6.add_file(_tf(label, paths[label], ts[6]))
    for label in ("I", "K"):
        snap6.add_file(_tf(label, paths[label], ts[6]))
    for label in ("J", "L"):
        snap6.add_file(
            _tf(
                label, _curated_path(label, track_prefix), ts[6],
                curation_status="curated",
            ),
        )
    for label in ("M", "N"):
        snap6.add_file(_tf(label, paths[label], ts[6]))
    state.snapshots.append(snap6)
    return state


# ── Scenario dispatcher ──────────────────────────────────────────────────────

_GENERATORS: dict[str, Callable[[int, int], _GeneratorState]] = {
    "ordered-evolution": _generate_ordered_evolution,
    "unordered-timestamps": _generate_unordered_timestamps,
    "branching": _generate_branching,
    "flash-drive-chain": _generate_flash_drive_chain,
    "corruption-mix": _generate_corruption_mix,
    "photo-curation": _generate_photo_curation,
}


# ── Main generate entry point ────────────────────────────────────────────────


def generate(
    scenario: str,
    num_snapshots: int | None = None,
    seed: int = 42,
    output_dir: Path = Path("./gen_dummy_out"),
    *,
    manifests: bool = False,
) -> Path:
    """Generate fixture tree for a given scenario.

    Parameters
    ----------
    scenario:
        One of the registered scenario names.
    num_snapshots:
        Number of snapshots to generate (uses scenario default if None).
    seed:
        Random seed for deterministic content.
    output_dir:
        Destination directory for generated fixture tree.
    manifests:
        If True, write ``manifests/S{name}.json`` files consumable by
        ``ingest_manifest()``.

    Returns
    -------
    Resolved Path to the output directory.

    """
    if num_snapshots is None:
        num_snapshots = SCENARIO_DEFAULTS.get(scenario, 5)

    generator = _GENERATORS.get(scenario)
    if generator is None:
        msg = (
            f"Unknown scenario: {scenario}. "
            f"Available: {list(_GENERATORS.keys())}"
        )
        raise ValueError(msg)

    state = generator(num_snapshots, seed)

    output_dir = output_dir.resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write each snapshot
    for i, snap in enumerate(state.snapshots):
        snap_name = state.snapshot_names[i]
        snap_dir = output_dir / snap_name
        snap_dir.mkdir(parents=True, exist_ok=True)

        for path, tf in snap.files.items():
            _ = _write_file(
                snap_dir, path, tf.content, tf.mtime_ns, tf.ctime_ns, tf.mode,
            )

    # Write manifests (after all snapshots are written)
    if manifests:
        _write_manifests(state, output_dir)

    # Write ground truth
    ground_truth: dict[str, object] = _derive_ground_truth(state)
    _ = (output_dir / ".meta.json").write_text(
        json.dumps(ground_truth, indent=2),
    )

    return output_dir


# ── CLI entry point ──────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point: parse arguments and dispatch generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic FolderHistory test fixtures",
    )
    _ = parser.add_argument(
        "--scenario",
        "-s",
        choices=list(_GENERATORS.keys()),
        help="Scenario to generate (required unless --list-scenarios)",
    )
    _ = parser.add_argument(
        "--snapshots",
        "-n",
        type=int,
        default=None,
        help="Number of snapshots (scenario-specific default)",
    )
    _ = parser.add_argument(
        "--seed", "-d", type=int, default=42, help="Random seed",
    )
    _ = parser.add_argument(
        "--output",
        "-o",
        default="./gen_dummy_out",
        type=Path,
        help="Output directory",
    )
    _ = parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing output directory",
    )
    _ = parser.add_argument(
        "--manifests",
        "-m",
        action="store_true",
        help="Write manifests/S{name}.json files consumable by ingest_manifest()",
    )
    _ = parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit",
    )

    args = parser.parse_args()

    # ── Scalar extraction helpers (argparse reports Any) ─────────────────
    scenario: str = cast("str", args.scenario)
    output: Path = cast("Path", args.output)
    force: bool = cast("bool", args.force)
    snapshots: int | None = cast("int | None", args.snapshots)
    seed_val: int = cast("int", args.seed)
    manifests_flag: bool = cast("bool", args.manifests)
    list_scenarios: bool = cast("bool", args.list_scenarios)

    # ── list-scenarios mode ──────────────────────────────────────────────
    if list_scenarios:
        print("Available scenarios:")  # noqa: T201
        for name in sorted(_GENERATORS.keys()):
            default_n = SCENARIO_DEFAULTS.get(name, 5)
            print(f"  {name}  (default snapshots: {default_n})")  # noqa: T201
        return

    if not scenario:
        parser.error(
            "--scenario/-s is required (use --list-scenarios for available scenarios)",
        )

    if output.exists():
        if force:
            shutil.rmtree(output)
        else:
            print(  # noqa: T201
                f"Error: {output} already exists. Use --force to overwrite.",
                file=sys.stderr,
            )
            sys.exit(1)

    result = generate(
        scenario=scenario,
        num_snapshots=snapshots,
        seed=seed_val,
        output_dir=output,
        manifests=manifests_flag,
    )

    file_count = sum(1 for p in result.rglob("*") if p.is_file())
    print(f"Generated {file_count} files in {result}")  # noqa: T201


if __name__ == "__main__":
    main()


# ── Random scenario generator ──────────────────────────────────────────────


def _generate_random(
    num_snapshots: int,
    seed: int,
    *,
    file_count: int = 8,
    p_create: float = 0.3,
    p_modify: float = 0.3,
    p_delete: float = 0.15,
    p_rename: float = 0.1,
    p_corrupt_time: float = 0.05,
    p_corrupt_binary: float = 0.05,
    p_nesting: float = 0.03,
    p_permission_error: float = 0.02,
) -> _GeneratorState:
    """Generate random snapshot sequence from configurable operation probabilities.

    Parameters
    ----------
    num_snapshots: Number of snapshots to generate.
    seed: RNG seed for reproducibility.
    file_count: Number of base files in S0.
    p_create: Probability of creating a new file between snapshots.
    p_modify: Probability of modifying an existing file.
    p_delete: Probability of deleting a file.
    p_rename: Probability of renaming a file.
    p_corrupt_time: Probability of zeroing/setting epoch timestamps.
    p_corrupt_binary: Probability of injecting binary content.
    p_nesting: Probability of creating deeply nested directories.
    p_permission_error: Probability of setting 0o000 permissions.
    """
    rng = random.Random(seed)
    state = _GeneratorState()
    state.seed = seed
    state.scenario = "random"
    state.snapshot_names = [f"S{i}" for i in range(num_snapshots)]
    baseline = 1700000000.0
    state.timestamps = [
        _compute_timestamp(baseline, i, seed) for i in range(num_snapshots)
    ]

    # File identity tracking
    file_ids: dict[str, str] = {}  # logical name → file_id
    next_file_num: int = 0

    def _next_fid() -> str:
        nonlocal next_file_num
        name = f"random_file_{next_file_num}"
        next_file_num += 1
        return name

    # S0: Create initial random file set
    snap0 = _SnapshotState()
    paths_available: list[str] = []
    paths_in_snapshot: set[str] = set()
    file_names: list[str] = []

    in_dirs = ["src", "docs", "data", ""]
    for i in range(file_count):
        d = rng.choice(in_dirs)
        fname = f"{_random_word(rng)}.{rng.choice(['py','md','txt','json','toml','yaml'])}"
        path = f"{d}/{fname}" if d else fname
        fname_logical = _next_fid()
        fid = _make_fid(fname_logical, seed)
        file_ids[fname_logical] = fid
        content = _make_content(path, 0, seed)
        ts = int(state.timestamps[0] * _NS_IN_SEC)
        snap0.add_file(_TrackedFile(fid, path, content, ts, ts, 0o644))
        paths_available.append(path)
        paths_in_snapshot.add(path)
        file_names.append(fname_logical)
    state.snapshots.append(snap0)

    if num_snapshots < 2:
        return state

    # S1..SN: Randomly apply operations
    for i in range(1, num_snapshots):
        prev_snap = state.snapshots[-1]
        new_snap = _SnapshotState()
        ts = int(state.timestamps[i] * _NS_IN_SEC)

        # Roll to see what happens to each file in the current snapshot
        files_prev = list(prev_snap.files.items())
        rng.shuffle(files_prev)

        deleted: set[str] = set()
        renamed_map: dict[str, str] = {}  # old_path → new_path

        for path, tf in files_prev:
            roll = rng.random()

            if roll < p_delete and len(new_snap.files) > 2:
                # DELETE — file is gone
                deleted.add(path)
                continue

            elif roll < p_delete + p_rename:
                # RENAME — new path, same file_id, same content level
                base = Path(path).stem
                ext = Path(path).suffix
                parent = Path(path).parent
                new_name = f"{base}_{rng.choice(['v2','copy','moved','renamed'])}{ext}"
                new_path = str(parent / new_name) if str(parent) != "." else new_name
                # Make sure we haven't used this path yet
                attempt = 0
                while new_path in [p for p, _ in files_prev] or new_path in renamed_map.values() or new_path in [f.path for f in new_snap.files.values()]:
                    new_name = f"{base}_{rng.choice(['alt','backup','revised','variant','mirror'])}{ext}"
                    new_path = str(parent / new_name) if str(parent) != "." else new_name
                    attempt += 1
                    if attempt > 5:
                        new_path = path
                        break
                renamed_map[path] = new_path
                new_snap.add_file(_TrackedFile(tf.file_id, new_path, tf.content, ts, ts, tf.mode))
                continue

            elif roll < p_delete + p_rename + p_modify:
                # MODIFY — advance version (content evolves by snapshot index)
                version = i
                new_content = _make_content(path, version + i, seed)
                new_snap.add_file(_TrackedFile(tf.file_id, path, new_content, ts, ts, tf.mode))
                continue

            else:
                # UNCHANGED — keep as-is
                new_snap.add_file(tf)
                continue

        # Apply timestamps for deleted files (log their deletion)
        for d in deleted:
            _ = d  # deleted file — just not in new_snap

        # Apply renames (already handled above via renamed_map)

        # MAYBE CREATE one or more new files
        p_create_snapshot = p_create * rng.randint(1, 3)
        if rng.random() < p_create_snapshot:
            new_fname = _next_fid()
            new_fid = _make_fid(new_fname, seed)
            file_ids[new_fname] = new_fid
            d = rng.choice(in_dirs)
            ext = rng.choice(['py','md','txt','json'])
            fname = f"{_random_word(rng)}.{ext}"
            path = f"{d}/{fname}" if d else fname
            content = _make_content(path, 0, seed + i)
            new_snap.add_file(_TrackedFile(new_fid, path, content, ts, ts, 0o644))
            paths_available.append(path)

        # MAYBE corrupt timestamps (set to epoch)
        if rng.random() < p_corrupt_time:
            files_list = list(new_snap.files.items())
            if files_list:
                target_path, target_tf = rng.choice(files_list)
                new_snap.files[target_path] = _TrackedFile(
                    target_tf.file_id, target_path, target_tf.content, 0, ts, target_tf.mode,
                )

        # MAYBE corrupt with binary content
        if rng.random() < p_corrupt_binary:
            files_list = list(new_snap.files.items())
            if files_list:
                target_path, target_tf = rng.choice(files_list)
                if not target_tf.content.startswith(b"\x00"):
                    bin_content = bytes(rng.randint(0, 255) for _ in range(rng.randint(16, 64)))
                    new_snap.files[target_path] = _TrackedFile(
                        target_tf.file_id, target_path, bin_content, ts, ts, target_tf.mode,
                    )

        # MAYBE add deeply nested file
        if rng.random() < p_nesting:
            depth = rng.randint(3, 7)
            nested_path = "/".join([_random_word(rng) for _ in range(depth)]) + "/deep.txt"
            nested_fid = _make_fid(f"nested_{i}_{rng.randint(0,999)}", seed)
            content = _make_content(nested_path, 0, seed)
            new_snap.add_file(_TrackedFile(nested_fid, nested_path, content, ts, ts, 0o644))

        # MAYBE set permission error (0o000)
        if rng.random() < p_permission_error:
            files_list = list(new_snap.files.items())
            if files_list:
                target_path, target_tf = rng.choice(files_list)
                new_snap.files[target_path] = _TrackedFile(
                    target_tf.file_id, target_path, target_tf.content, ts, ts, 0o000,
                )

        state.snapshots.append(new_snap)

    return state


def _random_word(rng: random.Random) -> str:
    """Generate a random-ish filename component."""
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    length = rng.randint(3, 8)
    chars = []
    for i in range(length):
        chars.append(rng.choice(consonants if i % 2 == 0 else vowels))
    return "".join(chars)


# ── Register random scenario ───────────────────────────────────────────────

_GENERATORS["random"] = _generate_random
SCENARIO_DEFAULTS["random"] = 5
