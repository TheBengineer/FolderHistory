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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from typing import cast

# ── Constants ────────────────────────────────────────────────────────────────

SCENARIO_DEFAULTS: dict[str, int] = {
    "ordered-evolution": 5,
    "unordered-timestamps": 5,
    "branching": 4,
    "flash-drive-chain": 5,
    "corruption-mix": 6,
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


# ── Content utilities ────────────────────────────────────────────────────────


def _make_content(path: str, version: int, seed: int) -> bytes:
    """Produce deterministic content via SHA256 hex digest."""
    h = hashlib.sha256(f"{path}:v{version}:{seed}".encode()).hexdigest()
    return h.encode()


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


# ── Ground truth derivation (private helpers) ────────────────────────────────


def _build_snapshot_info(state: _GeneratorState) -> list[dict[str, str | float]]:
    """Build snapshot metadata list from generator state."""
    info: list[dict[str, str | float]] = []
    for i, name in enumerate(state.snapshot_names):
        info.append({
            "id": name,
            "timestamp": state.timestamps[i],
            "source_path": name,
        })
    return info


def _build_identity_clusters(state: _GeneratorState) -> list[dict[str, object]]:
    """Build identity clusters from file_id observations across snapshots."""
    file_id_observations: dict[str, list[tuple[str, str]]] = {}
    for i, snap in enumerate(state.snapshots):
        snap_id = state.snapshot_names[i]
        for path, tf in snap.files.items():
            file_id_observations.setdefault(tf.file_id, []).append(
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
        clusters.append({  # type: ignore[reportUnknownMemberType]
            "uid": fid,
            "observations": observations,
            "canonical_path": canonical,
            "confidence": 1.0,
        })
    return clusters


def _build_operations(state: _GeneratorState) -> dict[str, list[dict[str, object]]]:
    """Derive edit operations between consecutive snapshot pairs."""
    operations: dict[str, list[dict[str, object]]] = {}
    for i in range(len(state.snapshots) - 1):
        snap_i = state.snapshot_names[i]
        snap_j = state.snapshot_names[i + 1]
        key = f"{snap_i}->{snap_j}"

        ops: list[dict[str, object]] = []

        files_i = {tf.file_id: tf for tf in state.snapshots[i].files.values()}
        files_j = {
            tf.file_id: tf for tf in state.snapshots[i + 1].files.values()
        }

        ids_i = set(files_i.keys())
        ids_j = set(files_j.keys())

        # Same file_id in both snapshots
        for fid in ids_i & ids_j:
            tf_i = files_i[fid]
            tf_j = files_j[fid]
            if tf_i.path == tf_j.path:
                if tf_i.content != tf_j.content:
                    ops.append({  # type: ignore[reportUnknownMemberType]
                        "op_type": "modify",
                        "file_id": fid,
                        "source_path": tf_i.path,
                        "target_path": tf_j.path,
                        "confidence": 1.0,
                    })
            else:
                ops.append({  # type: ignore[reportUnknownMemberType]
                    "op_type": "rename",
                    "file_id": fid,
                    "source_path": tf_i.path,
                    "target_path": tf_j.path,
                    "confidence": 1.0,
                })

        # Deleted files
        for fid in ids_i - ids_j:
            tf = files_i[fid]
            ops.append({  # type: ignore[reportUnknownMemberType]
                "op_type": "delete",
                "file_id": fid,
                "source_path": tf.path,
                "target_path": None,
                "confidence": 1.0,
            })

        # Created files
        for fid in ids_j - ids_i:
            tf = files_j[fid]
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
    }


# ── Shared stub helper ───────────────────────────────────────────────────────


def _make_stub_state(
    scenario: str,
    num_snapshots: int,
    seed: int,
) -> _GeneratorState:
    """Create minimal generator state for stub scenarios.

    Each snapshot contains a single placeholder file whose content
    changes deterministically with each version.  Every snapshot
    gets a unique file_id so ground-truth derivation exercises the
    create/delete path.
    """
    state = _GeneratorState()
    state.seed = seed
    state.scenario = scenario
    state.snapshot_names = [f"S{i}" for i in range(num_snapshots)]
    baseline = 1700000000.0
    state.timestamps = [
        _compute_timestamp(baseline, i, seed) for i in range(num_snapshots)
    ]
    rng = random.Random(seed)  # noqa: S311
    for i in range(num_snapshots):
        snap = _SnapshotState()
        content = _make_content("README.md", i, seed)
        snap.add_file(
            _TrackedFile(
                file_id=str(uuid.UUID(int=rng.getrandbits(128))),
                path="README.md",
                content=content,
                mtime_ns=int(state.timestamps[i] * _NS_IN_SEC),
                ctime_ns=int(state.timestamps[i] * _NS_IN_SEC),
                mode=0o644,
            ),
        )
        state.snapshots.append(snap)
    return state


# ── Scenario generators (stubs) ──────────────────────────────────────────────


def _generate_ordered_evolution(
    num_snapshots: int,
    seed: int,
) -> _GeneratorState:
    """Evolve files cleanly with clear create/modify/delete patterns.

    TODO: Implement full scenario logic with multi-file evolution.
    """
    return _make_stub_state("ordered-evolution", num_snapshots, seed)


def _generate_unordered_timestamps(
    num_snapshots: int,
    seed: int,
) -> _GeneratorState:
    """Produce snapshots with timestamps that don't match logical order.

    TODO: Implement full scenario logic with shuffled timestamps
    and content that conflicts with temporal ordering.
    """
    state = _make_stub_state("unordered-timestamps", num_snapshots, seed)
    rng = random.Random(seed)  # noqa: S311
    rng.shuffle(state.timestamps)
    return state


def _generate_branching(
    num_snapshots: int,
    seed: int,
) -> _GeneratorState:
    """Create branching history where snapshots diverge.

    TODO: Implement full scenario logic with a fork point and
    independently evolving branches.
    """
    return _make_stub_state("branching", num_snapshots, seed)


def _generate_flash_drive_chain(
    num_snapshots: int,
    seed: int,
) -> _GeneratorState:
    """Simulate flash drive backup chain with ctime resets.

    TODO: Implement full scenario logic with ctime-reset patterns
    and source_path changes across snapshots.
    """
    return _make_stub_state("flash-drive-chain", num_snapshots, seed)


def _generate_corruption_mix(
    num_snapshots: int,
    seed: int,
) -> _GeneratorState:
    """Corrupt files over time with various damage patterns.

    TODO: Implement full scenario logic with bit flips, truncations,
    zeroed content, and partial writes.
    """
    return _make_stub_state("corruption-mix", num_snapshots, seed)


# ── Scenario dispatcher ──────────────────────────────────────────────────────

_GENERATORS: dict[str, Callable[[int, int], _GeneratorState]] = {
    "ordered-evolution": _generate_ordered_evolution,
    "unordered-timestamps": _generate_unordered_timestamps,
    "branching": _generate_branching,
    "flash-drive-chain": _generate_flash_drive_chain,
    "corruption-mix": _generate_corruption_mix,
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
        If True, write a .manifest.json in each snapshot directory.

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

        # Optional per-snapshot manifest
        if manifests:
            manifest: dict[str, dict[str, object]] = {}
            for path, tf in snap.files.items():
                manifest[path] = {
                    "file_id": tf.file_id,
                    "size": len(tf.content),
                    "mtime_ns": tf.mtime_ns,
                    "ctime_ns": tf.ctime_ns,
                    "mode": tf.mode,
                }
            _ = (snap_dir / ".manifest.json").write_text(
                json.dumps(manifest, indent=2),
            )

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
        help="Write .manifest.json in each snapshot directory",
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
