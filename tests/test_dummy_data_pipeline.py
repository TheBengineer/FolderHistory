"""Integration tests: full FolderHistory pipeline on generated dummy data.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from collections.abc import Callable
from typing import cast

import pytest

from folderhistory.core.diff import derive_operations
from folderhistory.core.identity import assign_identities_exact
from folderhistory.core.ingest import ingest_snapshot
from folderhistory.core.timeline import build_timeline
from folderhistory.types import EditOperation, IdentityCluster, Snapshot

# Make tests/fixtures importable (no __init__.py in tests/).
_fixtures_dir = Path(__file__).resolve().parent / "fixtures"
if str(_fixtures_dir) not in sys.path:
    sys.path.insert(0, str(_fixtures_dir))

# Import gen_dummy_data via importlib so the type checker doesn't need
# to resolve a sys.path-based module at analysis time.
_gen_module = importlib.import_module("gen_dummy_data")

# Re-export with a typed wrapper so callers see ``Path`` return.
_generate_fn: Callable[..., Path] = cast(
    "Callable[..., Path]",
    _gen_module.generate,
)

def generate(
    scenario: str,
    num_snapshots: int | None = None,
    seed: int = 42,
    output_dir: Path = Path("./gen_dummy_out"),
) -> Path:
    """Wrap ``gen_dummy_data.generate()`` with a typed signature."""
    return _generate_fn(
        scenario=scenario,
        num_snapshots=num_snapshots,
        seed=seed,
        output_dir=output_dir,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _ingest_snapshots(output_dir: Path) -> list[Snapshot]:
    """Ingest all ``S{name}`` snapshot directories and return them sorted by id."""
    snaps: list[Snapshot] = []
    for child in sorted(output_dir.iterdir()):
        if child.is_dir() and child.name.startswith("S"):
            snaps.append(ingest_snapshot(child))
    snaps.sort(key=lambda s: s.id)
    return snaps


def _pipeline_op_key(op: EditOperation) -> tuple[str, str | None, str | None]:
    """Normalise a pipeline ``EditOperation`` to a comparable tuple."""
    return (op.op_type, op.source_path, op.target_path)


def _gt_op_key(op: dict[str, object]) -> tuple[str, str | None, str | None]:
    """Normalise a ground-truth operation dict to a comparable tuple."""
    op_type = cast(str, op["op_type"])
    source_path = cast("str | None", op.get("source_path"))
    target_path = cast("str | None", op.get("target_path"))
    return (op_type, source_path, target_path)


def _find_cluster_by_observation(
    clusters: dict[str, IdentityCluster],
    snap_id: str,
    path: str,
) -> IdentityCluster | None:
    """Find the first cluster containing the given ``(snap_id, path)`` observation."""
    for c in clusters.values():
        for sid, p in c.observations:
            if sid == snap_id and p == path:
                return c
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Test class
# ═══════════════════════════════════════════════════════════════════════════════


class TestPipelineOnDummyData:
    """Run full FolderHistory pipeline on generated dummy data."""

    # ── Scenario parametrisation ──────────────────────────────────────────────

    @pytest.mark.parametrize("scenario,count", [
        ("ordered-evolution", 3),
        ("unordered-timestamps", 4),
        ("branching", 4),
    ])
    def test_pipeline_produces_operations(
        self,
        scenario: str,
        count: int,
        tmp_path: Path,
    ) -> None:
        """Generate data, run full pipeline, verify operations exist."""
        output_dir = generate(
            scenario=scenario,
            num_snapshots=count,
            seed=42,
            output_dir=tmp_path / "data",
        )

        # Read ground truth
        meta: dict[str, object] = cast(
            "dict[str, object]",
            json.loads((output_dir / ".meta.json").read_text()),
        )
        assert cast(str, meta["scenario"]) == scenario

        # Ingest all snapshots
        snapshots = _ingest_snapshots(output_dir)
        assert len(snapshots) == count

        # Run identity and diff
        identities = assign_identities_exact(snapshots)
        operations = derive_operations(snapshots, identities)

        # Verify basic invariants
        assert len(operations) > 0
        assert len(operations) == len(snapshots) - 1

        # Every operation has a valid type
        valid_types = {"create", "delete", "modify", "rename", "move", "copy"}
        for pair_ops in operations:
            for op in pair_ops:
                assert op.op_type in valid_types, (
                    f"Unknown op_type: {op.op_type}"
                )
                assert op.confidence > 0.0

        # Timeline builds without error
        timeline = build_timeline(snapshots, operations)
        assert len(timeline.nodes) == len(snapshots)
        assert timeline.nodes[0].parent_id is None

    # ── Ground-truth operation comparison ────────────────────────────────────

    def test_ground_truth_operations_match(self, tmp_path: Path) -> None:
        """For ``ordered-evolution`` with 2 snapshots, pipeline ops match ground truth."""
        output_dir = generate(
            scenario="ordered-evolution",
            num_snapshots=2,
            seed=42,
            output_dir=tmp_path / "data",
        )

        meta: dict[str, object] = cast(
            "dict[str, object]",
            json.loads((output_dir / ".meta.json").read_text()),
        )

        # Ingest S0 and S1
        snap0 = ingest_snapshot(output_dir / "S0")
        snap1 = ingest_snapshot(output_dir / "S1")
        snapshots = [snap0, snap1]
        snapshots.sort(key=lambda s: s.id)

        # Run pipeline
        identities = assign_identities_exact(snapshots)
        ops_list = derive_operations(snapshots, identities)

        # 2 snapshots → 1 pair
        assert len(ops_list) == 1
        pipeline_ops = ops_list[0]

        # Ground truth for S0→S1
        expected_ops: dict[str, object] = cast(
            "dict[str, object]",
            meta["expected_operations"],
        )
        gt_ops: list[dict[str, object]] = cast(
            "list[dict[str, object]]",
            expected_ops["S0->S1"],
        )

        # Normalise both sides to (op_type, source_path, target_path) sets
        pipeline_tuples = {_pipeline_op_key(op) for op in pipeline_ops}
        gt_tuples = {_gt_op_key(op) for op in gt_ops}

        assert pipeline_tuples == gt_tuples, (
            f"Pipeline ops {pipeline_tuples} differ from ground truth {gt_tuples}"
        )

        # ── Also verify no-op files are truly absent from the diff ─────────
        # .gitignore and src/utils.py are unchanged between S0 and S1
        pipeline_paths = {op.source_path for op in pipeline_ops if op.source_path}
        unchanged = {".gitignore", "src/utils.py"}
        assert unchanged.isdisjoint(pipeline_paths), (
            f"Unchanged files {unchanged} unexpectedly appear in operations"
        )

    # ── Identity cluster comparison ──────────────────────────────────────────

    def test_identity_clusters_match_ground_truth(self, tmp_path: Path) -> None:
        """Pipeline identity clusters have the same observations as ground truth."""
        output_dir = generate(
            scenario="ordered-evolution",
            num_snapshots=3,
            seed=42,
            output_dir=tmp_path / "data",
        )

        meta: dict[str, object] = cast(
            "dict[str, object]",
            json.loads((output_dir / ".meta.json").read_text()),
        )
        snapshots = _ingest_snapshots(output_dir)

        # Run identity
        identities = assign_identities_exact(snapshots)

        # For every ground-truth cluster, find a pipeline cluster with the
        # exact same set of (snap_id, path) observations.
        gt_clusters: list[dict[str, object]] = cast(
            "list[dict[str, object]]",
            meta["identity_clusters"],
        )

        matched: int = 0
        for gt_cluster in gt_clusters:
            gt_obs_raw: list[list[str]] = cast(
                "list[list[str]]",
                gt_cluster["observations"],
            )
            # Convert each [snap_id, path] list to a (snap_id, path) tuple.
            gt_obs: set[tuple[str, str]] = {(o[0], o[1]) for o in gt_obs_raw}

            for pipe_cluster in identities.values():
                pipe_obs = set(pipe_cluster.observations)
                if pipe_obs == gt_obs:
                    matched += 1
                    break
            else:
                pytest.fail(
                    f"Pipeline missing cluster for observations: {gt_obs}",
                )

        # Every ground-truth cluster should have a match
        assert matched == len(gt_clusters), (
            f"Matched {matched}/{len(gt_clusters)} ground-truth clusters"
        )

        # ── Cross-check: every pipeline cluster should have a corresponding
        # ground-truth cluster (i.e. no spurious clusters).
        for pipe_cluster in identities.values():
            pipe_obs = set(pipe_cluster.observations)
            found = any(
                {(o[0], o[1]) for o in cast("list[list[str]]", gc["observations"])} == pipe_obs
                for gc in gt_clusters
            )
            assert found, (
                f"Pipeline has spurious cluster with observations: {pipe_obs}"
            )

    # ── Corruption scenario ──────────────────────────────────────────────────

    def test_pipeline_handles_corruption_scenario(self, tmp_path: Path) -> None:
        """Run full pipeline on ``corruption-mix`` — should not crash.

        This scenario includes binary files, epoch timestamps, deep nesting,
        restricted permissions, and CRLF-rename combinations.
        """
        output_dir = generate(
            scenario="corruption-mix",
            seed=42,
            output_dir=tmp_path / "data",
        )

        snapshots = _ingest_snapshots(output_dir)

        # Pipeline should complete without exceptions
        identities = assign_identities_exact(snapshots)
        ops_list = derive_operations(snapshots, identities)
        timeline = build_timeline(snapshots, ops_list)

        # Basic structural invariants
        assert len(ops_list) == len(snapshots) - 1
        assert len(timeline.nodes) == len(snapshots)

        # All operations should have valid types
        for pair_ops in ops_list:
            for op in pair_ops:
                assert op.op_type in {"create", "delete", "modify", "rename", "move", "copy"}
                assert op.confidence > 0.0

        # ── Specific edge-case checks ─────────────────────────────────────

        # Binary files (with null bytes) should have identity clusters.
        logo_cluster = _find_cluster_by_observation(identities, "S0", "assets/logo.png")
        assert logo_cluster is not None, (
            "Missing identity cluster for binary file assets/logo.png"
        )
        corrupted_bin_cluster = _find_cluster_by_observation(
            identities, "S3", "assets/corrupted.bin",
        )
        assert corrupted_bin_cluster is not None, (
            "Missing identity cluster for assets/corrupted.bin"
        )

        # Deeply nested path should have an identity.
        deep_cluster = _find_cluster_by_observation(identities, "S2", "a/b/c/d/e/f/g/deep.txt")
        assert deep_cluster is not None, (
            "Missing identity cluster for a/b/c/d/e/f/g/deep.txt"
        )

        # Files with mtime=0 (epoch) should still be ingestible.
        epoch_cluster = _find_cluster_by_observation(identities, "S1", "src/epoch_test.py")
        assert epoch_cluster is not None, (
            "Missing identity cluster for src/epoch_test.py (mtime=0)"
        )

        # NOTE: restricted/secret.txt has mode 0o000 — ingest_snapshot
        # correctly skips it with a PermissionError warning.  This is
        # expected corruption behaviour, not a bug.

    # ── Flash-drive chain identity ───────────────────────────────────────────

    def test_flash_drive_chain_cross_location_identity(self, tmp_path: Path) -> None:
        """``flash-drive-chain``: identity bridges ctime reset and CRLF changes.

        The scenario simulates a USB backup chain:
          S0 (laptop):    LF line endings, original ctimes
          S1 (USB copy):  ctime reset, main.py becomes CRLF
          S2 (USB week2): main.py content modified (CRLF stays)
          S3 (new laptop): all ctime reset again, content same as S2

        ``assign_identities_exact`` uses path-based identity keys, so every
        file at the same path across snapshots should be in the same cluster
        — even when raw_blake3 differs (CRLF conversion) or ctime is reset.
        """
        output_dir = generate(
            scenario="flash-drive-chain",
            num_snapshots=4,
            seed=42,
            output_dir=tmp_path / "data",
        )

        snapshots = _ingest_snapshots(output_dir)
        assert len(snapshots) == 4

        identities = assign_identities_exact(snapshots)

        # ── src/main.py bridges all 4 snapshots ──────────────────────────
        # S0: LF content, S1: CRLF content (different raw hash!)
        # Pipeline path-based identity should still bridge them.
        main_cluster = _find_cluster_by_observation(identities, "S0", "src/main.py")
        assert main_cluster is not None, (
            "Missing identity cluster for src/main.py"
        )

        main_sids = {sid for sid, _ in main_cluster.observations}
        assert main_sids == {"S0", "S1", "S2", "S3"}, (
            f"src/main.py should span all 4 snapshots, got {main_sids}"
        )

        # ── README.md bridges all 4 snapshots (same content, LF throughout) ──
        readme_cluster = _find_cluster_by_observation(identities, "S0", "README.md")
        assert readme_cluster is not None, (
            "Missing identity cluster for README.md"
        )

        readme_sids = {sid for sid, _ in readme_cluster.observations}
        assert readme_sids == {"S0", "S1", "S2", "S3"}, (
            f"README.md should span all 4 snapshots, got {readme_sids}"
        )

        # ── src/utils.py bridges all 4 snapshots ─────────────────────────
        utils_cluster = _find_cluster_by_observation(identities, "S0", "src/utils.py")
        assert utils_cluster is not None, (
            "Missing identity cluster for src/utils.py"
        )

        utils_sids = {sid for sid, _ in utils_cluster.observations}
        assert utils_sids == {"S0", "S1", "S2", "S3"}, (
            f"src/utils.py should span all 4 snapshots, got {utils_sids}"
        )

        # ── Full pipeline runs without error ─────────────────────────────
        ops_list = derive_operations(snapshots, identities)
        timeline = build_timeline(snapshots, ops_list)

        assert len(ops_list) == 3  # 4 snapshots → 3 pairs
        assert len(timeline.nodes) == 4

        # S0→S1 should detect a modify on src/main.py (LF→CRLF changes
        # raw_blake3 at the same path → modify).
        s0_s1_ops = ops_list[0]
        main_modifies = [
            op for op in s0_s1_ops
            if op.source_path == "src/main.py" and op.op_type == "modify"
        ]
        assert len(main_modifies) == 1, (
            "Expected 1 modify on src/main.py between S0 and S1 (LF→CRLF), "
            f"got {len(main_modifies)}: {[op.op_type for op in s0_s1_ops]}"
        )

    # ── Photo-curation pipeline ────────────────────────────────────────────

    def test_photo_curation_pipeline(self, tmp_path: Path) -> None:
        """Full pipeline on photo-curation scenario — verify copy ops and identity bridging."""
        output_dir = generate(
            scenario="photo-curation",
            num_snapshots=7,
            seed=42,
            output_dir=tmp_path / "data",
        )

        # Ingest all snapshots
        snapshots = _ingest_snapshots(output_dir)

        # Run identity and diff
        identities = assign_identities_exact(snapshots)
        operations = derive_operations(snapshots, identities)

        # Verify basic invariants
        assert len(operations) > 0
        assert len(operations) == len(snapshots) - 1

        # Verify: copy operations exist (from pipeline diff, not just ground truth)
        all_ops = [op for pair in operations for op in pair]
        copy_ops = [op for op in all_ops if op.op_type == "copy"]
        assert len(copy_ops) >= 1, (
            f"Expected copy ops, got {len(copy_ops)}"
        )

        # Verify: identity clusters bridge curated files
        # (same file_id across snapshots where content is identical)
        for cluster in identities.values():
            if len(cluster.observations) >= 2:
                paths = {p for _, p in cluster.observations}
                # Check that some clusters span both dated and curated paths
                if any("hotrod" in p for p in paths) and any("2024" in p for p in paths):
                    break
        else:
            pytest.fail("No identity cluster spans both dated and curated directories")

    def test_photo_curation_ground_truth(self, tmp_path: Path) -> None:
        """Verify photo-curation ground truth contains curation fields."""
        output_dir = generate(
            scenario="photo-curation",
            num_snapshots=7,
            seed=42,
            output_dir=tmp_path / "data",
        )

        meta: dict[str, object] = cast(
            "dict[str, object]",
            json.loads((output_dir / ".meta.json").read_text()),
        )

        # Check project_roots
        assert "project_roots" in meta

        # Check curation_status and project_uid in files metadata
        has_curated: bool = False
        has_project_uid: bool = False
        files_container: dict[str, object] = cast(
            "dict[str, object]", meta.get("files", {}),
        )
        for _snap_id, snap_files_raw in files_container.items():
            snap_files: dict[str, object] = cast("dict[str, object]", snap_files_raw)
            for _path, file_info_raw in snap_files.items():
                file_info: dict[str, object] = cast("dict[str, object]", file_info_raw)
                if file_info.get("curation_status") == "curated":
                    has_curated = True
                if file_info.get("project_uid"):
                    has_project_uid = True

        assert has_curated, "No curated files found in ground truth"
        assert has_project_uid, "No project_uid found in ground truth"

        # Check curated_observations in identity clusters
        identity_clusters_raw: object = meta.get("identity_clusters", [])
        assert isinstance(identity_clusters_raw, list)
        identity_clusters: list[dict[str, object]] = cast(
            "list[dict[str, object]]", identity_clusters_raw,
        )
        has_curated_obs: bool = any(
            c.get("curated_observations") for c in identity_clusters
        )
        assert has_curated_obs, "No curated_observations in identity clusters"
