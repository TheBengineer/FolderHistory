"""Unit tests for the FolderHistory dummy data generator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

_FIXTURES_DIR: str = str(Path(__file__).resolve().parent)
if _FIXTURES_DIR not in sys.path:
    sys.path.insert(0, _FIXTURES_DIR)

from gen_dummy_data import SCENARIO_DEFAULTS  # pyright: ignore[reportImplicitRelativeImport]

# ── Constants ────────────────────────────────────────────────────────────────

ALL_SCENARIOS: list[str] = sorted(SCENARIO_DEFAULTS.keys())

EXPECTED_SNAPSHOT_COUNTS: dict[str, int] = dict(SCENARIO_DEFAULTS)

REQUIRED_META_KEYS: set[str] = {
    "scenario",
    "seed",
    "snapshots",
    "expected_operations",
    "identity_clusters",
    "files",
}

SCRIPT_DIR: Path = Path(__file__).resolve().parent
SCRIPT_PATH: Path = SCRIPT_DIR / "gen_dummy_data.py"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run gen_dummy_data.py CLI with given arguments."""
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
    )
    return result


def _load_meta(output_dir: Path) -> dict[str, object]:
    """Load .meta.json as a typed dict."""
    meta_path: Path = output_dir / ".meta.json"
    assert meta_path.is_file(), f"{meta_path} not found"
    return cast("dict[str, object]", json.loads(meta_path.read_text()))


def _meta_str(meta: dict[str, object], key: str) -> str:
    val: object = meta[key]
    assert isinstance(val, str)
    return val


def _meta_int(meta: dict[str, object], key: str) -> int:
    val: object = meta[key]
    assert isinstance(val, int)
    return val


def _meta_dict(meta: dict[str, object], key: str) -> dict[str, object]:
    """Extract a nested dict, casting from the Any-narrowed type."""
    raw_val: object = meta[key]
    assert isinstance(raw_val, dict)
    return cast("dict[str, object]", raw_val)


def _meta_list_of_dicts(
    meta: dict[str, object], key: str,
) -> list[dict[str, object]]:
    """Extract a list-of-dicts, casting from the Any-narrowed type."""
    raw_val: object = meta[key]
    assert isinstance(raw_val, list)
    return cast("list[dict[str, object]]", raw_val)


def _compare_operations(
    ops_a: dict[str, object],
    ops_b: dict[str, object],
) -> None:
    """Order-independent comparison of expected_operations dicts."""
    assert ops_a.keys() == ops_b.keys()
    for tkey in ops_a:
        raw_a: object = ops_a[tkey]
        raw_b: object = ops_b[tkey]
        assert isinstance(raw_a, list)
        assert isinstance(raw_b, list)
        assert len(cast("list[object]", raw_a)) == len(cast("list[object]", raw_b))

        items_a: list[dict[str, object]] = cast(
            "list[dict[str, object]]", raw_a,
        )
        items_b: list[dict[str, object]] = cast(
            "list[dict[str, object]]", raw_b,
        )

        set_a: set[frozenset[tuple[str, object]]] = set()
        for op in items_a:
            set_a.add(frozenset(op.items()))
        set_b: set[frozenset[tuple[str, object]]] = set()
        for op in items_b:
            set_b.add(frozenset(op.items()))
        assert set_a == set_b, f"Operation set differs for {tkey}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Determinism
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Same seed + same args → identical output."""

    def test_same_seed_identical_snapshot_files(self, tmp_path: Path) -> None:
        """Two runs with same seed produce bit-identical snapshot files."""
        out1: Path = tmp_path / "run1"
        out2: Path = tmp_path / "run2"

        assert _run_cli(
            "-s", "ordered-evolution", "-n", "3", "-d", "42", "-o", str(out1),
        ).returncode == 0
        assert _run_cli(
            "-s", "ordered-evolution", "-n", "3", "-d", "42", "-o", str(out2),
        ).returncode == 0

        files1: set[Path] = {
            p.relative_to(out1) for p in out1.rglob("*")
            if p.is_file() and p.name != ".meta.json"
        }
        files2: set[Path] = {
            p.relative_to(out2) for p in out2.rglob("*")
            if p.is_file() and p.name != ".meta.json"
        }
        assert files1 == files2
        for rel in files1:
            assert (out1 / rel).read_bytes() == (out2 / rel).read_bytes()

    def test_same_seed_identical_meta_semantics(self, tmp_path: Path) -> None:
        """Two runs with same seed produce semantically identical .meta.json."""
        out1: Path = tmp_path / "meta1"
        out2: Path = tmp_path / "meta2"

        assert _run_cli(
            "-s", "ordered-evolution", "-n", "3", "-d", "42", "-o", str(out1),
        ).returncode == 0
        assert _run_cli(
            "-s", "ordered-evolution", "-n", "3", "-d", "42", "-o", str(out2),
        ).returncode == 0

        meta1: dict[str, object] = _load_meta(out1)
        meta2: dict[str, object] = _load_meta(out2)

        for key in ("scenario", "seed", "snapshots", "identity_clusters", "files"):
            assert meta1.get(key) == meta2.get(key)

        _compare_operations(
            _meta_dict(meta1, "expected_operations"),
            _meta_dict(meta2, "expected_operations"),
        )

    def test_different_seeds_different_content(self, tmp_path: Path) -> None:
        """Different seeds produce different raw_blake3 values."""
        out1: Path = tmp_path / "seed42"
        out2: Path = tmp_path / "seed99"

        assert _run_cli(
            "-s", "ordered-evolution", "-n", "3", "-d", "42", "-o", str(out1),
        ).returncode == 0
        assert _run_cli(
            "-s", "ordered-evolution", "-n", "3", "-d", "99", "-o", str(out2),
        ).returncode == 0

        meta1: dict[str, object] = _load_meta(out1)
        meta2: dict[str, object] = _load_meta(out2)

        files1_raw: dict[str, object] = _meta_dict(meta1, "files")
        files2_raw: dict[str, object] = _meta_dict(meta2, "files")

        hashes1: set[str] = set()
        for snap_val in files1_raw.values():
            assert isinstance(snap_val, dict)
            snap_dict_a: dict[str, object] = cast("dict[str, object]", snap_val)
            for info_val in snap_dict_a.values():
                assert isinstance(info_val, dict)
                info_dict_a: dict[str, object] = cast("dict[str, object]", info_val)
                h_a: object = info_dict_a.get("raw_blake3", "")
                assert isinstance(h_a, str)
                hashes1.add(h_a)

        hashes2: set[str] = set()
        for snap_val in files2_raw.values():
            assert isinstance(snap_val, dict)
            snap_dict_b: dict[str, object] = cast("dict[str, object]", snap_val)
            for info_val in snap_dict_b.values():
                assert isinstance(info_val, dict)
                info_dict_b: dict[str, object] = cast("dict[str, object]", info_val)
                h_b: object = info_dict_b.get("raw_blake3", "")
                assert isinstance(h_b, str)
                hashes2.add(h_b)

        assert hashes1 != hashes2


# ═══════════════════════════════════════════════════════════════════════════════
#  Scenario output correctness
# ═══════════════════════════════════════════════════════════════════════════════


class TestScenarioOutput:
    """Each scenario produces the correct snapshot structure."""

    @pytest.mark.parametrize("scenario", ALL_SCENARIOS)
    def test_snapshot_count(self, scenario: str, tmp_path: Path) -> None:
        """Each scenario produces expected number of snapshot directories."""
        out: Path = tmp_path / scenario
        assert _run_cli("-s", scenario, "-o", str(out)).returncode == 0

        snapshot_dirs: list[str] = sorted(
            d.name for d in out.iterdir() if d.is_dir()
        )
        expected: int = EXPECTED_SNAPSHOT_COUNTS[scenario]
        assert len(snapshot_dirs) == expected
        for i, name in enumerate(snapshot_dirs):
            assert name == f"S{i}"

    @pytest.mark.parametrize("scenario", ALL_SCENARIOS)
    def test_meta_json_present(self, scenario: str, tmp_path: Path) -> None:
        """Each scenario produces .meta.json with required top-level keys."""
        out: Path = tmp_path / scenario
        assert _run_cli("-s", scenario, "-o", str(out)).returncode == 0
        assert (out / ".meta.json").is_file()

        meta: dict[str, object] = _load_meta(out)
        for key in REQUIRED_META_KEYS:
            assert key in meta
        assert _meta_str(meta, "scenario") == scenario
        assert isinstance(_meta_int(meta, "seed"), int)

    @pytest.mark.parametrize("scenario", ALL_SCENARIOS)
    def test_snapshot_info_entries(self, scenario: str, tmp_path: Path) -> None:
        """Each snapshot entry has id/timestamp/source_path."""
        out: Path = tmp_path / scenario
        assert _run_cli("-s", scenario, "-o", str(out)).returncode == 0

        snapshots: list[dict[str, object]] = _meta_list_of_dicts(
            _load_meta(out), "snapshots",
        )
        for entry_obj in snapshots:
            for key in ("id", "timestamp", "source_path"):
                assert key in entry_obj

    @pytest.mark.parametrize("scenario", ALL_SCENARIOS)
    def test_snapshot_dirs_contain_files(self, scenario: str, tmp_path: Path) -> None:
        """Each snapshot directory contains actual files."""
        out: Path = tmp_path / scenario
        assert _run_cli("-s", scenario, "-o", str(out)).returncode == 0

        for snap_dir in out.iterdir():
            if not snap_dir.is_dir():
                continue
            files_count: int = sum(
                1 for p in snap_dir.rglob("*") if p.is_file()
            )
            assert files_count > 0

    def test_default_snapshot_count_matches_constant(self, tmp_path: Path) -> None:
        """generate() without --snapshots uses scenario default."""
        for scenario in ALL_SCENARIOS:
            out: Path = tmp_path / scenario
            assert _run_cli("-s", scenario, "-o", str(out)).returncode == 0

            expected: int = EXPECTED_SNAPSHOT_COUNTS[scenario]
            snapshots: list[dict[str, object]] = _meta_list_of_dicts(
                _load_meta(out), "snapshots",
            )
            assert len(snapshots) == expected

    def test_photo_curation_default_count(self) -> None:
        """Photo-curation scenario has default count of 7."""
        assert "photo-curation" in SCENARIO_DEFAULTS
        assert SCENARIO_DEFAULTS["photo-curation"] == 7


# ═══════════════════════════════════════════════════════════════════════════════
#  Ground truth correctness
# ═══════════════════════════════════════════════════════════════════════════════


class TestGroundTruth:
    """Derived ground truth matches expected scenario semantics."""

    def test_ordered_evolution_s0_to_s1_operations(self, tmp_path: Path) -> None:
        """S0->S1 has 2 modify operations: README.md and src/main.py."""
        out: Path = tmp_path / "oe_ops"
        assert _run_cli(
            "-s", "ordered-evolution", "-n", "2", "-d", "42", "-o", str(out),
        ).returncode == 0

        ops_container: dict[str, object] = _meta_dict(
            _load_meta(out), "expected_operations",
        )
        raw_s0s1: object = ops_container.get("S0->S1", [])
        assert isinstance(raw_s0s1, list)
        s0s1: list[dict[str, object]] = cast("list[dict[str, object]]", raw_s0s1)

        assert len(s0s1) == 2
        for op in s0s1:
            assert op.get("op_type") == "modify"

        paths: set[str] = set()
        for op in s0s1:
            sp_raw: object = op.get("source_path", "")
            assert isinstance(sp_raw, str)
            paths.add(sp_raw)
        assert "README.md" in paths
        assert "src/main.py" in paths

    def test_ordered_evolution_rename_identity(self, tmp_path: Path) -> None:
        """Renamed file (src/utils.py -> src/helpers.py) shares file_id."""
        out: Path = tmp_path / "oe_rename"
        assert _run_cli(
            "-s", "ordered-evolution", "-n", "4", "-d", "42", "-o", str(out),
        ).returncode == 0

        clusters: list[dict[str, object]] = _meta_list_of_dicts(
            _load_meta(out), "identity_clusters",
        )

        rename_cluster: dict[str, object] | None = None
        for cl_obj in clusters:
            raw_obs: object = cl_obj.get("observations", [])
            assert isinstance(raw_obs, list)
            obs_as_list: list[object] = cast("list[object]", raw_obs)
            obs_paths: list[str] = []
            for ob in obs_as_list:
                assert isinstance(ob, list)
                ob_list: list[object] = cast("list[object]", ob)
                assert len(ob_list) >= 2
                ob_path: object = ob_list[1]
                assert isinstance(ob_path, str)
                obs_paths.append(ob_path)
            if "src/utils.py" in obs_paths and "src/helpers.py" in obs_paths:
                rename_cluster = cl_obj
                break

        assert rename_cluster is not None

        raw_obs_final: object = rename_cluster.get("observations", [])
        assert isinstance(raw_obs_final, list)
        obs_final: list[object] = cast("list[object]", raw_obs_final)
        obs_snaps: set[str] = set()
        for ob in obs_final:
            assert isinstance(ob, list)
            ob_list_final: list[object] = cast("list[object]", ob)
            assert len(ob_list_final) >= 1
            obs_sid: object = ob_list_final[0]
            assert isinstance(obs_sid, str)
            obs_snaps.add(obs_sid)
        for i in range(4):
            assert f"S{i}" in obs_snaps

    def test_branching_merge_contains_all_files(self, tmp_path: Path) -> None:
        """Branching S3 has all files from S1 and S2 branches."""
        out: Path = tmp_path / "branch_merge"
        assert _run_cli("-s", "branching", "-o", str(out)).returncode == 0

        files_container: dict[str, object] = _meta_dict(
            _load_meta(out), "files",
        )

        def _snap_paths(snap_id: str) -> set[str]:
            raw_s: object = files_container.get(snap_id, {})
            assert isinstance(raw_s, dict)
            s_dict: dict[str, object] = cast("dict[str, object]", raw_s)
            result: set[str] = set()
            for k in s_dict:
                result.add(k)
            return result

        s3_paths: set[str] = _snap_paths("S3")
        s1_paths: set[str] = _snap_paths("S1")
        s2_paths: set[str] = _snap_paths("S2")
        assert s1_paths.issubset(s3_paths)
        assert s2_paths.issubset(s3_paths)

    def test_corruption_mix_epoch_mtime(self, tmp_path: Path) -> None:
        """S1 has mtime_ns=0 for README.md and src/epoch_test.py."""
        out: Path = tmp_path / "corrupt_time"
        assert _run_cli("-s", "corruption-mix", "-o", str(out)).returncode == 0

        files_container: dict[str, object] = _meta_dict(
            _load_meta(out), "files",
        )
        raw_s1: object = files_container.get("S1", {})
        assert isinstance(raw_s1, dict)
        s1_dict: dict[str, object] = cast("dict[str, object]", raw_s1)

        for fpath in ("README.md", "src/epoch_test.py"):
            raw_info: object = s1_dict.get(fpath, {})
            assert isinstance(raw_info, dict)
            info_dict: dict[str, object] = cast("dict[str, object]", raw_info)
            mtime: object = info_dict.get("mtime_ns", -1)
            assert mtime == 0

    def test_flash_drive_source_paths(self, tmp_path: Path) -> None:
        """Flash-drive-chain snapshots have expected source_path values."""
        out: Path = tmp_path / "flash_source"
        assert _run_cli("-s", "flash-drive-chain", "-o", str(out)).returncode == 0

        snapshots: list[dict[str, object]] = _meta_list_of_dicts(
            _load_meta(out), "snapshots",
        )
        expected: list[str] = ["laptop", "usb_drive", "usb_drive", "new_laptop"]
        for i, entry_obj in enumerate(snapshots):
            sp_raw: object = entry_obj.get("source_path", "")
            assert sp_raw == expected[i]


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI behavior
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLI:
    """CLI argument parsing and integration behavior."""

    def test_list_scenarios(self) -> None:
        """--list-scenarios outputs all 5 scenario names."""
        result: subprocess.CompletedProcess[str] = _run_cli("--list-scenarios")
        assert result.returncode == 0
        for scenario in ALL_SCENARIOS:
            assert scenario in result.stdout

    def test_generate_via_cli(self, tmp_path: Path) -> None:
        """Basic CLI generate produces output with snapshots and .meta.json."""
        out: Path = tmp_path / "cli_gen"
        result: subprocess.CompletedProcess[str] = _run_cli(
            "-s", "ordered-evolution", "-n", "2", "-d", "42", "-o", str(out),
        )
        assert result.returncode == 0
        assert out.is_dir()
        assert (out / ".meta.json").is_file()
        assert (out / "S0").is_dir()
        assert (out / "S1").is_dir()

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        """--force overwrites an existing output directory."""
        out: Path = tmp_path / "force_test"
        _ = out.mkdir()
        _ = (out / "placeholder.txt").write_text("garbage")

        result: subprocess.CompletedProcess[str] = _run_cli(
            "-s", "ordered-evolution", "-n", "2", "-o", str(out), "--force",
        )
        assert result.returncode == 0
        assert not (out / "placeholder.txt").exists()
        assert (out / ".meta.json").is_file()

    def test_no_force_exits_on_existing(self, tmp_path: Path) -> None:
        """Without --force, exiting when dir exists is an error."""
        out: Path = tmp_path / "no_force"
        assert _run_cli(
            "-s", "ordered-evolution", "-n", "2", "-o", str(out),
        ).returncode == 0

        result: subprocess.CompletedProcess[str] = _run_cli(
            "-s", "ordered-evolution", "-n", "2", "-o", str(out),
        )
        assert result.returncode != 0
        assert "already exists" in result.stderr

    def test_default_seed_cli(self, tmp_path: Path) -> None:
        """CLI without --seed uses seed 42 (identical to --seed 42)."""
        out_default: Path = tmp_path / "default"
        out_explicit: Path = tmp_path / "explicit"

        assert _run_cli(
            "-s", "ordered-evolution", "-n", "2", "-o", str(out_default),
        ).returncode == 0
        assert _run_cli(
            "-s", "ordered-evolution", "-n", "2", "-d", "42", "-o", str(out_explicit),
        ).returncode == 0

        meta_default: dict[str, object] = _load_meta(out_default)
        meta_explicit: dict[str, object] = _load_meta(out_explicit)

        assert _meta_int(meta_default, "seed") == 42
        assert _meta_int(meta_explicit, "seed") == 42

        for key in ("scenario", "snapshots", "identity_clusters", "files"):
            assert meta_default.get(key) == meta_explicit.get(key)

        _compare_operations(
            _meta_dict(meta_default, "expected_operations"),
            _meta_dict(meta_explicit, "expected_operations"),
        )

    def test_manifests_via_cli(self, tmp_path: Path) -> None:
        """--manifests creates manifests/S{name}.json files."""
        out: Path = tmp_path / "manifests_cli"
        assert _run_cli(
            "-s", "ordered-evolution", "-n", "2", "-o", str(out), "--manifests",
        ).returncode == 0

        manifests_dir: Path = out / "manifests"
        assert manifests_dir.is_dir()

        for snap in ("S0", "S1"):
            manifest_path: Path = manifests_dir / f"{snap}.json"
            assert manifest_path.is_file()

            manifest_data: dict[str, object] = cast(
                "dict[str, object]", json.loads(manifest_path.read_text()),
            )
            raw_files: object = manifest_data.get("files", [])
            assert isinstance(raw_files, list)
            assert len(cast("list[object]", raw_files)) > 0

    def test_invalid_scenario_exits(self, tmp_path: Path) -> None:
        """Unknown scenario name causes non-zero exit."""
        result: subprocess.CompletedProcess[str] = _run_cli(
            "-s", "nonexistent-scenario", "-n", "2", "-o", str(tmp_path / "bad"),
        )
        assert result.returncode != 0

    def test_missing_scenario_exits(self) -> None:
        """No --scenario and no --list-scenarios causes error exit."""
        result: subprocess.CompletedProcess[str] = _run_cli("-n", "2")
        assert result.returncode != 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Programmatic generate() API
# ═══════════════════════════════════════════════════════════════════════════════

from gen_dummy_data import generate as _generate  # pyright: ignore[reportImplicitRelativeImport]


class TestProgrammaticAPI:
    """Tests for the programmatic generate() entry point."""

    def test_generate_returns_resolved_path(self, tmp_path: Path) -> None:
        """generate() returns a resolved Path to the output directory."""
        out: Path = tmp_path / "prog_api"
        result: Path = _generate(
            "ordered-evolution", num_snapshots=2, seed=42, output_dir=out,
        )
        assert result == out.resolve()
        assert (result / ".meta.json").is_file()

    def test_generate_raises_on_unknown_scenario(self, tmp_path: Path) -> None:
        """generate() raises ValueError for unknown scenario."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            _ = _generate("nonexistent", output_dir=tmp_path / "bad")

    def test_generate_default_snapshot_count(self, tmp_path: Path) -> None:
        """generate() with num_snapshots=None uses scenario default."""
        out: Path = tmp_path / "default_n"
        _ = _generate("ordered-evolution", output_dir=out)
        snapshots: list[dict[str, object]] = _meta_list_of_dicts(
            _load_meta(out), "snapshots",
        )
        assert len(snapshots) == EXPECTED_SNAPSHOT_COUNTS["ordered-evolution"]

    def test_generate_default_seed(self, tmp_path: Path) -> None:
        """generate() without seed argument uses seed=42."""
        out_default: Path = tmp_path / "default_seed"
        out_explicit: Path = tmp_path / "explicit_seed"
        _ = _generate("ordered-evolution", num_snapshots=2, output_dir=out_default)
        _ = _generate(
            "ordered-evolution", num_snapshots=2, seed=42, output_dir=out_explicit,
        )
        assert _load_meta(out_default) == _load_meta(out_explicit)

    def test_generate_manifests(self, tmp_path: Path) -> None:
        """generate(..., manifests=True) creates manifests/S{name}.json."""
        out: Path = tmp_path / "prog_manifests"
        _ = _generate(
            "ordered-evolution", num_snapshots=2, output_dir=out, manifests=True,
        )
        manifests_dir: Path = out / "manifests"
        assert manifests_dir.is_dir()
        for snap in ("S0", "S1"):
            assert (manifests_dir / f"{snap}.json").is_file()

        out_no: Path = tmp_path / "no_manifests"
        _ = _generate("ordered-evolution", num_snapshots=2, output_dir=out_no)
        assert not (out_no / "manifests").exists()

    def test_generate_clears_existing(self, tmp_path: Path) -> None:
        """generate() clears existing output directory and rewrites it."""
        out: Path = tmp_path / "regen"
        _ = out.mkdir()
        _ = (out / "leftover.txt").write_text("should be gone")
        _ = _generate(
            "ordered-evolution", num_snapshots=2, seed=42, output_dir=out,
        )
        assert not (out / "leftover.txt").exists()
        assert (out / ".meta.json").is_file()
