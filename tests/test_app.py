"""Tests for ``folderhistory.app`` — CLI commands and flag parsing."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from folderhistory.app import app

runner = CliRunner()


# ── analyze --help ─────────────────────────────────────────────────────────────


class TestAnalyzeHelp:
    """Verify that ``analyze --help`` reports every new flag."""

    def test_help_exit_code(self) -> None:
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0

    def test_help_shows_mode_flag(self) -> None:
        result = runner.invoke(app, ["analyze", "--help"])
        assert "--mode" in result.stdout

    def test_help_shows_apply_flag(self) -> None:
        result = runner.invoke(app, ["analyze", "--help"])
        assert "--apply" in result.stdout

    def test_help_shows_kb_rollback_flag(self) -> None:
        result = runner.invoke(app, ["analyze", "--help"])
        assert "--kb-rollback" in result.stdout

    def test_help_shows_kb_path_flag(self) -> None:
        result = runner.invoke(app, ["analyze", "--help"])
        assert "--kb-path" in result.stdout


# ── kb-status subcommand ──────────────────────────────────────────────────────


class TestKbStatus:
    """Verify the ``kb-status`` subcommand is registered and functional."""

    def test_subcommand_registered(self) -> None:
        """--help output mentions kb-status as an available command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "kb-status" in result.stdout

    def test_help_exit_code(self) -> None:
        result = runner.invoke(app, ["kb-status", "--help"])
        assert result.exit_code == 0

    def test_help_shows_kb_path_flag(self) -> None:
        result = runner.invoke(app, ["kb-status", "--help"])
        assert "--kb-path" in result.stdout

    def test_missing_kb_reports_error(self, tmp_path: Path) -> None:
        """Running kb-status on a non-existent KB file reports an error."""
        missing = tmp_path / "nope" / "kb.json"
        result = runner.invoke(app, ["kb-status", "--kb-path", str(missing)])
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_kb_status_with_empty_kb(self, tmp_path: Path) -> None:
        """A freshly created KB shows zero counts but does not crash."""
        kb_file = tmp_path / "kb.json"
        _ = kb_file.write_text(
            '{"version":{"schema_version":1,"algorithm_version":"1.0.0"}'
            + ',"identities":{},"projects":{},"frequencies":{}'
            + ',"contradictions":[],"history":[]}'
        )
        result = runner.invoke(app, ["kb-status", "--kb-path", str(kb_file)])
        assert result.exit_code == 0
        assert "IGS" in result.stdout
        assert "Identities: 0" in result.stdout
        assert "Contradict: 0" in result.stdout

    def test_kb_status_with_data(self, tmp_path: Path) -> None:
        """A KB with identities and contradictions shows expected counts."""
        kb_file = tmp_path / "kb.json"
        _ = kb_file.write_text(
            '{"version":{"schema_version":1,"algorithm_version":"1.0.0"'
            + ',"run_count":3}'
            + ',"identities":{"h1":{"cluster_uid":"c1","confidence":0.95}}'
            + ',"projects":{"p1":{"fingerprint":{"h1":0.5}}}'
            + ',"contradictions":[{"hash":"h1","identity_a":"c1","identity_b":"c2"'
            + ',"strategy_applied":"CONFIDENCE_WEIGHT"}]'
            + ',"frequencies":{"h1":5},"history":[]}'
        )
        result = runner.invoke(app, ["kb-status", "--kb-path", str(kb_file)])
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert "Identities: 1" in result.stdout
        assert "Projects:   1" in result.stdout
        assert "Contradict: 1" in result.stdout
        assert "Run count:  3" in result.stdout
        assert "IGS:" in result.stdout


# ── analyze --mode parsing ────────────────────────────────────────────────────


class TestAnalyzeMode:
    """``--mode`` accepts *full*, *delta*, and *auto*."""

    def test_mode_full(self, tmp_path: Path) -> None:
        """--mode full parses and is accepted (fails on missing snapshots, err to stderr)."""
        result = runner.invoke(app, ["analyze", str(tmp_path), "--mode", "full"])
        assert result.exit_code != 0
        assert "No snapshot directories found" in result.stderr

    def test_mode_delta(self, tmp_path: Path) -> None:
        """--mode delta parses and is accepted."""
        result = runner.invoke(app, ["analyze", str(tmp_path), "--mode", "delta"])
        assert result.exit_code != 0
        assert "No snapshot directories found" in result.stderr

    def test_mode_auto(self, tmp_path: Path) -> None:
        """--mode auto parses and is accepted (default)."""
        result = runner.invoke(app, ["analyze", str(tmp_path), "--mode", "auto"])
        assert result.exit_code != 0
        assert "No snapshot directories found" in result.stderr

    def test_default_mode_is_auto(self, tmp_path: Path) -> None:
        """Calling analyze without --mode defaults to auto behaviour."""
        result = runner.invoke(app, ["analyze", str(tmp_path)])
        assert result.exit_code != 0

    def test_mode_with_kb_path(self, tmp_path: Path) -> None:
        """--mode full combined with --kb-path parses without error."""
        kb_file = tmp_path / "kb.json"
        result = runner.invoke(
            app,
            ["analyze", str(tmp_path), "--mode", "full", "--kb-path", str(kb_file)],
        )
        assert result.exit_code != 0
        assert "No snapshot directories found" in result.stderr


# ── analyze --apply parsing ───────────────────────────────────────────────────


class TestAnalyzeApply:
    """``--apply`` accepts a corrections JSON file path."""

    def test_apply_with_valid_json(self, tmp_path: Path) -> None:
        """--apply accepts a well-formed corrections file."""
        corr_file = tmp_path / "corrections.json"
        _ = corr_file.write_text("[]")
        result = runner.invoke(
            app,
            ["analyze", str(tmp_path), "--apply", str(corr_file)],
        )
        # Corrections message goes to stdout, snapshots error goes to stderr
        assert "Applied 0 correction(s) to KB" in result.stdout
        assert "No snapshot directories found" in result.stderr

    def test_apply_rejects_nonexistent_file(self, tmp_path: Path) -> None:
        """--apply with a missing file causes a parse error (Typer validates exists=True)."""
        result = runner.invoke(
            app,
            ["analyze", str(tmp_path), "--apply", str(tmp_path / "missing.json")],
        )
        assert result.exit_code == 2  # Typer/Click exits with code 2 on bad param
        assert "does not exist" in result.output or "Error" in result.output

    def test_apply_with_delta_mode(self, tmp_path: Path) -> None:
        """--apply combined with --mode delta parses correctly."""
        corr_file = tmp_path / "corrections.json"
        _ = corr_file.write_text("[]")
        result = runner.invoke(
            app,
            [
                "analyze",
                str(tmp_path),
                "--mode",
                "delta",
                "--apply",
                str(corr_file),
                "--kb-path",
                str(tmp_path / "kb.json"),
            ],
        )
        assert "Applied 0 correction(s) to KB" in result.stdout
        assert "No snapshot directories found" in result.stderr


# ── analyze --kb-rollback parsing ─────────────────────────────────────────────


class TestAnalyzeKbRollback:
    """``--kb-rollback`` accepts an integer."""

    def test_kb_rollback_accepts_integer(self, tmp_path: Path) -> None:
        """--kb-rollback accepts an integer argument."""
        kb_file = tmp_path / "kb.json"
        result = runner.invoke(
            app,
            ["analyze", str(tmp_path), "--kb-rollback", "1", "--kb-path", str(kb_file)],
        )
        assert result.exit_code == 1
        assert "KB rollback failed" in result.stderr

    def test_kb_rollback_rejects_negative(self, tmp_path: Path) -> None:
        """--kb-rollback with a negative number is rejected by the KB layer."""
        kb_file = tmp_path / "kb.json"
        result = runner.invoke(
            app,
            [
                "analyze",
                str(tmp_path),
                "--kb-rollback",
                "-1",
                "--kb-path",
                str(kb_file),
            ],
        )
        assert result.exit_code != 0
        assert "KB rollback failed" in result.stderr or "Error" in result.output

    def test_kb_rollback_with_mode_full(self, tmp_path: Path) -> None:
        """--kb-rollback works with --mode full (rollback is honoured before mode check)."""
        kb_file = tmp_path / "kb.json"
        result = runner.invoke(
            app,
            [
                "analyze",
                str(tmp_path),
                "--mode",
                "full",
                "--kb-rollback",
                "1",
                "--kb-path",
                str(kb_file),
            ],
        )
        assert result.exit_code == 1
        assert "KB rollback failed" in result.stderr


# ── analyze --recursive / --max-depth ─────────────────────────────────────────


class TestAnalyzeRecursive:
    """Recursive snapshot discovery via ``--recursive`` and ``--max-depth`` flags."""

    def test_flat_mode_backward_compatible(self, tmp_path: Path) -> None:
        """Without any flags, ``analyze`` discovers only immediate subdirectories (flat mode)."""
        (tmp_path / "v1").mkdir()
        _ = (tmp_path / "v1" / "f.txt").write_text("data")
        (tmp_path / "v2").mkdir()
        _ = (tmp_path / "v2" / "f.txt").write_text("data")

        out = tmp_path / "out.json"
        result = runner.invoke(app, ["analyze", str(tmp_path), "--out", str(out)])
        assert result.exit_code == 0
        assert out.exists()

        import typing
        import orjson

        raw = typing.cast("dict[str, object]", orjson.loads(out.read_bytes()))
        raw_nodes = typing.cast("list[object]", raw.get("nodes", []))
        assert len(raw_nodes) == 2
        node_ids = {
            typing.cast("str", typing.cast("dict[str, object]", n)["snapshot_id"])
            for n in raw_nodes
        }
        assert node_ids == {"v1", "v2"}

    def test_recursive_discovers_nested(self, tmp_path: Path) -> None:
        """``--recursive`` discovers snapshots inside nested subdirectories."""
        proj_v1 = tmp_path / "project_v1"
        proj_v1.mkdir()
        _ = (proj_v1 / "file.txt").write_text("data")

        proj_v0 = tmp_path / "archive" / "project_v0"
        proj_v0.mkdir(parents=True)
        _ = (proj_v0 / "old.txt").write_text("data")

        out = tmp_path / "out.json"
        result = runner.invoke(
            app, ["analyze", str(tmp_path), "--recursive", "--out", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()

        import typing
        import orjson

        raw = typing.cast("dict[str, object]", orjson.loads(out.read_bytes()))
        raw_nodes = typing.cast("list[object]", raw.get("nodes", []))
        snapshot_ids = {
            typing.cast("str", typing.cast("dict[str, object]", n)["snapshot_id"])
            for n in raw_nodes
        }
        assert snapshot_ids == {"project_v1", "archive/project_v0"}

    def test_max_depth_limits(self, tmp_path: Path) -> None:
        """``--max-depth 1`` uses flat discovery, returning all immediate subdirs."""
        proj_v1 = tmp_path / "project_v1"
        proj_v1.mkdir()
        _ = (proj_v1 / "file.txt").write_text("data")

        archive = tmp_path / "archive" / "deep"
        archive.mkdir(parents=True)
        _ = (archive / "old.txt").write_text("data")

        out = tmp_path / "out.json"
        result = runner.invoke(
            app,
            [
                "analyze", str(tmp_path),
                "--recursive", "--max-depth", "1",
                "--out", str(out),
            ],
        )
        assert result.exit_code == 0
        assert out.exists()

        import typing
        import orjson

        raw = typing.cast("dict[str, object]", orjson.loads(out.read_bytes()))
        raw_nodes = typing.cast("list[object]", raw.get("nodes", []))
        snapshot_ids = {
            typing.cast("str", typing.cast("dict[str, object]", n)["snapshot_id"])
            for n in raw_nodes
        }
        # max-depth=1 is flat mode — returns all immediate subdirs including containers
        assert snapshot_ids == {"archive", "project_v1"}

    def test_recursive_produces_valid_timeline(self, tmp_path: Path) -> None:
        """Full analyze pipeline with ``--recursive`` produces valid JSON timeline."""
        proj_v1 = tmp_path / "project_v1"
        proj_v1.mkdir()
        _ = (proj_v1 / "file.txt").write_text("data")

        proj_v0 = tmp_path / "archive" / "project_v0"
        proj_v0.mkdir(parents=True)
        _ = (proj_v0 / "old.txt").write_text("data")

        out = tmp_path / "out.json"
        result = runner.invoke(
            app, ["analyze", str(tmp_path), "--recursive", "--out", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()

        import typing
        import orjson

        raw = typing.cast("dict[str, object]", orjson.loads(out.read_bytes()))
        raw_nodes = typing.cast("list[object]", raw.get("nodes", []))
        assert len(raw_nodes) == 2
        node_ids = {
            typing.cast("str", typing.cast("dict[str, object]", n)["snapshot_id"])
            for n in raw_nodes
        }
        assert "project_v1" in node_ids
        assert "archive/project_v0" in node_ids
        for raw_node in raw_nodes:
            node = typing.cast("dict[str, object]", raw_node)
            sid = typing.cast("str", node["snapshot_id"])
            assert isinstance(sid, str) and sid
            ops = typing.cast("list[object]", node.get("operations", []))
            assert isinstance(ops, list)

    def test_recursive_no_snapshots(self, tmp_path: Path) -> None:
        """``--recursive`` on a tree with no snapshots reports an appropriate error."""
        (tmp_path / "empty_sub" / "deeper").mkdir(parents=True)
        result = runner.invoke(app, ["analyze", str(tmp_path), "--recursive"])
        assert result.exit_code == 1
        assert "No snapshot directories found" in result.stderr
