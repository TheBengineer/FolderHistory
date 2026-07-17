"""Typer CLI for FolderHistory.

Provides ``analyze``, ``diff``, ``log``, and ``kb-status`` commands for
reconstructing and inspecting folder history from a collection of snapshots,
with Knowledge Base support for iterative feedback loops.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import typer

from folderhistory.core.diff import derive_operations
from folderhistory.core.dir_align import build_inverted_index, resolve_project_identities
from folderhistory.core.identity import assign_identities_exact
from folderhistory.core.ingest import ingest_manifest, ingest_snapshot
from folderhistory.core.timeline import Timeline, TimelineNode, build_timeline
from folderhistory.io.output import format_gitlog, format_json, format_jsonlines
from folderhistory.knowledge.json_kb import JSONKnowledgeBase
from folderhistory.knowledge.types import VersionMeta
from folderhistory.knowledge.update import compute_igs
from folderhistory.types import EditOperation, IdentityCluster, Snapshot


app = typer.Typer(
    name="folderhistory",
    help="Reconstruct git-like version history from folder backups.",
)

_DEFAULT_KB_PATH = Path.home() / ".folderhistory" / "kb.json"


def _resolve_kb_path(kb_path: Path | None) -> Path:
    """Resolve the KB file path, falling back to *~/.folderhistory/kb.json*."""
    return kb_path.resolve() if kb_path is not None else _DEFAULT_KB_PATH


def _kb_has_data(kb: JSONKnowledgeBase) -> bool:
    """Check whether the KB already contains persisted identity data."""
    return _count_kb_identities(kb) > 0


def _count_kb_identities(kb: JSONKnowledgeBase) -> int:
    """Number of identity entries in a JSONKnowledgeBase (internal inspection)."""
    import typing

    data: dict[str, object] = typing.cast("dict[str, object]", getattr(kb, "_data", {}))
    raw_val: object = data.get("identities", {})
    return len(typing.cast("dict[str, object]", raw_val)) if isinstance(raw_val, dict) else 0


def _count_kb_projects(kb: JSONKnowledgeBase) -> int:
    """Number of project entries in a JSONKnowledgeBase (internal inspection)."""
    import typing

    data: dict[str, object] = typing.cast("dict[str, object]", getattr(kb, "_data", {}))
    raw_val: object = data.get("projects", {})
    return len(typing.cast("dict[str, object]", raw_val)) if isinstance(raw_val, dict) else 0


def _apply_corrections(kb: JSONKnowledgeBase, corrections_path: Path) -> None:
    """Load a corrections JSON array and write each entry into the KB."""
    import json

    raw_text = corrections_path.read_text(encoding="utf-8")
    parsed: object = json.loads(raw_text)  # pyright: ignore[reportAny]
    if not isinstance(parsed, list):
        typer.echo("Corrections file must be a JSON array", err=True)
        raise typer.Exit(code=1)

    applied = 0
    parsed_list = cast("list[object]", parsed)
    for elem in parsed_list:
        if not isinstance(elem, dict):
            continue
        entry = cast("dict[str, object]", elem)
        hash_val: object = entry.get("hash", "")
        cluster_uid: object = entry.get("override_cluster_uid", "")
        if isinstance(hash_val, str) and isinstance(cluster_uid, str) and hash_val and cluster_uid:
            kb.set_identity(hash_val, cluster_uid, 1.0)
            applied += 1

    typer.echo(f"Applied {applied} correction(s) to KB")


# ── analyze ────────────────────────────────────────────────────────────────────


@app.command()
def analyze(
    snapshots_dir: Path = typer.Argument(  # type: ignore  [reportCallInDefaultInitializer]
        ...,
        help="Directory containing snapshot folders",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    output: Path = typer.Option(  # type: ignore  [reportCallInDefaultInitializer]
        "timeline.json",
        "--out",
        "-o",
        help="Output file path",
    ),
    output_format: str = typer.Option(  # type: ignore  [reportCallInDefaultInitializer]
        "json",
        "--format",
        "-f",
        help="Output format: json, gitlog, jsonlines",
    ),
    mode: str = typer.Option(  # type: ignore  [reportCallInDefaultInitializer]
        "auto",
        "--mode",
        "-m",
        help="Processing mode: full, delta, auto",
    ),
    apply: Path | None = typer.Option(  # type: ignore  [reportCallInDefaultInitializer]
        None,
        "--apply",
        help="Apply user corrections JSON file to identity assignment",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    kb_rollback: int | None = typer.Option(  # type: ignore  [reportCallInDefaultInitializer]
        None,
        "--kb-rollback",
        help="Restore KB version N before running",
    ),
    kb_path: Path | None = typer.Option(  # type: ignore  [reportCallInDefaultInitializer]
        None,
        "--kb-path",
        help="Custom KB file path",
    ),
    recursive: bool = typer.Option(  # type: ignore  [reportCallInDefaultInitializer]  # noqa: FBT001
        False,  # noqa: FBT003
        "--recursive",
        "-r",
        help="Recursively discover snapshots in subdirectories",
    ),
    max_depth: int | None = typer.Option(  # type: ignore  [reportCallInDefaultInitializer]
        None,
        "--max-depth",
        "-d",
        help="Maximum directory depth for recursive discovery (default: 1, -1 = infinite)",
    ),
) -> None:
    """Analyze snapshots and reconstruct folder history.

    If --apply corrections.json: load user corrections and apply to identity assignment.
    If --kb-rollback N: restore KB version N before running.
    If --mode full: force full reprocessing (ignore KB).
    If --mode delta: only process unmatched hashes via KB.
    If --mode auto (default): auto-detect based on KB state.
    """
    snapshots_dir_resolved = snapshots_dir.resolve()

    # ── KB initialisation ──────────────────────────────────────────────
    kb_path_resolved = _resolve_kb_path(kb_path)

    # Rollback (honoured regardless of mode)
    if kb_rollback is not None:
        try:
            JSONKnowledgeBase.rollback(kb_path_resolved, kb_rollback)
            typer.echo(f"KB rolled back to version {kb_rollback}")
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(f"KB rollback failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    # Open KB when needed for corrections or identity seeding
    use_kb = mode != "full" or apply is not None or kb_rollback is not None
    kb: JSONKnowledgeBase | None = None

    if use_kb:
        kb = JSONKnowledgeBase(kb_path_resolved)
        kb.open()

        if apply is not None:
            _apply_corrections(kb, apply)

        # Auto mode: close KB if it has no data and no corrections applied
        if mode == "auto" and not _kb_has_data(kb):
            kb.close()
            kb = None

    # ── Snapshot processing ────────────────────────────────────────────
    from folderhistory.core.discovery import discover_snapshots

    # Determine effective depth:
    #   --recursive without --max-depth → infinite (-1)
    #   --max-depth N → N
    #   default (no flags) → 1 (flat)
    if recursive and max_depth is None:
        effective_depth = -1
    elif max_depth is not None:
        effective_depth = max_depth
    else:
        effective_depth = 1

    discovered = discover_snapshots(snapshots_dir_resolved, max_depth=effective_depth)
    if not discovered:
        typer.echo(
            "No snapshot directories found",
            err=True,
        )
        raise typer.Exit(code=1)

    snapshots: list[Snapshot] = []
    for sd in discovered:
        s = ingest_snapshot(sd.path, snapshot_id=sd.relative_id)
        snapshots.append(s)
    snapshots.sort(key=lambda s: s.id)

    # Group snapshots by project
    inverted_index = build_inverted_index(snapshots)
    project_groups = resolve_project_identities(
        snapshots, inverted_index, kb=kb, mode=mode,
    )

    # Run identity per project group
    all_identities: dict[str, IdentityCluster] = {}
    for project_uid, snap_ids in project_groups.items():
        group_snaps = [s for s in snapshots if s.id in snap_ids]
        if not group_snaps:
            continue
        group_identities = assign_identities_exact(
            group_snaps, kb=kb, project_uid=project_uid,
        )
        all_identities.update(group_identities)

    identities = all_identities
    operations = derive_operations(snapshots, identities)
    timeline = build_timeline(snapshots, operations)
    output_text = _format_timeline(timeline, output_format)
    _ = output.write_text(output_text)
    typer.echo(f"Timeline written to {output}")

    # ── Clean-up ───────────────────────────────────────────────────────
    if kb is not None:
        kb.close()


# ── diff ───────────────────────────────────────────────────────────────────────


@app.command()
def diff(
    before: Path = typer.Option(  # type: ignore  [reportCallInDefaultInitializer]
        ...,
        "--before",
        "-b",
        help="Before snapshot (directory or manifest.json)",
        exists=True,
        readable=True,
    ),
    after: Path = typer.Option(  # type: ignore  [reportCallInDefaultInitializer]
        ...,
        "--after",
        "-a",
        help="After snapshot (directory or manifest.json)",
        exists=True,
        readable=True,
    ),
    output_format: str = typer.Option(  # type: ignore  [reportCallInDefaultInitializer]
        "json",
        "--format",
        "-f",
        help="Output format: json, gitlog, jsonlines",
    ),
) -> None:
    snap_before = _load_snapshot(before)
    snap_after = _load_snapshot(after)
    identities = assign_identities_exact([snap_before, snap_after])
    ops_list = derive_operations([snap_before, snap_after], identities)
    pair_ops = ops_list[0] if ops_list else []

    if pair_ops:
        timeline = build_timeline([snap_before, snap_after], [pair_ops])
    else:
        timeline = build_timeline([snap_before, snap_after], [[]])

    output_text = _format_timeline(timeline, output_format)
    _ = typer.echo(output_text)


# ── log ────────────────────────────────────────────────────────────────────────


@app.command()
def log(
    manifest: Path = typer.Argument(  # type: ignore  [reportCallInDefaultInitializer]
        ...,
        help="Path to timeline JSON file (output of 'analyze')",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    import orjson

    raw = cast("dict[str, object]", orjson.loads(manifest.read_bytes()))
    timeline = _parse_timeline_json(raw)
    _ = typer.echo(format_gitlog(timeline))


# ── kb-status ────────────────────────────────────────────────────────────────


@app.command(name="kb-status")
def kb_status(
    kb_path: Path | None = typer.Option(  # type: ignore  [reportCallInDefaultInitializer]
        None,
        "--kb-path",
        help="Custom KB file path",
    ),
) -> None:
    """Display Knowledge Base status: version, run count, identity/project/contradiction counts, IGS."""
    kb_path_resolved = _resolve_kb_path(kb_path)

    if not kb_path_resolved.exists():
        typer.echo(f"Knowledge Base not found at {kb_path_resolved}")
        raise typer.Exit(code=1)

    kb = JSONKnowledgeBase(kb_path_resolved)
    kb.open()

    try:
        version_meta = cast(VersionMeta, kb.version())
        contradictions = kb.get_contradictions()
        # Suppress internal IGS warning — the value is printed below
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            igs = compute_igs(kb)

        identity_count = _count_kb_identities(kb)
        project_count = _count_kb_projects(kb)

        typer.echo(f"KB path:    {kb_path_resolved}")
        typer.echo(f"Schema v:   {version_meta.schema_version}")
        typer.echo(f"Algorithm:  {version_meta.algorithm_version}")
        typer.echo(f"Last run:   {version_meta.last_run_id or '(never)'}")
        typer.echo(f"Run count:  {version_meta.run_count}")
        typer.echo(f"Identities: {identity_count}")
        typer.echo(f"Projects:   {project_count}")
        typer.echo(f"Contradict: {len(contradictions)}")
        typer.echo(f"IGS:        {igs:.4f}")
    finally:
        kb.close()


def main() -> None:
    app()


# ── Internal helpers ───────────────────────────────────────────────────────────


def _format_timeline(timeline: Timeline, fmt: str) -> str:
    if fmt == "gitlog":
        return format_gitlog(timeline)
    if fmt == "jsonlines":
        return format_jsonlines(timeline)
    if fmt == "working-copy":
        from folderhistory.io.working_copy import export_working_copy
        import tempfile
        td = tempfile.mkdtemp(prefix="fh_wc_")
        result = export_working_copy(timeline, {}, Path(td))
        return f"Working copy exported to {result}"
    return format_json(timeline)


def _load_snapshot(path: Path) -> Snapshot:
    if path.is_dir():
        return ingest_snapshot(path)
    return ingest_manifest(path)


# ── JSON deserialisation (log command) ──────────────────────────────────────────


def _parse_timeline_json(raw: dict[str, object]) -> Timeline:
    import typing

    root_changes_raw: list[dict[str, str]] = []
    rc_raw: list[object] = typing.cast("list[object]", raw.get("root_changes", []))
    for item in rc_raw:
        d: dict[str, object] = typing.cast("dict[str, object]", item)
        root_changes_raw.append(
            {
                "snapshot_id": str(d.get("snapshot_id", "")),
                "from_root": str(d.get("from_root", "")),
                "to_root": str(d.get("to_root", "")),
            }
        )

    nodes: list[TimelineNode] = []
    ns_raw: list[object] = typing.cast("list[object]", raw.get("nodes", []))
    for item in ns_raw:
        n: dict[str, object] = typing.cast("dict[str, object]", item)
        node = _parse_node_dict(n)
        if node is not None:
            nodes.append(node)

    return Timeline(nodes=nodes, root_changes=root_changes_raw)


def _parse_node_dict(d: dict[str, object]) -> TimelineNode | None:
    import typing

    ops_list: list[EditOperation] = []
    ops_raw: list[object] = typing.cast(
        "list[object]",
        d.get("operations", []),
    )
    for item in ops_raw:
        op_dict: dict[str, object] = typing.cast("dict[str, object]", item)
        op = _parse_op_dict(op_dict)
        if op is not None:
            ops_list.append(op)

    raw_ts = d.get("timestamp")
    timestamp_val: float | None = None
    if raw_ts is not None:
        timestamp_val = float(typing.cast("float | int", raw_ts))

    raw_pid = d.get("parent_id")
    parent_id_val: str | None = str(raw_pid) if raw_pid is not None else None

    raw_rc = d.get("root_change")
    root_change_val: str | None = str(raw_rc) if raw_rc is not None else None

    raw_sid = d.get("snapshot_id")
    snapshot_id_val = str(raw_sid) if raw_sid is not None else ""

    raw_sp = d.get("source_path")
    source_path_val = str(raw_sp) if raw_sp is not None else ""

    return TimelineNode(
        snapshot_id=snapshot_id_val,
        timestamp=timestamp_val,
        source_path=source_path_val,
        operations=ops_list,
        parent_id=parent_id_val,
        root_change=root_change_val,
    )


def _parse_op_dict(d: dict[str, object]) -> EditOperation | None:
    import typing
    from typing import Literal

    raw_optype = d.get("op_type")
    op_type_str = str(raw_optype) if raw_optype is not None else "modify"

    raw_sp = d.get("source_path")
    source_path_val: str | None = str(raw_sp) if raw_sp is not None else None

    raw_tp = d.get("target_path")
    target_path_val: str | None = str(raw_tp) if raw_tp is not None else None

    raw_oh = d.get("old_hash")
    old_hash_val: str | None = str(raw_oh) if raw_oh is not None else None

    raw_nh = d.get("new_hash")
    new_hash_val: str | None = str(raw_nh) if raw_nh is not None else None

    raw_conf = d.get("confidence")
    confidence_val: float = 1.0
    if raw_conf is not None:
        confidence_val = float(typing.cast("float | int", raw_conf))

    raw_fid = d.get("file_id")
    file_id_val = str(raw_fid) if raw_fid is not None else ""

    return EditOperation(
        op_type=typing.cast(
            Literal["create", "delete", "modify", "rename", "move", "copy"],
            op_type_str,
        ),
        file_id=file_id_val,
        source_path=source_path_val,
        target_path=target_path_val,
        old_hash=old_hash_val,
        new_hash=new_hash_val,
        confidence=confidence_val,
    )


if __name__ == "__main__":
    main()
