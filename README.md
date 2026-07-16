# FolderHistory

**Reconstruct git-like version history from messy folder backups — using only file metadata.**

When you have a pile of backup snapshots of a directory — taken at different times, possibly overlapping, with timestamps that may be unreliable — FolderHistory reconstructs the most likely sequence of changes: which files were created, modified, moved, renamed, or deleted, and in what order.

## Quick Start

```bash
# Install
pip install folderhistory

# Point at a directory of backup snapshots
folderhistory analyze ~/backups/ --out history.json

# View as git-style log
folderhistory log history.json

# Export as browsable working copy
python -m folderhistory.app analyze ~/backups/ --format working-copy --out ./project

# Or export as real git repo
python -m folderhistory.app analyze ~/backups/ --out timeline.json
# (then use the git export module programmatically)
```

## The Problem

You back up a project folder manually, via cron, or with a cloud sync tool. Over months or years you accumulate N near-identical copies of the same directory. Each snapshot has a directory tree with file metadata — but that metadata is flawed:

- **ctime** resets when copied between filesystems
- **mtime** may be preserved or stripped depending on the tool
- **Filenames** change when files are renamed or moved
- **Content** drifts as files are edited across devices
- **Snapshots** may be incomplete if backups were interrupted

FolderHistory solves this by treating it as a **global file-identity assignment problem**: figure out which files across different snapshots represent the same logical entity, despite name changes, moves, and metadata corruption.

## Pipeline

```
Backup snapshots (directories)
        │
        ▼
┌──────────────────┐
│    Ingestion      │  Walk directories, compute BLAKE3 hashes (raw + normalized),
│                   │  detect line endings, normalize paths (NFC)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Dir Alignment   │  Inverted hash index → IDF-weighted Jaccard → find project
│                   │  roots across differently-located snapshots
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Identity        │  Union-Find over (project_uid, relative_path) keys.
│  Assignment      │  Content hash + path matching. Optional KB pre-seed.
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Diff            │  Derive operations: create, delete, modify, rename,
│                   │  move, copy — with confidence scores
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Consistency     │  Transitive closure check, cross-location validation,
│  Check           │  inner loop refinement (max 2 iterations)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Timeline        │  Linear or branching DAG with root-change annotations
│  Output          │  → JSON / gitlog / Working Copy / git repos
└──────────────────┘
```

## Features

### Ingestion
- **Raw + normalized BLAKE3 hashing**: Detects identical files despite CRLF/LF differences
- **Line-ending detection**: Classifies files as LF/CRLF/mixed/binary
- **NFC path normalization**: Cross-platform path matching
- **Streaming hashing**: Handles files >50MB without loading into memory
- **Symlink tracking**: Records symlinks without following them
- **Permission errors**: Gracefully skips unreadable files

### Identity Assignment
- **Exact matching**: Content hash + path via Union-Find
- **Multi-signal fusion**: Content, path (Jaro-Winkler), and metadata similarity
- **Size-blocked variant**: Blocks by size → xxhash64 → BLAKE3 for efficiency
- **Cross-location detection**: Finds project subtrees across different root directories

### Operation Derivation
- 6 operation types: `create`, `delete`, `modify`, `rename`, `move`, `copy`
- Rename vs. move classification (basename change vs. parent directory change)
- Normalized-hash fallback for cross-platform content identity
- Full confidence tracking per operation

### Timeline Output
- **Linear timeline**: Ordered snapshot chain with parent/child links
- **Branching DAG**: Fork/merge support for non-linear histories
- **Root-change annotations**: Detects when project location changed
- **Git export**: Real git repos with commit history + metadata notes
- **Working Copy export**: Browsable directory trees with incremental snapshot storage

### Knowledge Base & Feedback Loop
- **Cross-run state persistence**: Identities and fingerprints cached between runs
- **JSON Knowledge Base**: Zero-dependency KV store in `.folderhistory/`
- **Contradiction resolution**: 4 strategies (merge/rename, split, confidence-weight, evidence-weight)
- **IGS metric**: Identity Graph Stability tracking
- **Inner loop**: Up to 2 iterations of consistency threshold relaxation

### Cross-Location Project Detection
- **Content-probe subtree location**: Finds project roots inside larger snapshots
- **IDF-weighted Jaccard**: Correctly discounts shared dependencies (node_modules, etc.)
- **Root-relative path normalization**: Identity keys survive location changes

## Output Formats

| Format | Command | Description |
|--------|---------|-------------|
| JSON | `--format json` | Machine-readable timeline with operations and root changes |
| Git log | `--format gitlog` | Human-readable commit-style history |
| Working Copy | `--format working-copy` | `Content/` + `Snapshots/dated/` directory tree |
| Git repos | (via `git_export.py`) | Real git repositories with metadata via `refs/notes/fh/*` |

### Working Copy Format

```
<output>/
├── timeline.json              # Root manifest
├── Content/                   # BLAKE3-deduplicated file store
│   └── <prefix>/<hash>        # One file per unique content
└── Snapshots/
    ├── latest → <id>/         # Convenience symlink
    ├── S0/
    │   ├── CHANGES.txt        # Machine JSON + human-readable summary
    │   ├── src/main.py → ../../Content/ab/abc...  # Symlinks into Content/
    │   └── .fh-deleted/       # Deleted file content preserved
    ├── S1/
    │   └── ...
    └── ...
```

### Git Export (programmatic)

```python
from folderhistory.io.git_export import export_git
export_git(timeline, snapshots, Path("./my-repo"))
```

Produces a real git repository with:
- One commit per snapshot
- Metadata preserved via `refs/notes/fh/metadata`
- Confidence scores via `refs/notes/fh/confidence`
- Full git tooling support: `git log`, `git diff`, `git blame`

## CLI Reference

```bash
# Analyze snapshots and produce timeline
folderhistory analyze <SNAPSHOTS_DIR>
  --out PATH                  Output path (default: timeline.json)
  --format json|gitlog|jsonlines|working-copy
  --mode full|delta|auto     Processing mode (default: auto)
  --apply corrections.json   User-provided identity corrections
  --kb-rollback N            Restore KB version N

# Compare two snapshots directly
folderhistory diff --before <DIR> --after <DIR>
  --format json|gitlog|jsonlines

# View timeline as git log
folderhistory log <timeline.json>

# Inspect cross-run cache
folderhistory kb-status
```

## Project Structure

```
src/folderhistory/
├── app.py                   # Typer CLI
├── types.py                 # Core dataclasses (Snapshot, FileRecord, EditOperation)
├── models.py                # Project-level types (ProjectRecord, ProjectMatch)
├── core/
│   ├── ingest.py            # Snapshot ingestion (walk dirs, compute hashes)
│   ├── hash.py              # BLAKE3 + xxhash64 (streaming + normalized)
│   ├── dir_align.py         # Cross-location project detection
│   ├── identity.py          # Union-Find identity assignment
│   ├── match.py             # Pairwise snapshot comparison
│   ├── diff.py              # Operation derivation (6 types)
│   ├── consistency.py       # Timeline consistency checking
│   └── timeline.py          # Linear + branching DAG timeline
├── signals/
│   ├── content.py           # Content similarity signals
│   ├── path.py              # Path similarity (Jaro-Winkler)
│   ├── metadata.py          # Metadata similarity signals
│   └── fusion.py            # Multi-signal fusion
├── io/
│   ├── output.py            # JSON/gitlog/jsonlines formatters
│   ├── working_copy.py      # Working Copy export
│   └── git_export.py        # Git repo export
├── knowledge/
│   ├── types.py             # KB types (ObservationRecord, ContradictionRecord)
│   ├── json_kb.py           # JSON Knowledge Base implementation
│   └── update.py            # KB update rules, contradiction resolution
├── feedback/
│   ├── analyzer.py          # Post-run quality analysis
│   └── refine.py            # Refinement loop stub (Phase 2)
└── tests/                   # 399+ tests
```

## Requirements

- **Python 3.12+**
- No external databases, services, or daemons
- Git optional (for git export format)

### Runtime dependencies
`blake3`, `xxhash`, `typer`, `orjson`, `scipy`, `networkx`, `jaro-winkler`

## Development

```bash
# Setup
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Type check
.venv/bin/python -m basedpyright src/

# Test
.venv/bin/python -m pytest tests/ -v

# Generate dummy test data
python tests/fixtures/gen_dummy_data.py --scenario ordered-evolution --snapshots 5 --seed 42 --output /tmp/demo

# Run full pipeline on generated data
python -m folderhistory.app analyze /tmp/demo --out /tmp/demo/timeline.json
python -m folderhistory.app log /tmp/demo/timeline.json
```

## License

MIT
