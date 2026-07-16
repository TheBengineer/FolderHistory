# Scenario Catalog for FolderHistory Dummy Data

## Ground Truth Contract

Every scenario defines its ground truth as a **sequence of snapshots** (S₀, S₁, …, Sₙ)  
and a **known edit script** (the TrueOperations ✅ between each consecutive pair).  
FolderHistory's output (identity clusters + derived operations) is compared against
this ground truth.

### Ground Truth Encoding

Each scenario provides two machine-readable data structures:

1. **`ground_truth/snapshots.json`** — an ordered list of snapshot metadata:
   ```json
   [
     {"id": "S0", "timestamp": 1700000000.0, "source_path": "/tmp/project/v1/"},
     {"id": "S1", "timestamp": 1700003600.0, "source_path": "/tmp/project/v2/"}
   ]
   ```

2. **`ground_truth/operations.json`** — the known operations between consecutive snapshots:
   ```json
   [
     {
       "from": "S0", "to": "S1",
       "operations": [
         {"op_type": "modify", "file_id": "main.py",
          "source_path": "main.py", "target_path": "main.py",
          "old_hash": "aaaa", "new_hash": "bbbb"},
         {"op_type": "create", "file_id": "utils.py",
          "target_path": "utils.py", "new_hash": "cccc"}
       ]
     }
   ]
   ```

   These are the **true operations** we know happened. The test checks that
   FolderHistory derives the same (or a semantically equivalent) set.

3. **`ground_truth/file_identity.json`** — the ground-truth identity mapping:
   ```json
   {
     "aaaa": "uid_main_py",
     "bbbb": "uid_main_py",
     "cccc": "uid_utils_py"
   }
   ```
   Maps content hashes to their canonical identity UIDs. A hash not listed is
   a singleton identity. This lets tests verify that identity clusters are
   correctly formed.

---

## Metadata Mutation Catalog

### Flash Drive Copy Semantics

When a user copies files to a USB flash drive and back:

| Tool       | mtime preserved? | ctime changed? | atime?       | Notes                           |
|------------|------------------|----------------|--------------|---------------------------------|
| `cp -p`    | ✅ yes           | ❌ reset       | preserved    | Common Linux manual copy        |
| `rsync -a` | ✅ yes           | ❌ reset       | preserved    | Standard backup tool            |
| macOS Finder drag | 🟡 usually yes | ❌ reset | reset | Depends on volume format        |
| Windows Copy | ❌ no          | ❌ reset       | reset        | Default Explorer copy           |
| `robocopy` | ✅ yes (default) | ❌ reset      | preserved    | With `/COPY:DATSO` flag         |
| Tar extract| ✅ yes (in tar)  | ❌ reset       | reset        | Depends on umask/preserve flags |

**Key insight for testing**: A flash-drive copy always resets `ctime_ns` (since
the file is created new on the destination filesystem). `mtime_ns` may or may
not be preserved. The `source_path` also changes. These metadata shifts should
NOT prevent identity matching if content hashes align.

### Corruption Patterns

| Pattern | Description | Detection Method |
|---------|-------------|-----------------|
| `truncated` | File ends prematurely (last N bytes missing) | Size change + hash change |
| `bit_flip` | Single or multiple random bytes changed | Hash change, same size |
| `line_ending_corrupt` | Mixed LF/CRLF introduced | Raw hash changes, normalized hash stable |
| `zeroed` | File content replaced with null bytes | Size preserved, hash → zero-hash |
| `partial_write` | First K bytes correct, then garbage | Hash change, possible size change |
| `header_stripped` | First N bytes removed (e.g., BOM, magic bytes) | Size decrease + hash change |
| `block_swapped` | Two blocks of content swapped | Size preserved, hash change |

---

## Scenario 1: Flash Drive Backup Chain

### Narrative

A developer backs up their project to a USB flash drive every few days over
two weeks. Each backup is a `cp -r` copy. The flash drive accumulates a chain
of increasingly outdated snapshots. The final snapshot is a copy *back* from
the flash drive to a new laptop after the original laptop is lost.

### Realistic Context

- **S₀**: Project root on original laptop — "alive" working directory
- **S₁**: First USB backup (cp -r preserves mtime, resets ctime)
- **S₂**: Second USB backup one week later (some files changed between)
- **S₃**: Third USB backup (more changes, one file deleted, one created)
- **S₄**: Copy back to a new laptop — source_path changes completely

### Starting State (S₀)

```
project/
├── README.md              "Project Alpha"
├── src/
│   ├── main.py            "print('hello')"
│   └── utils.py           "def helper(): pass"
├── tests/
│   └── test_main.py       "assert True"
└── data/
    └── config.json        '{"version": 1}'
```

### Operation Sequence

| Step | Action | Affected | Ground Truth |
|------|--------|----------|-------------|
| S₀→S₁ | `cp -r` to flash drive | All files | No content changes; all ctime reset; source_path changed; ✗ root_change annotation |
| S₁→S₂ | Edit `src/main.py` | `src/main.py` | MODIFY on `src/main.py` (hash changes, same path) |
| S₁→S₂ | Create `src/new_feature.py` | `src/new_feature.py` | CREATE on `src/new_feature.py` |
| S₂→S₃ | Delete `tests/test_main.py` | `tests/test_main.py` | DELETE on `tests/test_main.py` |
| S₂→S₃ | Rename `data/config.json` → `settings.json` | moved | MOVE on `config.json` (parent changes) |
| S₂→S₃ | Edit `src/utils.py` | `src/utils.py` | MODIFY on `src/utils.py` |
| S₃→S₄ | Copy back from USB to new laptop | All files | No content changes; all ctime reset; source_path changed; root_change annotation |

### Expected FolderHistory Output

```
Snapshots: S₀(original) → S₁(usb-v1) → S₂(usb-v2) → S₃(usb-v3) → S₄(new-laptop)
Root changes: S₁ (source → /media/usb/), S₄ (/media/usb/ → ~/Documents/)

Operations:
  S₀→S₁: [root_change]  (no file ops — identical content)
  S₁→S₂: MODIFY main.py, CREATE new_feature.py
  S₂→S₃: DELETE test_main.py, RENAME config.json→settings.json, MODIFY utils.py
  S₃→S₄: [root_change]  (no file ops)
```

### Edge Cases Covered

- ctime reset should NOT break identity matching (content hashes match)
- mtime preserved across copies verifies timestamp-based heuristics
- Source path changes trigger root_change annotations
- S₄ on new laptop has entirely different filesystem metadata

---

## Scenario 2: Device Migration with Divergent Evolution

### Narrative

A project lives on a Desktop machine. The user migrates it to a Laptop via
USB stick (cp -r). Before the migration, some files were modified on the
Desktop but the user forgot to copy them. After migration, work continues
on both machines independently for a few days — the "branching" scenario.

The USB transfer preserves mtimes but resets ctimes (standard cp -r behavior).

### Starting State (S₀)

```
migration_project/
├── src/
│   ├── app.py             "v1 app"
│   ├── db.py              "v1 db"  
│   └── helpers.py          "v1 helpers"
├── docs/
│   └── README.md           "v1 readme"
└── Makefile                "v1 makefile"
```

### Operation Sequence

This demonstrates a **fork**: S₁ is the USB copy snapshot, then S₂ₐ (Desktop)
and S₂ᵦ (Laptop) diverge.

| Step | Action | Affected | Ground Truth |
|------|--------|----------|-------------|
| S₀→S₁ | `cp -r` to USB | All files | No content changes; ctime reset; source_path changes |
| S₁→S₂ₐ | Edit `src/app.py` on Desktop | `src/app.py` | MODIFY `src/app.py` on Desktop side |
| S₁→S₂ₐ | Create `src/legacy.sh` on Desktop | `src/legacy.sh` | CREATE `src/legacy.sh` on Desktop |
| S₁→S₂ₐ | Delete `docs/README.md` on Desktop | `docs/README.md` | DELETE `docs/README.md` on Desktop |
| S₁→S₂ᵦ | Edit `src/db.py` on Laptop | `src/db.py` | MODIFY `src/db.py` on Laptop |
| S₁→S₂ᵦ | Edit `Makefile` on Laptop | `Makefile` | MODIFY `Makefile` on Laptop |
| S₁→S₂ᵦ | Create `src/api.py` on Laptop | `src/api.py` | CREATE `src/api.py` on Laptop |

### Expected FolderHistory Output (Linear mode — assumes order known)

If snapshots are presented in order S₀, S₁, S₂ₐ, S₂ᵦ:

```
S₀→S₁: [root_change]           (USB copy — 0 file ops)
S₁→S₂ₐ: MODIFY app.py, CREATE legacy.sh, DELETE docs/README.md
S₂ₐ→S₂ᵦ: No identity bridging (different content at same paths → delete+create)
         or: (if multi-signal) MODIFY db.py, MODIFY Makefile, DELETE app.py,
         CREATE api.py, CREATE app.py (v2)
```

### Expected FolderHistory Output (Branching mode)

```
S₀ → S₁ → S₂ₐ (Desktop branch)
         └→ S₂ᵦ (Laptop branch)
```

### Ground Truth Structure

3 ground-truth truth tables:
- `desktop_truth.json`: operations for S₀→S₁→S₂ₐ
- `laptop_truth.json`: operations for S₀→S₁→S₂ᵦ
- `merged_truth.json`: full DAG with fork at S₁

### Edge Cases Covered

- Divergent edits on same file: `src/app.py` edited on Desktop, unchanged on Laptop
- Different files created on each branch
- One branch deletes a file the other preserves
- Tests identity clustering must NOT bridge Desktop `app.py`(v2) with Laptop `app.py`(v1)  
  if fuzzy matching is attempted

---

## Scenario 3: File Corruption Over Time (Bit Rot)

### Narrative

An old backup drive has been sitting in a drawer for years. Over time, sectors
degrade. A media project's files gradually corrupt across snapshots. The same
file appears in multiple snapshots with increasingly corrupted content.

### Starting State (S₀)

```
media_project/
├── assets/
│   ├── header.png          PNG image (valid)
│   ├── icon.png            PNG image (valid)
│   └── sprite_sheet.png    PNG image (valid)
├── src/
│   ├── main.c              C source file
│   └── shaders.glsl        GLSL shader code
└── config.ini              INI config file
```

### Corruption Sequence

| Step | Corruption | Affected | Pattern | Ground Truth |
|------|-----------|----------|---------|-------------|
| S₀→S₁ | ✅ No corruption — clean backup | — | — | No ops |
| S₁→S₂ | ⚠️ `header.png` bit flip at offset 512 | `assets/header.png` | `bit_flip` | MODIFY `header.png` (hash changes, size same) |
| S₂→S₃ | ⚠️ `sprite_sheet.png` truncated to 1KB | `assets/sprite_sheet.png` | `truncated` | MODIFY `sprite_sheet.png` (hash changes, size smaller) |
| S₃→S₄ | ⚠️ `config.ini` zeroed to 4KB of `\0` | `config.ini` | `zeroed` | MODIFY `config.ini` (hash → zero-hash, size preserved) |
| S₃→S₄ | ⚠️ `shaders.glsl` partial write (first 50 bytes OK, then garbage) | `src/shaders.glsl` | `partial_write` | MODIFY `shaders.glsl` (hash changes) |
| S₄→S₅ | ❌ `header.png` fully zeroed (complete data loss) | `assets/header.png` | `zeroed` | MODIFY `header.png` (hash → zero-hash) |
| S₄→S₅ | ❌ `icon.png` bit flip at offset 1024 | `assets/icon.png` | `bit_flip` | MODIFY `icon.png` |

### Ground Truth Corpus

Alongside the snapshots, we generate a `ground_truth/corruption_log.json`:
```json
[
  {"snapshot": "S2", "path": "assets/header.png", "pattern": "bit_flip",
   "offset": 512, "original_hash": "aaaa", "corrupted_hash": "bbbb"},
  {"snapshot": "S3", "path": "assets/sprite_sheet.png", "pattern": "truncated",
   "original_size": 40960, "new_size": 1024}
]
```

### Edge Cases Covered

- Multiple corruption patterns on the same file across snapshots
- Binary files (PNG) that should have `normalized_blake3 = None`
- Binary corruption where some files are still valid (not all files corrupt)
- Progressive degradation — same file corrupts again in later snapshot

---

## Scenario 4: Cross-Platform Backup Chaos

### Narrative

A Python project is developed on macOS, then backed up to a Windows NTFS drive,
then copied to a Linux server, then downloaded back to macOS. Each transfer
mangles line endings differently. The project also contains binary assets
that must NOT be normalized.

### Snapshot Farm

| Snapshot | Platform | Line Endings | Notes |
|----------|----------|-------------|-------|
| S₀ | macOS (APFS) | LF | Original development |
| S₁ | Windows (NTFS, via Git checkout) | CRLF | `git clone` on Windows with `autocrlf=true` |
| S₂ | Linux (ext4, via rsync from Windows) | Mixed | rsync preserves CRLF from Windows |
| S₃ | macOS again (via USB from Linux) | LF | Normalized by macOS tools |

### File Inventory

```
cross_platform_project/
├── src/
│   ├── main.py              Python text
│   ├── utils.py             Python text  
│   └── binary_data.bin      Binary (null bytes)
├── scripts/
│   └── build.sh             Shell script (text, no line-ending normalization needed)
├── docs/
│   ├── README.md            Markdown text
│   └── CHANGELOG.md         Markdown text
├── assets/
│   ├── logo.png             Binary PNG
│   └── icon.ico             Binary ICO
└── .gitattributes           `* text=auto` + `*.bin binary`
```

### Operation Sequence

| Step | What happens | Ground Truth |
|------|-------------|-------------|
| S₀→S₁ | All text files get CRLF line endings; binary files unchanged | ✅ normalized_blake3 matches for text files; raw_blake3 differs; binary raw_blake3 unchanged |
| S₁→S₂ | rsync preserves CRLF; `build.sh` on Linux may get LF if touched | MODIFY `build.sh` if line endings change; all else MODIFY only if raw_blake3 differs |
| S₂→S₃ | USB copy to macOS; Finder may normalize some text to LF | Same as S₀→S₁ in reverse |

### Truth Table

```json
{
  "normalized_hash_invariants": {
    "src/main.py": ["uid_main_py", "uid_main_py", "uid_main_py", "uid_main_py"],
    "src/utils.py": ["uid_utils_py", "uid_utils_py", "uid_utils_py", "uid_utils_py"],
    "assets/logo.png": ["uid_logo", null, null, null]
  },
  "raw_hash_invariants": {
    "src/main.py": ["raw_a", "raw_b", "raw_b", "raw_a"],
    "assets/logo.png": ["raw_logo", "raw_logo", "raw_logo", "raw_logo"]
  }
}
```

The null in normalized hash for binary files means `normalized_blake3 is None`.
Each "raw_a", "raw_b" etc. is a known fixed hash computed before generation.

### Edge Cases Covered

- Normalized blake3 provides identity across different line endings
- Binary files are immune to line-ending normalization
- Mixed-line-ending files (`build.sh` may have LF even on Windows)
- Round-trip: S₀→S₃ returns to original platform; identity should bridge
- Cross-platform identity assignment relies on normalized hash, not raw

---

## Scenario 5: Monorepo Extraction with Independent Evolution

### Narrative

A monorepo contains both a frontend and backend package. The frontend is
extracted into a standalone repository. Both the monorepo and the standalone
frontend evolve independently after the split.

### Starting State (S₀ — Complete Monorepo)

```
monorepo/
├── packages/
│   ├── frontend/
│   │   ├── package.json       "frontend v1"
│   │   ├── src/
│   │   │   ├── App.tsx         "App v1"
│   │   │   └── index.tsx       "index v1"  
│   │   └── public/
│   │       └── favicon.ico     (binary)
│   └── backend/
│       ├── package.json        "backend v1"
│       └── src/
│           ├── server.ts       "server v1"
│           └── routes.ts       "routes v1"
├── tsconfig.json               "root tsconfig"
└── README.md                   "monorepo readme"
```

### Operation Timeline

| Step | Description | Ground Truth Ops |
|------|------------|-----------------|
| S₀→S₁ | Frontend extracted: `packages/frontend/` → standalone `frontend/` | Identity cluster bridges `packages/frontend/App.tsx` ↔ `App.tsx` (different relative path) |
| S₁→S₂ (monorepo) | Backend adds `src/middleware.ts` | CREATE `src/middleware.ts` |
| S₁→S₂ (monorepo) | Backend modifies `src/server.ts` | MODIFY `src/server.ts` |
| S₁→S₂ (frontend) | Frontend updates `App.tsx` to v2 | MODIFY `App.tsx` (different hash than monorepo's) |
| S₁→S₂ (frontend) | Frontend adds `src/components/Button.tsx` | CREATE `src/components/Button.tsx` |

### Truth Table Complexities

After extraction (S₁), the identity cluster for `App.tsx` spans:
- Monorepo: `packages/frontend/src/App.tsx`
- Standalone: `src/App.tsx`

These have different relative paths BUT their content is identical at S₁.
FolderHistory's `dir_align` module should detect this via content probing.

At S₂, the two copies diverge:
- Monorepo's `packages/frontend/src/App.tsx` still has content from S₁
- Standalone's `src/App.tsx` has new content (v2)
- They should NOT be in the same identity cluster at S₂

### Ground Truth Structure

```json
{
  "project_groups": {
    "monorepo": ["S0_monorepo", "S1_monorepo", "S2_monorepo"],
    "frontend": ["S1_frontend", "S2_frontend"]
  },
  "expected_projects": 2,
  "split_snapshot": "S1"
}
```

### Edge Cases Covered

- Changed relative paths after extraction (subtree detected via content probing)
- Content divergence after split (same cluster at S₁, different clusters at S₂)
- Binary files (favicon.ico) survive extraction with no change
- Partial content overlap: standalone has fewer files than monorepo subtree

---

## Scenario 6: Accidental Overwrite / Restore from Backup

### Narrative

A developer accidentally overwrites a file with an older version from a backup,
then later restores a newer version. The timeline shows a "regression" followed
by a "recovery."

### Starting State (S₀)

```
overwrite_project/
├── src/
│   ├── main.py              "v3 — latest feature"
│   ├── utils.py             "stable v2"
│   └── config.py            "v1 initial"
└── README.md                "v2 readme updated"
```

### Operation Timeline

| Step | Action | Ground Truth |
|------|--------|-------------|
| S₀→S₁ | Accidental `cp backup/main.py src/main.py` (older version overwrites) | MODIFY `src/main.py` — hash reverts to older hash = `hash_backup` |
| S₁→S₂ | Realize mistake, restore from `.git`: `git checkout HEAD -- src/main.py` | MODIFY `src/main.py` — hash returns to `hash_v3` |
| S₁→S₂ | Also create `src/main_patched.py` with combined fix | CREATE `src/main_patched.py` |

### Key Property

Between S₀ and S₂, `src/main.py` has the same hash (v3 content), but the
intermediate S₁ has the backup (v1 content). FolderHistory should correctly
derive:

```
S₀→S₁: MODIFY main.py (hash v3 → v1)
S₁→S₂: MODIFY main.py (hash v1 → v3), CREATE main_patched.py
```

### Edge Cases Covered

- Content "regression" — newer snapshot has older content
- Content round-trip — same hash in non-consecutive snapshots
- Identity must bridge the file across all three states (same path throughout)

---

## Scenario 7: Deep Rename + Restructure

### Narrative

A project undergoes major directory restructuring: files are moved between
directories, renamed, and some split. This tests FolderHistory's rename/move
detection.

### Starting State (S₀)

```
messy_project/
├── old_stuff/
│   ├── run.sh               "run script v1"
│   ├── old_config.cfg       "old config"
│   └── garbage.txt          "temporary notes"
├── src/
│   ├── core.py              "core logic v1"
│   └── helpers.py           "helpers v1"
└── README.md                "readme v1"
```

### Operation Timeline

| Step | Action | Ground Truth |
|------|--------|-------------|
| S₀→S₁ | Move `old_stuff/run.sh` → `scripts/run.sh` | MOVE `old_stuff/run.sh` → `scripts/run.sh` (same basename, different parent) |
| S₀→S₁ | Rename `old_stuff/old_config.cfg` → `config/settings.cfg` | MOVE + RENAME `old_config.cfg` → `settings.cfg` (but in Frame- work: path change detection) |
| S₀→S₁ | Rename `src/helpers.py` → `src/utils.py` | RENAME `helpers.py` → `utils.py` (different basename, same parent) |
| S₀→S₁ | Create `scripts/deploy.sh` (new) | CREATE `scripts/deploy.sh` |
| S₀→S₁ | Delete `old_stuff/garbage.txt` | DELETE `old_stuff/garbage.txt` |
| S₀→S₁ | Split `src/core.py` → keep `core.py` with 50% content, create `src/advanced.py` | MODIFY `core.py` (content changed), CREATE `advanced.py` |

### Truth Table (Rename/Move Detection)

```json
{
  "expected_rename_ops": [
    {"file_id": "old_stuff/run.sh", "from": "old_stuff/run.sh", "to": "scripts/run.sh"},
    {"file_id": "old_stuff/old_config.cfg", "from": "old_stuff/old_config.cfg", "to": "config/settings.cfg"},
    {"file_id": "src/helpers.py", "from": "src/helpers.py", "to": "src/utils.py"}
  ],
  "rename_confidence_threshold": 0.8,
  "identity_bridges_across_rename": true
}
```

### Edge Cases Covered

- Pure rename (same parent, different name): `helpers.py` → `utils.py`
- Pure move (different parent, same name): `old_stuff/run.sh` → `scripts/run.sh`
- Combined rename + move: `old_stuff/old_config.cfg` → `config/settings.cfg`
- File split: one source file becomes two (content partially shared)
- The `split` case is the hardest — `core.py`'s identity after split may not
  bridge perfectly with either resulting file

---

## Summary Matrix

| # | Scenario | Snapshots | Branching? | Corruption? | Cross-Location? | Cross-Platform? | Key Test |
|---|----------|-----------|-----------|-------------|----------------|-----------------|----------|
| 1 | Flash Drive Chain | 5 | No | No | Yes (USB) | No | ctime reset does not break identity |
| 2 | Device Migration | 4 | Yes (fork) | No | Yes (desk→laptop) | No | Branching timeline with divergent edits |
| 3 | Bit Rot | 5+ | No | Yes (6 patterns) | No | No | Progressive file corruption detection |
| 4 | Cross-Platform Chaos | 4 | No | Line-endings | Yes (OS→OS) | Yes (LF/CRLF) | Normalized hash cross-platform identity |
| 5 | Monorepo Extraction | 5 | Yes (split) | No | Yes (monorepo→standalone) | No | Subtree detection + independent evolution |
| 6 | Accidental Overwrite | 3 | No | Content regression | No | No | Non-monotonic content tracking |
| 7 | Deep Restructure | 2 | No | No | No | No | Rename/move detection |

---

## Implementation Guide for Data Engineers

### File Generation Script Pattern

Each scenario lives in its own directory under `tests/fixtures/scenarios/`:

```
tests/fixtures/scenarios/
├── SCENARIO_CATALOG.md              ← this file
├── gen_flash_drive_chain.py         ← Scenario 1 generator
├── gen_device_migration.py          ← Scenario 2 generator
├── gen_corruption.py                ← Scenario 3 generator
├── gen_cross_platform.py            ← Scenario 4 generator
├── gen_monorepo_extraction.py       ← Scenario 5 generator
├── gen_overwrite.py                 ← Scenario 6 generator
├── gen_restructure.py               ← Scenario 7 generator
├── shared.py                        ← Shared helpers (hash computation, ground truth writing)
└── __init__.py
```

### Generator Interface

Every `gen_*.py` implements:

```python
def generate(output_dir: Path, *, overwrite: bool = False) -> dict:
    """Generate scenario directories and ground truth JSON.

    Returns:
        A dict with keys:
          - "snapshot_dirs": list of Path, one per snapshot
          - "ground_truth": Path to ground_truth/ directory
          - "scenario_name": str
    """
```

### Ground Truth Verification

A test fixture in `tests/conftest.py` provides:

```python
@pytest.fixture
def verify_scenario(scenario_dir: Path):
    """Compare FolderHistory output against ground truth."""
    # 1. Load ground_truth/operations.json
    # 2. Ingest all snapshot dirs
    # 3. Run full FolderHistory pipeline
    # 4. Assert derived operations == ground truth operations
    # 5. Assert identity clusters match ground_truth/file_identity.json
```

### Corruption Helpers (for Scenario 3)

```python
def apply_bit_flip(content: bytes, offset: int) -> bytes:
    """Flip one random byte at the given offset."""
    byte_arr = bytearray(content)
    byte_arr[offset] ^= 0xFF  # flip all bits
    return bytes(byte_arr)

def apply_truncation(content: bytes, new_size: int) -> bytes:
    """Truncate content to new_size bytes."""
    return content[:new_size]

def apply_zeroed(content: bytes) -> bytes:
    """Replace all bytes with null bytes."""
    return b'\x00' * len(content)

def apply_partial_write(content: bytes, good_prefix: int) -> bytes:
    """Keep first good_prefix bytes, rest becomes garbage."""
    import random
    garbage = bytes(random.randint(0, 255) for _ in range(len(content) - good_prefix))
    return content[:good_prefix] + garbage
```
