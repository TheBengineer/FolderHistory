# FolderHistory

**Reconstruct git-like version history from messy folder backups — using only file metadata.**

When you have a pile of backup snapshots of a directory — taken at different times, possibly overlapping, with timestamps that may be unreliable — FolderHistory aims to reconstruct the most likely sequence of changes: which files were created, modified, moved, renamed, or deleted, and in what order.

## The Problem

You back up a project folder manually, or via cron, or with a cloud sync tool. Over months or years you accumulate N near-identical copies of the same directory tree. Each snapshot has:

- A **directory tree** (files & folders at that point in time)
- **File metadata**: creation time, modification time, size, permissions
- Possibly **content hashes** (if you opt in)

But metadata is **flawed**:

- `ctime` (creation time) may be reset when files are copied between filesystems
- `mtime` (modification time) may be preserved or stripped depending on the backup tool
- Files may be renamed, moved, or duplicated across snapshots with no trace
- Snapshot timestamps themselves may be imprecise or lost

The challenge: given only these noisy snapshots, reconstruct the most plausible **linear or branching history** of how the folder evolved over time.

## Project Phases

### Phase 1 — Deterministic Reconstruction (known ground truth)

Build an algorithm that works when metadata is **reliable**. Given ordered snapshots with accurate timestamps, reconstruct a precise sequence of file-system operations (create, modify, delete, rename, move) between each pair of consecutive snapshots. This serves as the ground-truth baseline.

### Phase 2 — Probabilistic Reconstruction (flawed metadata)

Extend the algorithm to handle **uncertainty**:

- Unknown snapshot ordering → infer temporal order
- Inconsistent timestamps → weight evidence probabilistically
- Missing data → interpolate between known states
- Rename/move detection → use content similarity when metadata is ambiguous

Output a ranked set of possible histories with confidence scores, rather than a single deterministic answer.

## Algorithmic Approaches

This project synthesises ideas from several research areas. Here are the major families under consideration:

### 1. Tree Edit Distance (TED)

Classic approach: model each snapshot as a **rooted, labelled tree** and compute the minimum-cost edit script that transforms one tree into another. Operations: insert node, delete node, rename node.

- **Strengths**: Well-studied, provable optimality for ordered trees (Zhang-Shasha, O(n³)).
- **Weaknesses**: Unordered trees are MAX-SNP hard; rename detection requires subtree similarity, not just identity; doesn't natively handle moves or copies.
- **References**: Zhang & Shasha (1989), "Simple Fast Algorithms for the Editing Distance Between Trees and Related Problems."

### 2. MH-DIFF — Meaningful Change Detection (Edge Cover model)

Represents nodes as a **bipartite graph** and finds a minimum-cost edge cover to match nodes between two tree snapshots. Supports **move and copy** operations — not just insert/delete/update. The edge cover is then decoded into an edit script.

- **Strengths**: Detects moved/copied subtrees (semantically meaningful); produces compact edit scripts.
- **Weaknesses**: Heuristic; doesn't guarantee optimal edit distance.
- **References**: Chawathe et al. (1996), "Change Detection in Hierarchically Structured Information" (SIGMOD).

### 3. RWS-Diff — Random Walk Similarity Diff

Represents each subtree as a **d-dimensional feature vector** via random walks, then uses nearest-neighbour search to find similar (not identical) subtrees across snapshots. Runs in O(n log n).

- **Strengths**: Handles both ordered and unordered trees; similarity-aware (matches near-identical subtrees even when labels differ); O(n log n) scalable.
- **Weaknesses**: Probabilistic approximation; may miss exact minimal edit scripts.
- **References**: Finis et al. (2013), "RWS-Diff: Flexible and Efficient Change Detection in Hierarchical Data."

### 4. Multi-Signal Provenance Reconstruction (VU Amsterdam)

Treats history reconstruction as a **multi-signal fusion pipeline**: extract signals from content similarity, metadata similarity, temporal ordering, and domain-specific heuristics; generate candidate provenance graphs; prune inconsistent hypotheses; aggregate and rank.

- **Strengths**: Combines multiple weak signals for robustness; extensible to new signals; handles mixed data types.
- **Weaknesses**: Requires tuning signal weights; computationally expensive for large snapshot sets.
- **References**: Niels de Vries (2012), "Reconstructing Provenance of Files from Filesystem Metadata" (VU Amsterdam).

### 5. Backwards Timestamp Reasoning (NTFS)

Rather than comparing snapshots forward, **work backwards** from final timestamps. Given a file's current NTFS timestamps and a known set of possible filesystem operations, constrain which sequences of operations could have produced that state. Build a tree of possible timelines.

- **Strengths**: Can reconstruct history from a single snapshot; accounts for timestamp-altering operations.
- **Weaknesses**: Limited to NTFS timestamp semantics; combinatorial explosion of possible histories.
- **References**: Bouma et al. (2023), "Reconstructing Timelines: From NTFS Timestamps to File Histories."

### 6. Document-Level Revision Detection

Models document revision detection as a **minimum-cost branching problem** on a directed graph of documents, using semantic distances. Proposes wDTW (word vector-based Dynamic Time Warping) and wTED (word vector-based Tree Edit Distance) for document-level comparison.

- **Strengths**: Unsupervised; handles large corpora; semantic understanding.
- **Weaknesses**: Document-focused, not folder-tree focused; requires vector embeddings.
- **References**: Zhu, Klabjan, Bless (2017), "Semantic Document Distance Measures and Unsupervised Document Revision Detection."

## Architecture (Proposed)

```
Snapshots (N folders)
       │
       ▼
┌──────────────────────┐
│  Snapshot Ingestion   │  Parse directory tree, extract metadata,
│                       │  optionally hash file contents
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Pairwise Comparison  │  Compare every pair of snapshots using
│                       │  one or more matching strategies
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Graph Construction   │  Build a weighted DAG where nodes are
│                       │  snapshot states, edges are edit scripts
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Path Optimisation    │  Find most likely global history path(s)
│                       │  through the snapshot graph
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  History Output       │  Render as git-like log, timeline,
│                       │  or structured diff stream
└──────────────────────┘
```

### Key Design Decisions (to be resolved)

| Decision | Options | Notes |
|----------|---------|-------|
| Pairwise matching algorithm | TED / MH-DIFF / RWS-Diff / content hash | Phase 1: simple hash + path matching. Phase 2: similarity-based. |
| History model | Linear chain / DAG / branching tree | Phase 1: assume linear (if snapshot order known). Phase 2: allow branching. |
| Rename detection | Path-based / hash-based / similarity-based | Path change + content match = rename hypothesis. |
| Timestamp confidence model | None / weight by filesystem / Bayesian | Phase 2 only. |
| Output format | Git-like log / JSON diff stream / Graphviz DOT | User-facing vs machine-readable. |

## Development Roadmap

### Milestone 1 — Snapshot diff engine
- [ ] Implement pairwise snapshot comparison (path + hash matching)
- [ ] Detect: created files, deleted files, modified files, moved/renamed files
- [ ] Output structured diffs (create/delete/modify/rename operations)

### Milestone 2 — Linear history reconstruction
- [ ] Accept ordered snapshot sequence
- [ ] Chain pairwise diffs into a linear timeline
- [ ] Render as git-style log

### Milestone 3 — Unordered / probabilistic reconstruction
- [ ] Infer snapshot ordering from timestamps and content
- [ ] Add similarity-based matching (RWS-Diff or multi-signal)
- [ ] Score and rank alternative histories
- [ ] Handle missing snapshots (interpolation)

### Milestone 4 — Robustness & real-world testing
- [ ] Test with deliberately corrupted / shifted timestamps
- [ ] Test with cross-filesystem copies (ctime reset)
- [ ] Fuzz test against random directory mutations
- [ ] Benchmark on large directory trees

## Related Work & Further Reading

- **Git internals** — `git diff-tree`, rename detection (`-M`, `-C`), similarity index heuristics
- **fs-tree-diff** (Stefan Penner) — minimal patch calculation between two filesystem trees
- **btrfs-file-history** (fkzys) — track file lifecycle across btrfs snapshots using UUID linkage
- **foldiff** (yellowsink) — efficient binary folder diffing for backup storage
- **VU provenance reconstruction** — multi-signal pipeline for reconstructing file dependencies in shared folders
- **Btr-Diff** (Pimpale et al., 2014) — kernel-level Btrfs snapshot diff using COW B-tree structures

## License

MIT
