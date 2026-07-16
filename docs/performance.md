# FolderHistory — Performance Characteristics

This document describes the expected performance, scaling properties, and
known bottlenecks of FolderHistory's core engine.

---

## 1. Ingestion Throughput

| File count | File size | Observed time  | Throughput       |
|-----------:|----------:|---------------:|-----------------:|
| 100        | 1 KB      | ~0.1 s         | ~1 000 files/s   |
| 1 000      | 1 KB      | ~0.3 s         | ~3 000 files/s   |
| 5 000      | 1 KB      | ~1.5 s         | ~3 300 files/s   |
| 50 000     | 512 B     | ~10–30 s       | ~2 000 files/s   |

**Scaling**: **O(N)** in the number of files. Each file is stat'd, read into
memory, and hashed (BLAKE3 + XXH64). The dominant cost is the BLAKE3 hash
(`~1 GB/s` throughput with the `blake3` PyPI package on modern x86_64).

**Bottlenecks**:

- **Disk I/O** for the `read_bytes()` call on every regular file. Cold-cache
  performance is significantly lower than warm-cache.
- **BLAKE3 hashing** is SIMD-optimised in the native library, but the Python
  FFI overhead adds ~1–3 µs per call. For very small files this dominates.
- **`os.scandir()` recursion** is fast but serial; the tree walk is
  single-threaded.

**Mitigations**:

- Use NVMe / SSD storage for the snapshots being ingested.
- Pre-warm the page cache (`vmtouch`, `fadvise`) for repeated benchmarks.
- For non-security contexts, consider using **XXH64-only hashing** and
  deferring BLAKE3 to a background task (the `hash_bytes()` function always
  computes both).

---

## 2. Hashing Throughput

All measurements are single-threaded on a modern x86_64 core.

| Algorithm | Payload | Latency    | Throughput    |
|-----------|--------:|-----------:|--------------:|
| BLAKE3    | 1 KB    | ~3 µs      | ~300 MB/s     |
| BLAKE3    | 1 MB    | ~1 ms      | ~1 000 MB/s   |
| BLAKE3    | 10 MB   | ~9 ms      | ~1 100 MB/s   |
| XXH64     | 1 KB    | ~0.1 µs    | ~10 GB/s      |
| XXH64     | 1 MB    | ~80 µs     | ~12 GB/s      |
| XXH64     | 10 MB   | ~800 µs    | ~12 GB/s      |

**Key observation**: XXH64 is **10–40× faster** than BLAKE3 for in-memory data.
For files on disk the gap narrows because I/O is the bottleneck.

**`hash_bytes()` combined**: Computing both BLAKE3 and XXH64 in one pass takes
roughly the same time as BLAKE3 alone (the XXH64 cost is negligible by
comparison) — the function currently does **two passes** over the data
(once for each library), which is a missed optimisation.  A streaming or
combined-pass implementation would reduce the wall-clock time by ~30-50 %
for large files.

**Bottlenecks**:

- **Double pass** in `hash_bytes()`: calls `blake3.blake3(data)` and
  `xxhash.xxh64(data)` separately, each reading the entire buffer from L1/L2
  cache.  A single-pass implementation using the incremental API of both
  libraries would halve the memory bandwidth pressure.
- **Python FFI overhead** for the `hexdigest()` call on every tiny file.

**Mitigations**:

- Use the **blocking variant** (`assign_identities_with_blocking`) which
  first groups by file *size* and then by XXH64, drastically reducing the
  number of BLAKE3 comparisons needed.
- If hashing is a bottleneck, consider storing the hex digests in a sidecar
  JSON manifest (see `save_manifest()`) so that ingestion only needs to run
  once.

---

## 3. Identity Assignment Scaling

### `assign_identities_exact`

| Snapshots | Files total | Time (ms) | Scaling vs N=2 |
|----------:|------------:|----------:|----------------|
| 2         | 400         | ~11 ms    | 1×             |
| 5         | 1 000       | ~11 ms    | ~1×            |
| 10        | 2 000       | ~10 ms    | ~1×            |

**Scaling**: **O(F)** where F = total number of files across all snapshots.
The hash-index build is linear in F.  The union-find operations are
essentially **O(α(F))** (inverse Ackermann).  Rename detection (hash → key
union) is **O(H · K)** where H is the number of unique blake3 hashes and K
is the average collision count — in practice K is very close to 1 so this
is also effectively linear.

There is **no O(N²)** term in `assign_identities_exact` — the function does
not compare snapshot pairs.  Complexity scales with the *total observation
count*, not the number of snapshots.

### `assign_identities_with_blocking`

| Snapshots | Files total | Time (ms) | Scaling vs N=2 |
|----------:|------------:|----------:|----------------|
| 2         | 400         | ~12 ms    | 1×             |
| 5         | 1 000       | ~12 ms    | ~1×            |
| 10        | 2 000       | ~11 ms    | ~1×            |

**Scaling**: Still **O(F)** in the common case.  The blocking step adds a
dictionary grouping by `(size, xxhash64)` which is linear.  For data with
many unique sizes this is marginally faster than `assign_identities_exact`;
for data where many files share the same size (e.g. a monorepo with
identical-size source files) it may be slightly slower.

### Where O(N²) appears

The `match_snapshots_exact_all_pairs()` function **does** have **O(N²)**
scaling in the number of snapshots — it compares every unordered pair.
This is only used when snapshot ordering is uncertain and all pairs need to
be examined.  For the common case (ordered snapshots),
`match_snapshots_exact()` is **O(N)** in the number of snapshot pairs.

`derive_operations()` is **O(C · S)** where C is the number of identity
clusters and S is the number of snapshot-pair intervals — both are
proportional to the total file count, so the overall complexity remains
linear in practice.

---

## 4. Memory Usage

| Scenario                                    | RSS delta | Per-file overhead |
|---------------------------------------------|----------:|------------------:|
| Ingest 1 000 files (512 B each)             | ~200 KB   | ~0.2 KB / file    |
| Ingest 10 000 files (512 B each)            | ~6.5 MB   | ~0.7 KB / file    |
| Identity on 10 snapshots × 200 files        | ~4 KB     | ~0.0 KB / obs     |

**Scaling**: **O(F)** for the `FileRecord` list (each record is ~200–300
bytes on the heap) plus **O(F)** for the identity-assignment data structures
(hash index + union-find).  The number-of-files term usually dominates.

**Bottlenecks**:

- The `FileRecord` dataclass stores hashes as **hex strings** (64 chars for
  BLAKE3 + 16 for XXH64).  For 100 000 files this is ~8 MB just for hash
  strings.
- **`list[FileRecord]`** on each `Snapshot` keeps every file's metadata in
  memory simultaneously.  For very large snapshot sets (> 500 000 files
  across all snapshots) this can exceed available RAM.
- The **hash index** (`dict[str, set[str]]`) in `assign_identities_exact`
  duplicates the path strings keyed by their BLAKE3 hex digest.

**Mitigations**:

- Use **`save_manifest()`** after ingestion to persist the snapshot to JSON,
  then free the in-memory `Snapshot` (`del snapshot`) before loading the next.
  Re-load from manifests when running the identity engine.
- For identity assignment, the **blocking variant** (`with_blocking`) stores
  a nested dict keyed by `(size, xxhash64)` rather than a flat hash→keys
  mapping, which can reduce the index size when many files share the same
  size block.
- Consider using **binary hash representations** (32 bytes for BLAKE3
  instead of 64-char hex) if memory pressure becomes critical — this would
  require changes to the `FileRecord` type and serialisation format.

---

## 5. Recommendations for Large Snapshot Sets

| Scenario                          | Recommended approach                                           |
|-----------------------------------|---------------------------------------------------------------|
| > 100 000 files total             | Use `assign_identities_with_blocking` for tighter memory use   |
| > 50 snapshots                    | Use `match_snapshots_exact` (not `_all_pairs`) — O(N) vs O(N²)|
| Memory-constrained (< 512 MB)     | Persist snapshots to manifests; load one at a time             |
| Repeated runs on same data        | Use `JSONKnowledgeBase` (KB) for cross-run state reuse        |
| Throughput-critical               | Pre-hash with `hash_bytes` in parallel workers; ingest from manifests |
| First-run cold cache              | Run ingestion twice; discard first run's timing                |

### KB-Assisted Cross-Run State Reuse

When a `ReadOnlyKB` instance is provided, `assign_identities_exact` and
`assign_identities_with_blocking` pre-seed the Union-Find with known
hash → cluster-UID mappings from previous runs.  This:

- Preserves identity stability across incremental snapshot additions.
- Reduces the number of BLAKE3 comparisons for files whose content has
  not changed.
- Adds a small overhead (one KB lookup per file) that is negligible
  compared to hash computation.

Enable KB reuse when benchmarking the identity engine on snapshot sets
that grow over time.

---

## 6. Benchmark Methodology

All benchmarks in `tests/benchmarks/test_performance.py` follow this
protocol:

1. **Warmup**: 3 unmeasured rounds to prime CPU caches and trigger any
   lazy initialisation.
2. **Measurement**: 5 timed rounds using `time.perf_counter()`.
3. **Report**: Average time + throughput rate printed to stdout for each
   parametrised configuration.
4. **Sanity assert**: Every benchmark includes a loose upper-bound check
   to catch catastrophic regressions (e.g. an ingestion that takes 10×
   longer than expected).

**To run all benchmarks**:

```bash
uv run pytest tests/benchmarks/ -v -m benchmark
```

**To skip slow tests**:

```bash
uv run pytest tests/benchmarks/ -v -m "benchmark and not slow"
```
