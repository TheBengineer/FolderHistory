"""Performance benchmarks for FolderHistory.

Measures ingestion throughput, hashing throughput, identity assignment
scaling, and memory usage — using ``time.perf_counter()`` with warmup rounds
(no ``pytest-benchmark`` dependency required).

SIZE_OK: 332 pure LOC.  Benchmark collection with 5 test classes sharing
common helpers.  Splitting per-class would add more import/setup boilerplate
than value for a ``benchmarks/`` directory.
"""

from __future__ import annotations

import gc
import resource
import time
from pathlib import Path

import blake3
import pytest
import xxhash

from folderhistory.core.hash import hash_bytes
from folderhistory.core.identity import (
    assign_identities_exact,
    assign_identities_with_blocking,
)
from folderhistory.core.ingest import ingest_snapshot
from folderhistory.types import FileRecord, Snapshot

# ── Benchmark constants ───────────────────────────────────────────────────────

WARMUP_ROUNDS: int = 3
"""Number of unmeasured warmup iterations before timing begins."""

MEASURED_ROUNDS: int = 5
"""Number of timed rounds used to compute the final average."""

#: Small content payload (1 KB) — fits in L1 cache for most CPUs.
_PAYLOAD_1KB: bytes = b"x" * 1024

#: Medium content payload (1 MB).
_PAYLOAD_1MB: bytes = b"x" * (1024 * 1024)

#: Large content payload (10 MB).
_PAYLOAD_10MB: bytes = b"x" * (10 * 1024 * 1024)


# ── Internal helpers ─────────────────────────────────────────────────────────


def _write_file(src: Path, rel: str, content: bytes) -> None:
    """Create a file at *src / rel* with *content*, creating parent dirs."""
    p = src / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def _generate_tree(src: Path, num_files: int, file_size: int = 1024) -> None:
    """Generate *num_files* files under *src* with *file_size* bytes each.

    Files are spread across 16 subdirectories to create a non-trivial tree
    depth.  All files contain the same repeated byte — BLAKE3 / XXH64 will
    still hash the full content on every invocation.
    """
    content = b"x" * file_size
    files_per_dir = max(1, num_files // 16)
    for i in range(num_files):
        sub = i // files_per_dir
        _write_file(src, f"dir_{sub:04d}/file_{i:06d}.bin", content)


def _make_file_record(path: str, blake3_hash: str, size: int = 100) -> FileRecord:
    """Create a ``FileRecord`` for identity-benchmark snapshot data."""
    xxh: str = xxhash.xxh64(blake3_hash.encode()).hexdigest()
    return FileRecord(
        path=path,
        size=size,
        mode=0o100644,
        mtime_ns=1_000_000_000,
        ctime_ns=1_000_000_000,
        raw_blake3=blake3_hash,
        normalized_blake3=blake3_hash,
        line_ending="lf",
        xxhash64=xxh,
        is_symlink=False,
    )


def _make_identity_snapshots(
    num_snapshots: int,
    files_per_snapshot: int = 100,
) -> list[Snapshot]:
    """Generate *num_snapshots* in-memory snapshots for identity benchmarking.

    Between consecutive snapshots roughly 20 % of files change content hash
    and 10 % change path (simulating renames), to exercise both the
    path-based and hash-based identity-union code paths.
    """
    snapshots: list[Snapshot] = []

    for s_idx in range(num_snapshots):
        snap_id = f"v{s_idx}"
        files: list[FileRecord] = []

        for f_idx in range(files_per_snapshot):
            cat = f_idx % 10

            if s_idx == 0:
                # First snapshot: every file gets a unique hash.
                h = f"{f_idx:064d}"[:64]
                path = f"file_{f_idx:04d}.txt"
            elif cat < 7:
                # 70 % unchanged — same hash, same path.
                h = f"{f_idx:064d}"[:64]
                path = f"file_{f_idx:04d}.txt"
            elif cat < 9:
                # 20 % modified — new hash at the same path.
                h = f"{f_idx + s_idx * 10_000:064d}"[:64]
                path = f"file_{f_idx:04d}.txt"
            else:
                # 10 % renamed — same hash, different path.
                h = f"{f_idx:064d}"[:64]
                path = f"renamed_{f_idx:04d}_v{s_idx}.txt"

            files.append(_make_file_record(path, h, size=100))

        snapshots.append(Snapshot(
            id=snap_id,
            timestamp=float(s_idx),
            source_path=Path(f"/fake/{snap_id}"),
            files=files,
        ))

    return snapshots


def _measure(*, gc_before: bool = True) -> float:
    """Return a ``time.perf_counter()`` tick, optionally GC-ing first."""
    if gc_before:
        gc.collect()
    return time.perf_counter()


def _report(name: str, elapsed: float, count: float, unit: str) -> str:
    """Format a benchmark line for ``print()`` output."""
    rate = count / elapsed if elapsed > 0 else float("inf")
    return f"  {name}: {elapsed:.3f} s  ({rate:,.0f} {unit}/s)"


# ── Benchmark 1: Ingestion throughput ─────────────────────────────────────────


class TestIngestionThroughput:
    """Benchmark ``ingest_snapshot()`` for trees of varying sizes."""

    @staticmethod
    def _run_ingestion(src: Path) -> tuple[float, int]:
        """Run ``ingest_snapshot`` and return ``(elapsed_s, file_count)``."""
        t0 = _measure()
        snap = ingest_snapshot(src, snapshot_id="bench")
        elapsed = _measure() - t0
        return elapsed, len(snap.files)

    # ── Parametrized: 100, 1 000, 5 000 files ─────────────────────────────
    # (100K would be too slow for routine test runs; use the "slow" marker
    #  variant below for the heavy case.)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("num_files", [100, 1_000, 5_000], ids=lambda v: f"{v}files")
    def test_ingestion_throughput(self, num_files: int, tmp_path: Path) -> None:
        """Measure files / second via ``ingest_snapshot``."""
        _generate_tree(tmp_path, num_files, file_size=1024)

        # Warmup (unmeasured).
        for _ in range(WARMUP_ROUNDS):
            _ = ingest_snapshot(tmp_path, snapshot_id="warmup")

        # Measured rounds.
        times: list[float] = []
        file_counts: list[int] = []
        for _ in range(MEASURED_ROUNDS):
            elapsed, count = self._run_ingestion(tmp_path)
            times.append(elapsed)
            file_counts.append(count)

        avg_time = sum(times) / len(times)
        avg_files = sum(file_counts) / len(file_counts) if file_counts else 0.0

        print()
        print(f"  files={num_files}, file_size=1KB")
        print(_report("ingest", avg_time, avg_files, "files"))
        print(f"  avg_time={avg_time*1000:.1f} ms")

        # Sanity: ingestion must complete in reasonable time.
        # 5 000 files should take < 30 seconds even on a slow machine.
        assert avg_time < 30.0, (
            f"Ingestion of {num_files} files took {avg_time:.1f}s "
            f"(expected < 30 s)"
        )

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_ingestion_large_tree(self, tmp_path: Path) -> None:
        """Benchmark a 50 000-file tree (slow — only run with ``-m benchmark``)."""
        num_files = 50_000
        _generate_tree(tmp_path, num_files, file_size=512)

        # Warmup (1 round — generation + walk is heavy).
        for _ in range(1):
            _ = ingest_snapshot(tmp_path, snapshot_id="warmup")

        # Single measured round.
        elapsed, count = self._run_ingestion(tmp_path)

        print()
        print(f"  files={num_files}, file_size=512B")
        print(_report("ingest", elapsed, count, "files"))

        assert elapsed < 120.0, (
            f"Large-tree ingestion took {elapsed:.1f}s (expected < 120 s)"
        )


# ── Benchmark 2: Hashing throughput ──────────────────────────────────────────


class TestHashingThroughput:
    """Benchmark BLAKE3 and XXH64 individually for various payload sizes."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize(
        ("payload", "label"),
        [
            (_PAYLOAD_1KB, "1KB"),
            (_PAYLOAD_1MB, "1MB"),
            (_PAYLOAD_10MB, "10MB"),
        ],
    )
    def test_blake3_throughput(self, payload: bytes, label: str) -> None:
        """Measure BLAKE3 hashes / second."""
        # Warmup
        for _ in range(WARMUP_ROUNDS):
            _ = blake3.blake3(payload).hexdigest()

        times: list[float] = []
        for _ in range(MEASURED_ROUNDS):
            t0 = _measure()
            _ = blake3.blake3(payload).hexdigest()
            elapsed = _measure() - t0
            times.append(elapsed)

        avg = sum(times) / len(times)
        ops_per_sec = 1.0 / avg if avg > 0 else 0.0
        throughput = len(payload) / avg if avg > 0 else 0.0

        print()
        print(f"  size={label}, algorithm=BLAKE3")
        print(f"  avg_time={avg*1000:.3f} ms  ({ops_per_sec:,.0f} ops/s)")
        print(f"  throughput={throughput / 1024 / 1024:.1f} MB/s")

    @pytest.mark.benchmark
    @pytest.mark.parametrize(
        ("payload", "label"),
        [
            (_PAYLOAD_1KB, "1KB"),
            (_PAYLOAD_1MB, "1MB"),
            (_PAYLOAD_10MB, "10MB"),
        ],
    )
    def test_xxhash64_throughput(self, payload: bytes, label: str) -> None:
        """Measure XXH64 hashes / second."""
        for _ in range(WARMUP_ROUNDS):
            _ = xxhash.xxh64(payload).hexdigest()

        times: list[float] = []
        for _ in range(MEASURED_ROUNDS):
            t0 = _measure()
            _ = xxhash.xxh64(payload).hexdigest()
            elapsed = _measure() - t0
            times.append(elapsed)

        avg = sum(times) / len(times)
        ops_per_sec = 1.0 / avg if avg > 0 else 0.0
        throughput = len(payload) / avg if avg > 0 else 0.0

        print()
        print(f"  size={label}, algorithm=XXH64")
        print(f"  avg_time={avg*1000:.3f} ms  ({ops_per_sec:,.0f} ops/s)")
        print(f"  throughput={throughput / 1024 / 1024:.1f} MB/s")

    @pytest.mark.benchmark
    def test_hash_bytes_combined(self) -> None:
        """Measure ``hash_bytes()`` (BLAKE3 + XXH64 in one pass) for 1 MB."""
        payload = _PAYLOAD_1MB

        for _ in range(WARMUP_ROUNDS):
            _ = hash_bytes(payload)

        times: list[float] = []
        for _ in range(MEASURED_ROUNDS):
            t0 = _measure()
            _ = hash_bytes(payload)
            elapsed = _measure() - t0
            times.append(elapsed)

        avg = sum(times) / len(times)
        ops_per_sec = 1.0 / avg if avg > 0 else 0.0
        print()
        print("  size=1MB, algorithm=BLAKE3+XXH64 (hash_bytes)")  # noqa: T201
        print(f"  avg_time={avg*1000:.3f} ms  ({ops_per_sec:,.0f} ops/s)")


# ── Benchmark 3: Identity assignment scaling ─────────────────────────────────


class TestIdentityScaling:
    """Benchmark identity assignment vs. number of snapshots.

    Tests both ``assign_identities_exact`` and ``assign_identities_with_blocking``.
    """

    FILES_PER_SNAPSHOT: int = 200

    @pytest.mark.benchmark
    @pytest.mark.parametrize("num_snapshots", [2, 5, 10], ids=lambda v: f"N={v}")
    def test_identity_exact_scaling(self, num_snapshots: int) -> None:
        """Measure time for ``assign_identities_exact`` with *num_snapshots*."""
        snapshots = _make_identity_snapshots(
            num_snapshots,
            self.FILES_PER_SNAPSHOT,
        )

        # Warmup.
        for _ in range(WARMUP_ROUNDS):
            _ = assign_identities_exact(snapshots, kb=None)

        times: list[float] = []
        clusters_count: int = 0
        for _ in range(MEASURED_ROUNDS):
            t0 = _measure()
            clusters = assign_identities_exact(snapshots, kb=None)
            elapsed = _measure() - t0
            times.append(elapsed)
            clusters_count = len(clusters)

        avg = sum(times) / len(times)
        total_obs = num_snapshots * self.FILES_PER_SNAPSHOT

        print()
        print(f"  snapshots={num_snapshots}, files={total_obs}")
        print(f"  avg_time={avg*1000:.1f} ms, clusters={clusters_count}")
        print(_report("exact", avg, total_obs, "obs"))

    @pytest.mark.benchmark
    @pytest.mark.parametrize("num_snapshots", [2, 5, 10], ids=lambda v: f"N={v}")
    def test_identity_blocking_scaling(self, num_snapshots: int) -> None:
        """Measure time for ``assign_identities_with_blocking`` with *num_snapshots*."""
        snapshots = _make_identity_snapshots(
            num_snapshots,
            self.FILES_PER_SNAPSHOT,
        )

        for _ in range(WARMUP_ROUNDS):
            _ = assign_identities_with_blocking(snapshots, kb=None)

        times: list[float] = []
        for _ in range(MEASURED_ROUNDS):
            t0 = _measure()
            clusters = assign_identities_with_blocking(snapshots, kb=None)
            elapsed = _measure() - t0
            times.append(elapsed)

        avg = sum(times) / len(times)
        total_obs = num_snapshots * self.FILES_PER_SNAPSHOT

        print()
        print(f"  snapshots={num_snapshots}, files={total_obs}")
        print(f"  avg_time={avg*1000:.1f} ms, clusters={len(clusters)}")
        print(_report("blocking", avg, total_obs, "obs"))


# ── Benchmark 4: Memory usage ────────────────────────────────────────────────


class TestMemoryUsage:
    """Benchmark peak RSS for ingestion and identity assignment of sizable trees."""

    @staticmethod
    def _current_rss_kb() -> int:
        """Return the process's *current* resident set size in KB.

        Reads ``/proc/self/status`` (Linux-specific) for the instantaneous
        VmRSS value.  Falls back to ``resource.getrusage`` for non-Linux OS,
        noting that ``ru_maxrss`` is a process-lifetime peak and will be
        less accurate for per-test deltas.
        """
        try:
            with Path("/proc/self/status").open() as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        # Line format: "VmRSS:   12345 kB"
                        parts = line.strip().split()
                        return int(parts[1]) if len(parts) >= 2 else 0
        except (FileNotFoundError, OSError, ValueError):
            pass
        # Fallback: ru_maxrss (peak, not current — delta will be ~0 in
        # sequence if a previous test already touched memory).
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    @pytest.mark.benchmark
    @pytest.mark.parametrize("num_files", [1_000, 10_000], ids=lambda v: f"{v}files")
    def test_ingestion_memory(self, num_files: int, tmp_path: Path) -> None:
        """Measure current RSS delta after ingesting a *num_files*-file tree."""
        _generate_tree(tmp_path, num_files, file_size=512)

        # Baseline: GC then record current RSS.
        gc.collect()
        baseline = self._current_rss_kb()

        # Ingest.
        _snap = ingest_snapshot(tmp_path, snapshot_id="mem")

        # Post-ingestion: GC and record again.
        gc.collect()
        post = self._current_rss_kb()

        delta_kb = post - baseline
        per_file = delta_kb / num_files if num_files > 0 else 0.0

        print()
        print(f"  files={num_files}, file_size=512B")
        print(f"  rss_delta={delta_kb:,} KB ({delta_kb / 1024:.2f} MB)")
        print(f"  per_file={per_file:.1f} KB/file")

        # Sanity: each 512 B file should not cost more than ~4 KB of RSS
        # overhead (inode + FileRecord + path string).
        assert per_file < 10.0, (
            f"Per-file RSS overhead {per_file:.1f} KB seems excessive"
            f" (expected < 10 KB / file)"
        )

    @pytest.mark.benchmark
    def test_identity_memory(self) -> None:
        """Measure current RSS delta for identity assignment (10 snapshots x 200 files)."""
        num_snapshots = 10
        files_per = 200
        snapshots = _make_identity_snapshots(num_snapshots, files_per)

        gc.collect()
        baseline = self._current_rss_kb()

        clusters = assign_identities_exact(snapshots, kb=None)

        gc.collect()
        post = self._current_rss_kb()

        delta_kb = post - baseline
        total_obs = num_snapshots * files_per
        per_observation = delta_kb / total_obs if total_obs > 0 else 0.0

        print()
        print(f"  snapshots={num_snapshots}, files={total_obs}")
        print(f"  clusters={len(clusters)}")
        print(f"  rss_delta={delta_kb:,} KB ({delta_kb / 1024:.2f} MB)")
        print(f"  per_observation={per_observation:.1f} KB/obs")

        # Each observation is a tuple + FileRecord reference; overhead
        # should be well under 1 KB / observation.
        assert per_observation < 5.0, (
            f"Per-observation RSS overhead {per_observation:.1f} KB is too high"
        )
