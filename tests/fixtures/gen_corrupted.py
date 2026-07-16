#!/usr/bin/env python3
"""Programmatic corrupted-metadata fixture generator.

Generates a directory tree with deliberately corrupted metadata for
testing FolderHistory's robustness.  Each sub-tree exercises a different
corruption category.

Usage
-----
    uv run python tests/fixtures/gen_corrupted.py <output_dir>

The generated tree will look like::

    <output_dir>/
    ├── shifted_timestamps/
    │   ├── one_year_past.txt        mtime = now - 365 days
    │   ├── one_year_future.txt      mtime = now + 365 days
    │   ├── five_years_past.txt      mtime = now - 1825 days
    │   └── epoch.txt                mtime = 1970-01-01
    ├── zero_timestamps/
    │   ├── zero_mtime.txt           mtime = 0
    │   └── zero_mtime_atime.txt     atime = mtime = 0
    ├── ctime_gt_mtime/
    │   ├── copied.txt               mtime in past, ctime current (via chmod)
    │   └── restored.txt             mtime in past (via utime, chmod)
    ├── extreme_values/
    │   ├── huge.txt                 mtime near INT64_MAX
    │   └── empty.txt                size = 0, all timestamps = 0
    ├── cross_platform/
    │   ├── lf.txt                   LF line endings
    │   ├── crlf.txt                 CRLF line endings
    │   ├── mixed.txt                Mixed LF/CRLF line endings
    │   ├── binary.bin               Binary content (null byte)
    │   └── script.py                Text override (null byte + CRLF)
    ├── unicode_paths/
    │   ├── café.txt                 NFD path (if platform allows)
    │   ├── 中文.txt                  CJK
    │   ├── 😀.txt                    Emoji
    │   └── العربية.txt              RTL
    ├── empty_file.txt               Empty file
    ├── large_file.bin               10 MB of zeros (for large-tree tests)
    └── deep_nesting/
        └── a/b/c/d/e/f/g/h/i/j/deep.txt   Deeply nested file

After generation, you can snapshot the tree with::

    python -c "from folderhistory.core.ingest import ingest_snapshot; \\
        from pathlib import Path; \\
        snap = ingest_snapshot(Path('<output_dir>')); \\
        print(f'{len(snap.files)} files ingested')"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────

_NS_IN_SEC = 1_000_000_000
_DAY_NS = 86400 * _NS_IN_SEC


def _create_file(path: Path, content: bytes = b"") -> None:
    """Create a file, ensuring parent directories exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _set_mtime(path: Path, mtime_ns: int) -> None:
    """Set both atime and mtime of *path* to *mtime_ns* (nanoseconds)."""
    os.utime(path, ns=(mtime_ns, mtime_ns))


def _touch_via_chmod(path: Path) -> None:
    """Perform a chmod to advance ctime without changing mtime."""
    st = path.stat()
    os.chmod(str(path), st.st_mode)


# ── Generator functions ──────────────────────────────────────────────────────


def gen_shifted_timestamps(root: Path) -> None:
    """Files with mtime shifted far from the present."""
    now_ns = int(time.time() * _NS_IN_SEC)

    # 1 year in the past
    f = root / "one_year_past.txt"
    _create_file(f, b"past content")
    _set_mtime(f, now_ns - 365 * _DAY_NS)

    # 1 year in the future
    f = root / "one_year_future.txt"
    _create_file(f, b"future content")
    _set_mtime(f, now_ns + 365 * _DAY_NS)

    # 5 years in the past
    f = root / "five_years_past.txt"
    _create_file(f, b"very old content")
    _set_mtime(f, now_ns - 1825 * _DAY_NS)

    # Epoch (1970-01-01)
    f = root / "epoch.txt"
    _create_file(f, b"epoch content")
    _set_mtime(f, 0)


def gen_zero_timestamps(root: Path) -> None:
    """Files with timestamps at or near zero."""

    f = root / "zero_mtime.txt"
    _create_file(f, b"zero mtime")
    _set_mtime(f, 0)

    f = root / "zero_mtime_atime.txt"
    _create_file(f, b"zero atime and mtime")
    _set_mtime(f, 0)


def gen_ctime_gt_mtime(root: Path) -> None:
    """Files where ctime > mtime (common after copy/restore)."""

    # Create file, set mtime to past, then chmod to advance ctime
    f = root / "copied.txt"
    _create_file(f, b"this file was copied")
    past_ns = int((time.time() - 90 * 86400) * _NS_IN_SEC)
    _set_mtime(f, past_ns)
    # chmod advances ctime but keeps mtime
    _touch_via_chmod(f)

    f = root / "restored.txt"
    _create_file(f, b"restored from backup")
    older_ns = int((time.time() - 365 * 86400) * _NS_IN_SEC)
    _set_mtime(f, older_ns)
    _touch_via_chmod(f)


def gen_extreme_values(root: Path) -> None:
    """Files with extreme or boundary metadata values."""

    # Near-INT64_MAX mtime (year ~292277)
    f = root / "huge.txt"
    _create_file(f, b"extreme future")
    _set_mtime(f, 9_223_372_036_854_775_807)

    # Empty file with all-zero timestamps
    f = root / "empty.txt"
    _create_file(f, b"")
    _set_mtime(f, 0)


def gen_cross_platform(root: Path) -> None:
    """Files with various line-ending styles for hash matching tests."""
    root.mkdir(parents=True, exist_ok=True)

    (root / "lf.txt").write_bytes(b"line1\nline2\nline3\n")
    (root / "crlf.txt").write_bytes(b"line1\r\nline2\r\nline3\r\n")
    (root / "mixed.txt").write_bytes(b"line1\r\nline2\nline3\r\n")
    # Binary: has null byte in first 512 bytes
    (root / "binary.bin").write_bytes(b"\x00\x01\x02\r\n")
    # Script with null byte but known-text extension
    (root / "script.py").write_bytes(b"#!/usr/bin/env python3\x00stuff\n")


def gen_unicode_paths(root: Path) -> None:
    """Files with various Unicode path types."""

    # NFD path — on Linux this creates a literal NFD filename
    # (unlike macOS which may normalise)
    _create_file(root / "cafe\u0301.txt", b"nfd cafe")

    # CJK
    _create_file(root / "\u4e2d\u6587.txt", b"chinese")  # 中文
    _create_file(root / "\u65e5\u672c\u8a9e.txt", b"japanese")  # 日本語
    _create_file(root / "\ud55c\uad6d\uc5b4.txt", b"korean")  # 한국어

    # Emoji
    _create_file(root / "\U0001f600.txt", b"emoji")  # 😀
    _create_file(root / "\U0001f680.txt", b"rocket")  # 🚀

    # RTL
    _create_file(root / "\u0627\u0644\u0639\u0631\u0628\u064a\u0629.txt", b"arabic")  # العربية
    _create_file(root / "\u05e2\u05d1\u05e8\u05d9\u05ea.txt", b"hebrew")  # עברית


def gen_deep_nesting(root: Path) -> None:
    """Deeply nested directory structure."""
    deep_path = root / "a" / "b" / "c" / "d" / "e" / "f" / "g" / "h" / "i" / "j" / "deep.txt"
    _create_file(deep_path, b"deeply nested file")


# ── Main generator ───────────────────────────────────────────────────────────


def generate(output_dir: Path, *, overwrite: bool = False) -> Path:
    """Generate the full corrupted-metadata fixture tree.

    Parameters
    ----------
    output_dir:
        Destination directory for the generated tree.
    overwrite:
        If True, remove *output_dir* if it already exists.

    Returns
    -------
    The *output_dir* path (resolved).
    """
    output_dir = output_dir.resolve()

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            print(f"Error: {output_dir} already exists. Use --force to overwrite.", file=sys.stderr)
            sys.exit(1)

    output_dir.mkdir(parents=True)

    # Top-level empty file
    _create_file(output_dir / "empty_file.txt", b"")

    # 10 MB zero file (for large-tree stress tests)
    _create_file(output_dir / "large_file.bin", b"\x00" * 10_000_000)

    # Categories
    gen_shifted_timestamps(output_dir / "shifted_timestamps")
    gen_zero_timestamps(output_dir / "zero_timestamps")
    gen_ctime_gt_mtime(output_dir / "ctime_gt_mtime")
    gen_extreme_values(output_dir / "extreme_values")
    gen_cross_platform(output_dir / "cross_platform")
    gen_unicode_paths(output_dir / "unicode_paths")
    gen_deep_nesting(output_dir / "deep_nesting")

    return output_dir


# ── CLI entry point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate corrupted-metadata fixture tree for FolderHistory testing",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Destination directory for the generated tree",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing output directory",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print summary of what will be generated and exit",
    )

    args = parser.parse_args()

    if args.info:
        print("Corrupted metadata fixture generator")
        print("=" * 40)
        print("Categories:")
        print("  shifted_timestamps/  — mtime ±1y, ±5y, epoch")
        print("  zero_timestamps/     — mtime = 0")
        print("  ctime_gt_mtime/      — ctime > mtime scenarios")
        print("  extreme_values/      — near-INT64_MAX mtime, zero-size")
        print("  cross_platform/      — LF, CRLF, mixed, binary, text-override")
        print("  unicode_paths/       — NFD, CJK, emoji, RTL")
        print("  deep_nesting/        — 10-level deep nesting")
        print()
        print("Plus: empty_file.txt, large_file.bin")
        print()
        print("Estimated files: 22")
        print("Estimated size:  ~10 MB")
        return

    if args.output_dir is None:
        parser.error("output_dir is required (use --info for help)")

    result = generate(args.output_dir, overwrite=args.force)

    # Count generated files
    file_count = sum(1 for p in result.rglob("*") if p.is_file())
    print(f"Generated {file_count} files in {result}")
    print(f"Run with: python -c \"from folderhistory.core.ingest import ingest_snapshot; "
          f"from pathlib import Path; "
          f"snap = ingest_snapshot(Path('{result}')); "
          f"print(f'{{len(snap.files)}} files ingested')\"")


if __name__ == "__main__":
    main()
