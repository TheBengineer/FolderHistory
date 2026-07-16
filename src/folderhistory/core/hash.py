"""Hashing utilities for FolderHistory.

Provides BLAKE3 and XXH64 hashing, plus CRLF-normalised content hashing
for text files.
"""

from __future__ import annotations

from pathlib import Path

import blake3
import xxhash

from folderhistory.types import LineEnding

# ── Public helpers ───────────────────────────────────────────────────────────


def hash_file(path: Path) -> tuple[str, str]:
    """Read *path* from disk and return ``(raw_blake3, xxhash64)`` hex digests."""
    data = path.read_bytes()
    return hash_bytes(data)


def hash_bytes(data: bytes) -> tuple[str, str]:
    """Return ``(raw_blake3, xxhash64)`` hex digests for arbitrary byte data."""
    b3: str = blake3.blake3(data).hexdigest()
    xx: str = xxhash.xxh64(data).hexdigest()
    return (b3, xx)


# ── Normalised content hashing ───────────────────────────────────────────────

_NULL_BYTE_CHECK_SIZE: int = 512
"""Number of bytes to scan for a null byte when detecting binary content."""

_LE_SAMPLE_SIZE: int = 8192
"""Number of bytes to sample for line-ending classification."""


def compute_normalized_hash(
    data: bytes,
    *,
    known_text: bool = False,
) -> tuple[str | None, LineEnding]:
    """Compute BLAKE3 after CRLF → LF normalisation, and classify line endings.

    Parameters
    ----------
    data:
        Raw file content to analyse.
    known_text:
        When *True*, the null-byte binary-detection check is skipped.
        Useful when the caller already knows the file type (e.g. by extension).

    Returns
    -------
    ``(normalized_blake3 or None, line_ending)`` where *line_ending* is one
    of ``"binary"``, ``"crlf"``, ``"lf"``, or ``"mixed"``.
    """
    # ── Binary-content detection ──────────────────────────────────────────
    if not known_text and data:
        # Check the first _NULL_BYTE_CHECK_SIZE bytes for a null byte.
        if b"\x00" in data[:_NULL_BYTE_CHECK_SIZE]:
            return (None, "binary")

    # ── Line-ending classification ────────────────────────────────────────
    sample = data[:_LE_SAMPLE_SIZE]
    crlf_count = sample.count(b"\r\n")
    # Bare \n = total \n minus those already counted as part of \r\n
    bare_lf_count = sample.count(b"\n") - crlf_count

    line_ending: LineEnding
    if crlf_count > 0 and bare_lf_count > 0:
        line_ending = "mixed"
    elif crlf_count > 0:
        line_ending = "crlf"
    else:
        # Covers bare \n only, no line endings at all, and empty files.
        line_ending = "lf"

    # ── Normalised hash ───────────────────────────────────────────────────
    normalised = data.replace(b"\r\n", b"\n")
    normalised_hash: str = blake3.blake3(normalised).hexdigest()

    return (normalised_hash, line_ending)
