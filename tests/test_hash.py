"""Tests for :mod:`folderhistory.core.hash`."""

from __future__ import annotations

from pathlib import Path

import pytest

from folderhistory.core.hash import (
    compute_normalized_hash,
    hash_bytes,
    hash_file,
)

# ── Known test vectors ───────────────────────────────────────────────────────

BLAKE3_EMPTY = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
BLAKE3_HELLO = "ea8f163db38682925e4491c5e58d4bb3506ef8c14eb78a86e908c5624a67200f"
XXH64_EMPTY = "ef46db3751d8e999"
XXH64_HELLO = "26c7827d889f6da3"
BLAKE3_HELLO_LF = BLAKE3_HELLO  # "hello" has no CRLF → same as raw


# ── hash_bytes ───────────────────────────────────────────────────────────────


class TestHashBytes:
    def test_empty_bytes(self) -> None:
        b3, xx = hash_bytes(b"")
        assert b3 == BLAKE3_EMPTY
        assert xx == XXH64_EMPTY

    def test_hello(self) -> None:
        b3, xx = hash_bytes(b"hello")
        assert b3 == BLAKE3_HELLO
        assert xx == XXH64_HELLO

    def test_deterministic(self) -> None:
        """Same input always produces the same output."""
        b3_a, xx_a = hash_bytes(b"the quick brown fox")
        b3_b, xx_b = hash_bytes(b"the quick brown fox")
        assert b3_a == b3_b
        assert xx_a == xx_b

    def test_different_inputs_different_hashes(self) -> None:
        b3_a, _ = hash_bytes(b"hello")
        b3_b, _ = hash_bytes(b"world")
        assert b3_a != b3_b


# ── hash_file ────────────────────────────────────────────────────────────────


class TestHashFile:
    def test_roundtrip_identical_to_hash_bytes(self, tmp_path: Path) -> None:
        p = tmp_path / "test.txt"
        p.write_bytes(b"hello")
        file_b3, file_xx = hash_file(p)
        byte_b3, byte_xx = hash_bytes(b"hello")
        assert file_b3 == byte_b3
        assert file_xx == byte_xx

    def test_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope"
        with pytest.raises(FileNotFoundError):
            hash_file(missing)


# ── compute_normalized_hash ──────────────────────────────────────────────────


class TestComputeNormalizedHash:
    """CRLF → LF normalisation and line-ending classification."""

    # ── LF content (no change) ────────────────────────────────────────────

    def test_lf_no_change(self) -> None:
        """LF-only content → normalized hash matches raw hash."""
        data = b"hello\nworld\n"
        normalized_hash, line_ending = compute_normalized_hash(data)
        raw_hash, _ = hash_bytes(data)
        assert normalized_hash == raw_hash
        assert line_ending == "lf"

    def test_no_newlines(self) -> None:
        """Content with no newlines → lf (default for text)."""
        data = b"hello world"
        normalized_hash, line_ending = compute_normalized_hash(data)
        raw_hash, _ = hash_bytes(data)
        assert normalized_hash == raw_hash
        assert line_ending == "lf"

    def test_empty(self) -> None:
        """Empty content → lf."""
        normalized_hash, line_ending = compute_normalized_hash(b"")
        raw_hash, _ = hash_bytes(b"")
        assert normalized_hash == raw_hash
        assert line_ending == "lf"

    # ── CRLF content ──────────────────────────────────────────────────────

    def test_crlf_normalised_matches_lf_hash(self) -> None:
        """CRLF → LF normalised hash equals hash of LF version."""
        crlf_data = b"hello\r\nworld\r\n"
        lf_data = b"hello\nworld\n"

        crlf_normalized, line_ending = compute_normalized_hash(crlf_data)
        lf_normalized, _ = compute_normalized_hash(lf_data)

        assert crlf_normalized == lf_normalized
        assert line_ending == "crlf"

    def test_crlf_raw_differs_from_lf(self) -> None:
        """Raw hash of CRLF file differs from raw hash of LF file."""
        crlf_hash, _ = hash_bytes(b"hello\r\nworld\r\n")
        lf_hash, _ = hash_bytes(b"hello\nworld\n")
        assert crlf_hash != lf_hash

    # ── Binary detection ──────────────────────────────────────────────────

    def test_binary_with_null_byte(self) -> None:
        """Null byte in first 512 bytes → (None, 'binary')."""
        data = b"AB\x00CD"
        normalized_hash, line_ending = compute_normalized_hash(data)
        assert normalized_hash is None
        assert line_ending == "binary"

    def test_binary_longer(self) -> None:
        """Binary file longer than 512 bytes with null early on."""
        data = b"\x00" + b"A" * 1000
        normalized_hash, line_ending = compute_normalized_hash(data)
        assert normalized_hash is None
        assert line_ending == "binary"

    def test_binary_no_null_in_first_512(self) -> None:
        """Null byte beyond 512 bytes → NOT treated as binary."""
        # 600 bytes: first 512 are clean, null at position 550
        data = b"A" * 550 + b"\x00" + b"B" * 49
        normalized_hash, line_ending = compute_normalized_hash(data)
        assert normalized_hash is not None
        assert line_ending == "lf"

    # ── known_text override ───────────────────────────────────────────────

    def test_known_text_skips_null_byte_check(self) -> None:
        """known_text=True → null byte ignored, text processing proceeds."""
        data = b"hello\x00world\n"
        normalized_hash, line_ending = compute_normalized_hash(
            data,
            known_text=True,
        )
        assert normalized_hash is not None
        assert line_ending == "lf"

    # ── Mixed line endings ────────────────────────────────────────────────

    def test_mixed_line_endings(self) -> None:
        """Both CRLF and bare LF → 'mixed'."""
        data = b"line1\r\nline2\nline3\r\n"
        normalized_hash, line_ending = compute_normalized_hash(data)
        assert normalized_hash is not None
        assert line_ending == "mixed"
        # Verify the normalised hash matches the all-LF version
        lf_data = b"line1\nline2\nline3\n"
        lf_hash, _ = hash_bytes(lf_data)
        assert normalized_hash == lf_hash

    def test_crlf_only(self) -> None:
        """Only CRLF endings → 'crlf'."""
        data = b"a\r\nb\r\n"
        normalized_hash, line_ending = compute_normalized_hash(data)
        assert line_ending == "crlf"
        lf_data = b"a\nb\n"
        lf_hash, _ = hash_bytes(lf_data)
        assert normalized_hash == lf_hash

    # ── Large content ─────────────────────────────────────────────────────

    def test_large_file_sampling(self) -> None:
        """Line-ending detection on first 8KB, but normalisation covers whole file."""
        # Build 20KB of LF content, then a CRLF at the very end (beyond 8KB sample)
        body = b"line\n" * 2000  # ~12KB
        data = body + b"last\r\n"
        normalized_hash, line_ending = compute_normalized_hash(data)
        # Sample (first 8KB) sees only LF → classifies as "lf"
        assert line_ending == "lf"
        # But normalisation still replaces ALL CRLFs → hash matches all-LF
        lf_data = body + b"last\n"
        lf_hash, _ = hash_bytes(lf_data)
        assert normalized_hash == lf_hash
