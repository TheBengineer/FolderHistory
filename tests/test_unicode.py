"""Tests for Unicode path handling in :mod:`folderhistory.core.ingest`.

Covers NFC / NFD normalisation, CJK characters, emoji, RTL text,
and mixed-encoding directory trees.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from folderhistory.core.ingest import ingest_snapshot


# ── Helpers ──────────────────────────────────────────────────────────────────


def _nfc(s: str) -> str:
    """Return *s* NFC-normalised."""
    return unicodedata.normalize("NFC", s)


def _write(src: Path, rel: str, content: bytes = b"") -> None:
    """Create a file at *src / rel* with *content*, creating parent dirs."""
    p = src / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


# ── NFC normalisation (complements TestNFCNormalization in test_ingest.py) ───


class TestNFCNormalisationExtended:
    """Deeper coverage of NFC/NFD normalisation behaviour."""

    def test_hangul_nfd_to_nfc(self, tmp_path: Path) -> None:
        """Korean hangul syllable in NFD is normalised to NFC."""
        # "한" in NFD = ᄒ(1112) + ᅡ(1161) + ᆫ(11AB)
        nfd = "\u1112\u1161\u11ab"
        nfc = _nfc(nfd)
        _write(tmp_path, f"{nfd}.txt", b"korean")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert nfc + ".txt" in paths
        assert nfd + ".txt" not in paths

    def test_mixed_nfc_nfd_same_directory(self, tmp_path: Path) -> None:
        """Files in the same dir, some NFC some NFD, all appear as NFC."""
        nfd_name = "cafe\u0301.txt"  # NFD "café"
        nfc_name = _nfc(nfd_name)
        _write(tmp_path, nfd_name, b"nfd")
        _write(tmp_path, "normal.txt", b"normal")
        # macOS may have already given us NFC, but forcing NFD via bytes can
        # produce a distinct directory entry on case-sensitive + NFD systems.
        # On Linux the kernel doesn't normalise, so both forms may coexist.
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        # At minimum the NFC form of café must appear.
        assert nfc_name in paths
        assert "normal.txt" in paths

    def test_deeply_nested_nfd(self, tmp_path: Path) -> None:
        """NFD names at multiple nesting levels are all NFC-normalised."""
        nfd_a = "de\u0301"  # "dé" NFD
        nfd_b = "ja\u0301"  # "já" NFD
        nfc_a = _nfc(nfd_a)
        nfc_b = _nfc(nfd_b)
        _write(tmp_path, f"{nfd_a}/{nfd_b}/f.txt", b"deep")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        expected = f"{nfc_a}/{nfc_b}/f.txt"
        assert expected in paths

    def test_multiple_combining_marks(self, tmp_path: Path) -> None:
        """Characters with multiple combining marks are NFC-normalised."""
        # "ấ" = a + combining circumflex + combining acute
        nfd = "a\u0302\u0301"
        nfc = _nfc(nfd)
        _write(tmp_path, f"{nfd}.txt", b"marks")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        # On most systems this composes to "ấ"
        assert nfc + ".txt" in paths, f"Expected {nfc}.txt in {paths}"

    def test_file_content_preserved_across_normalisation(self, tmp_path: Path) -> None:
        """Content hashing is unaffected by path normalisation."""
        nfd_name = "cafe\u0301.txt"
        _write(tmp_path, nfd_name, b"hello caf\xe9")
        snapshot = ingest_snapshot(tmp_path)
        rec = snapshot.files[0]
        assert rec.size == 10
        assert rec.raw_blake3 != ""
        assert rec.normalized_blake3 is not None


# ── CJK characters ──────────────────────────────────────────────────────────


class TestCJK:
    """Chinese, Japanese, and Korean characters in filenames."""

    @pytest.mark.parametrize(
        ("filename", "desc"),
        [
            ("中文.txt", "Chinese (simplified)"),
            ("繁體字.txt", "Chinese (traditional)"),
            ("日本語.txt", "Japanese"),
            ("한국어.txt", "Korean"),
            ("どうもありがとう.txt", "Japanese hiragana"),
            ("サーベイ.txt", "Japanese katakana"),
        ],
    )
    def test_cjk_filenames(self, tmp_path: Path, filename: str, desc: str) -> None:
        _write(tmp_path, filename, desc.encode())
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert filename in paths, f"Expected CJK filename {filename!r} to be present"

    def test_cjk_nested_directories(self, tmp_path: Path) -> None:
        """CJK directory names at multiple levels."""
        _write(tmp_path, "项目/文档/report.txt", b"report")
        _write(tmp_path, "项目/代码/main.py", b"print('hello')")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert "项目/文档/report.txt" in paths
        assert "项目/代码/main.py" in paths

    def test_cjk_mixed_with_ascii(self, tmp_path: Path) -> None:
        _write(tmp_path, "ファイル-2024-08-15.txt", b"mixed")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert "ファイル-2024-08-15.txt" in paths

    def test_cjk_with_spaces(self, tmp_path: Path) -> None:
        _write(tmp_path, "我的 文档.txt", b"spaces")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert "我的 文档.txt" in paths


# ── Emoji in filenames ───────────────────────────────────────────────────────


class TestEmoji:
    """Emoji and other extended Unicode characters in filenames."""

    @pytest.mark.parametrize(
        ("filename", "desc"),
        [
            ("😀.txt", "basic smiley"),
            ("🚀.txt", "rocket"),
            ("📁.txt", "folder emoji"),
            ("🎉.txt", "party popper"),
            ("❤️.txt", "heart + VS16"),
            ("🧑‍💻.txt", "ZWJ sequence (technologist)"),
            ("🏳️‍🌈.txt", "rainbow flag ZWJ sequence"),
        ],
    )
    def test_emoji_filenames(self, tmp_path: Path, filename: str, desc: str) -> None:
        _write(tmp_path, filename, desc.encode())
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert filename in paths, f"Expected emoji filename {filename!r} to be present"

    def test_emoji_in_directory_name(self, tmp_path: Path) -> None:
        _write(tmp_path, "images/🚀/launch.txt", b"go")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert "images/🚀/launch.txt" in paths

    def test_emoji_with_variation_selectors(self, tmp_path: Path) -> None:
        """Emoji with variation selector-16 (text vs emoji style)."""
        # "⚠️" = warning sign + VS16
        filename = "\u26a0\ufe0f.txt"
        _write(tmp_path, filename, b"warning")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert filename in paths


# ── Right-to-left text ───────────────────────────────────────────────────────


class TestRTL:
    """Right-to-left (Arabic, Hebrew, etc.) characters in filenames."""

    @pytest.mark.parametrize(
        ("filename", "desc"),
        [
            ("العربية.txt", "Arabic"),
            ("עברית.txt", "Hebrew"),
            ("فارسی.txt", "Persian"),
            ("اردو.txt", "Urdu"),
            ("پښتو.txt", "Pashto"),
        ],
    )
    def test_rtl_filenames(self, tmp_path: Path, filename: str, desc: str) -> None:
        _write(tmp_path, filename, desc.encode())
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert filename in paths, f"Expected RTL filename {filename!r} to be present"

    def test_rtl_nested(self, tmp_path: Path) -> None:
        _write(tmp_path, "docs/العربية/file.txt", b"rtl nested")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert "docs/العربية/file.txt" in paths

    def test_rtl_mixed_with_ltr(self, tmp_path: Path) -> None:
        """Bidirectional text in the same filename."""
        # Arabic + English
        filename = "ملف-report-v1.txt"
        _write(tmp_path, filename, b"bidi")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert filename in paths


# ── Zero-width and special Unicode characters ────────────────────────────────


class TestSpecialUnicode:
    """Zero-width characters, invisible chars, and other edge cases."""

    def test_zero_width_space(self, tmp_path: Path) -> None:
        """Zero-width space (U+200B) in filename."""
        filename = "hello\u200bworld.txt"
        _write(tmp_path, filename, b"zwsp")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert filename in paths

    def test_zero_width_non_joiner(self, tmp_path: Path) -> None:
        filename = "a\u200cb.txt"
        _write(tmp_path, filename, b"zwnj")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert filename in paths

    def test_bom_in_filename(self, tmp_path: Path) -> None:
        """Byte-order mark (U+FEFF) as filename character."""
        filename = "\ufefffile.txt"
        _write(tmp_path, filename, b"bom")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert filename in paths

    def test_soft_hyphen(self, tmp_path: Path) -> None:
        filename = "shy\xad.txt"
        _write(tmp_path, filename, b"soft hyphen")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert filename in paths


# ── Very long Unicode filenames ──────────────────────────────────────────────


class TestLongUnicodeFilenames:
    """Unicode filenames near filesystem length limits."""

    def test_long_cjk_filename(self, tmp_path: Path) -> None:
        """A filename composed of 80 CJK characters (fits in 255-byte limit)."""
        # Each CJK char is 3 bytes in UTF-8 → 80 × 3 = 240 bytes + ".txt" = 244
        long_name = "文" * 80 + ".txt"
        _write(tmp_path, long_name, b"long cjk")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert long_name in paths

    def test_long_emoji_filename(self, tmp_path: Path) -> None:
        """Emoji are 4 bytes each; 50 emoji = 200 bytes, safe on most FS."""
        long_emoji = "😀" * 50 + ".txt"
        try:
            _write(tmp_path, long_emoji, b"long emoji")
        except OSError:
            pytest.skip("Filesystem does not support long emoji filenames")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert long_emoji in paths

    def test_deep_unicode_nesting(self, tmp_path: Path) -> None:
        """Deep directory nesting with unicode names."""
        parts = ["レベル1", "レベル2", "レベル3", "レベル4"]
        path = "/".join(parts) + "/file.txt"
        _write(tmp_path, path, b"deep")
        snapshot = ingest_snapshot(tmp_path)
        paths = {f.path for f in snapshot.files}
        assert path in paths


# ── Cross-platform path separator normalisation ──────────────────────────────


class TestPathSeparatorNormalisation:
    """Ingested paths use forward slashes regardless of platform."""

    def test_all_forward_slashes(self, tmp_path: Path) -> None:
        _write(tmp_path, "a/b/c.txt", b"nested")
        snapshot = ingest_snapshot(tmp_path)
        assert all("/" in f.path or f.path.count("/") == 0 for f in snapshot.files)
        # The nested file should have forward slashes
        assert "a/b/c.txt" in {f.path for f in snapshot.files}

    def test_no_backslash_in_paths(self, tmp_path: Path) -> None:
        _write(tmp_path, "a/deep/file.txt", b"check")
        snapshot = ingest_snapshot(tmp_path)
        for f in snapshot.files:
            assert "\\" not in f.path, f"Backslash found in path: {f.path}"
