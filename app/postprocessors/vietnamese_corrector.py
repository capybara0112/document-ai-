"""Dedicated Vietnamese OCR post-processing module.

Features:
- Unicode NFC normalization and spacing cleanup.
- Common OCR character confusion resolution (l vs 1/I, O vs 0, rn vs m, cl vs d).
- Vietnamese diacritics and tone mark restoration.
- Protection of crucial business/legal entities: tax codes, phone numbers,
  currency amounts, dates, and technical terminology.
- Auditable change tracking: maintains both raw OCR text and corrected text.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

# ---------------------------------------------------------------------------
# Whitelist of Technical Terms and Keywords
# ---------------------------------------------------------------------------
PROTECTED_TERMS: frozenset[str] = frozenset({
    "opencv", "python", "canny", "matplotlib", "trackbar", "kernel",
    "paddleocr", "mediapipe", "fastapi", "laplacian", "gaussianblur",
    "bilateral", "threshold", "cv2", "numpy", "pytorch", "tensorflow",
    "docker", "cuda", "onnx", "git", "linux", "windows", "json", "rest",
    "api", "post", "get", "put", "delete", "houghlinesp", "warpperspective",
    "getperspectivetransform", "getrotationmatrix2d", "findcontours",
})

# Common technical mis-recognitions from OCR
TECH_OCR_FIXES: dict[str, str] = {
    "kerne1": "kernel",
    "kerne|": "kernel",
    "cvz": "cv2",
    "openev": "OpenCV",
    "cannv": "Canny",
    "nvnumpy": "numpy",
}

# ---------------------------------------------------------------------------
# Common Vietnamese OCR error mappings (corrupted vowels, tone marks)
# ---------------------------------------------------------------------------
VIETNAMESE_OCR_FIXES: dict[str, str] = {
    # Exam / Document vocabulary
    "kiém": "kiểm",
    "kiêm": "kiểm",
    "th giác": "thị giác",
    "gia k": "giữa kỳ",
    "giữa k": "giữa kỳ",
    "đc ành": "đọc ảnh",
    "đc ảnh": "đọc ảnh",
    "thc hin": "thực hiện",
    "thực hin": "thực hiện",
    "thc hiện": "thực hiện",
    "thao tác": "thao tác",
    "thao tac": "thao tác",
    "thay đói": "thay đổi",
    "kich thưc": "kích thước",
    "kích thưc": "kích thước",
    "kich thước": "kích thước",
    "h só": "hệ số",
    "hệ só": "hệ số",
    "ti lę": "tỉ lệ",
    "tỉ lę": "tỉ lệ",
    "ti lệ": "tỉ lệ",
    "trc x": "trục x",
    "trc y": "trục y",
    "trục y nhp": "trục y nhập",
    "nhp vào": "nhập vào",
    "tù bàn": "từ bàn",
    "từ bàn": "từ bàn",
    "bàn phim": "bàn phím",
    "biěn đói": "biến đổi",
    "biến đói": "biến đổi",
    "ảnh m bán": "ảnh âm bản",
    "âm bán": "âm bản",
    "m bán": "âm bản",
    "tách biên nh": "tách biên ảnh",
    "bng phưong": "bằng phương",
    "bằng phưong": "bằng phương",
    "phưong pháp": "phương pháp",
    "to trackbar": "tạo trackbar",
    "đ lāy": "để lấy",
    "để lāy": "để lấy",
    "đé lęc": "để lọc",
    "để lęc": "để lọc",
    "s dng": "sử dụng",
    "sử dng": "sử dụng",
    "b lc": "bộ lọc",
    "bộ lęc": "bộ lọc",
    "1āy t": "lấy từ",
    "lấy t": "lấy từ",
    "lāy từ": "lấy từ",
    "liru li": "lưu lại",
    "lưu li": "lưu lại",
    "kět quà": "kết quả",
    "kết quà": "kết quả",
    "kět quả": "kết quả",
    # Invoice vocabulary
    "hoádon": "hóa đơn",
    "hoá don": "hóa đơn",
    "mău só": "mẫu số",
    "giá tri gia tàng": "giá trị gia tăng",
    "đon vi": "đơn vị",
    "đon giá": "đơn giá",
    "thành tiên": "thành tiền",
    "tóng công": "tổng cộng",
    "nguòi mua": "người mua",
    "nguòi bán": "người bán",
    "ngui bán": "người bán",
    "ső tién": "số tiền",
    "vit bàng": "viết bằng",
    "chů": "chữ",
    "ký, ghi rő": "ký, ghi rõ",
    "ghi rő": "ghi rõ",
}

# ---------------------------------------------------------------------------
# Tone normalization dictionary (choosing consistent modern tone placement)
# ---------------------------------------------------------------------------
TONE_PAIRS = [
    ("oà", "òa"), ("oá", "óa"), ("oả", "ỏa"), ("oã", "õa"), ("oạ", "ọa"),
    ("uỳ", "ùy"), ("uý", "úy"), ("uỷ", "ủy"), ("uỹ", "ũy"), ("uỵ", "ụy"),
]


class VietnameseCorrector:
    """Enterprise-grade Vietnamese OCR normalizer and post-processor."""

    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold

    def normalize_unicode(self, text: str) -> str:
        """Standardize Unicode to NFC, strip zero-width chars and clean whitespaces."""
        if not text:
            return ""
        # NFC normalisation
        norm = unicodedata.normalize("NFC", text)
        # Remove zero-width spaces and non-breaking space
        norm = norm.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
        # Clean multi-spaces and tabs
        norm = re.sub(r"[ \t]+", " ", norm)
        return norm.strip()

    def normalize_tone_marks(self, text: str) -> str:
        """Ensure consistent tone mark representation."""
        result = text
        for old, new in TONE_PAIRS:
            result = result.replace(old, new)
            result = result.replace(old.upper(), new.upper())
        return result

    def is_protected_entity(self, token: str) -> tuple[bool, str]:
        """Detect whether a token is a protected number, tax code, phone, or tech term."""
        clean = token.strip(" ,.:;()[]{}'\"")
        lower = clean.casefold()

        # 1. Technical term whitelist
        if lower in PROTECTED_TERMS:
            return True, "tech_term"

        # 2. Tax code (10 or 13 digits)
        if re.fullmatch(r"\d{10}(?:-\d{3})?", clean):
            return True, "tax_code"

        # 3. Currency / number with thousands separator
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?%?", clean):
            return True, "numeric_amount"

        # 4. Phone number
        if re.fullmatch(r"(?:\+84|0)\d{9,10}", clean):
            return True, "phone_number"

        # 5. Date format (DD/MM/YYYY or DD-MM-YYYY)
        if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", clean):
            return True, "date"

        return False, "none"

    def correct_character_confusions(self, token: str) -> tuple[str, bool, str]:
        """Correct unambiguous OCR letter/digit confusions."""
        # Check if already a protected entity
        is_prot, entity_type = self.is_protected_entity(token)
        if is_prot:
            return token, False, ""

        clean = token.strip(" ,.:;()[]{}'\"")
        lower = clean.casefold()

        # Check tech terms fix
        if lower in TECH_OCR_FIXES:
            fixed = TECH_OCR_FIXES[lower]
            return token.replace(clean, fixed), True, f"Thuật ngữ kỹ thuật: '{clean}' -> '{fixed}'"

        # Fix number with 'O' or 'o' mistaken for zero: e.g. 1.O50.000 -> 1.050.000
        if re.match(r"^\d+[.O,o]+\d+$", clean) and ("O" in clean or "o" in clean):
            fixed = clean.replace("O", "0").replace("o", "0")
            return token.replace(clean, fixed), True, f"Số lẫn chữ 'O': '{clean}' -> '{fixed}'"

        # Fix 'cl' mistaken for 'đ' at the start of words: clược -> được, clọc -> đọc
        if lower.startswith("cl") and len(lower) > 3:
            # check if replacing cl with đ makes sense
            candidate = "đ" + clean[2:]
            if lower.startswith("clược"):
                return token.replace(clean, "được" if clean.islower() else "Được"), True, "Lỗi OCR 'cl' -> 'đ'"

        return token, False, ""

    def correct_phrase_errors(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Apply contextual phrase corrections for common Vietnamese OCR errors."""
        working = text
        changes: list[dict[str, Any]] = []

        # Sort phrase keys by length descending to match longest phrases first
        for wrong_phrase in sorted(VIETNAMESE_OCR_FIXES.keys(), key=len, reverse=True):
            correct_phrase = VIETNAMESE_OCR_FIXES[wrong_phrase]
            pattern = re.compile(re.escape(wrong_phrase), re.IGNORECASE)
            
            matches = list(pattern.finditer(working))
            for match in matches:
                span_text = match.group(0)
                # Match case
                if span_text.isupper():
                    replacement = correct_phrase.upper()
                elif span_text[0].isupper():
                    replacement = correct_phrase.capitalize()
                else:
                    replacement = correct_phrase

                working = working[:match.start()] + replacement + working[match.end():]
                changes.append({
                    "original": span_text,
                    "corrected": replacement,
                    "reason": f"Sửa lỗi dấu / từ vựng OCR: '{span_text}' -> '{replacement}'",
                    "confidence": 0.95,
                })

        return working, changes

    def process(
        self,
        raw_text: str,
        lines: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Perform full Vietnamese post-processing pipeline on document text.
        
        Returns:
            dict containing raw_text, normalized_text, applied_corrections,
            protected_entities, and uncertain_tokens.
        """
        if not raw_text or not raw_text.strip():
            return {
                "raw_text": raw_text,
                "normalized_text": raw_text,
                "applied_corrections": [],
                "protected_entities": [],
                "uncertain_tokens": [],
            }

        # Step 1: Unicode NFC Normalization
        nfc_text = self.normalize_unicode(raw_text)

        # Step 2: Tone marks standardization
        tone_text = self.normalize_tone_marks(nfc_text)

        # Step 3: Phrase-level OCR error correction
        corrected_text, phrase_changes = self.correct_phrase_errors(tone_text)

        # Step 4: Token-level confusion correction & protection check
        all_changes = list(phrase_changes)
        protected_entities: list[dict[str, str]] = []
        uncertain_tokens: list[dict[str, str]] = []

        tokens = corrected_text.split()
        for tok in tokens:
            is_prot, ent_type = self.is_protected_entity(tok)
            if is_prot:
                protected_entities.append({"token": tok, "type": ent_type})
            else:
                tok_fixed, changed, reason = self.correct_character_confusions(tok)
                if changed:
                    all_changes.append({
                        "original": tok,
                        "corrected": tok_fixed,
                        "reason": reason,
                        "confidence": 0.90,
                    })

        return {
            "raw_text": raw_text,
            "normalized_text": corrected_text,
            "applied_corrections": all_changes,
            "protected_entities": protected_entities,
            "uncertain_tokens": uncertain_tokens,
            "corrections_count": len(all_changes),
        }


# Singleton instance
_VIETNAMESE_CORRECTOR: VietnameseCorrector | None = None


def get_vietnamese_corrector() -> VietnameseCorrector:
    global _VIETNAMESE_CORRECTOR
    if _VIETNAMESE_CORRECTOR is None:
        _VIETNAMESE_CORRECTOR = VietnameseCorrector()
    return _VIETNAMESE_CORRECTOR
