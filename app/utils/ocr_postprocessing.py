"""Reading order, conservative normalization, and auditable OCR review signals."""
from __future__ import annotations

import re
import unicodedata
from difflib import get_close_matches
from statistics import median
from typing import Any

from app.correction import get_correction_service
from app.postprocessors import get_vietnamese_corrector

# ---------------------------------------------------------------------------
# Known technical terms to detect near-misses via fuzzy matching
# ---------------------------------------------------------------------------
_TECHNICAL_TERMS = (
    "OpenCV", "Matplotlib", "Gaussian Blur", "Bilateral Filter", "Canny",
    "Laplacian", "PaddleOCR", "MediaPipe", "Trackbar", "Kernel",
    "GaussianBlur", "cvtColor", "imread", "imshow", "imwrite",
    "threshold", "findContours", "drawContours", "warpPerspective",
    "getPerspectiveTransform", "getRotationMatrix2D", "HoughLinesP",
)

# ---------------------------------------------------------------------------
# Extended Vietnamese unaccented → accented mapping.
# These are *review signals* only – the system never auto-replaces.
# Covers common words that OCR tends to strip diacritics from.
# ---------------------------------------------------------------------------
_UNACCENTED_VIETNAMESE: dict[str, str] = {
    # Pronouns / conjunctions / prepositions
    "la": "là", "co": "có", "tu": "từ", "den": "đến", "tren": "trên",
    "duoi": "dưới", "trong": "trong", "ngoai": "ngoài", "va": "và",
    "hay": "hay", "hoac": "hoặc", "neu": "nếu", "khi": "khi",
    "voi": "với", "ve": "về", "cho": "cho", "cua": "của", "ma": "mà",
    "thi": "thì", "de": "để", "sau": "sau", "truoc": "trước",
    "nhan": "nhận", "gui": "gửi", "xem": "xem", "biet": "biết",
    # Common nouns / verbs
    "anh": "ảnh", "cau": "câu", "doc": "đọc", "thuc": "thực",
    "hien": "hiện", "cac": "các", "phep": "phép", "bien": "biên",
    "dung": "dùng", "nhap": "nhập", "ban": "bản", "phim": "phím",
    "goc": "góc", "trai": "trái", "chieu": "chiều", "rong": "rộng",
    "thao": "thao", "tac": "tác", "tach": "tách", "hinh": "hình",
    "doi": "đổi", "bai": "bài", "tap": "tập", "ket": "kết",
    "qua": "quả", "diem": "điểm", "cong": "công", "viet": "viết",
    "nam": "năm", "hoc": "học", "sinh": "sinh", "giao": "giáo",
    "vien": "viên", "truong": "trường", "lop": "lớp", "mon": "môn",
    "toan": "toán", "ly": "lý", "su": "sử", "dia": "địa",
    "ngu": "ngữ", "van": "văn", "lich": "lịch", "khoa": "khoa",
    "ky": "kỳ", "thi": "thi", "kiem": "kiểm", "tra": "tra",
    "bao": "báo", "cao": "cáo", "don": "đơn", "vi": "vị",
    "so": "số", "ngay": "ngày", "thang": "tháng", "gio": "giờ",
    "phut": "phút", "luong": "lượng", "gia": "giá", "tri": "trị",
    "tong": "tổng", "tiền": "tiền", "hang": "hàng", "hoa": "hóa",
    "don": "đơn", "nhan": "nhãn", "ma": "mã", "so": "số",
    # Computer vision / CV terms in Vietnamese context
    "bien": "biên", "canh": "cạnh", "goc": "góc", "vung": "vùng",
    "phat": "phát", "hien": "hiện", "nhan": "nhận", "dang": "dạng",
    "mau": "màu", "sac": "sắc", "do": "độ", "sang": "sáng",
    "net": "nét", "ro": "rõ", "min": "mịn", "nhieu": "nhiễu",
    "loc": "lọc", "xu": "xử", "ly": "lý", "anh": "ảnh",
    "bo": "bộ", "luu": "lưu", "dich": "dịch", "chuyen": "chuyển",
    "phuong": "phương", "phap": "pháp", "thuat": "thuật", "toan": "toán",
}

# Normalize keys to casefold for O(1) lookup
_UNACCENTED_KEYS: frozenset[str] = frozenset(_UNACCENTED_VIETNAMESE)


# ---------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------

def _box(line: dict[str, Any]) -> list[float] | None:
    value = line.get("bounding_box")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def _line_height(line: dict[str, Any]) -> float:
    box = _box(line)
    return max(1.0, box[3] - box[1]) if box else 1.0


# ---------------------------------------------------------------------------
# Reading order: row grouping + two-column detection
# ---------------------------------------------------------------------------

def _row_order(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group lines into visual rows then sort left-to-right within each row.

    Tolerance is computed per-row using the *minimum* height of the lines
    already assigned to that row, rather than the global median.  This prevents
    large headings from absorbing adjacent body-text lines.
    """
    # Sort by vertical centre first, horizontal second
    pending = sorted(
        lines,
        key=lambda ln: ((_box(ln)[1] + _box(ln)[3]) / 2, _box(ln)[0]),
    )
    rows: list[list[dict[str, Any]]] = []
    row_centers: list[float] = []

    for line in pending:
        box = _box(line)
        center_y = (box[1] + box[3]) / 2
        line_h = box[3] - box[1]

        placed = False
        for row_index, (row, row_center) in enumerate(zip(rows, row_centers)):
            # Use the *minimum* height in the candidate row to set tolerance –
            # avoids absorbing a small line into a large-heading row.
            min_h_in_row = min(_line_height(r) for r in row)
            tolerance = max(4.0, min(min_h_in_row, line_h) * 0.60)
            if abs(center_y - row_center) <= tolerance:
                rows[row_index].append(line)
                # Update running median center for the row
                row_centers[row_index] = median(
                    (_box(r)[1] + _box(r)[3]) / 2 for r in rows[row_index]
                )
                placed = True
                break
        if not placed:
            rows.append([line])
            row_centers.append(center_y)

    return [item for row in rows for item in sorted(row, key=lambda ln: _box(ln)[0])]


def _column_groups(
    lines: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]] | None, str]:
    """Detect genuine multi-column layout; return None for single-column documents.

    A document is genuinely two-column only if:
    1. At least 6 positioned lines.
    2. There is a distinct vertical gutter between left and right columns.
    3. No lines (or at most 1 anomaly) cross across the gutter between left and right.
    4. Each column has at least 3 lines.
    5. The two columns are NOT vertically aligned as in a table.
    """
    if len(lines) < 6:
        return None, "rows_left_to_right"

    boxes = [_box(ln) for ln in lines if _box(ln)]
    if len(boxes) < 6:
        return None, "rows_left_to_right"

    left = min(b[0] for b in boxes)
    right = max(b[2] for b in boxes)
    page_width = right - left
    if page_width <= 0:
        return None, "rows_left_to_right"

    # Search for a clean vertical splitting line between 30% and 70% of page width
    best_split = None
    min_crossing = len(lines)

    for frac in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        split_x = left + frac * page_width
        left_sub = [ln for ln in lines if _box(ln)[2] <= split_x + 0.04 * page_width]
        right_sub = [ln for ln in lines if _box(ln)[0] >= split_x - 0.04 * page_width]
        crossing = [
            ln for ln in lines
            if _box(ln)[0] < split_x - 0.04 * page_width and _box(ln)[2] > split_x + 0.04 * page_width
        ]
        if len(left_sub) >= 3 and len(right_sub) >= 3:
            if len(crossing) <= max(1, int(len(lines) * 0.08)) and len(crossing) < min_crossing:
                max_left_x = max(_box(ln)[2] for ln in left_sub)
                min_right_x = min(_box(ln)[0] for ln in right_sub)
                gutter = min_right_x - max_left_x
                if gutter >= page_width * 0.04:
                    min_crossing = len(crossing)
                    best_split = (left_sub, right_sub)

    if best_split is None:
        return None, "rows_left_to_right"

    left_group, right_group = best_split

    # Reject table-like layouts where lines in both groups share Y positions
    heights = [_line_height(ln) for ln in lines]
    tolerance = max(5.0, median(heights) * 0.5)
    first_y = [(_box(ln)[1] + _box(ln)[3]) / 2 for ln in left_group]
    second_y = [(_box(ln)[1] + _box(ln)[3]) / 2 for ln in right_group]
    aligned = sum(
        any(abs(y - peer) <= tolerance for peer in second_y) for y in first_y
    ) / max(1, len(first_y))
    if aligned >= 0.70:
        return None, "rows_left_to_right"

    return [left_group, right_group], "two_columns_left_then_right"


def order_lines(
    lines: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Order lines by visual reading order:
    1. Separate top header banner if present so headers are not scattered.
    2. Detect genuine 2-column body or fall back to single-column row order.
    3. Return combined ordered lines and reading strategy.
    """
    positioned = [ln for ln in lines if _box(ln)]
    without_boxes = [ln for ln in lines if not _box(ln)]
    if not positioned:
        return [*lines], "source_detection_order"

    page_top = min(_box(ln)[1] for ln in positioned)
    page_bottom = max(_box(ln)[3] for ln in positioned)
    page_height = page_bottom - page_top
    page_left = min(_box(ln)[0] for ln in positioned)
    page_right = max(_box(ln)[2] for ln in positioned)
    page_width = page_right - page_left

    # Check for a distinct top header region (top 22% of page)
    header_cutoff = page_top + 0.22 * page_height
    header_lines: list[dict[str, Any]] = []
    body_lines: list[dict[str, Any]] = []

    for ln in positioned:
        b = _box(ln)
        text_sample = str(ln.get("text", "")).casefold()
        is_top = b[3] <= header_cutoff
        is_header_kw = bool(
            re.search(r"\b(?:kiểm tra|thi|thời gian|bộ giáo dục|trường|khoa|hóa đơn|cộng hòa)\b", text_sample)
        )
        if is_top and (is_header_kw or (b[2] - b[0]) >= 0.45 * page_width):
            header_lines.append(ln)
        else:
            body_lines.append(ln)

    # If header and body were partitioned cleanly
    if header_lines and body_lines:
        ordered_header = _row_order(header_lines)
        columns, body_strategy = _column_groups(body_lines)
        if columns is None:
            ordered_body = _row_order(body_lines)
            strategy = "rows_left_to_right"
        else:
            ordered_body = [item for column in columns for item in _row_order(column)]
            strategy = "two_columns_left_then_right"
        return [*ordered_header, *ordered_body, *without_boxes], strategy

    # Otherwise treat as unified block
    columns, strategy = _column_groups(positioned)
    if columns is None:
        return [*_row_order(positioned), *without_boxes], strategy
    ordered = [item for column in columns for item in _row_order(column)]
    return [*ordered, *without_boxes], strategy


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """NFC normalisation + whitespace cleanup only; no word-level changes."""
    text = unicodedata.normalize("NFC", text).replace("\u00a0", " ").replace("\u200b", "")
    return re.sub(r"[\t\r ]+", " ", text).strip()


# ---------------------------------------------------------------------------
# Language consistency heuristic (review signal, never rewrites)
# ---------------------------------------------------------------------------

def _language_consistency(text: str) -> dict[str, Any]:
    """Heuristic review signal. Never rewrites OCR text."""
    normalized = _normalize(text)
    if not normalized:
        return {"score_0_1": 0.0, "signals": ["empty_text"], "possible_unaccented_tokens": []}

    printable = sum(ch.isprintable() for ch in normalized) / len(normalized)
    tokens = re.findall(r"[A-Za-zÀ-ỹĐđ]+", normalized)
    weird = normalized.count("") + len(re.findall(r"([^\w\s])\1{2,}", normalized))

    # Detect tokens that match the unaccented Vietnamese word list
    unaccented: list[str] = []
    for token in tokens:
        cf = token.casefold()
        if cf in _UNACCENTED_KEYS:
            # Skip if the *accented* form is already correct
            expected_accented = _UNACCENTED_VIETNAMESE.get(cf, "")
            if token != expected_accented:
                unaccented.append(token)

    # Penalty: each unaccented word reduces score by ~0.05 (capped at 0.35)
    unaccented_penalty = min(0.35, len(unaccented) * 0.05)
    score = max(0.0, min(1.0, printable - min(0.35, weird * 0.12) - unaccented_penalty))

    signals: list[str] = []
    if weird:
        signals.append("unexpected_symbol_sequence")
    if unaccented:
        signals.append("possible_missing_vietnamese_diacritics")
    return {
        "score_0_1": round(score, 3),
        "signals": signals,
        "possible_unaccented_tokens": unaccented,
    }


# ---------------------------------------------------------------------------
# Candidate summary (used by ocr_adapter for candidate selection)
# ---------------------------------------------------------------------------

def candidate_summary(lines: list[dict[str, Any]]) -> dict[str, Any]:
    text = "\n".join(
        str(ln.get("text", "")).strip()
        for ln in lines
        if str(ln.get("text", "")).strip()
    )
    confidences: list[float] = []
    for ln in lines:
        try:
            confidences.append(float(ln["confidence"]))
        except (KeyError, TypeError, ValueError):
            pass
    language = _language_consistency(text)
    return {
        "recognized_line_count": sum(
            1 for ln in lines if str(ln.get("text", "")).strip()
        ),
        "recognized_character_count": len(re.sub(r"\s+", "", text)),
        "ocr_confidence_mean": (
            round(sum(confidences) / len(confidences), 4) if confidences else None
        ),
        "language_consistency": language,
    }


# ---------------------------------------------------------------------------
# Smart correction suggestions (never applied automatically)
# ---------------------------------------------------------------------------

def _technical_suggestions(text: str) -> list[dict[str, Any]]:
    """Flag tokens that look like mis-recognised technical terms."""
    suggestions: list[dict[str, Any]] = []
    known = [term.casefold().replace(" ", "") for term in _TECHNICAL_TERMS]
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text):
        compact = token.casefold().replace(" ", "")
        matches = get_close_matches(compact, known, n=1, cutoff=0.80)
        if matches and matches[0] != compact:
            candidate = next(
                term
                for term in _TECHNICAL_TERMS
                if term.casefold().replace(" ", "") == matches[0]
            )
            suggestions.append({
                "original": token,
                "proposed": candidate,
                "correction_confidence": 0.60,
                "reason": "Từ gần thuật ngữ kỹ thuật đã biết; chưa tự thay thế.",
            })
    return suggestions


def _accent_suggestions(text: str) -> list[dict[str, Any]]:
    """Flag tokens that appear to be Vietnamese words with missing diacritics."""
    suggestions: list[dict[str, Any]] = []
    for token in re.findall(r"[A-Za-zÀ-ỹĐđ]+", text):
        cf = token.casefold()
        suggestion = _UNACCENTED_VIETNAMESE.get(cf)
        if suggestion and suggestion != token:
            suggestions.append({
                "original": token,
                "proposed": suggestion,
                "correction_confidence": 0.55,
                "reason": (
                    "Dạng không dấu khớp từ tiếng Việt phổ biến; "
                    "cần đối chiếu ảnh gốc trước khi chấp nhận."
                ),
            })
    return suggestions


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Line type classifier and block hierarchy
# ---------------------------------------------------------------------------

def _classify(text: str) -> str:
    """Classify a text line into a structural role."""
    # Question: "Câu 1", "Cau 1", "câu 1", "Bài 1", "Bai 1", "Phần 1"
    if re.match(r"(?i)^(?:c[aâ]u|b[aà]i|phần)\s*\d+", text):
        return "question"
    # Numbered / bulleted list items: "1.", "2)", "a.", "a)", "-", "•", "*", "+"
    if re.match(r"^(?:\d+[.)]\s*|[a-zA-Z][.)]\s*|[-•*+–—]\s*)", text):
        return "list_item"
    # Heading: >= 75 % uppercase letters (ignoring digits / punctuation)
    letters = [ch for ch in text if ch.isalpha()]
    if letters and sum(ch.isupper() for ch in letters) / len(letters) >= 0.75:
        return "heading"
    return "paragraph"


# ---------------------------------------------------------------------------
# Main result builder with Document Hierarchy and AI Language Correction
# ---------------------------------------------------------------------------

def build_analysis_result(
    lines: list[dict[str, Any]],
    image_analysis: dict[str, Any],
    confidence_threshold: float = 0.70,
    ocr_strategy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce separate raw, normalized, and AI-corrected layers with document hierarchy."""
    raw_lines = [
        str(ln.get("text", "")).strip()
        for ln in lines
        if str(ln.get("text", "")).strip()
    ]
    ordered, reading_strategy = order_lines(lines)

    normalized_lines: list[str] = []
    blocks: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    smart_corrections: list[dict[str, Any]] = []

    prev_box: list[float] | None = None
    prev_type: str | None = None

    for number, line in enumerate(ordered, start=1):
        raw = str(line.get("text", "")).strip()
        normalized = _normalize(raw)
        if not normalized:
            continue

        try:
            model_confidence = float(line.get("confidence"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            model_confidence = None

        curr_box = _box(line)
        line_type = _classify(normalized)

        # Detect continuation line: follows a list item / question with normal vertical gap
        is_continuation = False
        if (
            prev_box is not None
            and curr_box is not None
            and line_type == "paragraph"
            and prev_type in {"list_item", "question", "paragraph"}
        ):
            line_h = max(1.0, curr_box[3] - curr_box[1])
            vert_gap = curr_box[1] - prev_box[3]
            # If gap is within 1.6 lines and not jumping far to the left
            if 0 <= vert_gap <= 1.8 * line_h and curr_box[0] >= prev_box[0] - 25:
                is_continuation = True

        language = _language_consistency(normalized)
        corrections = [*_accent_suggestions(normalized), *_technical_suggestions(normalized)]

        normalized_lines.append(normalized)

        # Tokenize line into words for Document -> Page -> Block -> Line -> Word hierarchy
        words = [
            {"text": word, "index": w_idx, "confidence": model_confidence}
            for w_idx, word in enumerate(normalized.split())
        ]

        block_entry = {
            "line_number": number,
            "reading_order_index": number,
            "type": line_type,
            "text": normalized,
            "raw_text": raw,
            "confidence": model_confidence,
            "language_consistency": language,
            "bounding_box": curr_box,
            "polygon": line.get("polygon"),
            "ocr_bounding_box": line.get("ocr_bounding_box"),
            "candidate_id": line.get("candidate", line.get("candidate_id")),
            "coordinate_space": "source_image_coordinates",
            "is_continuation": is_continuation,
            "words": words,
        }
        blocks.append(block_entry)

        if corrections:
            smart_corrections.append({
                "line_number": number,
                "raw_text": raw,
                "suggestions": corrections,
            })

        # Build review entry when any signal is concerning
        reasons: list[str] = []
        if model_confidence is not None and model_confidence < confidence_threshold:
            reasons.append("Điểm confidence do OCR cung cấp thấp.")
        if language["signals"]:
            reasons.append("Dấu hiệu ngôn ngữ: " + ", ".join(language["signals"]) + ".")
        if corrections:
            reasons.append("Có đề xuất sửa, nhưng hệ thống không tự thay nội dung.")
        if reasons:
            review.append({
                "line_number": number,
                "raw_text": raw,
                "normalized_text": normalized,
                "model_confidence": model_confidence,
                "confidence": model_confidence,
                "language_consistency": language,
                "bounding_box": curr_box,
                "suggestions": corrections,
                "reason": " ".join(reasons),
                "status": "CẦN KIỂM TRA",
            })

        prev_box = curr_box
        prev_type = line_type

    # Construct Hierarchical Blocks (grouping continuation lines into logical items)
    hierarchical_blocks: list[dict[str, Any]] = []
    curr_h_block: dict[str, Any] | None = None
    for b in blocks:
        if curr_h_block is not None and b.get("is_continuation"):
            curr_h_block["text"] += " " + b["text"]
            curr_h_block["raw_text"] += " " + b["raw_text"]
            curr_h_block["lines"].append(b)
            # Expand bounding box
            cb = curr_h_block["bounding_box"]
            nb = b["bounding_box"]
            if cb and nb:
                curr_h_block["bounding_box"] = [
                    min(cb[0], nb[0]), min(cb[1], nb[1]),
                    max(cb[2], nb[2]), max(cb[3], nb[3])
                ]
        else:
            curr_h_block = {
                "block_id": len(hierarchical_blocks) + 1,
                "type": b["type"],
                "text": b["text"],
                "raw_text": b["raw_text"],
                "confidence": b["confidence"],
                "bounding_box": list(b["bounding_box"]) if b["bounding_box"] else None,
                "lines": [b],
            }
            hierarchical_blocks.append(curr_h_block)

    normalized_text = "\n".join(normalized_lines)
    raw_text = "\n".join(raw_lines)

    # Phase 5: Vietnamese Rule-based Correction
    vi_corrector = get_vietnamese_corrector()
    vi_result = vi_corrector.process(normalized_text, lines=blocks)
    vi_applied_text = vi_result.get("normalized_text", normalized_text)
    vi_changes = vi_result.get("applied_corrections", [])
    vi_protected = vi_result.get("protected_entities", [])

    # Phase 5: AI Language Correction
    correction_service = get_correction_service()
    ai_result = correction_service.correct_text(vi_applied_text, lines=blocks)
    if ai_result.correction_status == "CORRECTED":
        corrected_text = ai_result.corrected_text
    elif vi_changes:
        corrected_text = vi_applied_text
    else:
        corrected_text = normalized_text
    ai_correction_dict = ai_result.to_dict()

    # Structural metadata
    questions = sorted({
        int(m.group(1))
        for text in normalized_lines
        if (m := re.search(r"(?i)^\s*c[aâ]u\s*(\d+)", text))
    })
    duration = next(
        (
            m.group(0).strip()
            for text in normalized_lines
            if (m := re.search(r"(?i)thời\s*gian[^\n]{0,80}", text))
        ),
        None,
    )
    title = next(
        (block["text"] for block in blocks if block["type"] == "heading"), None
    )

    warnings = list(image_analysis.get("warnings", []))
    if not raw_lines:
        warnings.append(
            "Không nhận dạng được dòng chữ nào; kiểm tra OCR, định dạng và chất lượng ảnh."
        )
    if review:
        warnings.append(
            f"Có {len(review)} dòng cần kiểm tra; "
            "confidence OCR không được hiểu là độ chính xác tuyệt đối."
        )
    if ai_result.warnings:
        warnings.extend(ai_result.warnings)

    before = image_analysis.get("image_quality_before", {})
    normalized_section = {
        "text": normalized_text,
        "blocks": blocks,
        "hierarchical_blocks": hierarchical_blocks,
        "reading_order_strategy": reading_strategy,
        "note": (
            "Chỉ chuẩn hóa Unicode NFC và khoảng trắng, "
            "sau đó khôi phục thứ tự đọc theo bố cục; không tự sửa từ."
        ),
    }
    reconstructed_section = {
        "text": corrected_text,
        "normalized_text": normalized_text,
        "blocks": blocks,
        "hierarchical_blocks": hierarchical_blocks,
        "reading_order_strategy": reading_strategy,
        "ai_correction": ai_correction_dict,
        "vietnamese_corrections": vi_changes,
    }
    report = {
        "A_document_information": {
            "title_candidate": title,
            "duration_candidate": duration,
            "question_count_detected": len(questions),
            "question_numbers_detected": questions,
            "recognized_line_count": len(raw_lines),
            "image_quality": before.get("label", "UNKNOWN"),
            "image_analysis": image_analysis,
            "ocr_strategy": ocr_strategy or {},
        },
        "B_raw_ocr": {
            "text": raw_text,
            "lines_in_source_detection_order": raw_lines,
            "note": (
                "Nội dung OCR thô theo thứ tự mô hình trả về; "
                "không sửa chính tả hoặc sắp xếp."
            ),
        },
        "C_normalized_ocr": normalized_section,
        "C_reconstructed_content": reconstructed_section,
        "D_review_required": review,
        "D_uncertain_parts": review,
        "smart_corrections": smart_corrections,
        "vietnamese_corrections": vi_changes,
        "ai_correction": ai_correction_dict,
        "E_improvement_suggestions": [
            "Đối chiếu ảnh gốc cho chữ viết tay, dấu tiếng Việt, số tiền, "
            "ngày tháng, mã số và mọi dòng CẦN KIỂM TRA.",
            "Chụp lại với đủ sáng, tài liệu phẳng, thấy bốn góc trang "
            "và độ phân giải cao khi chữ nhỏ.",
            "Huấn luyện fine-tune nhận dạng trên dòng tiếng Việt đã được "
            "con người chép đúng nếu lỗi lặp lại theo mẫu tài liệu.",
        ],
        "provenance": {
            "raw_text_is_direct_ocr": True,
            "normalization_is_deterministic": True,
            "smart_corrections_are_not_applied": True,
            "vietnamese_rule_corrections_count": len(vi_changes),
            "ai_correction_applied": (ai_result.correction_status == "CORRECTED"),
            "ai_provider": ai_result.provider,
            "note": (
                "Confidence OCR, chất lượng ảnh và độ phù hợp ngôn ngữ là các tín hiệu "
                "riêng; không tín hiệu nào chứng minh văn bản đúng tuyệt đối."
            ),
        },
    }
    return {
        "text": corrected_text if (ai_result.correction_status == "CORRECTED" or vi_changes) else normalized_text,
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "corrected_text": corrected_text,
        "vietnamese_corrections": vi_changes,
        "protected_entities": vi_protected,
        "ai_correction": ai_correction_dict,
        "lines": ordered,
        "raw_lines": lines,
        "document_structure": {
            "title_candidate": title,
            "duration_candidate": duration,
            "question_numbers_detected": questions,
            "blocks": blocks,
            "hierarchical_blocks": hierarchical_blocks,
            "reading_order_strategy": reading_strategy,
        },
        "uncertain_lines": review,
        "smart_corrections": smart_corrections,
        "analysis_report": report,
        "warnings": list(dict.fromkeys(warnings)),
    }
