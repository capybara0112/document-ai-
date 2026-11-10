"""Create a searchable source scan, editable OCR reconstruction, and A–E report.

Page 1 preserves the uploaded image and adds an invisible OCR layer. Page 2
contains editable text/vector lines. Further pages hold the analysis report.
"""
from __future__ import annotations

from io import BytesIO
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from PIL import Image
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

_FONT_NAME = "OCRUnicodeEditable"
_MAX_PAGE_SIDE_PT = 792.0


def _ensure_unicode_font() -> str:
    """Register a Unicode TrueType font that includes Vietnamese glyphs."""
    if _FONT_NAME in pdfmetrics.getRegisteredFontNames():
        return _FONT_NAME

    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            pdfmetrics.registerFont(TTFont(_FONT_NAME, candidate))
            return _FONT_NAME

    raise RuntimeError(
        "Không tìm thấy font Unicode để tạo PDF có tiếng Việt. "
        "Trên Windows, hãy kiểm tra C:\\Windows\\Fonts\\arial.ttf."
    )


def _valid_box(
    value: Any,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Return a clipped (x1, y1, x2, y2) OCR box in top-left image coordinates."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not all(isfinite(v) for v in (x1, y1, x2, y2)):
        return None

    x1 = min(float(image_width), max(0.0, x1))
    y1 = min(float(image_height), max(0.0, y1))
    x2 = min(float(image_width), max(0.0, x2))
    y2 = min(float(image_height), max(0.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _collect_lines(
    lines: Sequence[dict[str, Any]] | None,
    image_width: int,
    image_height: int,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    collected = []
    for item in lines or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        box = _valid_box(item.get("bounding_box"), image_width, image_height)
        if text and box:
            collected.append((text, box))
    return sorted(collected, key=lambda row: (row[1][1], row[1][0]))


def _draw_invisible_text(
    pdf: canvas.Canvas,
    *,
    text: str,
    box: tuple[float, float, float, float],
    page_height: float,
    scale: float,
    font_name: str,
) -> None:
    """Put searchable OCR text over the source scan without changing its look."""
    x1, y1, x2, y2 = box
    box_width_pt = (x2 - x1) * scale
    box_height_pt = (y2 - y1) * scale
    if box_width_pt <= 0 or box_height_pt <= 0:
        return

    x_pt = x1 * scale
    y_pt = page_height - y2 * scale + box_height_pt * 0.12
    font_size = max(3.5, min(16.0, box_height_pt * 0.82))
    measured_width = pdfmetrics.stringWidth(text, font_name, font_size)
    horizontal_scale = 100.0
    if measured_width > box_width_pt and measured_width > 0:
        horizontal_scale = max(35.0, min(100.0, box_width_pt / measured_width * 100.0))

    obj = pdf.beginText(x_pt, y_pt)
    obj.setFont(font_name, font_size)
    obj.setTextRenderMode(3)
    if horizontal_scale < 100:
        obj.setHorizScale(horizontal_scale)
    obj.textOut(text)
    pdf.drawText(obj)


def _red_pixel(pixel: tuple[int, int, int]) -> bool:
    red, green, blue = pixel
    # Ignore the pale red paper tint; keep the saturated printed form rules.
    return red >= 105 and red - green >= 28 and red - blue >= 22


def _clusters(positions: list[int], max_gap: int = 2) -> list[list[int]]:
    if not positions:
        return []
    groups = [[positions[0]]]
    for position in positions[1:]:
        if position - groups[-1][-1] <= max_gap + 1:
            groups[-1].append(position)
        else:
            groups.append([position])
    return groups


def _detect_red_rules(image: Image.Image) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Find long red horizontal/vertical strokes that likely form invoice tables.

    The detector uses projection counts rather than a table-specific template, so
    it works with scanned forms whose rules are printed in a saturated red ink.
    Coordinates and extents returned are in original image pixels.
    """
    max_dimension = 1400
    factor = min(1.0, max_dimension / max(image.size))
    if factor < 1.0:
        work_width = max(1, round(image.width * factor))
        work_height = max(1, round(image.height * factor))
        work = image.resize((work_width, work_height), Image.Resampling.BILINEAR)
    else:
        work = image

    width, height = work.size
    pixels = work.load()
    horizontal_counts = [0] * height
    vertical_counts = [0] * width
    for y in range(height):
        for x in range(width):
            if _red_pixel(pixels[x, y][:3]):
                horizontal_counts[y] += 1
                vertical_counts[x] += 1

    horizontal_candidates = [
        y for y, count in enumerate(horizontal_counts) if count >= width * 0.30
    ]
    vertical_candidates = [
        x for x, count in enumerate(vertical_counts) if count >= height * 0.20
    ]

    horizontal_rules = []
    for group in _clusters(horizontal_candidates):
        y = min(group, key=lambda row: horizontal_counts[row])
        active_x = [x for x in range(width) if _red_pixel(pixels[x, y][:3])]
        if len(active_x) < width * 0.30:
            continue
        horizontal_rules.append((
            median(group) / factor,
            min(active_x) / factor,
            max(active_x) / factor,
        ))

    vertical_rules = []
    for group in _clusters(vertical_candidates):
        x = min(group, key=lambda col: vertical_counts[col])
        active_y = [y for y in range(height) if _red_pixel(pixels[x, y][:3])]
        if len(active_y) < height * 0.18:
            continue
        vertical_rules.append((
            median(group) / factor,
            min(active_y) / factor,
            max(active_y) / factor,
        ))

    return horizontal_rules, vertical_rules


def _estimate_ink_color(
    image: Image.Image,
    box: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Approximate the source ink color so red labels and blue handwriting stay distinct."""
    x1, y1, x2, y2 = box
    crop = image.crop((int(x1), int(y1), max(int(x1) + 1, int(x2)), max(int(y1) + 1, int(y2))))
    red_count = 0
    blue_count = 0
    ink_count = 0
    for red, green, blue in crop.convert("RGB").getdata():
        if max(red, green, blue) - min(red, green, blue) < 28:
            continue
        if red > green * 1.18 and red > blue * 1.12:
            red_count += 1
        elif blue > red * 1.08 and blue > green * 1.08:
            blue_count += 1
        ink_count += 1

    if ink_count and red_count > blue_count * 1.5:
        return 0.68, 0.08, 0.11
    if ink_count and blue_count > red_count * 1.3:
        return 0.12, 0.20, 0.48
    return 0.16, 0.16, 0.16


def _draw_editable_text(
    pdf: canvas.Canvas,
    *,
    text: str,
    box: tuple[float, float, float, float],
    image: Image.Image,
    page_height: float,
    scale: float,
    font_name: str,
) -> None:
    x1, y1, x2, y2 = box
    box_width_pt = (x2 - x1) * scale
    box_height_pt = (y2 - y1) * scale
    if box_width_pt <= 0 or box_height_pt <= 0:
        return

    font_size = max(4.0, min(30.0, box_height_pt * 0.78))
    measured_width = pdfmetrics.stringWidth(text, font_name, font_size)
    horizontal_scale = 100.0
    if measured_width > box_width_pt and measured_width > 0:
        horizontal_scale = max(35.0, min(100.0, box_width_pt / measured_width * 100.0))

    color = _estimate_ink_color(image, box)
    baseline = page_height - y2 * scale + max(0.0, box_height_pt - font_size) * 0.18
    obj = pdf.beginText(x1 * scale, baseline)
    obj.setFont(font_name, font_size)
    obj.setFillColorRGB(*color)
    obj.setTextRenderMode(0)
    if horizontal_scale < 100:
        obj.setHorizScale(horizontal_scale)
    obj.textOut(text)
    pdf.drawText(obj)


def _draw_fallback_text(
    pdf: canvas.Canvas,
    text: str,
    *,
    page_height: float,
    page_width: float,
    font_name: str,
    visible: bool = True,
) -> None:
    """Keep OCR output editable if a model returned text without coordinates."""
    font_size = 10
    margin = 24
    obj = pdf.beginText(margin, page_height - margin)
    obj.setFont(font_name, font_size)
    obj.setFillColorRGB(0.16, 0.16, 0.16)
    obj.setTextRenderMode(0 if visible else 3)
    max_width = page_width - margin * 2
    for line in text.splitlines():
        if pdfmetrics.stringWidth(line, font_name, font_size) <= max_width:
            obj.textLine(line)
            continue
        current = ""
        for word in line.split():
            candidate = f"{current} {word}".strip()
            if current and pdfmetrics.stringWidth(candidate, font_name, font_size) > max_width:
                obj.textLine(current)
                current = word
            else:
                current = candidate
        obj.textLine(current)
    pdf.drawText(obj)


def create_ocr_pdf(
    image_bytes: bytes,
    extracted_text: str = "",
    lines: Sequence[dict[str, Any]] | None = None,
    analysis_report: dict[str, Any] | None = None,
) -> bytes:
    """Return a PDF with the original scan and a visible editable reconstruction.

    The first page preserves the exact uploaded image and embeds OCR text
    invisibly for search/copy. The second page redraws recognized lines as visible
    text objects and long red table strokes as vector lines. It is a best-effort
    reconstruction: recognition errors, especially handwriting, need review.
    """
    font_name = _ensure_unicode_font()
    output = BytesIO()
    try:
        with Image.open(BytesIO(image_bytes)) as source_image:
            image = source_image.convert("RGB")
            image.load()
    except Exception as exc:
        raise ValueError("Không thể mở ảnh đầu vào để tạo PDF.") from exc

    # Keep the source aspect ratio while making the PDF page large enough to read.
    scale = _MAX_PAGE_SIDE_PT / max(image.width, image.height)
    page_width = image.width * scale
    page_height = image.height * scale
    pdf = canvas.Canvas(output, pagesize=(page_width, page_height), pageCompression=1)
    pdf.setTitle("PDF hóa đơn OCR có thể chỉnh sửa")
    pdf.setSubject("Ảnh gốc, bản OCR editable và báo cáo phân tích A–E")

    ocr_lines = _collect_lines(lines, image.width, image.height)

    # Page 1: pixel-faithful source image plus invisible searchable OCR text.
    pdf.drawImage(
        ImageReader(image), 0, 0,
        width=page_width, height=page_height,
        preserveAspectRatio=True, mask="auto",
    )
    for text, box in ocr_lines:
        _draw_invisible_text(
            pdf,
            text=text,
            box=box,
            page_height=page_height,
            scale=scale,
            font_name=font_name,
        )
    if not ocr_lines and extracted_text.strip():
        _draw_fallback_text(
            pdf, extracted_text, page_height=page_height, page_width=page_width,
            font_name=font_name, visible=False,
        )
    pdf.showPage()

    # Page 2: a clean vector/text page that can be edited in PDF editing software.
    pdf.setFillColorRGB(1, 1, 1)
    pdf.rect(0, 0, page_width, page_height, fill=1, stroke=0)
    horizontal_rules, vertical_rules = _detect_red_rules(image)
    pdf.saveState()
    pdf.setStrokeColorRGB(0.76, 0.27, 0.30)
    pdf.setLineWidth(max(0.35, scale * 0.7))
    for y, x1, x2 in horizontal_rules:
        pdf.line(x1 * scale, page_height - y * scale, x2 * scale, page_height - y * scale)
    for x, y1, y2 in vertical_rules:
        pdf.line(x * scale, page_height - y1 * scale, x * scale, page_height - y2 * scale)
    pdf.restoreState()

    if ocr_lines:
        for text, box in ocr_lines:
            _draw_editable_text(
                pdf,
                text=text,
                box=box,
                image=image,
                page_height=page_height,
                scale=scale,
                font_name=font_name,
            )
    elif extracted_text.strip():
        _draw_fallback_text(
            pdf, extracted_text, page_height=page_height, page_width=page_width,
            font_name=font_name,
        )
    else:
        _draw_fallback_text(
            pdf,
            "Không nhận dạng được chữ. Hãy kiểm tra lại ảnh hoặc cấu hình OCR tiếng Việt.",
            page_height=page_height,
            page_width=page_width,
            font_name=font_name,
        )

    if analysis_report:
        _draw_analysis_report(
            pdf,
            analysis_report,
            page_width=page_width,
            page_height=page_height,
            font_name=font_name,
        )

    pdf.save()
    return output.getvalue()


def _draw_analysis_report(
    pdf: canvas.Canvas,
    report: dict[str, Any],
    *,
    page_width: float,
    page_height: float,
    font_name: str,
) -> None:
    """Append a readable A–E report, creating extra pages when the text is long."""
    from reportlab.lib.colors import HexColor, white

    margin = 28.0
    max_width = page_width - margin * 2
    y = page_height - margin

    def new_report_page() -> float:
        pdf.showPage()
        pdf.setFillColor(white)
        pdf.rect(0, 0, page_width, page_height, fill=1, stroke=0)
        return page_height - margin

    def draw_wrapped(text: str, *, size: float = 9.0, leading: float = 12.0, bold: bool = False) -> None:
        nonlocal y
        value = str(text) if text is not None else "[KHÔNG RÕ]"
        lines = simpleSplit(value, font_name, size, max_width) or [""]
        pdf.setFont(font_name, size)
        for line in lines:
            if y < margin + leading:
                y = new_report_page()
            pdf.drawString(margin, y, line)
            y -= leading

    def heading(title: str) -> None:
        nonlocal y
        if y < margin + 28:
            y = new_report_page()
        y -= 4
        pdf.setFillColor(HexColor("#8C1D2C"))
        pdf.setFont(font_name, 12)
        pdf.drawString(margin, y, title)
        y -= 18
        pdf.setFillColor(HexColor("#202020"))

    pdf.showPage()
    pdf.setFillColor(white)
    pdf.rect(0, 0, page_width, page_height, fill=1, stroke=0)
    pdf.setFillColor(HexColor("#202020"))
    pdf.setFont(font_name, 16)
    pdf.drawString(margin, y, "Báo cáo phân tích ảnh và OCR (A–E)")
    y -= 28

    info = report.get("A_document_information", {})
    heading("A. Thông tin tài liệu")
    for label, key in [
        ("Tiêu đề nhận diện", "title_candidate"),
        ("Thời gian", "duration_candidate"),
        ("Số câu nhận diện", "question_count_detected"),
        ("Chất lượng ảnh", "image_quality"),
    ]:
        draw_wrapped(f"{label}: {info.get(key) or '[KHÔNG RÕ]'}")
    metrics = info.get("image_analysis", {})
    draw_wrapped(
        "Ảnh {} × {} px; nghiêng {}°; biên trang {}; sáng {}; tương phản {}; độ nét {}; chỉ số nhiễu {}.".format(
            metrics.get("width_px", "?"), metrics.get("height_px", "?"),
            metrics.get("estimated_skew_degrees", "?"),
            "đã thấy" if metrics.get("document_boundary_detected") else "chưa rõ",
            metrics.get("brightness_mean_0_255", "?"),
            metrics.get("contrast_stddev_0_255", "?"),
            metrics.get("blur_laplacian_variance", "?"),
            metrics.get("noise_indicator_median_residual", "?"),
        )
    )
    for warning in metrics.get("warnings", []):
        draw_wrapped(f"Cảnh báo: {warning}")

    heading("B. Nội dung OCR nguyên bản")
    raw_text = report.get("B_raw_ocr", {}).get("text", "")
    for line in raw_text.splitlines() or ["[Không nhận dạng được chữ]"]:
        draw_wrapped(line)

    heading("C. Nội dung đã chuẩn hóa và cấu trúc")
    normalized = report.get("C_reconstructed_content", {}).get("text", "")
    for line in normalized.splitlines() or ["[KHÔNG RÕ]"]:
        draw_wrapped(line)

    heading("D. Phần chưa chắc chắn")
    uncertain = report.get("D_uncertain_parts", [])
    if uncertain:
        for item in uncertain:
            candidates = ", ".join(
                suggestion.get("proposed", suggestion.get("candidate_term", ""))
                for suggestion in item.get("suggestions", item.get("candidate_terms", []))
            )
            draw_wrapped(
                "Dòng {} ({}): {} — {} {}".format(
                    item.get("line_number", "?"),
                    _format_report_confidence(item.get("confidence")),
                    item.get("raw_text", ""),
                    item.get("reason", "Cần rà soát ảnh gốc."),
                    f"Gợi ý: {candidates}" if candidates else "",
                )
            )
    else:
        draw_wrapped("Không có dòng dưới ngưỡng tin cậy; vẫn cần rà soát số liệu quan trọng.")

    heading("E. Đề xuất cải thiện")
    for recommendation in report.get("E_improvement_suggestions", []):
        draw_wrapped(f"• {recommendation}")
    provenance = report.get("provenance", {}).get("note")
    if provenance:
        draw_wrapped(f"Ghi chú: {provenance}")


def _format_report_confidence(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "không có điểm"
