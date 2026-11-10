"""Create editable Word and Excel exports from OCR results."""
from __future__ import annotations

from bisect import bisect_right
from io import BytesIO
from statistics import median
from typing import Any

from PIL import Image


def _normalized_lines(extracted: dict[str, Any]) -> list[dict[str, Any]]:
    lines = []
    for item in extracted.get("lines", []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        box = item.get("bounding_box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            box = None
        else:
            try:
                box = [float(value) for value in box]
            except (TypeError, ValueError):
                box = None
        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            confidence = None
        lines.append({
            "text": text,
            "confidence": confidence,
            "box": box,
            "reading_order_index": item.get("reading_order_index", len(lines) + 1),
            "type": item.get("type", "paragraph"),
        })

    if lines:
        # Preserve document layout reading order
        return lines

    return [
        {"text": text.strip(), "confidence": None, "box": None}
        for text in str(extracted.get("text", "")).splitlines()
        if text.strip()
    ]


def _format_confidence(value: float | None) -> str:
    return "" if value is None else f"{value:.1%}"


def create_ocr_docx(image_bytes: bytes, extracted: dict[str, Any]) -> bytes:
    """Create a Word document with the original scan and editable OCR rows."""
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    output = BytesIO()
    document = Document()
    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11.69)
    section.page_height = Inches(8.27)
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)
    section.top_margin = Inches(0.45)
    section.bottom_margin = Inches(0.45)

    document.core_properties.title = "Bản OCR có thể chỉnh sửa"
    document.add_heading("Ảnh gốc", level=1)
    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
    max_width, max_height = 10.4, 6.75
    image_scale = min(max_width / width, max_height / height)
    picture = document.add_paragraph()
    picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture.add_run().add_picture(
        BytesIO(image_bytes), width=Inches(width * image_scale)
    )

    document.add_page_break()
    document.add_heading("Nội dung OCR có thể chỉnh sửa", level=1)
    document.add_paragraph(
        "Mỗi hàng là một dòng OCR. Đối chiếu các trường viết tay, mã số thuế "
        "và số tiền với ảnh gốc trước khi sử dụng."
    )

    lines = _normalized_lines(extracted)
    table = document.add_table(rows=1, cols=4)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["STT", "Nội dung nhận dạng", "Độ tin cậy", "Vị trí ảnh (x, y)"]
    for cell, label in zip(table.rows[0].cells, headers):
        cell.text = label
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for run in cell.paragraphs[0].runs:
            run.bold = True

    for index, line in enumerate(lines, start=1):
        cells = table.add_row().cells
        box = line["box"]
        position = f"{box[0]:.0f}, {box[1]:.0f}" if box else ""
        values = [str(index), line["text"], _format_confidence(line["confidence"]), position]
        for cell, value in zip(cells, values):
            cell.text = value
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.name = "Arial"
                    run.font.size = Pt(9)

    with Image.open(BytesIO(image_bytes)) as source_image:
        source_image = source_image.convert("RGB")
        column_rules = _red_rule_columns(source_image)
        layout_rows = _layout_rows(lines, column_rules)
    if layout_rows:
        document.add_heading("Bảng theo vị trí trên ảnh (xấp xỉ)", level=2)
        layout_table = document.add_table(rows=0, cols=len(layout_rows[0]))
        layout_table.style = "Table Grid"
        for row in layout_rows:
            cells = layout_table.add_row().cells
            for cell, value in zip(cells, row):
                cell.text = value
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.name = "Arial"
                        run.font.size = Pt(8)

    report = extracted.get("analysis_report") or {}
    if report:
        document.add_page_break()
        document.add_heading("Báo cáo phân tích OCR (A–E)", level=1)
        info = report.get("A_document_information", {})
        document.add_heading("A. Thông tin tài liệu", level=2)
        for label, key in [
            ("Tiêu đề nhận diện", "title_candidate"),
            ("Thời gian", "duration_candidate"),
            ("Số câu nhận diện", "question_count_detected"),
            ("Chất lượng ảnh", "image_quality"),
        ]:
            document.add_paragraph(f"{label}: {info.get(key) or '[KHÔNG RÕ]'}")
        metrics = info.get("image_analysis", {})
        document.add_paragraph(
            "Kích thước: {} × {} px; nghiêng ước tính: {}°; biên tài liệu: {}; "
            "độ sáng: {}; tương phản: {}; độ nét Laplacian: {}; chỉ số nhiễu: {}".format(
                metrics.get("width_px", "?"), metrics.get("height_px", "?"),
                metrics.get("estimated_skew_degrees", "?"),
                "đã phát hiện" if metrics.get("document_boundary_detected") else "chưa rõ",
                metrics.get("brightness_mean_0_255", "?"),
                metrics.get("contrast_stddev_0_255", "?"),
                metrics.get("blur_laplacian_variance", "?"),
                metrics.get("noise_indicator_median_residual", "?"),
            )
        )

        document.add_heading("B. Nội dung OCR nguyên bản", level=2)
        raw_text = report.get("B_raw_ocr", {}).get("text", "")
        for raw_line in raw_text.splitlines() or ["[Không nhận dạng được chữ]"]:
            document.add_paragraph(raw_line)

        document.add_heading("C. Nội dung đã chuẩn hóa và cấu trúc", level=2)
        reconstructed = report.get("C_reconstructed_content", {}).get("text", "")
        for normalized_line in reconstructed.splitlines() or ["[KHÔNG RÕ]"]:
            document.add_paragraph(normalized_line)

        document.add_heading("D. Phần cần kiểm tra", level=2)
        uncertain = report.get("D_uncertain_parts", [])
        if uncertain:
            uncertainty_table = document.add_table(rows=1, cols=4)
            uncertainty_table.style = "Table Grid"
            for cell, label in zip(uncertainty_table.rows[0].cells, ["Dòng", "OCR đọc được", "Độ tin cậy", "Đề xuất"]):
                cell.text = label
            for item in uncertain:
                cells = uncertainty_table.add_row().cells
                cells[0].text = str(item.get("line_number", ""))
                cells[1].text = str(item.get("raw_text", ""))
                cells[2].text = _format_confidence(item.get("confidence"))
                cells[3].text = ", ".join(
                    suggestion.get("proposed", suggestion.get("candidate_term", ""))
                    for suggestion in item.get("suggestions", item.get("candidate_terms", []))
                ) or "[CẦN KIỂM TRA]"
        else:
            document.add_paragraph("Không có dòng nào dưới ngưỡng tin cậy; vẫn nên kiểm tra số liệu quan trọng.")

        document.add_heading("E. Đề xuất cải thiện", level=2)
        for recommendation in report.get("E_improvement_suggestions", []):
            document.add_paragraph(str(recommendation), style="List Bullet")

    output.seek(0)
    document.save(output)
    return output.getvalue()


def _excel_safe_text(cell: Any, value: str) -> None:
    cell.value = value
    # OCR text is data, never an Excel formula.
    cell.data_type = "s"


def _set_header_style(worksheet: Any, row_number: int = 1) -> None:
    from openpyxl.styles import Font, PatternFill

    for cell in worksheet[row_number]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")


def _red_rule_columns(image: Image.Image) -> list[float]:
    # Import locally so Word exports do not need to initialize PDF support.
    from app.utils.ocr_pdf_generator import _detect_red_rules

    _horizontal, vertical = _detect_red_rules(image)
    positions = sorted(x for x, _y1, _y2 in vertical)
    merged: list[float] = []
    for x in positions:
        if not merged or x - merged[-1] > 8:
            merged.append(x)
        else:
            merged[-1] = (merged[-1] + x) / 2
    return merged


def _layout_rows(lines: list[dict[str, Any]], column_rules: list[float]) -> list[list[str]]:
    positioned = [line for line in lines if line["box"]]
    if not positioned or len(column_rules) < 2:
        return []

    heights = [line["box"][3] - line["box"][1] for line in positioned]
    row_tolerance = max(8.0, median(heights) * 0.72)
    ordered = sorted(positioned, key=lambda line: (line["box"][1] + line["box"][3]) / 2)
    groups: list[list[dict[str, Any]]] = []
    centers: list[float] = []
    for line in ordered:
        center_y = (line["box"][1] + line["box"][3]) / 2
        if groups and center_y - centers[-1] <= row_tolerance:
            groups[-1].append(line)
            centers[-1] = median(
                (entry["box"][1] + entry["box"][3]) / 2 for entry in groups[-1]
            )
        else:
            groups.append([line])
            centers.append(center_y)

    column_count = len(column_rules) + 1
    rows: list[list[str]] = []
    for group in groups:
        row = [""] * column_count
        for line in sorted(group, key=lambda item: item["box"][0]):
            center_x = (line["box"][0] + line["box"][2]) / 2
            column_index = max(0, min(column_count - 1, bisect_right(column_rules, center_x)))
            row[column_index] = "\n".join(filter(None, [row[column_index], line["text"]]))
        rows.append(row)
    return rows


def create_ocr_xlsx(image_bytes: bytes, extracted: dict[str, Any]) -> bytes:
    """Create an Excel workbook with OCR rows, source image, and layout grid."""
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as SpreadsheetImage
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    output = BytesIO()
    workbook = Workbook()
    lines_sheet = workbook.active
    lines_sheet.title = "OCR Lines"
    headers = ["STT", "Nội dung", "Độ tin cậy", "X1", "Y1", "X2", "Y2"]
    lines_sheet.append(headers)
    _set_header_style(lines_sheet)
    lines = _normalized_lines(extracted)
    for index, line in enumerate(lines, start=1):
        row_number = lines_sheet.max_row + 1
        lines_sheet.cell(row=row_number, column=1, value=index)
        text_cell = lines_sheet.cell(row=row_number, column=2)
        _excel_safe_text(text_cell, line["text"])
        confidence = lines_sheet.cell(row=row_number, column=3, value=line["confidence"])
        if line["confidence"] is not None:
            confidence.number_format = "0.0%"
        for col_index, coordinate in enumerate(line["box"] or [None] * 4, start=4):
            lines_sheet.cell(row=row_number, column=col_index, value=coordinate)

    lines_sheet.freeze_panes = "A2"
    lines_sheet.auto_filter.ref = lines_sheet.dimensions
    for column, width in {"A": 8, "B": 64, "C": 14, "D": 12, "E": 12, "F": 12, "G": 12}.items():
        lines_sheet.column_dimensions[column].width = width
    if lines:
        table = Table(displayName="OcrLines", ref=f"A1:G{lines_sheet.max_row}")
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        lines_sheet.add_table(table)

    layout = workbook.create_sheet("Layout")
    layout.sheet_view.showGridLines = False
    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        column_rules = _red_rule_columns(image)
        grid_rows = _layout_rows(lines, column_rules)
        image_width, image_height = image.size
    if grid_rows:
        layout.append([f"Cột {index}" for index in range(1, len(grid_rows[0]) + 1)])
        _set_header_style(layout)
        for row in grid_rows:
            for column_index, value in enumerate(row, start=1):
                cell = layout.cell(row=layout.max_row + 1, column=column_index)
                _excel_safe_text(cell, value)
        for column_index in range(1, len(grid_rows[0]) + 1):
            layout.column_dimensions[get_column_letter(column_index)].width = 24
        layout.freeze_panes = "A2"
    else:
        layout.append(["OCR chưa có đủ tọa độ/đường bảng để dựng lưới; xem sheet OCR Lines."])

    report = extracted.get("analysis_report") or {}
    if report:
        from openpyxl.styles import Alignment

        analysis = workbook.create_sheet("Analysis")
        analysis.append(["Mục", "Trường", "Nội dung"])
        _set_header_style(analysis)
        info = report.get("A_document_information", {})
        image_info = info.get("image_analysis", {})
        for key, value in [
            ("Tiêu đề nhận diện", info.get("title_candidate")),
            ("Thời gian", info.get("duration_candidate")),
            ("Số câu nhận diện", info.get("question_count_detected")),
            ("Chất lượng ảnh", info.get("image_quality")),
            ("Kích thước ảnh", f"{image_info.get('width_px', '?')} × {image_info.get('height_px', '?')} px"),
            ("Góc nghiêng ước tính", image_info.get("estimated_skew_degrees")),
            ("Độ sáng trung bình", image_info.get("brightness_mean_0_255")),
            ("Độ tương phản", image_info.get("contrast_stddev_0_255")),
            ("Độ nét Laplacian", image_info.get("blur_laplacian_variance")),
            ("Chỉ số nhiễu nền", image_info.get("noise_indicator_median_residual")),
        ]:
            analysis.append(["A", key, "[KHÔNG RÕ]" if value is None else str(value)])

        for label, key in [("B", "B_raw_ocr"), ("C", "C_reconstructed_content")]:
            value = report.get(key, {}).get("text", "")
            analysis.append([label, "Nội dung", str(value)])
        analysis.append(["D", "Dòng", "OCR đọc được", "Độ tin cậy", "Lý do"])
        uncertain = report.get("D_uncertain_parts", [])
        for item in uncertain:
            analysis.append([
                "D", item.get("line_number"), str(item.get("raw_text", "")),
                item.get("confidence"), str(item.get("reason", "")) + (
                    " Gợi ý: " + ", ".join(
                        suggestion.get("proposed", suggestion.get("candidate_term", ""))
                        for suggestion in item.get("suggestions", item.get("candidate_terms", []))
                    ) if item.get("suggestions", item.get("candidate_terms", [])) else ""
                ),
            ])
        if not uncertain:
            analysis.append(["D", "", "Không có dòng dưới ngưỡng tin cậy", "", ""])
        for recommendation in report.get("E_improvement_suggestions", []):
            analysis.append(["E", "Đề xuất", str(recommendation)])
        for row in analysis.iter_rows(min_row=2):
            for cell in row:
                if cell.value is not None:
                    cell.data_type = "s"
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
        analysis.column_dimensions["A"].width = 10
        analysis.column_dimensions["B"].width = 25
        analysis.column_dimensions["C"].width = 72
        analysis.column_dimensions["D"].width = 16
        analysis.column_dimensions["E"].width = 58
        analysis.freeze_panes = "A2"

    source_sheet = workbook.create_sheet("Original Scan")
    source_sheet["A1"] = f"Ảnh gốc ({image_width} x {image_height} px)"
    source_sheet["A1"].font = Font(bold=True)
    source_sheet.add_image(SpreadsheetImage(BytesIO(image_bytes)), "A3")

    workbook.properties.title = "Kết quả OCR có thể chỉnh sửa"
    workbook.properties.subject = "OCR text and approximate page layout"
    workbook.save(output)
    return output.getvalue()
