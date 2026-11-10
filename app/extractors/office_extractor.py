from io import BytesIO
from docx import Document
from openpyxl import load_workbook


def extract_docx(content: bytes) -> dict:
    doc = Document(BytesIO(content))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    tables = []
    for table_index, table in enumerate(doc.tables, start=1):
        rows = []
        for row in table.rows:
            rows.append([cell.text.strip() for cell in row.cells])
        tables.append({"table_index": table_index, "rows": rows})
    return {
        "metadata": {"extraction_method": "python_docx"},
        "text": "\n".join(paragraphs),
        "paragraphs": paragraphs,
        "tables": tables,
        "fields": {},
    }


def extract_xlsx(content: bytes) -> dict:
    workbook = load_workbook(BytesIO(content), read_only=True, data_only=True)
    sheets = []
    try:
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                values = [normalize_cell(value) for value in row]
                # Ignore entirely empty rows but preserve internal blank cells.
                if any(value is not None and value != "" for value in values):
                    rows.append(values)
            sheets.append({
                "sheet_name": sheet.title,
                "rows": rows,
            })
    finally:
        workbook.close()

    return {
        "metadata": {"extraction_method": "openpyxl", "sheet_count": len(sheets)},
        "text": "",
        "sheets": sheets,
        "tables": [],
        "fields": {},
    }


def normalize_cell(value):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
