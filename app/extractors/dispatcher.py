from app.extractors.pdf_digital import extract_digital_pdf
from app.extractors.office_extractor import extract_docx, extract_xlsx
from app.extractors.ocr_adapter import extract_image_ocr, extract_scanned_pdf


def extract_document(filename, content, file_type, task_type, metadata):
    if file_type == "PDF_DIGITAL":
        result = extract_digital_pdf(content)
    elif file_type == "PDF_SCANNED":
        result = extract_scanned_pdf(content)
    elif file_type == "DOCX":
        result = extract_docx(content)
    elif file_type == "XLSX":
        result = extract_xlsx(content)
    elif file_type == "IMAGE":
        result = extract_image_ocr(content)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    result["metadata"] = {
        **result.get("metadata", {}),
        "source_filename": filename,
        "detected_file_type": file_type,
        "requested_task_type": task_type,
        **metadata,
    }
    result["document_type"] = infer_document_type(result, task_type)
    return result


def infer_document_type(result, task_type):
    if task_type == "INVOICE":
        return "HOA_DON_GTGT"
    if task_type == "FINANCIAL_REPORT":
        return "BAO_CAO_TAI_CHINH"
    if task_type == "GENERAL":
        return "GENERAL"
    # AUTO is only a routing hint here; do not claim semantic classification
    # without a classifier or validated business rules.
    return "UNCLASSIFIED"
