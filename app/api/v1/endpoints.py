import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.api.v1.schemas import AnalyzeResponse
from app.core.config import settings
from app.core.router import detect_file_type
from app.extractors.dispatcher import extract_document
from app.utils.ocr_exporters import create_ocr_docx, create_ocr_xlsx
from app.utils.ocr_pdf_generator import create_ocr_pdf

router = APIRouter()
ALLOWED_TASK_TYPES = {"AUTO", "INVOICE", "FINANCIAL_REPORT", "GENERAL"}
OUTPUT_TYPES = {
    "PDF": ("application/pdf", ".pdf"),
    "DOCX": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    "XLSX": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
}


async def _read_limited_upload(file: UploadFile) -> bytes:
    """Read and close an upload, enforcing the shared size and empty-file checks."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        max_bytes = settings.max_upload_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {settings.max_upload_mb} MB limit.",
            )
        return content
    finally:
        await file.close()


def _validate_task_type(value: str) -> str:
    task_type = value.upper().strip()
    if task_type not in ALLOWED_TASK_TYPES:
        raise HTTPException(status_code=422, detail="Invalid task_type.")
    return task_type


def _extract_upload(
    filename: str,
    content: bytes,
    task_type: str,
    expected_file_type: str | None = None,
) -> tuple[str, dict, dict]:
    file_type, metadata = detect_file_type(filename, content, settings.max_pdf_pages)
    if expected_file_type and file_type != expected_file_type:
        raise ValueError("This endpoint accepts PNG, JPG, or JPEG images only.")
    extracted = extract_document(
        filename=filename,
        content=content,
        file_type=file_type,
        task_type=task_type,
        metadata=metadata,
    )
    return file_type, metadata, extracted


def _to_download(content: bytes, filename: str, output_format: str) -> Response:
    media_type, extension = OUTPUT_TYPES[output_format]
    stem = Path(filename).stem or "document"
    safe_stem = "".join(char for char in stem if char.isalnum() or char in "-_") or "document"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{safe_stem}_ocr{extension}"'},
    )


def _render_output(
    image_bytes: bytes,
    extracted: dict,
    output_format: str,
) -> bytes:
    if output_format == "PDF":
        return create_ocr_pdf(
            image_bytes,
            extracted_text=extracted.get("text", ""),
            lines=extracted.get("lines", []),
            analysis_report=extracted.get("analysis_report"),
        )
    if output_format == "DOCX":
        return create_ocr_docx(image_bytes, extracted)
    return create_ocr_xlsx(image_bytes, extracted)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_document(
    file: UploadFile = File(...),
    task_type: str = Form("AUTO"),
):
    started = time.perf_counter()
    filename = file.filename or "upload.bin"
    content = await _read_limited_upload(file)
    task_type = _validate_task_type(task_type)

    try:
        file_type, metadata, extracted = _extract_upload(filename, content, task_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document extraction failed.") from exc

    return AnalyzeResponse(
        status="PARTIAL" if extracted.get("warnings") else "SUCCESS",
        file_type=file_type,
        task_type=task_type,
        processing_time_ms=round((time.perf_counter() - started) * 1000),
        pages_count=metadata.get("pages_count"),
        extracted_data=extracted,
        warnings=extracted.get("warnings", []),
    )


async def _convert_image_upload(
    file: UploadFile,
    task_type: str,
    output_format: str,
) -> Response:
    filename = file.filename or "upload.png"
    content = await _read_limited_upload(file)
    task_type = _validate_task_type(task_type)
    output_format = output_format.upper().strip()
    if output_format not in OUTPUT_TYPES:
        raise HTTPException(status_code=422, detail="Invalid output_format.")

    try:
        _file_type, _metadata, extracted = _extract_upload(
            filename, content, task_type, expected_file_type="IMAGE"
        )
        output = _render_output(content, extracted, output_format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Image OCR or {output_format} generation failed.",
        ) from exc
    return _to_download(output, filename, output_format)


@router.post(
    "/analyze/pdf",
    summary="Convert an image to a searchable and editable PDF",
    response_class=Response,
    responses={
        200: {
            "description": (
                "PDF with the original searchable scan, editable OCR reconstruction, and A–E analysis report."
            ),
            "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}},
        }
    },
)
async def analyze_image_to_pdf(
    file: UploadFile = File(...),
    task_type: str = Form("AUTO"),
):
    return await _convert_image_upload(file, task_type, "PDF")


@router.post(
    "/convert/image",
    summary="Convert an image to PDF, Word, or Excel",
    response_class=Response,
    responses={
        200: {
            "description": (
                "Downloads the selected file. Word/Excel include editable OCR and the A–E report; "
                "all formats retain the original scan for visual reference."
            ),
            "content": {
                media_type: {"schema": {"type": "string", "format": "binary"}}
                for media_type, _extension in OUTPUT_TYPES.values()
            },
        }
    },
)
async def convert_image(
    file: UploadFile = File(...),
    output_format: Literal["PDF", "DOCX", "XLSX"] = Form(
        "PDF",
        description="Choose the file format to download.",
    ),
    task_type: str = Form("AUTO"),
):
    return await _convert_image_upload(file, task_type, output_format)
