from io import BytesIO
from pathlib import Path
from typing import Any
import fitz
from PIL import Image

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _detect_image_signature(content: bytes) -> str | None:
    """Identify real image format from magic bytes without relying solely on extension."""
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if content.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if content.startswith(b"II*\x00") or content.startswith(b"MM\x00*"):
        return "TIFF"
    if content.startswith(b"RIFF") and len(content) > 12 and content[8:12] == b"WEBP":
        return "WEBP"
    if content.startswith(b"BM"):
        return "BMP"
    return None


def detect_file_type(filename: str, content: bytes, max_pdf_pages: int) -> tuple[str, dict[str, Any]]:
    """Determine the processing route using file signature, image format, and PDF content."""
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext or '(missing)'}")

    if ext == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise ValueError("File extension is PDF but PDF signature is invalid.")
        try:
            with fitz.open(stream=content, filetype="pdf") as pdf:
                if pdf.page_count > max_pdf_pages:
                    raise ValueError(f"PDF has {pdf.page_count} pages; limit is {max_pdf_pages}.")
                text_chars = 0
                for page in pdf:
                    text_chars += len(page.get_text("text").strip())
                    if text_chars >= 80:
                        break
                file_type = "PDF_DIGITAL" if text_chars >= 80 else "PDF_SCANNED"
                return file_type, {"pages_count": pdf.page_count, "digital_text_chars_sampled": text_chars}
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Cannot read PDF: {exc}") from exc

    if ext in {".docx", ".xlsx"}:
        if not content.startswith(b"PK"):
            raise ValueError(f"File signature does not match {ext} format.")
        return ("DOCX" if ext == ".docx" else "XLSX"), {}

    # Image formats (.jpg, .jpeg, .png, .tif, .tiff)
    sig_format = _detect_image_signature(content)
    if not sig_format:
        # Fallback verification with PIL to handle valid images with varied headers
        try:
            with Image.open(BytesIO(content)) as img:
                sig_format = img.format or "IMAGE"
        except Exception as exc:
            raise ValueError(f"File signature does not match {ext} format.") from exc

    pages_count = 1
    if sig_format == "TIFF":
        try:
            with Image.open(BytesIO(content)) as img:
                pages_count = getattr(img, "n_frames", 1)
        except Exception:
            pages_count = 1

    return "IMAGE", {"detected_format": sig_format, "pages_count": pages_count}
