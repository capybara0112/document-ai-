import fitz


def extract_digital_pdf(content: bytes) -> dict:
    pages = []
    with fitz.open(stream=content, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            pages.append({
                "page_number": page_number,
                "text": page.get_text("text"),
            })
    text = "\n".join(page["text"] for page in pages)
    return {
        "metadata": {"pages_count": len(pages), "extraction_method": "pymupdf_text"},
        "text": text,
        "pages": pages,
        "tables": [],
        "fields": {},
    }
