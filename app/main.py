from pathlib import Path
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.endpoints import router as v1_router
from app.core.config import settings

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Document AI Engine - Vietnamese OCR & Extraction",
    version="2.0.0",
    description=(
        "Hệ thống OCR và Trích xuất Tài liệu Tiếng Việt Thông minh (PaddleOCR PP-OCRv6, "
        "Document Layout Analysis, AI Diacritic Correction, và Xuất file Word/Excel/PDF).\n\n"
        "- **Giao diện Web trực quan**: Truy cập tại `/` hoặc `/ui`\n"
        "- **Health Check**: `/health`\n"
        "- **API chính**: `/api/v1/analyze`, `/api/v1/convert/image`"
    ),
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "docExpansion": "list",
        "filter": True,
        "tryItOutEnabled": True,
    },
)


def custom_openapi():
    """Generate OpenAPI 3.0.3 schema so Swagger UI renders file fields as proper file choosers.

    FastAPI defaults to OpenAPI 3.1 which uses contentMediaType/contentEncoding for binary
    fields. Swagger UI does NOT render those as <input type="file"> — it shows a text box.
    Downgrading to 3.0.3 and using type=string + format=binary is the standard fix.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # ── Force OpenAPI 3.0.3 so Swagger UI handles binary correctly ──────────
    openapi_schema["openapi"] = "3.0.3"

    # ── Patch every component schema: replace OAS3.1 binary encoding with
    #    OAS3.0 format:binary so Swagger UI renders a file chooser button ───
    for schema in openapi_schema.get("components", {}).get("schemas", {}).values():
        if not isinstance(schema, dict):
            continue
        for prop_name, prop in schema.get("properties", {}).items():
            if not isinstance(prop, dict):
                continue
            # OAS 3.1 marks binary uploads with contentMediaType
            if prop.get("contentMediaType") == "application/octet-stream" or prop_name == "file":
                # Remove OAS 3.1-only keys that confuse Swagger UI
                prop.pop("contentMediaType", None)
                prop.pop("contentEncoding", None)
                # Set the OAS 3.0 canonical binary representation
                prop["type"] = "string"
                prop["format"] = "binary"

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Mount static assets
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def serve_web_ui():
    """Serve the modern Document AI Web User Interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Document AI Engine</h1><p>Vui lòng truy cập <a href='/docs'>/docs</a></p>")


app.include_router(v1_router, prefix="/api/v1", tags=["document-analysis"])


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "document-ai-engine"}
