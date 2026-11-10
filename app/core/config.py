import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(os.getenv(name, str(default)))))
    except ValueError:
        return default


def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return min(maximum, max(minimum, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Document AI Engine")
    max_upload_mb: int = _int_env("MAX_UPLOAD_MB", 25)
    max_pdf_pages: int = _int_env("MAX_PDF_PAGES", 100)
    ocr_enabled: bool = os.getenv("OCR_ENABLED", "false").strip().lower() == "true"
    ocr_text_detection_model_dir: str | None = os.getenv("OCR_TEXT_DETECTION_MODEL_DIR") or None
    ocr_text_recognition_model_dir: str | None = os.getenv("OCR_TEXT_RECOGNITION_MODEL_DIR") or None
    ocr_preprocess_mode: str = os.getenv("OCR_PREPROCESS_MODE", "auto").strip().lower()
    ocr_low_confidence_threshold: float = _float_env("OCR_LOW_CONFIDENCE_THRESHOLD", 0.70)
    ocr_max_candidates: int = _bounded_int_env("OCR_MAX_CANDIDATES", 4, 1, 6)
    ocr_target_short_edge: int = _bounded_int_env("OCR_TARGET_SHORT_EDGE", 1200, 300, 2400)
    ocr_max_dimension: int = _bounded_int_env("OCR_MAX_DIMENSION", 3500, 800, 6000)
    ocr_debug_image_dir: str | None = os.getenv("OCR_DEBUG_IMAGE_DIR") or None
    ocr_llm_correction_enabled: bool = os.getenv("OCR_LLM_CORRECTION_ENABLED", "false").strip().lower() == "true"
    ocr_llm_provider: str = os.getenv("OCR_LLM_PROVIDER", "none").strip().lower()
    ocr_llm_model: str = os.getenv("OCR_LLM_MODEL", "").strip()
    ocr_llm_api_key: str = (
        os.getenv("OCR_LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or ""
    ).strip()
    ocr_llm_timeout_seconds: int = _bounded_int_env("OCR_LLM_TIMEOUT_SECONDS", 15, 1, 120)
    ocr_llm_max_chars: int = _bounded_int_env("OCR_LLM_MAX_CHARS", 8000, 100, 50000)
    languagetool_api_url: str = os.getenv(
        "LANGUAGETOOL_API_URL", "https://api.languagetool.org/v2/check"
    ).strip()


settings = Settings()
