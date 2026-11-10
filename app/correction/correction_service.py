"""Central correction service coordinating OCR text correction."""
from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings
from app.correction.base_corrector import BaseCorrector, CorrectionResult
from app.correction.gemini_corrector import GeminiCorrector
from app.correction.languagetool_corrector import LanguageToolCorrector
from app.correction.openai_corrector import OpenAICorrector

logger = logging.getLogger(__name__)


class CorrectionService:
    """Service orchestrating AI Language Correction for OCR text."""

    def __init__(self) -> None:
        self.enabled = settings.ocr_llm_correction_enabled
        self.provider_name = settings.ocr_llm_provider
        self._corrector: BaseCorrector | None = None

        if self.enabled:
            self._init_corrector()

    def _init_corrector(self) -> None:
        provider = self.provider_name.lower().strip()
        api_key = settings.ocr_llm_api_key
        model = settings.ocr_llm_model
        timeout = settings.ocr_llm_timeout_seconds

        if provider == "openai":
            self._corrector = OpenAICorrector(
                api_key=api_key,
                model=model or "gpt-4o-mini",
                timeout_seconds=timeout,
            )
        elif provider == "gemini":
            self._corrector = GeminiCorrector(
                api_key=api_key,
                model=model or "gemini-1.5-flash",
                timeout_seconds=timeout,
            )
        elif provider == "languagetool":
            self._corrector = LanguageToolCorrector(
                api_url=settings.languagetool_api_url,
                timeout_seconds=timeout,
            )
        else:
            logger.info("OCR LLM provider '%s' not recognized; AI correction disabled.", provider)
            self._corrector = None

    def correct_text(
        self,
        text: str,
        lines: list[dict[str, Any]] | None = None,
    ) -> CorrectionResult:
        """Run correction on OCR text. If disabled or fails, cleanly return fallback."""
        if not self.enabled:
            return CorrectionResult(
                raw_text=text,
                corrected_text=text,
                correction_status="SKIPPED",
                provider="none",
                warnings=["AI Language Correction đang tắt (OCR_LLM_CORRECTION_ENABLED=false)."],
            )

        if self._corrector is None:
            return CorrectionResult(
                raw_text=text,
                corrected_text=text,
                correction_status="SKIPPED",
                provider="none",
                warnings=[f"Không có provider khả dụng cho '{self.provider_name}'."],
            )

        try:
            return self._corrector.correct(text=text, lines=lines)
        except Exception as exc:
            logger.error("AI correction service unexpected error: %s", type(exc).__name__)
            return CorrectionResult(
                raw_text=text,
                corrected_text=text,
                correction_status="FAILED",
                provider=self.provider_name,
                warnings=[f"AI Correction gặp lỗi hệ thống: {type(exc).__name__}. Giữ nguyên kết quả OCR gốc."],
            )


_SERVICE: CorrectionService | None = None


def get_correction_service() -> CorrectionService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = CorrectionService()
    return _SERVICE
