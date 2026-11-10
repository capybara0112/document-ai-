"""LanguageTool HTTP API corrector and grammar checker for OCR text."""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from app.correction.base_corrector import BaseCorrector, CorrectionResult

logger = logging.getLogger(__name__)


class LanguageToolCorrector(BaseCorrector):
    """Corrector using LanguageTool HTTP API (rule-based spell and grammar check)."""

    def __init__(
        self,
        api_url: str = "https://api.languagetool.org/v2/check",
        language: str = "vi",
        timeout_seconds: int = 10,
    ):
        super().__init__(provider_name="languagetool", model="rule_engine", timeout_seconds=timeout_seconds)
        self.api_url = api_url
        self.language = language

    def correct(self, text: str, lines: list[dict[str, Any]] | None = None) -> CorrectionResult:
        started = time.perf_counter()
        if not text or not text.strip():
            return CorrectionResult(
                raw_text=text,
                corrected_text=text,
                correction_status="UNCHANGED",
                provider=self.provider_name,
                model=self.model,
                latency_ms=0,
            )

        post_data = urllib.parse.urlencode({
            "text": text[:5000],
            "language": self.language,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.api_url,
            data=post_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8")
                latency_ms = round((time.perf_counter() - started) * 1000)

            data = json.loads(body)
            matches = data.get("matches", [])
            changes: list[dict[str, str]] = []
            uncertain: list[dict[str, str]] = []

            # LanguageTool identifies errors by character offset
            # We sort matches in reverse offset order to replace in-place safely
            working_text = text
            for match in sorted(matches, key=lambda m: m.get("offset", 0), reverse=True):
                offset = match.get("offset", 0)
                length = match.get("length", 0)
                replacements = match.get("replacements", [])
                original_substr = working_text[offset : offset + length]

                if replacements:
                    best_repl = replacements[0].get("value")
                    if best_repl and best_repl != original_substr:
                        working_text = working_text[:offset] + best_repl + working_text[offset + length :]
                        changes.append({
                            "original": original_substr,
                            "corrected": best_repl,
                            "reason": match.get("message", "LanguageTool rule match"),
                        })
                else:
                    uncertain.append({
                        "text": original_substr,
                        "reason": match.get("message", "Nghi ngờ lỗi ngữ pháp / chính tả"),
                    })

            valid_text, valid_changes, validation_warnings = self.validate_and_sanitize(
                text, working_text, changes, uncertain
            )

            status = "CORRECTED" if (valid_text != text and valid_changes) else "UNCHANGED"
            return CorrectionResult(
                raw_text=text,
                corrected_text=valid_text,
                correction_status=status,
                correction_confidence=0.75 if status == "CORRECTED" else 1.0,
                changes=valid_changes,
                uncertain_segments=uncertain,
                provider=self.provider_name,
                model=self.model,
                latency_ms=latency_ms,
                warnings=[
                    *validation_warnings,
                    "LanguageTool có bộ quy tắc tiếng Việt hạn chế; chỉ sửa các lỗi ngữ pháp cơ bản.",
                ],
            )

        except urllib.error.URLError as exc:
            latency_ms = round((time.perf_counter() - started) * 1000)
            logger.warning("LanguageTool connection failed: %s", type(exc).__name__)
            return CorrectionResult(
                raw_text=text,
                corrected_text=text,
                correction_status="FAILED",
                provider=self.provider_name,
                model=self.model,
                latency_ms=latency_ms,
                warnings=[f"LanguageTool API không phản hồi: {type(exc).__name__}. Sử dụng kết quả OCR gốc."],
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000)
            return CorrectionResult(
                raw_text=text,
                corrected_text=text,
                correction_status="FAILED",
                provider=self.provider_name,
                model=self.model,
                latency_ms=latency_ms,
                warnings=[f"LanguageTool error: {str(exc)[:120]}. Sử dụng kết quả OCR gốc."],
            )
