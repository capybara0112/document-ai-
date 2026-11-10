"""Google Gemini Structured Outputs corrector for OCR text."""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from app.correction.base_corrector import BaseCorrector, CorrectionResult

logger = logging.getLogger(__name__)

GEMINI_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "corrected_text": {
            "type": "STRING",
            "description": "Văn bản đã khôi phục dấu tiếng Việt và sửa lỗi ký tự OCR. Giữ nguyên cấu trúc dòng."
        },
        "correction_confidence": {
            "type": "NUMBER",
            "description": "Độ tin cậy của việc sửa đổi từ 0.0 đến 1.0."
        },
        "changes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "original": {"type": "STRING"},
                    "corrected": {"type": "STRING"},
                    "reason": {"type": "STRING"}
                },
                "required": ["original", "corrected", "reason"]
            }
        },
        "uncertain_segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING"},
                    "reason": {"type": "STRING"}
                },
                "required": ["text", "reason"]
            }
        }
    },
    "required": ["corrected_text", "correction_confidence", "changes", "uncertain_segments"]
}

SYSTEM_INSTRUCTION = """Bạn là chuyên gia hậu xử lý OCR tiếng Việt (Vietnamese OCR Post-processor).
Nhiệm vụ: Khôi phục dấu tiếng Việt bị thiếu, sửa các lỗi nhận diện ký tự quang học.

QUY TẮC BẮT BUỘC:
1. KHÔNG tự ý sáng tác, thêm bớt ý hoặc tóm tắt.
2. KHÔNG thay đổi bất kỳ số liệu, ngày tháng, công thức, mã số hoặc dấu câu cấu trúc.
3. BẢO VỆ TUYỆT ĐỐI các thuật ngữ kỹ thuật tiếng Anh (OpenCV, Python, Canny, matplotlib, trackbar, kernel, OCR, API...).
4. Giữ nguyên thứ tự các dòng và các ký hiệu gạch đầu dòng (-, +, *, •), số thứ tự (Câu 1, 1., 2)...).
5. Nếu một đoạn quá mờ hoặc vô nghĩa và không đủ bằng chứng khôi phục chắc chắn, GIỮ NGUYÊN bản thô và đưa vào danh sách uncertain_segments."""


class GeminiCorrector(BaseCorrector):
    """Corrector using Google Gemini API with Structured Output."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-flash",
        timeout_seconds: int = 15,
    ):
        super().__init__(provider_name="gemini", model=model or "gemini-1.5-flash", timeout_seconds=timeout_seconds)
        self.api_key = api_key

    def correct(self, text: str, lines: list[dict[str, Any]] | None = None) -> CorrectionResult:
        started = time.perf_counter()
        if not self.api_key:
            return CorrectionResult(
                raw_text=text,
                corrected_text=text,
                correction_status="SKIPPED",
                provider=self.provider_name,
                model=self.model,
                latency_ms=0,
                warnings=["Gemini API key chưa được cấu hình. Bỏ qua LLM correction."],
            )

        truncated_text = text[:8000]
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )

        payload = {
            "system_instruction": {
                "parts": [{"text": SYSTEM_INSTRUCTION}]
            },
            "contents": [
                {
                    "parts": [{"text": f"Văn bản OCR thô cần khôi phục:\n\n{truncated_text}"}]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "response_schema": GEMINI_RESPONSE_SCHEMA,
                "temperature": 0.1,
            }
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                status_code = resp.status
                body = resp.read().decode("utf-8")
                latency_ms = round((time.perf_counter() - started) * 1000)

            if status_code != 200:
                return CorrectionResult(
                    raw_text=text,
                    corrected_text=text,
                    correction_status="FAILED",
                    provider=self.provider_name,
                    model=self.model,
                    latency_ms=latency_ms,
                    warnings=[f"Gemini HTTP error: {status_code}"],
                )

            res_json = json.loads(body)
            candidates = res_json.get("candidates", [])
            if not candidates:
                return CorrectionResult(
                    raw_text=text,
                    corrected_text=text,
                    correction_status="FAILED",
                    provider=self.provider_name,
                    model=self.model,
                    latency_ms=latency_ms,
                    warnings=["Gemini không trả về candidate nào."],
                )

            content_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
            parsed = json.loads(content_text)

            proposed_text = parsed.get("corrected_text", text)
            confidence = float(parsed.get("correction_confidence", 0.85))
            raw_changes = parsed.get("changes", [])
            uncertain = parsed.get("uncertain_segments", [])

            valid_text, valid_changes, validation_warnings = self.validate_and_sanitize(
                text, proposed_text, raw_changes, uncertain
            )

            status = "CORRECTED" if (valid_text != text and valid_changes) else "UNCHANGED"
            return CorrectionResult(
                raw_text=text,
                corrected_text=valid_text,
                correction_status=status,
                correction_confidence=confidence,
                changes=valid_changes,
                uncertain_segments=uncertain,
                provider=self.provider_name,
                model=self.model,
                latency_ms=latency_ms,
                warnings=validation_warnings,
            )

        except urllib.error.URLError as exc:
            latency_ms = round((time.perf_counter() - started) * 1000)
            logger.warning("Gemini API network error: %s", type(exc).__name__)
            return CorrectionResult(
                raw_text=text,
                corrected_text=text,
                correction_status="FAILED",
                provider=self.provider_name,
                model=self.model,
                latency_ms=latency_ms,
                warnings=[f"Gemini API kết nối thất bại: {type(exc).__name__}. Sử dụng kết quả OCR gốc."],
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000)
            logger.warning("Gemini correction parsing error: %s", type(exc).__name__)
            return CorrectionResult(
                raw_text=text,
                corrected_text=text,
                correction_status="FAILED",
                provider=self.provider_name,
                model=self.model,
                latency_ms=latency_ms,
                warnings=[f"Lỗi xử lý phản hồi từ Gemini: {str(exc)[:150]}. Sử dụng kết quả OCR gốc."],
            )
