"""Base abstractions and validation for AI-assisted OCR language correction."""
from __future__ import annotations

import abc
import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Protected technical terms that must NEVER be translated, removed, or mutilated
PROTECTED_TECHNICAL_TERMS: frozenset[str] = frozenset({
    "python", "opencv", "canny", "matplotlib", "trackbar", "kernel",
    "ocr", "api", "laplacian", "gaussianblur", "bilateral", "sobel",
    "threshold", "cv2", "numpy", "pytorch", "tensorflow", "fastapi",
    "docker", "cuda", "onnx", "git", "linux", "windows", "json", "rest",
})


@dataclass
class CorrectionResult:
    """Standardized result for OCR language correction."""
    raw_text: str
    corrected_text: str
    correction_status: str  # "CORRECTED", "UNCHANGED", "SKIPPED", "FAILED"
    correction_confidence: float | None = None
    changes: list[dict[str, str]] = field(default_factory=list)
    uncertain_segments: list[dict[str, str]] = field(default_factory=list)
    provider: str = "none"
    model: str | None = None
    latency_ms: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseCorrector(abc.ABC):
    """Abstract base corrector interface."""

    def __init__(self, provider_name: str, model: str | None = None, timeout_seconds: int = 15):
        self.provider_name = provider_name
        self.model = model
        self.timeout_seconds = timeout_seconds

    @abc.abstractmethod
    def correct(self, text: str, lines: list[dict[str, Any]] | None = None) -> CorrectionResult:
        """Correct OCR text using the configured provider."""
        raise NotImplementedError

    def validate_and_sanitize(
        self,
        raw_text: str,
        proposed_text: str,
        changes: list[dict[str, str]],
        uncertain_segments: list[dict[str, str]],
    ) -> tuple[str, list[dict[str, str]], list[str]]:
        """Verify that proposed corrections adhere to document fidelity rules:
        1. Length cannot wildly diverge (> 30% expansion or shrinkage).
        2. Numbers and equations must be preserved.
        3. Protected technical terms must be preserved.
        4. No conversational filler or AI preambles.
        """
        warnings: list[str] = []
        if not proposed_text or not proposed_text.strip():
            return raw_text, [], ["Mô hình trả về văn bản rỗng; giữ nguyên raw OCR text."]

        clean_proposed = proposed_text.strip()

        # Reject conversational preamble/postamble
        clean_proposed = re.sub(
            r"^(?:dưới đây là|đây là bản sửa|bản sửa đổi|kết quả sửa|văn bản đã sửa):\s*",
            "",
            clean_proposed,
            flags=re.IGNORECASE,
        ).strip()

        # Check length ratio
        raw_len = max(1, len(raw_text))
        prop_len = len(clean_proposed)
        ratio = prop_len / raw_len
        if ratio < 0.60 or ratio > 1.40:
            warnings.append(
                f"Tỷ lệ độ dài văn bản đề xuất ({ratio:.2f}) lệch quá 40%; "
                "nghi ngờ tóm tắt hoặc tự ý thêm thông tin. Giữ nguyên raw text."
            )
            return raw_text, [], warnings

        # Verify preserved numbers
        raw_numbers = sorted(re.findall(r"\b\d+(?:[.,]\d+)?\b", raw_text))
        prop_numbers = sorted(re.findall(r"\b\d+(?:[.,]\d+)?\b", clean_proposed))
        if raw_numbers != prop_numbers:
            # Check how many numbers differ
            missing_nums = set(raw_numbers) - set(prop_numbers)
            added_nums = set(prop_numbers) - set(raw_numbers)
            if missing_nums or added_nums:
                warnings.append(
                    f"Phát hiện số liệu bị thay đổi (thiếu: {missing_nums}, thừa: {added_nums}); "
                    "hủy thay đổi để bảo vệ dữ liệu."
                )
                return raw_text, [], warnings

        # Verify protected technical terms
        raw_lower = raw_text.casefold()
        prop_lower = clean_proposed.casefold()
        for term in PROTECTED_TECHNICAL_TERMS:
            if re.search(r"\b" + re.escape(term) + r"\b", raw_lower):
                if not re.search(r"\b" + re.escape(term) + r"\b", prop_lower):
                    warnings.append(
                        f"Thuật ngữ kỹ thuật '{term}' bị mất trong kết quả; hủy thay đổi."
                    )
                    return raw_text, [], warnings

        # Sanitize changes list
        sanitized_changes = []
        for change in changes:
            if isinstance(change, dict) and "original" in change and "corrected" in change:
                sanitized_changes.append({
                    "original": str(change["original"]),
                    "corrected": str(change["corrected"]),
                    "reason": str(change.get("reason", "Khôi phục dấu / chính tả")),
                })

        return clean_proposed, sanitized_changes, warnings
