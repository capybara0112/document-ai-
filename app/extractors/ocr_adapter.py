"""PaddleOCR adapter with bounded multi-strategy image OCR and auditable selection."""
from __future__ import annotations

from difflib import SequenceMatcher
from importlib.metadata import PackageNotFoundError, version
import logging
from typing import Any

from app.core.config import settings
from app.utils.image_preprocessor import (
    OcrImageCandidate,
    map_box_to_source,
    map_polygon_to_source,
    prepare_ocr_candidates,
)
from app.utils.ocr_postprocessing import build_analysis_result, candidate_summary

_PADDLE_OCR = None
logger = logging.getLogger(__name__)


def _require_ocr() -> None:
    if not settings.ocr_enabled:
        raise RuntimeError("OCR is disabled. Set OCR_ENABLED=true before starting the API.")


def _paddleocr_version() -> str | None:
    try:
        return version("paddleocr")
    except PackageNotFoundError:
        return None


def _get_paddle_ocr():
    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError("OCR is enabled but PaddleOCR is not installed in the active Python environment.") from exc
        _PADDLE_OCR = PaddleOCR(
            lang="vi",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_dir=settings.ocr_text_detection_model_dir,
            text_recognition_model_dir=settings.ocr_text_recognition_model_dir,
        )
    return _PADDLE_OCR


def extract_image_ocr(content: bytes) -> dict:
    """Run several suitable OCR inputs, then retain one complete candidate result."""
    _require_ocr()
    return _extract_image_content(content, _get_paddle_ocr())


def _sanitize_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types or foreign objects to Python native types."""
    if hasattr(obj, "item") and callable(obj.item):
        return _sanitize_jsonable(obj.item())
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return _sanitize_jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _sanitize_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_jsonable(v) for v in obj]
    return obj


def _extract_image_content(content: bytes, ocr: Any) -> dict:
    candidates, image_analysis = prepare_ocr_candidates(
        content,
        mode=settings.ocr_preprocess_mode,
        max_candidates=settings.ocr_max_candidates,
        target_short_edge=settings.ocr_target_short_edge,
        max_dimension=settings.ocr_max_dimension,
        debug_dir=settings.ocr_debug_image_dir,
    )
    source_size = (image_analysis["width_px"], image_analysis["height_px"])
    successful: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            lines = _parse_paddle_result(
                ocr.predict(candidate.image), candidate, source_size
            )
            successful.append({"candidate": candidate, "lines": lines, "summary": candidate_summary(lines)})
        except Exception as exc:
            failures.append({"candidate": candidate.name, "error": str(exc)[:300]})
            logger.warning("OCR candidate '%s' failed: %s", candidate.name, type(exc).__name__)
    if not successful:
        details = "; ".join(f"{item['candidate']}: {item['error']}" for item in failures)
        raise RuntimeError(f"PaddleOCR failed for every image candidate. {details}")

    selected, strategy = _select_candidate(successful, settings.ocr_low_confidence_threshold)
    strategy["failed_candidates"] = failures
    logger.info("Selected OCR candidate '%s' from %d successful candidate(s).", strategy["selected_candidate"], len(successful))
    selected_plan = next(
        (item for item in image_analysis["candidate_plan"] if item["name"] == strategy["selected_candidate"]),
        None,
    )
    if selected_plan:
        image_analysis["image_quality_selected_candidate"] = selected_plan["quality"]
    result = build_analysis_result(
        selected["lines"], image_analysis,
        confidence_threshold=settings.ocr_low_confidence_threshold,
        ocr_strategy=strategy,
    )
    if failures:
        result["warnings"] = [*result["warnings"], f"{len(failures)} OCR candidate(s) failed; selection used the successful candidates."]
    return _sanitize_jsonable({
        "metadata": {
            "pages_count": 1,
            "extraction_method": "paddleocr_multi_strategy",
            "paddleocr_version": _paddleocr_version(),
            "ocr_language": "vi",
            "image_analysis": image_analysis,
        },
        **result,
        "tables": [],
        "fields": {},
    })


def _candidate_text(lines: list[dict[str, Any]]) -> str:
    return " ".join(
        " ".join(str(line.get("text", "")).casefold().split())
        for line in lines if str(line.get("text", "")).strip()
    )


def _select_candidate(
    candidates: list[dict[str, Any]],
    confidence_threshold: float = 0.70,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose a whole candidate, never merge words/lines from competing OCR runs.

    Selection score weights (revised):
    - 0.35 OCR model confidence  (reduced: high confidence ≠ correct diacritics)
    - 0.30 Language consistency  (increased: Vietnamese diacritic signal)
    - 0.25 Inter-candidate agreement
    - 0.10 Relative character coverage
    """
    max_characters = max(item["summary"]["recognized_character_count"] for item in candidates) or 1
    texts = [_candidate_text(item["lines"]) for item in candidates]
    candidate_rows = []
    for index, item in enumerate(candidates):
        agreements = [
            SequenceMatcher(None, texts[index], other).ratio()
            for other_index, other in enumerate(texts)
            if other_index != index
        ]
        agreement = sum(agreements) / len(agreements) if agreements else 1.0
        summary = item["summary"]
        confidence = summary["ocr_confidence_mean"]
        confidence_component = confidence if confidence is not None else 0.0
        language_component = summary["language_consistency"]["score_0_1"]
        coverage = summary["recognized_character_count"] / max_characters

        # Penalise when a high-confidence run has many flagged unaccented tokens
        unaccented_count = len(summary["language_consistency"].get("possible_unaccented_tokens", []))
        unaccented_penalty = min(0.20, unaccented_count * 0.025)

        selection_score = round(
            0.35 * confidence_component
            + 0.30 * language_component
            + 0.25 * agreement
            + 0.10 * coverage
            - unaccented_penalty,
            4,
        )
        candidate_rows.append({
            "index": index,
            "name": item["candidate"].name,
            **summary,
            "inter_candidate_agreement": round(agreement, 4),
            "coverage_relative": round(coverage, 4),
            "unaccented_token_count": unaccented_count,
            "unaccented_penalty": round(unaccented_penalty, 4),
            "selection_score": selection_score,
        })
    best = max(
        candidate_rows,
        key=lambda row: (row["selection_score"], row["recognized_character_count"]),
    )
    return candidates[best["index"]], {
        "selected_candidate": best["name"],
        "candidate_scores": [
            {k: v for k, v in row.items() if k != "index"}
            for row in candidate_rows
        ],
        "selection_note": (
            "selection_score is a routing heuristic combining language-consistency "
            "(Vietnamese diacritic signals), OCR model confidence, inter-candidate "
            "agreement, and relative text coverage. "
            "High OCR confidence does not guarantee correct Vietnamese diacritics."
        ),
    }


def extract_scanned_pdf(content: bytes) -> dict:
    """Render each scanned page and use the same bounded image OCR pipeline."""
    _require_ocr()
    try:
        import cv2
        import fitz
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Scanned-PDF OCR requires PyMuPDF, OpenCV, and NumPy.") from exc
    ocr = _get_paddle_ocr()
    pages: list[dict[str, Any]] = []
    with fitz.open(stream=content, filetype="pdf") as pdf:
        if pdf.page_count > settings.max_pdf_pages:
            raise ValueError(f"PDF exceeds the {settings.max_pdf_pages} page limit.")
        for page_number, page in enumerate(pdf, start=1):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
            encoded_ok, encoded = cv2.imencode(".png", rgb[:, :, ::-1])
            if not encoded_ok:
                raise RuntimeError(f"Could not prepare PDF page {page_number} for OCR.")
            page_result = _extract_image_content(encoded.tobytes(), ocr)
            page_result["page_number"] = page_number
            pages.append(page_result)

    raw_text = "\n\n".join(page["raw_text"] for page in pages if page["raw_text"])
    normalized_text = "\n\n".join(page["normalized_text"] for page in pages if page["normalized_text"])
    review = [{"page_number": page["page_number"], **item} for page in pages for item in page["uncertain_lines"]]
    warnings = list(dict.fromkeys(warning for page in pages for warning in page.get("warnings", [])))
    normalized_section = {"text": normalized_text, "pages": [page["document_structure"] for page in pages]}
    report = {
        "A_document_information": {"page_count": len(pages), "image_quality_by_page": [page["analysis_report"]["A_document_information"] for page in pages]},
        "B_raw_ocr": {"text": raw_text, "pages": [page["raw_text"] for page in pages]},
        "C_normalized_ocr": normalized_section,
        "C_reconstructed_content": normalized_section,
        "D_review_required": review,
        "D_uncertain_parts": review,
        "smart_corrections": [{"page_number": page["page_number"], **item} for page in pages for item in page.get("smart_corrections", [])],
        "E_improvement_suggestions": list(dict.fromkeys(suggestion for page in pages for suggestion in page["analysis_report"]["E_improvement_suggestions"])),
        "provenance": {"raw_text_is_direct_ocr": True, "normalization_is_deterministic": True, "smart_corrections_are_not_applied": True},
    }
    return _sanitize_jsonable({"metadata": {"pages_count": len(pages), "extraction_method": "paddleocr_multi_strategy", "paddleocr_version": _paddleocr_version(), "image_analysis_by_page": [page["metadata"]["image_analysis"] for page in pages]}, "text": normalized_text, "raw_text": raw_text, "normalized_text": normalized_text, "pages": pages, "tables": [], "fields": {}, "uncertain_lines": review, "analysis_report": report, "warnings": warnings})


def _as_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _box_to_xyxy(box: Any) -> list[float] | None:
    box = _as_python(box)
    if not isinstance(box, (list, tuple)):
        return None
    if len(box) == 4 and all(isinstance(value, (int, float)) for value in box):
        return [float(value) for value in box]
    points = []
    for point in box:
        point = _as_python(point)
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
    if not points:
        return None
    return [min(x for x, _y in points), min(y for _x, y in points), max(x for x, _y in points), max(y for _x, y in points)]


def _parse_paddle_result(result: Any, candidate: OcrImageCandidate, source_size: tuple[int, int]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for page_result in result or []:
        payload = getattr(page_result, "json", None)
        payload = payload() if callable(payload) else payload
        if not isinstance(payload, dict) and isinstance(page_result, dict):
            payload = page_result
        payload = payload.get("res", payload) if isinstance(payload, dict) else None
        if not isinstance(payload, dict):
            continue
        texts, scores = payload.get("rec_texts"), payload.get("rec_scores")
        boxes = payload.get("rec_boxes") if payload.get("rec_boxes") is not None else payload.get("rec_polys")
        polys = payload.get("rec_polys")
        texts, scores, boxes = ([] if texts is None else texts), ([] if scores is None else scores), ([] if boxes is None else boxes)
        for index, recognized in enumerate(texts):
            text = str(recognized).strip()
            if not text:
                continue
            try:
                confidence = float(_as_python(scores[index])) if index < len(scores) else None
            except (TypeError, ValueError):
                confidence = None
            ocr_box = _box_to_xyxy(boxes[index]) if index < len(boxes) else None
            ocr_poly = _as_python(polys[index]) if (polys is not None and index < len(polys)) else None
            bounding_box = map_box_to_source(ocr_box, candidate.source_to_candidate, source_size)
            polygon = map_polygon_to_source(ocr_poly, candidate.source_to_candidate, source_size) if ocr_poly else None
            detections.append({
                "text": text,
                "confidence": confidence,
                "bounding_box": bounding_box,
                "ocr_bounding_box": ocr_box,
                "polygon": polygon,
                "ocr_polygon": ocr_poly,
                "candidate": candidate.name,
                "candidate_id": candidate.name,
                "coordinate_space": "source_image_coordinates",
                "source_detection_index": len(detections),
            })
    return detections
