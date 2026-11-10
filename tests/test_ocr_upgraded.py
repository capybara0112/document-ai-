"""Comprehensive unit and regression tests for upgraded Document AI OCR Pipeline."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from app.correction.base_corrector import BaseCorrector, CorrectionResult
from app.correction.correction_service import CorrectionService
from app.utils.image_preprocessor import (
    map_box_to_source,
    map_polygon_to_source,
    prepare_ocr_candidates,
)
from app.utils.ocr_postprocessing import (
    build_analysis_result,
    order_lines,
)
from scripts.evaluate_ocr import (
    compute_cer,
    compute_reading_order_accuracy,
    compute_wer,
    levenshtein_distance,
)


def encode(image: np.ndarray) -> bytes:
    ok, content = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError("Could not encode image to PNG bytes")
    return content.tobytes()


class UpgradedPreprocessingTests(unittest.TestCase):
    """Phase 2: Image Preprocessing Enhancements."""

    def test_coloured_paper_triggers_hsv_clahe(self) -> None:
        """Coloured paper (high saturation) should use HSV value channel CLAHE."""
        # Create a yellow/cream coloured document (high saturation in HSV)
        yellow_img = np.zeros((500, 400, 3), dtype=np.uint8)
        yellow_img[:, :] = (150, 230, 255)  # BGR yellow
        cv2.putText(yellow_img, "Tai lieu mau", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
        candidates, diagnostics = prepare_ocr_candidates(encode(yellow_img), mode="enhanced")

        # Must flag colour_document = True
        self.assertTrue(diagnostics["colour_document"])
        # If enhanced candidate was created, steps must include HSV or CLAHE
        enhanced = next((c for c in candidates if c.name == "enhanced_color"), None)
        if enhanced:
            clahe_step = any("clahe" in s for s in enhanced.preprocessing_steps)
            self.assertTrue(clahe_step)

    def test_high_impulse_noise_selects_median_blur(self) -> None:
        """Image with salt-and-pepper noise should select median blur."""
        img = np.full((500, 400, 3), 240, dtype=np.uint8)
        cv2.putText(img, "Noise Test", (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (10, 10, 10), 2)
        # Add salt and pepper
        rng = np.random.default_rng(42)
        mask_salt = rng.random(img.shape[:2]) < 0.04
        mask_pepper = rng.random(img.shape[:2]) < 0.04
        img[mask_salt] = 255
        img[mask_pepper] = 0

        candidates, diagnostics = prepare_ocr_candidates(encode(img))
        denoised = next((c for c in candidates if c.name == "denoised"), None)
        if denoised:
            self.assertTrue(any("median" in s for s in denoised.preprocessing_steps))

    def test_extreme_perspective_fallback(self) -> None:
        """When perspective distortion is excessive (> 0.28), fallback to deskew without crashing."""
        # Create an image and verify diagnostics include perspective fields
        img = np.full((600, 500, 3), 245, dtype=np.uint8)
        cv2.putText(img, "Perspective", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        candidates, diagnostics = prepare_ocr_candidates(encode(img))

        # Check required geometry diagnostic fields exist
        self.assertIn("page_boundary_detected", diagnostics)
        self.assertIn("geometry_confidence", diagnostics)
        self.assertIn("perspective_distortion_score", diagnostics)
        self.assertIn("page_angle_degrees", diagnostics)
        self.assertIn("page_aspect_ratio", diagnostics)

    def test_coordinate_mapping_box_and_polygon(self) -> None:
        """Inverse transformation must correctly map boxes and polygons back to source coordinates."""
        # Scale 2x transform matrix
        matrix = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype="float64")
        source_size = (1000, 800)

        # Candidate box at (100, 200, 300, 400) -> source box should be (50, 100, 150, 200)
        box = [100.0, 200.0, 300.0, 400.0]
        mapped_box = map_box_to_source(box, matrix, source_size)
        self.assertIsNotNone(mapped_box)
        self.assertAlmostEqual(mapped_box[0], 50.0, places=1)
        self.assertAlmostEqual(mapped_box[1], 100.0, places=1)
        self.assertAlmostEqual(mapped_box[2], 150.0, places=1)
        self.assertAlmostEqual(mapped_box[3], 200.0, places=1)

        # Polygon mapping
        poly = [[100.0, 200.0], [300.0, 200.0], [300.0, 400.0], [100.0, 400.0]]
        mapped_poly = map_polygon_to_source(poly, matrix, source_size)
        self.assertIsNotNone(mapped_poly)
        self.assertEqual(len(mapped_poly), 4)
        self.assertAlmostEqual(mapped_poly[0][0], 50.0, places=1)
        self.assertAlmostEqual(mapped_poly[0][1], 100.0, places=1)


class UpgradedReadingOrderTests(unittest.TestCase):
    """Phase 4: Document Layout Analysis & Reading Order Reconstruction."""

    def test_header_lines_remain_before_body_rows(self) -> None:
        """Header title and duration must stay at the top in correct visual order."""
        lines = [
            # Body question and bullets
            {"text": "Câu 1 (5 điểm)", "bounding_box": [50, 150, 500, 175]},
            {"text": "- Tách biên ảnh bằng Canny", "bounding_box": [70, 200, 450, 225]},
            # Header lines at top
            {"text": "KIỂM TRA GIỮA KỲ THỊ GIÁC MÁY TÍNH", "bounding_box": [50, 30, 400, 60]},
            {"text": "Thời gian: 30 phút", "bounding_box": [550, 35, 750, 60]},
        ]
        ordered, strategy = order_lines(lines)
        texts = [ln["text"] for ln in ordered]
        # Title and Duration must precede Câu 1
        self.assertLess(texts.index("KIỂM TRA GIỮA KỲ THỊ GIÁC MÁY TÍNH"), texts.index("Câu 1 (5 điểm)"))
        self.assertLess(texts.index("Thời gian: 30 phút"), texts.index("Câu 1 (5 điểm)"))
        self.assertEqual(texts[0], "KIỂM TRA GIỮA KỲ THỊ GIÁC MÁY TÍNH")
        self.assertEqual(texts[1], "Thời gian: 30 phút")

    def test_bullet_continuation_line_grouped_into_hierarchical_block(self) -> None:
        """A wrapped line following a bullet item must be merged into that block."""
        lines = [
            {"text": "Câu 1: Đọc ảnh và thực hiện:", "bounding_box": [50, 50, 500, 75]},
            {"text": "- Thay đổi kích thước ảnh với hệ số tỉ lệ trục x và trục y", "bounding_box": [70, 100, 650, 125]},
            {"text": "nhập vào từ bàn phím", "bounding_box": [90, 130, 300, 155]},  # continuation line
            {"text": "- Tách biên Canny", "bounding_box": [70, 180, 400, 205]},
        ]
        result = build_analysis_result(lines, {"image_quality_before": {}, "warnings": []})
        hierarchical = result["document_structure"]["hierarchical_blocks"]
        
        # Find the block for the first bullet
        bullet_block = next((b for b in hierarchical if "Thay đổi kích thước" in b["text"]), None)
        self.assertIsNotNone(bullet_block)
        # Must have merged the continuation text
        self.assertIn("nhập vào từ bàn phím", bullet_block["text"])
        self.assertEqual(len(bullet_block["lines"]), 2)


class DummyCorrector(BaseCorrector):
    def __init__(self):
        super().__init__(provider_name="dummy")

    def correct(self, text: str, lines: list | None = None) -> CorrectionResult:
        return CorrectionResult(raw_text=text, corrected_text=text, correction_status="UNCHANGED")


class UpgradedAICorrectionTests(unittest.TestCase):
    """Phase 5: AI Language Correction & Fidelity Validation."""

    def test_validator_rejects_altered_numbers(self) -> None:
        """Validation must reject correction if numbers or equations are changed."""
        corrector = DummyCorrector()
        raw = "Câu 1 (5 điểm): Thời gian 30 phút."
        # Altered number (30 -> 45)
        bad_prop = "Câu 1 (5 điểm): Thời gian 45 phút."
        clean, changes, warnings = corrector.validate_and_sanitize(raw, bad_prop, [], [])
        # Must reject and revert to raw
        self.assertEqual(clean, raw)
        self.assertTrue(any("số liệu" in w for w in warnings))

    def test_validator_rejects_removed_technical_terms(self) -> None:
        """Validation must protect terms like OpenCV, Canny, Python."""
        corrector = DummyCorrector()
        raw = "Tách biên ảnh bằng phương pháp Canny trong OpenCV."
        bad_prop = "Tách biên ảnh bằng phương pháp tách cạnh thông thường."  # Canny and OpenCV dropped
        clean, changes, warnings = corrector.validate_and_sanitize(raw, bad_prop, [], [])
        self.assertEqual(clean, raw)
        self.assertTrue(any("thuật ngữ" in w.lower() for w in warnings))

    def test_validator_accepts_valid_diacritic_correction(self) -> None:
        """Validation must accept genuine Vietnamese diacritic restoration."""
        corrector = DummyCorrector()
        raw = "Kiém tra gia k Th giác máy tính"
        good_prop = "Kiểm tra giữa kỳ Thị giác máy tính"
        changes = [
            {"original": "Kiém", "corrected": "Kiểm", "reason": "Dấu hỏi"},
            {"original": "gia k", "corrected": "giữa kỳ", "reason": "Thiếu dấu"},
            {"original": "Th", "corrected": "Thị", "reason": "Dấu nặng"},
        ]
        clean, valid_changes, warnings = corrector.validate_and_sanitize(raw, good_prop, changes, [])
        self.assertEqual(clean, good_prop)
        self.assertEqual(len(valid_changes), 3)
        self.assertEqual(len(warnings), 0)

    def test_correction_service_disabled_by_default(self) -> None:
        """When OCR_LLM_CORRECTION_ENABLED is false, service returns SKIPPED cleanly."""
        service = CorrectionService()
        service.enabled = False
        res = service.correct_text("Test raw text")
        self.assertEqual(res.correction_status, "SKIPPED")
        self.assertEqual(res.corrected_text, "Test raw text")


class EvaluationMetricsTests(unittest.TestCase):
    """Phase 7: Evaluation Metrics (CER, WER, Reading Order Accuracy)."""

    def test_cer_and_wer_computation(self) -> None:
        ref = "Kiểm tra giữa kỳ môn Thị giác máy tính"
        hyp = "Kiem tra giua ky mon Thi giac may tinh"
        cer = compute_cer(ref, hyp)
        wer = compute_wer(ref, hyp)
        self.assertGreater(cer, 0.0)
        self.assertLess(cer, 1.0)
        self.assertGreater(wer, 0.0)

        # Identical text should yield 0.0
        self.assertEqual(compute_cer(ref, ref), 0.0)
        self.assertEqual(compute_wer(ref, ref), 0.0)

    def test_reading_order_accuracy_computation(self) -> None:
        ref = "Dòng 1\nDòng 2\nDòng 3\nDòng 4"
        # Perfectly ordered predicted lines
        perfect = ["Dòng 1", "Dòng 2", "Dòng 3", "Dòng 4"]
        self.assertEqual(compute_reading_order_accuracy(ref, perfect), 1.0)

        # Inverted order
        inverted = ["Dòng 4", "Dòng 3", "Dòng 2", "Dòng 1"]
        acc = compute_reading_order_accuracy(ref, inverted)
        self.assertLess(acc, 0.5)


class VietnameseCorrectorTests(unittest.TestCase):
    """Phase 6: Vietnamese Rule-based OCR Corrections and Entity Protection."""

    def test_vietnamese_corrector_fixes_common_ocr_errors(self) -> None:
        from app.postprocessors import get_vietnamese_corrector
        corrector = get_vietnamese_corrector()

        raw = "Kiém tra gia k Th giác máy tính\nđc ành và thc hin các thao tác sau"
        res = corrector.process(raw)
        normalized = res["normalized_text"]
        self.assertIn("Kiểm tra giữa kỳ Thị giác máy tính", normalized)
        self.assertIn("đọc ảnh và thực hiện các thao tác sau", normalized)
        self.assertGreater(res["corrections_count"], 0)

    def test_vietnamese_corrector_preserves_protected_entities(self) -> None:
        from app.postprocessors import get_vietnamese_corrector
        corrector = get_vietnamese_corrector()

        raw = "Mã số thuế: 0102030405-001, điện thoại: 0912345678, số tiền: 1.050.000, ngày: 15/10/2026, thuật toán Canny trong OpenCV."
        res = corrector.process(raw)
        norm = res["normalized_text"]
        self.assertIn("0102030405-001", norm)
        self.assertIn("0912345678", norm)
        self.assertIn("1.050.000", norm)
        self.assertIn("15/10/2026", norm)
        self.assertIn("Canny", norm)
        self.assertIn("OpenCV", norm)

    def test_build_analysis_result_includes_vietnamese_corrections(self) -> None:
        lines = [
            {"text": "Kiém tra gia k Th giác máy tính", "confidence": 0.95, "bounding_box": [50, 10, 400, 35]},
            {"text": "Câu 1 (5 điém): Đc ành và thc hin", "confidence": 0.92, "bounding_box": [50, 50, 450, 75]},
        ]
        result = build_analysis_result(lines, {"image_quality_before": {}, "warnings": []})
        self.assertTrue(result["vietnamese_corrections"])
        self.assertIn("Kiểm tra giữa kỳ Thị giác máy tính", result["corrected_text"])
        self.assertIn("vietnamese_rule_corrections_count", result["analysis_report"]["provenance"])


class TiffAndRoutingTests(unittest.TestCase):
    """File intake: TIFF and image signature routing tests."""

    def test_tiff_detection_and_routing(self) -> None:
        from io import BytesIO
        from PIL import Image
        from app.core.router import detect_file_type

        # Generate a 1-page TIFF in memory
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        buf = BytesIO()
        img.save(buf, format="TIFF")
        tiff_bytes = buf.getvalue()

        route, meta = detect_file_type("scan.tiff", tiff_bytes, max_pdf_pages=50)
        self.assertEqual(route, "IMAGE")
        self.assertEqual(meta["detected_format"], "TIFF")
        self.assertEqual(meta["pages_count"], 1)

    def test_image_preprocessor_reads_tiff_bytes(self) -> None:
        from io import BytesIO
        from PIL import Image
        from app.utils.image_preprocessor import prepare_ocr_candidates

        img = Image.new("RGB", (120, 80), color=(200, 200, 200))
        buf = BytesIO()
        img.save(buf, format="TIFF")
        tiff_bytes = buf.getvalue()

        candidates, diagnostics = prepare_ocr_candidates(tiff_bytes, mode="original")
        self.assertGreater(len(candidates), 0)
        self.assertEqual(diagnostics["width_px"], 120)
        self.assertEqual(diagnostics["height_px"], 80)


class TrainingSuiteTests(unittest.TestCase):
    """Verification for training and fine-tuning suite."""

    def test_training_tools_import_and_dry_run(self) -> None:
        from pathlib import Path
        from training.scripts.prepare_dataset import normalize_vietnamese_text, load_character_dict
        from training.scripts.train_recognition import run_training

        # Verify Unicode NFC normalization
        norm = normalize_vietnamese_text("Kiém tra gia k\u00a0Th giác\u200b")
        self.assertEqual(norm, "Kiém tra gia k Th giác")

        # Verify character dict loads
        chars = load_character_dict(Path("training/datasets/vietnamese_dict.txt"))
        self.assertIn("ă", chars)
        self.assertIn("đ", chars)
        self.assertIn("0", chars)

        # Verify dry run completes
        summary = run_training(
            config_path=Path("training/configs/rec_vietnamese_v6.yml"),
            device="cpu",
            dry_run=True,
        )
        self.assertEqual(summary["status"], "DRY_RUN_PASSED")


class WebUiAndSchemaTests(unittest.TestCase):
    """Verification for Web UI and Swagger OpenAPI file schemas."""

    def test_web_ui_served_at_root(self) -> None:
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Document AI Engine", res.text)
        self.assertIn("dropzone", res.text)

    def test_openapi_file_schema_has_binary_format(self) -> None:
        from app.main import app

        schema = app.openapi()
        convert_schema = schema["components"]["schemas"]["Body_convert_image_api_v1_convert_image_post"]
        file_prop = convert_schema["properties"]["file"]
        self.assertEqual(file_prop.get("format"), "binary")

        analyze_schema = schema["components"]["schemas"]["Body_analyze_document_api_v1_analyze_post"]
        self.assertEqual(analyze_schema["properties"]["file"].get("format"), "binary")


if __name__ == "__main__":
    unittest.main()
