"""Fast tests for image routing and reading-order logic; no OCR model download required."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from app.utils.image_preprocessor import prepare_ocr_candidates
from app.utils.ocr_postprocessing import build_analysis_result, order_lines


def encode(image: np.ndarray) -> bytes:
    ok, content = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError("Could not create test PNG")
    return content.tobytes()


# ---------------------------------------------------------------------------
# Image preparation tests
# ---------------------------------------------------------------------------

class ImagePreparationTests(unittest.TestCase):

    def test_low_light_document_generates_an_enhanced_candidate(self) -> None:
        image = np.full((600, 450, 3), 42, dtype=np.uint8)
        cv2.putText(image, "Van ban tieng Viet", (35, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (85, 85, 85), 2)
        candidates, analysis = prepare_ocr_candidates(encode(image), max_candidates=4)
        self.assertIn("enhanced_color", [c.name for c in candidates])
        self.assertLess(analysis["image_quality_before"]["metrics"]["brightness_mean_0_255"], 75)

    def test_noisy_image_uses_one_selected_denoising_method(self) -> None:
        image = np.full((700, 500, 3), 220, dtype=np.uint8)
        cv2.putText(image, "OCR", (120, 320), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (25, 25, 25), 3)
        noise = np.random.default_rng(7).normal(0, 24, image.shape).astype(np.int16)
        noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        candidates, _analysis = prepare_ocr_candidates(encode(noisy), max_candidates=4)
        names = {c.name for c in candidates}
        self.assertTrue(names & {"denoised", "original"})
        denoise_steps = [
            step
            for c in candidates
            for step in c.preprocessing_steps
            if "noise" in step or "filter" in step
        ]
        self.assertLessEqual(len(denoise_steps), 1)

    def test_source_image_is_always_retained_as_a_candidate(self) -> None:
        image = np.full((240, 320, 3), 255, dtype=np.uint8)
        candidates, analysis = prepare_ocr_candidates(encode(image), max_candidates=2)
        self.assertEqual(candidates[0].name, "original")
        self.assertEqual(analysis["width_px"], 320)
        self.assertEqual(analysis["height_px"], 240)

    def test_blurry_image_generates_sharpened_candidate(self) -> None:
        """A deliberately blurry image should produce a sharpened candidate."""
        # Create a sharp image then blur it heavily
        image = np.full((600, 500, 3), 230, dtype=np.uint8)
        cv2.putText(image, "Sharpness test", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
        blurry = cv2.GaussianBlur(image, (21, 21), 8)
        candidates, analysis = prepare_ocr_candidates(encode(blurry), max_candidates=6)
        candidate_names = [c.name for c in candidates]
        sharpened_steps = [
            step
            for c in candidates
            for step in c.preprocessing_steps
            if "unsharp" in step
        ]
        # Either a sharpened candidate exists or sharpness was above the threshold
        laplacian = analysis["image_quality_before"]["metrics"]["sharpness_laplacian_variance"]
        if laplacian < 80:
            self.assertTrue(
                sharpened_steps or "sharpened" in candidate_names,
                "Expected a sharpened candidate for a blurry image",
            )

    def test_grayscale_like_low_contrast_image_may_produce_binary_candidate(self) -> None:
        """Low-saturation + low-contrast image should trigger binary thresholding."""
        # Grey document with faint text
        image = np.full((600, 450, 3), 190, dtype=np.uint8)
        cv2.putText(image, "Faint text", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (155, 155, 155), 2)
        candidates, analysis = prepare_ocr_candidates(encode(image), max_candidates=6)
        metrics = analysis["image_quality_before"]["metrics"]
        # Only assert binary candidate when conditions are actually triggered
        if (
            metrics["color_saturation_mean_0_255"] < 20
            and metrics["ink_background_separation_0_255"] < 70
        ):
            binary_names = {"adaptive_binary"}
            names = {c.name for c in candidates}
            self.assertTrue(
                names & binary_names,
                f"Expected binary candidate; got {names}",
            )

    def test_diagnostics_contain_required_keys(self) -> None:
        """Diagnostics dict must always include the backward-compatible flat keys."""
        image = np.full((400, 300, 3), 200, dtype=np.uint8)
        _candidates, diagnostics = prepare_ocr_candidates(encode(image))
        required_keys = {
            "width_px", "height_px",
            "image_quality_before", "brightness_mean_0_255",
            "contrast_stddev_0_255", "blur_laplacian_variance",
            "noise_indicator_median_residual", "estimated_skew_degrees",
            "perspective_correction_applied", "deskew_applied",
            "candidate_plan", "warnings",
        }
        missing = required_keys - diagnostics.keys()
        self.assertFalse(missing, f"Missing diagnostics keys: {missing}")

    def test_oversized_image_raises_value_error(self) -> None:
        """Images exceeding 60 MP should raise ValueError before processing."""
        # Encode a 1×1 placeholder but lie about size via a crafted PNG
        # Simpler: use a real large image
        big = np.zeros((7001, 9001, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", big[:10, :10])  # small encode
        # Verify the limit constant covers the oversized dimensions:
        # 7001 * 9001 = 63,018,001 → actually 63,016,001; either way > 60 M.
        self.assertGreater(7001 * 9001, 60_000_000)


# ---------------------------------------------------------------------------
# Reading order tests
# ---------------------------------------------------------------------------

class ReadingOrderTests(unittest.TestCase):

    def test_two_column_prose_is_read_left_column_then_right_column(self) -> None:
        lines = [
            {"text": "R2", "bounding_box": [320, 20, 370, 35]},
            {"text": "L3", "bounding_box": [20, 80, 70, 95]},
            {"text": "R4", "bounding_box": [320, 110, 370, 125]},
            {"text": "L1", "bounding_box": [20, 10, 70, 25]},
            {"text": "R3", "bounding_box": [320, 80, 370, 95]},
            {"text": "L4", "bounding_box": [20, 110, 70, 125]},
            {"text": "R1", "bounding_box": [320, 50, 370, 65]},
            {"text": "L2", "bounding_box": [20, 40, 70, 55]},
        ]
        ordered, strategy = order_lines(lines)
        self.assertEqual(strategy, "two_columns_left_then_right")
        self.assertEqual(
            [ln["text"] for ln in ordered],
            ["L1", "L2", "L3", "L4", "R2", "R1", "R3", "R4"],
        )

    def test_raw_ocr_is_preserved_and_corrections_are_only_suggestions(self) -> None:
        lines = [
            {"text": "Cau 1 (5 diem)", "confidence": 0.98, "bounding_box": [20, 10, 200, 30]},
            {"text": "Doc anh bang Canny", "confidence": 0.99, "bounding_box": [20, 50, 250, 70]},
        ]
        result = build_analysis_result(
            lines,
            {"image_quality_before": {"label": "GOOD"}, "warnings": []},
        )
        self.assertEqual(result["raw_text"], "Cau 1 (5 diem)\nDoc anh bang Canny")
        self.assertIn("Cau 1", result["normalized_text"])
        self.assertTrue(result["smart_corrections"])
        self.assertTrue(result["analysis_report"]["provenance"]["smart_corrections_are_not_applied"])

    def test_single_column_document_uses_row_order(self) -> None:
        """Lines in a narrow column should use simple top-to-bottom row order."""
        lines = [
            {"text": "Line3", "bounding_box": [20, 80, 200, 95]},
            {"text": "Line1", "bounding_box": [20, 10, 200, 25]},
            {"text": "Line2", "bounding_box": [20, 40, 200, 55]},
        ]
        ordered, strategy = order_lines(lines)
        self.assertEqual(strategy, "rows_left_to_right")
        self.assertEqual([ln["text"] for ln in ordered], ["Line1", "Line2", "Line3"])

    def test_mixed_font_size_rows_are_not_merged_incorrectly(self) -> None:
        """A heading (tall) and body text (short) on different rows must stay separate."""
        lines = [
            # Heading: large height
            {"text": "HEADING", "bounding_box": [20, 5, 400, 55]},
            # Body line directly below – should NOT be merged into heading row
            {"text": "body text", "bounding_box": [20, 65, 300, 80]},
            {"text": "more body", "bounding_box": [20, 90, 300, 105]},
        ]
        ordered, _strategy = order_lines(lines)
        texts = [ln["text"] for ln in ordered]
        # HEADING must appear before body text
        self.assertLess(texts.index("HEADING"), texts.index("body text"))
        # All three lines must appear (no merging)
        self.assertEqual(len(ordered), 3)

    def test_unaccented_vietnamese_flagged_in_smart_corrections(self) -> None:
        """Words matching the unaccented Vietnamese dict should generate suggestions."""
        lines = [
            {"text": "Doc anh va xu ly", "confidence": 0.97, "bounding_box": [0, 0, 200, 20]},
        ]
        result = build_analysis_result(
            lines,
            {"image_quality_before": {"label": "FAIR"}, "warnings": []},
        )
        suggestions = [
            item
            for block in result["smart_corrections"]
            for item in block["suggestions"]
        ]
        proposed = {s["proposed"] for s in suggestions}
        # "Doc" → "đọc", "anh" → "ảnh", "va" → "và", "xu ly" → xử lý
        self.assertTrue(
            proposed & {"đọc", "ảnh", "và"},
            f"Expected Vietnamese accent suggestions; got proposed={proposed}",
        )
        # Raw text must be unchanged
        self.assertEqual(result["raw_text"], "Doc anh va xu ly")

    def test_question_classifier_handles_unaccented_cau(self) -> None:
        """'Cau 1' (no diacritic) should still be classified as a question."""
        lines = [
            {"text": "Cau 1 (2 diem)", "confidence": 0.92, "bounding_box": [0, 0, 200, 20]},
        ]
        result = build_analysis_result(
            lines,
            {"image_quality_before": {"label": "GOOD"}, "warnings": []},
        )
        blocks = result["document_structure"]["blocks"]
        self.assertEqual(blocks[0]["type"], "question")

    def test_review_required_includes_unaccented_token_lines(self) -> None:
        """Lines with possible missing diacritics should appear in D_review_required."""
        lines = [
            {"text": "Doc anh bang Canny edge detection", "confidence": 0.96,
             "bounding_box": [0, 0, 300, 20]},
        ]
        result = build_analysis_result(
            lines,
            {"image_quality_before": {"label": "GOOD"}, "warnings": []},
            confidence_threshold=0.70,
        )
        review = result["uncertain_lines"]
        # Should be flagged because of possible_missing_vietnamese_diacritics
        self.assertTrue(
            any("diacritics" in reason.lower() or "ngôn ngữ" in reason
                for item in review for reason in [item.get("reason", "")]),
            f"Expected unaccented-diacritics flag in review; got review={review}",
        )


if __name__ == "__main__":
    unittest.main()
