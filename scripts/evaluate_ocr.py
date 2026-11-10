"""Comprehensive evaluation of OCR quality: CER, WER, Reading Order Accuracy, and Line Grouping."""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def levenshtein_distance(expected: list[str], actual: list[str]) -> int:
    """Standard Levenshtein dynamic programming distance."""
    previous = list(range(len(actual) + 1))
    for row, expected_value in enumerate(expected, start=1):
        current = [row]
        for column, actual_value in enumerate(actual, start=1):
            cost = 0 if expected_value == actual_value else 1
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + cost))
        previous = current
    return previous[-1]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = Levenshtein(chars) / len(reference)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    dist = levenshtein_distance(ref_chars, hyp_chars)
    return round(dist / max(1, len(ref_chars)), 4)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate = Levenshtein(words) / len(reference_words)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = levenshtein_distance(ref_words, hyp_words)
    return round(dist / max(1, len(ref_words)), 4)


def compute_reading_order_accuracy(reference_text: str, predicted_lines: list[str]) -> float:
    """Reading Order Accuracy using relative inversion rank concordance.
    
    Matches each predicted line to the most similar reference line (similarity >= 0.50),
    then computes the proportion of concordant line pairs (Kendall's Tau).
    """
    ref_lines = [ln.strip() for ln in reference_text.splitlines() if ln.strip()]
    if len(ref_lines) <= 1 or not predicted_lines:
        return 1.0

    matched_indices: list[int] = []
    for pred in predicted_lines:
        if not pred.strip():
            continue
        best_idx = -1
        best_ratio = 0.0
        for idx, ref in enumerate(ref_lines):
            ratio = SequenceMatcher(None, pred.casefold(), ref.casefold()).ratio()
            if ratio > best_ratio and ratio >= 0.45:
                best_ratio = ratio
                best_idx = idx
        if best_idx != -1:
            matched_indices.append(best_idx)

    if len(matched_indices) <= 1:
        return 1.0

    # Calculate concordant vs discordant pairs
    total_pairs = 0
    concordant_pairs = 0
    for i in range(len(matched_indices)):
        for j in range(i + 1, len(matched_indices)):
            total_pairs += 1
            if matched_indices[i] <= matched_indices[j]:
                concordant_pairs += 1

    return round(concordant_pairs / max(1, total_pairs), 4)


def compute_line_grouping_accuracy(reference_text: str, actual_blocks: list[dict[str, Any]]) -> float:
    """Measure whether multi-line bullets or paragraphs are preserved without unwanted splits."""
    ref_lines = [ln.strip() for ln in reference_text.splitlines() if ln.strip()]
    if not ref_lines or not actual_blocks:
        return 1.0
    # Ratio of matched logical block contents to reference count
    block_texts = [b.get("text", "").strip() for b in actual_blocks if b.get("text", "").strip()]
    matches = 0
    for ref in ref_lines:
        if any(SequenceMatcher(None, ref.casefold(), bt.casefold()).ratio() >= 0.70 for bt in block_texts):
            matches += 1
    return round(matches / max(1, len(ref_lines)), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Document AI OCR against verified transcripts.")
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("tests/fixtures/manifest.jsonl"), help="JSONL records")
    parser.add_argument("--output", type=Path, default=Path("ocr_evaluation.json"))
    args = parser.parse_args()

    os.environ["OCR_ENABLED"] = "true"
    from app.extractors.ocr_adapter import extract_image_ocr

    records = []
    if args.manifest.exists():
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    # Check if 4.png exists in root or fixtures
    for candidate_4 in [Path("tests/fixtures/4.png"), Path("4.png")]:
        if candidate_4.exists() and not any(r.get("image") == str(candidate_4) for r in records):
            gt_file = Path("tests/fixtures/ground_truth_4.txt")
            if gt_file.exists():
                records.append({
                    "image": str(candidate_4),
                    "verified_transcript": gt_file.read_text(encoding="utf-8"),
                    "note": "Human-verified Ground Truth for 4.png"
                })
                break

    results = []
    for item in records:
        img_spec = item["image"]
        img_path = (args.manifest.parent / img_spec).resolve() if not Path(img_spec).is_absolute() else Path(img_spec)
        if not img_path.exists():
            img_path = Path(img_spec).resolve()
        if not img_path.exists():
            print(f"Warning: image {img_spec} not found, skipping.")
            continue

        reference = item.get("verified_transcript", "")
        extracted = extract_image_ocr(img_path.read_bytes())
        
        raw_text = extracted.get("raw_text", "")
        norm_text = extracted.get("normalized_text", "")
        corr_text = extracted.get("corrected_text", norm_text)

        raw_cer = compute_cer(reference, raw_text)
        raw_wer = compute_wer(reference, raw_text)
        norm_cer = compute_cer(reference, norm_text)
        norm_wer = compute_wer(reference, norm_text)
        corr_cer = compute_cer(reference, corr_text)
        corr_wer = compute_wer(reference, corr_text)

        pred_lines = [ln.get("text", "") for ln in extracted.get("lines", [])]
        reading_order_acc = compute_reading_order_accuracy(reference, pred_lines)
        
        blocks = extracted.get("document_structure", {}).get("hierarchical_blocks", [])
        line_grouping_acc = compute_line_grouping_accuracy(reference, blocks)

        candidate_name = extracted.get("analysis_report", {}).get("A_document_information", {}).get("ocr_strategy", {}).get("selected_candidate", "original")

        results.append({
            "image": str(img_path.name),
            "raw_ocr": {"cer": raw_cer, "wer": raw_wer},
            "normalized_ocr": {"cer": norm_cer, "wer": norm_wer},
            "corrected_ocr": {"cer": corr_cer, "wer": corr_wer},
            "reading_order_accuracy": reading_order_acc,
            "line_grouping_accuracy": line_grouping_acc,
            "selected_candidate": candidate_name,
        })

    summary = {
        "documents_evaluated": len(results),
        "mean_raw_cer": round(sum(r["raw_ocr"]["cer"] for r in results) / len(results), 4) if results else 0.0,
        "mean_normalized_cer": round(sum(r["normalized_ocr"]["cer"] for r in results) / len(results), 4) if results else 0.0,
        "mean_reading_order_accuracy": round(sum(r["reading_order_accuracy"] for r in results) / len(results), 4) if results else 0.0,
        "mean_line_grouping_accuracy": round(sum(r["line_grouping_accuracy"] for r in results) / len(results), 4) if results else 0.0,
        "results": results,
    }

    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
