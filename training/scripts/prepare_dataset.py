"""Dataset Preparation and Validation Tool for Vietnamese OCR Fine-Tuning.

Usage:
    python training/scripts/prepare_dataset.py --input_dir /path/to/crops --output_dir training/datasets
    python training/scripts/prepare_dataset.py --generate_sample_crops
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def normalize_vietnamese_text(text: str) -> str:
    """Normalize text to Unicode NFC and clean abnormal spaces."""
    if not text:
        return ""
    # Standardize Unicode NFC
    norm = unicodedata.normalize("NFC", text)
    # Remove control/invisible chars
    norm = norm.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    # Single whitespace
    norm = re.sub(r"[ \t]+", " ", norm).strip()
    return norm


def load_character_dict(dict_path: Path) -> set[str]:
    """Load character vocabulary set from file."""
    if not dict_path.is_file():
        return set()
    chars = set()
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            c = line.strip("\r\n")
            if c:
                chars.add(c)
    return chars


def validate_image(image_path: Path) -> tuple[bool, str, tuple[int, int]]:
    """Check image readability and geometry."""
    if not image_path.is_file():
        return False, "File does not exist", (0, 0)
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            w, h = img.size
            if w < 6 or h < 6:
                return False, f"Image too small ({w}x{h})", (w, h)
            return True, "OK", (w, h)
    except Exception as exc:
        return False, f"Corrupted or invalid image: {exc}", (0, 0)


def generate_synthetic_sample_crops(output_dir: Path) -> list[tuple[str, str]]:
    """Generate sample Vietnamese line crops for immediate training verification."""
    output_crops_dir = output_dir / "sample_crops"
    output_crops_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        ("line_0001.png", "KIỂM TRA GIỮA KỲ THỊ GIÁC MÁY TÍNH"),
        ("line_0002.png", "Thời gian làm bài: 30 phút"),
        ("line_0003.png", "Câu 1 (5 điểm): Đọc ảnh và thực hiện các thao tác sau"),
        ("line_0004.png", "- Thay đổi kích thước ảnh với hệ số tỉ lệ"),
        ("line_0005.png", "trục x và trục y nhập vào từ bàn phím"),
        ("line_0006.png", "- Biến đổi ảnh và cắt thành ảnh âm bản"),
        ("line_0007.png", "Câu 2 (5 điểm): Tách biên ảnh bằng phương pháp Canny"),
        ("line_0008.png", "- Tạo trackbar để lấy giá trị threshold1, threshold2"),
        ("line_0009.png", "- Sử dụng bộ lọc Gaussian Blur làm mịn ảnh trước khi tách biên"),
        ("line_0010.png", "- Ấn phím 's' hoặc 'c' để lưu lại ảnh kết quả"),
        ("line_0011.png", "ĐƠN VỊ BÁN HÀNG: CÔNG TY TNHH GIẢI PHÁP CÔNG NGHỆ"),
        ("line_0012.png", "Mã số thuế: 0102030405-001"),
        ("line_0013.png", "Địa chỉ: Số 123 Đường Giải Phóng, Quận Hai Bà Trưng, Hà Nội"),
        ("line_0014.png", "Số tài khoản: 19028374659012 tại Techcombank"),
        ("line_0015.png", "Họ tên người mua hàng: Nguyễn Văn An"),
        ("line_0016.png", "Hình thức thanh toán: Chuyển khoản / Tiền mặt"),
        ("line_0017.png", "Tên hàng hóa: Máy tính xách tay ThinkPad T14 Gen 4"),
        ("line_0018.png", "Đơn vị tính: Chiếc, Số lượng: 02, Đơn giá: 25.500.000"),
        ("line_0019.png", "Thành tiền: 51.000.000 VNĐ"),
        ("line_0020.png", "Thuế suất GTGT: 10%, Tiền thuế GTGT: 5.100.000 VNĐ"),
        ("line_0021.png", "Tổng cộng tiền thanh toán: 56.100.000 VNĐ"),
        ("line_0022.png", "Số tiền viết bằng chữ: Năm mươi sáu triệu một trăm nghìn đồng."),
        ("line_0023.png", "Người mua hàng ký và ghi rõ họ tên"),
        ("line_0024.png", "Thủ trưởng đơn vị ký đóng dấu chứng thực"),
    ]

    generated: list[tuple[str, str]] = []
    for filename, text in samples:
        img_path = output_crops_dir / filename
        # Estimate image dimensions
        char_w = 16
        w = max(240, len(text) * char_w + 30)
        h = 48

        # Create realistic document background (slight off-white / noise)
        bg_val = random.randint(245, 255)
        img = Image.new("RGB", (w, h), color=(bg_val, bg_val, bg_val - random.randint(0, 5)))
        draw = ImageDraw.Draw(img)

        # Draw subtle grid or document paper line
        if random.random() > 0.6:
            draw.line([(0, h - 2), (w, h - 2)], fill=(220, 225, 230), width=1)

        # Attempt to use system font or fallback default font
        font = None
        font_names = ["arial.ttf", "calibri.ttf", "times.ttf", "segoeui.ttf"]
        for fn in font_names:
            try:
                font = ImageFont.truetype(fn, 22)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        # Text color: dark charcoal or dark blue
        text_color = (random.randint(15, 35), random.randint(15, 35), random.randint(20, 45))
        draw.text((15, 10), text, fill=text_color, font=font)

        img.save(img_path, format="PNG")
        rel_path = f"sample_crops/{filename}"
        generated.append((rel_path, text))

    return generated


def prepare_dataset(
    input_file: Path | None,
    crops_dir: Path,
    output_dir: Path,
    dict_file: Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Process annotations, validate images, normalize text, and split into train/val/test."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dict_chars = load_character_dict(dict_file)

    pairs: list[tuple[str, str, str]] = []  # (rel_image_path, text, doc_id)

    if input_file and input_file.is_file():
        # Read from tab-separated file: image_path\ttext or image_path\ttext\tdoc_id
        with open(input_file, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                raw_line = raw_line.strip("\r\n")
                if not raw_line or raw_line.startswith("#"):
                    continue
                parts = raw_line.split("\t")
                if len(parts) >= 2:
                    rel_img = parts[0].strip()
                    txt = parts[1].strip()
                    doc_id = parts[2].strip() if len(parts) >= 3 else Path(rel_img).stem
                    pairs.append((rel_img, txt, doc_id))
    else:
        # Scan crops directory directly for image + .txt pairs
        image_exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
        for img_path in crops_dir.rglob("*"):
            if img_path.suffix.lower() in image_exts:
                txt_path = img_path.with_suffix(".txt")
                if txt_path.is_file():
                    with open(txt_path, "r", encoding="utf-8") as tf:
                        txt = tf.read().strip()
                    rel_path = str(img_path.relative_to(output_dir)).replace("\\", "/")
                    doc_id = img_path.parent.name
                    pairs.append((rel_path, txt, doc_id))

    if not pairs:
        print("[INFO] No external pairs found. Generating synthetic Vietnamese sample crops...")
        synthetic = generate_synthetic_sample_crops(output_dir)
        pairs = [(p, t, "sample_doc") for p, t in synthetic]

    # Validate and clean
    valid_samples: list[dict[str, Any]] = []
    skipped_count = 0
    missing_chars: set[str] = set()

    for rel_img, raw_text, doc_id in pairs:
        full_img_path = output_dir / rel_img
        if not full_img_path.is_file():
            # Try relative to crops_dir
            full_img_path = crops_dir / rel_img
        
        ok, msg, dims = validate_image(full_img_path)
        if not ok:
            print(f"[WARN] Skipped {rel_img}: {msg}", file=sys.stderr)
            skipped_count += 1
            continue

        norm_text = normalize_vietnamese_text(raw_text)
        if not norm_text:
            skipped_count += 1
            continue

        # Check vocabulary coverage
        if dict_chars:
            for ch in norm_text:
                if ch not in dict_chars and ch != " ":
                    missing_chars.add(ch)

        valid_samples.append({
            "image": rel_img,
            "text": norm_text,
            "doc_id": doc_id,
            "width": dims[0],
            "height": dims[1],
        })

    # Group-based split (group by doc_id to avoid data leakage)
    doc_groups: dict[str, list[dict[str, Any]]] = {}
    for sample in valid_samples:
        doc_groups.setdefault(sample["doc_id"], []).append(sample)

    doc_ids = sorted(doc_groups.keys())
    rng = random.Random(random_seed)
    rng.shuffle(doc_ids)

    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []
    test_samples: list[dict[str, Any]] = []

    # If single document or small set, split sample-wise
    if len(doc_ids) <= 3:
        all_shuffled = list(valid_samples)
        rng.shuffle(all_shuffled)
        n_total = len(all_shuffled)
        n_val = max(1, int(n_total * val_ratio))
        n_test = max(1, int(n_total * test_ratio))
        val_samples = all_shuffled[:n_val]
        test_samples = all_shuffled[n_val : n_val + n_test]
        train_samples = all_shuffled[n_val + n_test :]
    else:
        n_docs = len(doc_ids)
        n_val_docs = max(1, int(n_docs * val_ratio))
        n_test_docs = max(1, int(n_docs * test_ratio))

        val_doc_ids = set(doc_ids[:n_val_docs])
        test_doc_ids = set(doc_ids[n_val_docs : n_val_docs + n_test_docs])

        for doc_id, group in doc_groups.items():
            if doc_id in val_doc_ids:
                val_samples.extend(group)
            elif doc_id in test_doc_ids:
                test_samples.extend(group)
            else:
                train_samples.extend(group)

    # Write split files (relative image path \t text)
    splits = [
        ("train.txt", train_samples),
        ("val.txt", val_samples),
        ("test.txt", test_samples),
    ]

    for split_filename, split_data in splits:
        split_path = output_dir / split_filename
        with open(split_path, "w", encoding="utf-8") as sf:
            for item in split_data:
                sf.write(f"{item['image']}\t{item['text']}\n")

    report = {
        "total_input_pairs": len(pairs),
        "valid_samples": len(valid_samples),
        "skipped_samples": skipped_count,
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "unique_characters_observed": len({c for s in valid_samples for c in s["text"]}),
        "unrecognized_characters_count": len(missing_chars),
        "unrecognized_characters": sorted(list(missing_chars)),
        "output_directory": str(output_dir),
    }

    report_path = output_dir / "dataset_summary.json"
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset for Vietnamese OCR fine-tuning.")
    parser.add_argument("--input_file", type=Path, default=None, help="Path to raw annotation file.")
    parser.add_argument("--crops_dir", type=Path, default=Path("training/datasets"), help="Directory containing crops.")
    parser.add_argument("--output_dir", type=Path, default=Path("training/datasets"), help="Output directory for dataset splits.")
    parser.add_argument("--dict_file", type=Path, default=Path("training/datasets/vietnamese_dict.txt"), help="Character vocabulary file.")
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate_sample_crops", action="store_true", help="Generate sample synthetic crops.")

    args = parser.parse_args()

    summary = prepare_dataset(
        input_file=args.input_file,
        crops_dir=args.crops_dir,
        output_dir=args.output_dir,
        dict_file=args.dict_file,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )

    print("[SUCCESS] Dataset preparation complete:")
    print(f"  - Train: {summary['train_count']} samples -> {args.output_dir / 'train.txt'}")
    print(f"  - Val:   {summary['val_count']} samples -> {args.output_dir / 'val.txt'}")
    print(f"  - Test:  {summary['test_count']} samples -> {args.output_dir / 'test.txt'}")
    print(f"  - Characters: {summary['unique_characters_observed']} unique characters")
    if summary["unrecognized_characters"]:
        print(f"  - [WARN] Unrecognized in dict: {summary['unrecognized_characters']}")


if __name__ == "__main__":
    main()
