"""Generate realistic synthetic test document fixtures for Vietnamese OCR evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FIXTURES_DIR = Path("tests/fixtures")
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def _get_font(size: int = 24):
    for font_name in ["arial.ttf", "segoeui.ttf", "tahoma.ttf", "times.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_base_canvas(width: int = 900, height: int = 1100, bg_color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (width, height), bg_color)


def generate_all_fixtures() -> None:
    font_title = _get_font(28)
    font_body = _get_font(20)

    manifest_records = []

    # 1. Coloured paper document (Warm yellow/cream document)
    img_colored = create_base_canvas(bg_color=(255, 240, 200))
    draw = ImageDraw.Draw(img_colored)
    draw.text((80, 60), "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", font=font_title, fill=(30, 30, 30))
    draw.text((80, 110), "Độc lập - Tự do - Hạnh phúc", font=font_body, fill=(30, 30, 30))
    draw.text((80, 170), "THÔNG BÁO VỀ KỲ THI THỊ GIÁC MÁY TÍNH", font=font_title, fill=(20, 20, 20))
    draw.text((80, 230), "Thời gian thi chính thức: 45 phút", font=font_body, fill=(30, 30, 30))
    draw.text((80, 270), "Thí sinh sử dụng Python và OpenCV để thực hiện bài thi.", font=font_body, fill=(30, 30, 30))
    colored_path = FIXTURES_DIR / "fixture_colored_paper.png"
    img_colored.save(colored_path)
    colored_text = (
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
        "Độc lập - Tự do - Hạnh phúc\n"
        "THÔNG BÁO VỀ KỲ THI THỊ GIÁC MÁY TÍNH\n"
        "Thời gian thi chính thức: 45 phút\n"
        "Thí sinh sử dụng Python và OpenCV để thực hiện bài thi."
    )
    (FIXTURES_DIR / "fixture_colored_paper.txt").write_text(colored_text, encoding="utf-8")
    manifest_records.append({"image": "fixture_colored_paper.png", "verified_transcript": colored_text})

    # 2. Skewed document (Rotated by 5.5 degrees)
    img_skew = create_base_canvas(width=900, height=700)
    draw_skew = ImageDraw.Draw(img_skew)
    draw_skew.text((60, 50), "BÁO CÁO KỸ THUẬT XỬ LÝ ẢNH", font=font_title, fill=(0, 0, 0))
    draw_skew.text((60, 110), "1. Khử nhiễu bằng bộ lọc Gaussian và Bilateral.", font=font_body, fill=(0, 0, 0))
    draw_skew.text((60, 150), "2. Tách biên bằng thuật toán Canny với hai ngưỡng.", font=font_body, fill=(0, 0, 0))
    draw_skew.text((60, 190), "3. Sử dụng trackbar để điều chỉnh tham số ngưỡng.", font=font_body, fill=(0, 0, 0))
    
    cv_skew = cv2.cvtColor(np.array(img_skew), cv2.COLOR_RGB2BGR)
    center = (cv_skew.shape[1] // 2, cv_skew.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, 5.5, 1.0)
    cv_rotated = cv2.warpAffine(cv_skew, rot_mat, (cv_skew.shape[1], cv_skew.shape[0]), borderValue=(255, 255, 255))
    skew_path = FIXTURES_DIR / "fixture_skewed.png"
    cv2.imwrite(str(skew_path), cv_rotated)
    skew_text = (
        "BÁO CÁO KỸ THUẬT XỬ LÝ ẢNH\n"
        "1. Khử nhiễu bằng bộ lọc Gaussian và Bilateral.\n"
        "2. Tách biên bằng thuật toán Canny với hai ngưỡng.\n"
        "3. Sử dụng trackbar để điều chỉnh tham số ngưỡng."
    )
    (FIXTURES_DIR / "fixture_skewed.txt").write_text(skew_text, encoding="utf-8")
    manifest_records.append({"image": "fixture_skewed.png", "verified_transcript": skew_text})

    # 3. Noisy document (Gaussian + impulse noise)
    img_noise = create_base_canvas(width=850, height=600)
    draw_noise = ImageDraw.Draw(img_noise)
    draw_noise.text((60, 60), "HƯỚNG DẪN THỰC HÀNH PADDLEOCR", font=font_title, fill=(0, 0, 0))
    draw_noise.text((60, 120), "Kiểm tra độ chính xác nhận dạng văn bản tiếng Việt.", font=font_body, fill=(0, 0, 0))
    draw_noise.text((60, 160), "Bảo toàn các dấu thanh: sắc, huyền, hỏi, ngã, nặng.", font=font_body, fill=(0, 0, 0))
    cv_noise = cv2.cvtColor(np.array(img_noise), cv2.COLOR_RGB2BGR).astype(np.float32)
    noise = np.random.normal(0, 18, cv_noise.shape)
    cv_noisy = np.clip(cv_noise + noise, 0, 255).astype(np.uint8)
    noise_path = FIXTURES_DIR / "fixture_noisy.png"
    cv2.imwrite(str(noise_path), cv_noisy)
    noisy_text = (
        "HƯỚNG DẪN THỰC HÀNH PADDLEOCR\n"
        "Kiểm tra độ chính xác nhận dạng văn bản tiếng Việt.\n"
        "Bảo toàn các dấu thanh: sắc, huyền, hỏi, ngã, nặng."
    )
    (FIXTURES_DIR / "fixture_noisy.txt").write_text(noisy_text, encoding="utf-8")
    manifest_records.append({"image": "fixture_noisy.png", "verified_transcript": noisy_text})

    # 4. Multi-line wrapped bullet points & questions (Matching 4.png structure)
    img_exam = create_base_canvas(width=950, height=850)
    draw_exam = ImageDraw.Draw(img_exam)
    draw_exam.text((80, 50), "KIỂM TRA GIỮA KỲ THỊ GIÁC MÁY TÍNH", font=font_title, fill=(0, 0, 0))
    draw_exam.text((650, 55), "Thời gian: 30 phút", font=font_body, fill=(0, 0, 0))
    draw_exam.text((80, 120), "Câu 1 (5 điểm) Đọc ảnh và thực hiện các thao tác sau:", font=font_body, fill=(0, 0, 0))
    draw_exam.text((100, 170), "- Thay đổi kích thước ảnh với hệ số tỉ lệ trục x và trục y", font=font_body, fill=(0, 0, 0))
    draw_exam.text((120, 205), "nhập vào từ bàn phím", font=font_body, fill=(0, 0, 0))
    draw_exam.text((100, 255), "- Biến đổi ảnh và cắt thành ảnh âm bản", font=font_body, fill=(0, 0, 0))
    draw_exam.text((100, 305), "- Tách biên ảnh bằng phương pháp Canny", font=font_body, fill=(0, 0, 0))
    draw_exam.text((100, 355), "- Tạo trackbar để lấy kích thước cho kernel.", font=font_body, fill=(0, 0, 0))
    draw_exam.text((100, 405), "- Sử dụng bộ lọc trung bình để lọc ảnh với kích thước bộ lọc", font=font_body, fill=(0, 0, 0))
    draw_exam.text((120, 440), "lấy từ trackbar", font=font_body, fill=(0, 0, 0))
    draw_exam.text((100, 490), "- Ấn phím C để lưu lại ảnh kết quả", font=font_body, fill=(0, 0, 0))
    exam_path = FIXTURES_DIR / "fixture_exam_cv.png"
    img_exam.save(exam_path)

    exam_gt = (
        "KIỂM TRA GIỮA KỲ THỊ GIÁC MÁY TÍNH\n"
        "Thời gian: 30 phút\n"
        "Câu 1 (5 điểm) Đọc ảnh và thực hiện các thao tác sau:\n"
        "- Thay đổi kích thước ảnh với hệ số tỉ lệ trục x và trục y nhập vào từ bàn phím\n"
        "- Biến đổi ảnh và cắt thành ảnh âm bản\n"
        "- Tách biên ảnh bằng phương pháp Canny\n"
        "- Tạo trackbar để lấy kích thước cho kernel.\n"
        "- Sử dụng bộ lọc trung bình để lọc ảnh với kích thước bộ lọc lấy từ trackbar\n"
        "- Ấn phím C để lưu lại ảnh kết quả"
    )
    (FIXTURES_DIR / "ground_truth_4.txt").write_text(exam_gt, encoding="utf-8")
    (FIXTURES_DIR / "fixture_exam_cv.txt").write_text(exam_gt, encoding="utf-8")
    manifest_records.append({"image": "fixture_exam_cv.png", "verified_transcript": exam_gt})

    # Save manifest.jsonl
    manifest_file = FIXTURES_DIR / "manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Also create README in tests/fixtures explaining 4.png placement
    readme_text = """# Test Fixtures & Evaluation Data

Thư mục này chứa các dữ liệu thử nghiệm thực tế:
1. `ground_truth_4.txt`: Nội dung ground truth đã kiểm chứng cho đề thi Thị giác máy tính (`4.png`).
2. `fixture_exam_cv.png`: Ảnh tổng hợp mô phỏng cấu trúc đề thi `4.png` với đầy đủ dấu tiếng Việt chuẩn.
3. `fixture_colored_paper.png`: Ảnh tài liệu nền giấy màu vàng/kem (kiểm thử Colour-aware CLAHE).
4. `fixture_skewed.png`: Ảnh tài liệu bị nghiêng 5.5 độ (kiểm thử Deskew).
5. `fixture_noisy.png`: Ảnh tài liệu có nhiễu Gaussian (kiểm thử Adaptive Denoise).
6. `manifest.jsonl`: Danh sách ánh xạ ảnh và ground truth phục vụ tính CER, WER, Reading Order Accuracy.

## Vị trí đặt ảnh 4.png gốc từ người dùng:
Để đánh giá trực tiếp trên ảnh chụp thực tế `4.png`, bạn có thể đặt file `4.png` tại một trong hai đường dẫn:
- `D:\\code\\document-ai-engine\\4.png`
- `D:\\code\\document-ai-engine\\tests\\fixtures\\4.png`

Khi phát hiện file, pipeline đánh giá sẽ tự động chạy OCR trên ảnh này và so sánh với `ground_truth_4.txt`.
"""
    (FIXTURES_DIR / "README.md").write_text(readme_text, encoding="utf-8")
    print(f"Generated {len(manifest_records)} fixtures successfully in {FIXTURES_DIR}")


if __name__ == "__main__":
    generate_all_fixtures()
