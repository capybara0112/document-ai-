# Hướng Dẫn Huấn Luyện & Fine-Tuning OCR Tiếng Việt (PP-OCRv6)
## Document AI Engine - Text Recognition Training Suite

Tài liệu này hướng dẫn chi tiết quy trình chuẩn bị dữ liệu, cấu hình, huấn luyện (fine-tune) và triển khai mô hình nhận dạng ký tự tiếng Việt (Text Recognition) dựa trên kiến trúc **PP-OCRv6** (SVTR_LCNet) cho hệ thống **Document AI Engine**.

---

## 1. Tổng Quan Kiến Trúc & Mục Tiêu

Hệ thống mặc định sử dụng PaddleOCR đa ngôn ngữ hỗ trợ tiếng Việt (`lang="vi"`). Khi xử lý các tài liệu đặc thù (hóa đơn VAT, đề thi in mờ, font chữ đặc thù, văn bản nghiêng, chữ viết tay), việc fine-tuning mô hình Recognition sẽ giải quyết triệt để các lỗi:
- Mất dấu thanh (`kiém` thay vì `kiểm`, `thc hin` thay vì `thực hiện`).
- Nhầm lẫn nguyên âm tiếng Việt (`ă`, `â`, `ê`, `ơ`, `ư`, `đ`).
- Nhầm lẫn số và chữ cái (`O` và `0`, `l` và `1`).

### Cấu Trúc Bộ Công Cụ Huấn Luyện

```
training/
├── configs/
│   └── rec_vietnamese_v6.yml       # Cấu hình huấn luyện PaddleOCR PP-OCRv6
├── datasets/
│   ├── sample_crops/              # Ảnh crop từng dòng văn bản
│   ├── train.txt                  # Danh sách mẫu train (<image_rel_path>\t<text>)
│   ├── val.txt                    # Danh sách mẫu validation
│   ├── test.txt                   # Danh sách mẫu test độc lập
│   ├── vietnamese_dict.txt        # Bảng từ vựng ký tự tiếng Việt đầy đủ
│   └── dataset_summary.json       # Báo cáo thống kê dữ liệu
├── scripts/
│   ├── prepare_dataset.py         # Kiểm tra ảnh, chuẩn hóa NFC, chia tập không rò rỉ
│   ├── train_recognition.py       # Trình điều khiển huấn luyện, early stopping, metrics
│   └── export_model.py            # Chuyển đổi checkpoint sang định dạng inference
└── TRAINING_GUIDE.md              # Hướng dẫn chi tiết này
```

---

## 2. Quy Trình Chuẩn Bị Dữ Liệu (Dataset Preparation)

### Định Dạng Nhãn (Annotation Format)
Mỗi dòng văn bản crop được lưu thành 1 file ảnh (`.png`, `.jpg`, `.tiff`), gán nhãn trong file văn bản UTF-8 phân cách bằng ký tự **TAB (`\t`)**:
```text
sample_crops/line_0001.png<TAB>KIỂM TRA GIỮA KỲ THỊ GIÁC MÁY TÍNH
sample_crops/line_0002.png<TAB>Thời gian làm bài: 30 phút
sample_crops/line_0003.png<TAB>Câu 1 (5 điểm): Đọc ảnh và thực hiện các thao tác sau
```

### Nguyên Tắc Phân Chia Tránh Rò Rỉ Dữ Liệu (Data Leakage)
1. **Chia theo Document/Writer ID**: Không chia ngẫu nhiên các dòng crop từ cùng một trang hóa đơn/đề thi vào cả tập train và test. Các dòng từ cùng một tài liệu phải nằm trọn vẹn trong một tập.
2. **Tỷ lệ chuẩn**: 80% Train, 10% Validation, 10% Test.
3. **Chuẩn hóa Unicode NFC**: Toàn bộ nhãn được chuẩn hóa NFC để tránh xung đột giữa Unicode dựng sẵn và tổ hợp.

### Lệnh Chuẩn Bị & Kiểm Định Dữ Liệu
Chạy script `prepare_dataset.py` để tự động kiểm tra kích thước ảnh, độ nét, chuẩn hóa nhãn và sinh các file split:

```powershell
# Chế độ kiểm tra dữ liệu hiện có hoặc sinh mẫu thử nghiệm:
.venv311\Scripts\python training/scripts/prepare_dataset.py --generate_sample_crops

# Chế độ xử lý từ thư mục ảnh crop thực tế:
.venv311\Scripts\python training/scripts/prepare_dataset.py `
    --crops_dir "D:\datasets\vietnamese_invoices\crops" `
    --output_dir "training/datasets" `
    --train_ratio 0.80 `
    --val_ratio 0.10 `
    --test_ratio 0.10
```

---

## 3. Huấn Luyện Mô Hình (Model Fine-Tuning)

### Huấn Luyện Cục Bộ (Local Machine)

Sử dụng script `train_recognition.py` với cấu hình định sẵn:

```powershell
# Kiểm tra cấu hình và dữ liệu trước khi train (Dry-run):
.venv311\Scripts\python training/scripts/train_recognition.py --config training/configs/rec_vietnamese_v6.yml --dry_run

# Huấn luyện trên GPU (khuyến nghị cho tập dữ liệu > 1,000 ảnh):
.venv311\Scripts\python training/scripts/train_recognition.py `
    --config training/configs/rec_vietnamese_v6.yml `
    --device gpu `
    --epochs 100 `
    --batch_size 64 `
    --learning_rate 0.0005 `
    --early_stopping_patience 10

# Huấn luyện thử nghiệm trên CPU:
.venv311\Scripts\python training/scripts/train_recognition.py `
    --config training/configs/rec_vietnamese_v6.yml `
    --device cpu `
    --epochs 10
```

### Huấn Luyện Miễn Phí Trên Google Colab / Kaggle (NVIDIA T4 / P100 GPU)

1. Tải thư mục `training/` lên Google Drive hoặc Kaggle Dataset.
2. Khởi tạo Colab Notebook với GPU:
```python
# Cài đặt PaddlePaddle GPU và PaddleOCR
!pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
!git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleOCR.git
!pip install -r PaddleOCR/requirements.txt
!pip install pyyaml pillow

# Tải Pretrained Weights của PP-OCRv6 Recognition
!wget -nc https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/multilingual_PP-OCRv3_rec_train.tar
!tar -xf multilingual_PP-OCRv3_rec_train.tar

# Chạy huấn luyện
!python training/scripts/train_recognition.py \
    --config training/configs/rec_vietnamese_v6.yml \
    --pretrained_model multilingual_PP-OCRv3_rec_train/best_accuracy \
    --device gpu \
    --epochs 100 \
    --batch_size 128
```

---

## 4. Đóng Gói & Xuất Mô Hình (Model Export)

Mô hình huấn luyện xong lưu checkpoint tại `output/rec_vietnamese_v6/best_accuracy`.
Trước khi sử dụng trong API production, cần chuyển đổi sang định dạng Paddle Inference (`inference.pdmodel`, `inference.pdiparams`):

```powershell
.venv311\Scripts\python training/scripts/export_model.py `
    --config training/configs/rec_vietnamese_v6.yml `
    --checkpoints output/rec_vietnamese_v6/best_accuracy `
    --output_dir output/rec_inference_vietnamese
```

Kết quả sinh ra thư mục `output/rec_inference_vietnamese/` gồm:
- `inference.pdmodel`: Đồ thị tính toán đã được tối ưu hóa (frozen graph).
- `inference.pdiparams`: Trọng số mô hình.
- `vietnamese_dict.txt`: Bảng từ điển ký tự tiếng Việt đồng bộ.
- `inference_metadata.json`: Thông số kỹ thuật của bản export.

---

## 5. Tích Hợp Vào Document AI Engine

Để hệ thống `Document AI Engine` tự động nạp mô hình vừa fine-tune thay cho mô hình mặc định:

### Cách 1: Thiết Lập Trong File `.env` (Khuyến nghị)
Mở file `.env` tại thư mục gốc dự án và cập nhật:
```env
OCR_ENABLED=true
OCR_TEXT_RECOGNITION_MODEL_DIR=D:\code\document-ai-engine\output\rec_inference_vietnamese
OCR_TEXT_DETECTION_MODEL_DIR=
```

### Cách 2: Thiết Lập Bằng Biến Môi Trường PowerShell
```powershell
$env:OCR_TEXT_RECOGNITION_MODEL_DIR = "D:\code\document-ai-engine\output\rec_inference_vietnamese"
$env:OCR_ENABLED = "true"

# Khởi chạy lại server FastAPI:
.venv311\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 6. Đánh Giá & Đo Lường Hiệu Năng (Evaluation)

Sau khi tích hợp mô hình mới, chạy công cụ đánh giá `scripts/evaluate_ocr.py`:

```powershell
.venv311\Scripts\python scripts/evaluate_ocr.py --manifest tests/fixtures/manifest.jsonl
```

### Tiêu Chuẩn Đạt Yêu Cầu (Target Metrics)
- **Character Error Rate (CER)**: Giảm xuống dưới **0.05** (< 5%).
- **Reading Order Accuracy**: Giữ vững **100%** (1.000).
- **Exact Field Accuracy** (Mã số thuế, Ngày tháng, Số tiền): Đạt **100%** nhờ tầng bảo vệ `is_protected_entity` trong `VietnameseCorrector`.
