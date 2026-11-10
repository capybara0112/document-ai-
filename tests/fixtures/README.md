# Test Fixtures & Evaluation Data

Thư mục này chứa các dữ liệu thử nghiệm thực tế:
1. `ground_truth_4.txt`: Nội dung ground truth đã kiểm chứng cho đề thi Thị giác máy tính (`4.png`).
2. `fixture_exam_cv.png`: Ảnh tổng hợp mô phỏng cấu trúc đề thi `4.png` với đầy đủ dấu tiếng Việt chuẩn.
3. `fixture_colored_paper.png`: Ảnh tài liệu nền giấy màu vàng/kem (kiểm thử Colour-aware CLAHE).
4. `fixture_skewed.png`: Ảnh tài liệu bị nghiêng 5.5 độ (kiểm thử Deskew).
5. `fixture_noisy.png`: Ảnh tài liệu có nhiễu Gaussian (kiểm thử Adaptive Denoise).
6. `manifest.jsonl`: Danh sách ánh xạ ảnh và ground truth phục vụ tính CER, WER, Reading Order Accuracy.

## Vị trí đặt ảnh 4.png gốc từ người dùng:
Để đánh giá trực tiếp trên ảnh chụp thực tế `4.png`, bạn có thể đặt file `4.png` tại một trong hai đường dẫn:
- `D:\code\document-ai-engine\4.png`
- `D:\code\document-ai-engine\tests\fixtures\4.png`

Khi phát hiện file, pipeline đánh giá sẽ tự động chạy OCR trên ảnh này và so sánh với `ground_truth_4.txt`.
