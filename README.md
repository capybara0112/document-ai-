# Document AI Engine (Python/FastAPI)

MVP service for the Java Spring Boot application:
- Accepts PDF, DOCX, XLSX, PNG, JPG/JPEG.
- Detects digital-text PDFs vs scanned/image PDFs.
- Extracts digital PDF text with PyMuPDF.
- Extracts DOCX paragraphs/tables with python-docx.
- Extracts XLSX sheets/cells with openpyxl.
- Provides an optional PaddleOCR 3.x adapter for images/scanned PDFs.
- Converts PNG/JPG images into a searchable source page, editable OCR
  reconstruction, and a document-analysis report.
- Provides a format selector in `/docs` to export an image as PDF, DOCX, or XLSX.
- Returns normalized JSON suitable for Java to persist in MySQL.

## 1. Requirements
- Python 3.11 recommended.
- Windows PowerShell commands below.
- For the first integration test, CPU mode and digital PDFs work without OCR.

## 2. Create environment and install
```powershell
cd document-ai-engine
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```
Open `http://127.0.0.1:8001/` for the Web UI or `http://127.0.0.1:8001/docs` for Swagger API.

## 4. Test
```powershell
curl.exe -X POST "http://127.0.0.1:8001/api/v1/analyze" `
  -F "file=@C:\path\to\sample.pdf" `
  -F "task_type=AUTO"
```

Supported task_type: `AUTO`, `INVOICE`, `FINANCIAL_REPORT`, `GENERAL`.

## 5. OCR
OCR is optional so the base service can run without downloading OCR models. For a
CPU setup on Windows, install PaddlePaddle and PaddleOCR in the active virtual
environment, then enable OCR before starting the API:

```powershell
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install "paddleocr>=3.0,<4.0"
$env:OCR_ENABLED = "true"
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

For GPU or another Python/runtime combination, use the matching installation
instructions from the [PaddleOCR documentation](https://www.paddleocr.ai/main/en/quick_start.html).
The first OCR run may download model files. After updating this project, run
`pip install -r requirements.txt` to install OpenCV image preprocessing. The
service returns a clear error when OCR is disabled or PaddleOCR is not installed. See
`app/extractors/ocr_adapter.py`.

### Convert an image to an editable PDF

```powershell
curl.exe -X POST "http://127.0.0.1:8001/api/v1/analyze/pdf" `
  -F "file=@.\test.png" `
  -F "task_type=INVOICE" `
  --output .\test_ocr.pdf
```

The response contains three or more pages:

1. The uploaded image, preserved as-is, with an invisible OCR layer for search
   and copy.
2. A reconstruction with visible editable OCR text and detected red form rules.
3. A report containing image diagnostics and OCR sections A–E. Additional pages
   are added if the report text needs more space.

Text and vector lines on page 2 can be edited in software such as Acrobat Pro or
LibreOffice Draw. Page 3 records raw OCR, normalized text, uncertain lines, and
image quality; it does not silently replace uncertain words.

This is a best-effort reconstruction, not a trained handwriting recognizer.
PaddleOCR may misread handwriting, stamps, or low-quality text; verify names,
tax IDs, dates, quantities, and amounts against the first page before using the
editable page as a record. The editable page recreates detected rules and OCR
text positions; complex visual elements such as stamps, signatures, logos, and
paper texture remain on the original scan page.

### Choose Word or Excel

Open `http://127.0.0.1:8001/docs`, choose `POST /api/v1/convert/image`, upload
the PNG/JPG, then select `PDF`, `DOCX`, or `XLSX` in the `output_format` list.
The API returns a downloadable file. The PDF route above remains available for
clients that only need PDF.

- **DOCX:** embeds the source scan, an editable OCR table, and a report with
  sections A–E.
- **XLSX:** includes `OCR Lines`, an approximate `Layout` grid, an `Analysis`
  sheet with sections A–E, and `Original Scan`.

Word and Excel retain editable OCR text, but they do not recreate every stamp,
signature, logo, or table merge from the scan. Review the recognized values and
adjust the table in Word/Excel where needed.

### Image analysis and OCR report

`POST /api/v1/analyze` returns `extracted_data.analysis_report`:

- **A:** image dimensions, estimated skew, page-boundary detection, brightness,
  contrast, blur/noise indicators, and a coarse quality label.
- **B:** OCR text as recognized, in reading order, before spelling normalization.
- **C:** Unicode/spacing normalization and rule-based blocks for headings,
  questions, list items, and paragraphs. Low-confidence lines are marked
  `[KHÔNG RÕ]`.
- **D:** low-confidence lines and possible technical-term matches such as
  `Canny` or `Laplacian`. These are review suggestions and are never silently
  substituted for the OCR result.
- **E:** image recapture and manual-review suggestions.

Preprocessing estimates document corners, corrects perspective and small skew
when the page boundary is reliable, enlarges small images, and applies only one
quality-selected denoising method. It can generate a bounded set of complete OCR
candidates: source, geometry-corrected, denoised, enhanced-color,
adaptive-binary, or sharpened. The service chooses one complete result rather
than merging text from different candidates. Text boxes are mapped back to the
uploaded image coordinates for export.

`OCR_PREPROCESS_MODE` accepts `auto`, `enhanced`, `threshold`, and `original`.
`auto` is the default. `OCR_MAX_CANDIDATES` is `4` by default; reduce it when
CPU latency or memory is more important than trying more image variants.
`OCR_TARGET_SHORT_EDGE`, `OCR_MAX_DIMENSION`, and
`OCR_LOW_CONFIDENCE_THRESHOLD` are documented in `.env.example`.

This is a transparent OCR pipeline, not a general language model: it does not
infer missing text or automatically correct Vietnamese spelling. It can flag
possible unreadable lines using OCR confidence, but cannot prove that the text
detector found every character or recognize handwriting/signatures reliably.
For higher accuracy, fine-tune PaddleOCR with corrected Vietnamese line crops
and validate on documents that were not used in training; see
[`OCR_TRAINING_VI.md`](OCR_TRAINING_VI.md).

### OCR candidate selection and correction layers

The response keeps three separate layers:

- `B_raw_ocr`: the selected PaddleOCR candidate in original model order.
- `C_normalized_ocr`: NFC/whitespace normalization plus layout-based reading
  order. It does not substitute OCR words.
- `D_review_required` and `smart_corrections`: low confidence, unusual language
  signals, and proposed Vietnamese/technical-term corrections. Every proposal
  includes the source token and reason, and is marked for human review.

`ocr_strategy.candidate_scores` records the mean model confidence, agreement
between OCR candidates, language-consistency signals, coverage, and the derived
selection score. The selection score is a routing heuristic, never a claim that
the text is correct.

### Test and evaluate

Run the fast, model-free pipeline checks:

```powershell
.\.venv311\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

For CER and WER, copy `evaluation/manifest.example.jsonl`, place the eight
listed fixture types beside it, and replace every `verified_transcript` with a
human-approved transcription. Then run:

```powershell
$env:OCR_ENABLED = "true"
.\.venv311\Scripts\python.exe -X utf8 scripts\evaluate_ocr.py .\evaluation\manifest.jsonl --output .\ocr_evaluation.json
```

The evaluator rejects records without verified text, so it cannot emit invented
CER/WER values. It reports a per-document and mean result only after real OCR
has run.

## Improve Vietnamese OCR with fine-tuning

The current adapter already requests `lang="vi"`, and the model shown in your
server log is PP-OCRv6. Vietnamese is supported by that model family; errors on
your invoice are more likely to come from handwriting, pale/red printed
backgrounds, low contrast, and dense table lines than from an unset language.
The setup and data requirements for specialization are in
[`OCR_TRAINING_VI.md`](OCR_TRAINING_VI.md).

After exporting a fine-tuned model to PaddleOCR inference format, set one or
both model directories before starting the API:

```powershell
$env:OCR_TEXT_RECOGNITION_MODEL_DIR = "D:\models\invoice-vietnamese\rec_inference"
# Set this only if you also fine-tuned the text detector:
$env:OCR_TEXT_DETECTION_MODEL_DIR = "D:\models\invoice-vietnamese\det_inference"
$env:OCR_ENABLED = "true"
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

The recognition model directory is the main improvement for misread words.
Fine-tune the detector only when text boxes are missing or merged.

## 6. Architecture
`api/v1/endpoints.py` receives upload -> validates extension/signature/size ->
`core/router.py` detects type -> `ocr_adapter.py` creates and scores bounded OCR
candidates -> `image_preprocessor.py` supplies geometry/quality analysis ->
`ocr_postprocessing.py` keeps raw, normalized, and review layers -> exporters
produce PDF, Word, or Excel.

## Important limits
- This is an integration-ready MVP, not a production OCR model trained for Vietnamese VAT invoices or financial statements.
- Digital text extraction is not the same as semantic understanding. Field extraction for invoices should be evaluated against labeled samples and improved with rules/model output.
- Do not expose this service publicly without authentication/network controls. The Java service should be the only caller.
- Uploaded files are processed in memory and not permanently stored by this service. Java remains responsible for storage and task lifecycle.
