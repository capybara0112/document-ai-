# Fine-tune PaddleOCR for Vietnamese invoices

The API uses PaddleOCR with `lang="vi"`. The log in the current run reports
PP-OCRv6, whose multilingual models include Vietnamese. Fine-tuning is still
useful for the invoice's handwriting, fonts, print quality, and form layout.
The current dataset does not include the user's invoice samples, so this project
does not claim a trained custom model yet.

Training the same images repeatedly is not a reliable improvement method. It
can overfit the samples without improving new invoices. Use each training cycle
to add corrected examples that represent a real error type, then compare the
new checkpoint against a fixed validation set and an untouched test set.

## 1. Decide which model needs work

- Fine-tune **recognition** when a text box is detected but its characters are
  wrong. For this filled-in invoice, start here and include handwritten
  Vietnamese word/line crops.
- Fine-tune **detection** when text is missed, neighboring fields are combined,
  or the red grid is mistaken for text. Annotate complete text-line boxes on
  full invoice images.
- Keep recognition and detection validation sets separate. Changes to one
  model should not hide regressions in the other.

## 2. Build a representative, labeled dataset

Collect scans from the invoice formats and scanners that will actually be used.
Include printed text, handwriting styles, red/blue ink, stamps, faint scans,
different rotations, and clean examples. Remove or mask personal and financial
data unless you have permission to use it for training.

For recognition, crop each text line and pair the image with the exact
transcription, preserving Vietnamese accents, punctuation, and digits. PaddleOCR
uses a UTF-8 label file with one image path and its text separated by a tab:
replace `<TAB>` below with one actual tab character.

```text
train/line_0001.png<TAB>Đơn vị bán hàng
train/line_0002.png<TAB>1.050.000
```

For detection, annotate each text line with its four image-coordinate corners
and the exact transcription. Use PaddleOCR's documented detection annotation
format. Split train/validation/test by **whole invoice and preferably writer or
source**, not by randomly splitting crops from the same page; otherwise the
reported score will look better than real-world performance.

As an initial target, the PaddleOCR fine-tuning guide recommends roughly 5,000
recognition crops and 500 detection pages for general fine-tuning. These are
guidelines, not guarantees; handwritten Vietnamese may need more varied samples.
See the [fine-tuning data recommendations](https://www.paddleocr.ai/v2.10.0/en/ppocr/model_train/finetune.html).

## 3. Iterate without overfitting

1. Run the current model on a representative batch and save the image, OCR text,
   confidence, and bounding boxes.
2. Manually correct the text and label the error type: missing line, wrong
   character/diacritic, handwriting, red grid interference, stamp, or number.
3. Add only verified examples to the training set, while retaining earlier
   samples so the model does not forget printed text and other invoice styles.
4. Fine-tune from the best prior checkpoint with a small learning rate and
   evaluate after each training cycle. Stop when held-out metrics stop improving
   or regress.
5. Keep the final test set untouched until model selection is complete. Never
   tune against the test set.

For amounts, dates, invoice numbers, and tax IDs, add deterministic validation
after OCR (format checks, arithmetic checks, and human review for low confidence).
The recognition model alone cannot guarantee these fields are correct.

## 4. Fine-tune and export

Use the training configuration and pretrained checkpoint that match the model
version selected in the installed PaddleOCR release. Train and evaluate on the
held-out invoice set, then export the best checkpoint to PaddleOCR inference
format. The official docs include the dataset schema, model configuration,
training, evaluation, and export steps:

- [PaddleOCR text recognition and training](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html)
- [PaddleOCR OCR pipeline options](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html)
- [PaddleOCR OCR dataset formats](https://www.paddleocr.ai/v3.7.0/en/datasets/ocr_datasets.html)

Do not point the API at a training checkpoint such as `best_accuracy.pdparams`.
Set the environment variable to the exported **inference model directory**:

```powershell
$env:OCR_TEXT_RECOGNITION_MODEL_DIR = "D:\models\invoice-vietnamese\rec_inference"
$env:OCR_TEXT_DETECTION_MODEL_DIR = "D:\models\invoice-vietnamese\det_inference"
$env:OCR_ENABLED = "true"
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

Only set the detection variable if a custom detector was exported. Restart the
API after changing these values. The adapter loads the custom model directories
on its next process start.

## 5. Measure whether the model improved

Keep a fixed test set that was never used in training. Track character error
rate (CER), word error rate (WER), and exact-match accuracy for critical invoice
fields such as tax IDs, invoice numbers, dates, quantities, and amounts. Review
results separately for printed text and handwriting. A lower average OCR error
does not guarantee that financial fields are safe to accept automatically.

## What model training cannot restore

OCR training improves text recognition. It does not recreate the original ink
color, paper texture, table cell merges, stamp artwork, or signatures. Keep the
original image embedded for visual fidelity: the PDF's first page, the Word
document's scan page, and the Excel `Original Scan` sheet do this. In the
editable reconstruction, table geometry is approximate and stamps/signatures
remain image content. If seal wording must be searchable, crop and rectify the
seal, then evaluate a dedicated seal-text recognition workflow separately from
the main invoice OCR.
