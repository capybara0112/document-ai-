# OCR evaluation dataset

`manifest.example.jsonl` names the required evaluation categories:

1. skewed document
2. perspective-distorted document
3. low-light document
4. noisy document
5. small text
6. Vietnamese diacritics
7. invoice
8. exam or numbered-list document

Place the matching images under `evaluation/fixtures/` and copy the manifest to
`manifest.jsonl`. Every `verified_transcript` must be typed or reviewed by a
human. Keep the final test split separate from images used to tune preprocessing
or train a custom PaddleOCR model.

Run the evaluator from the project root with `OCR_ENABLED=true`:

```powershell
.\.venv311\Scripts\python.exe -X utf8 scripts\evaluate_ocr.py .\evaluation\manifest.jsonl --output .\ocr_evaluation.json
```

The JSON output contains CER, WER, and the candidate selected for each image.
It does not calculate text-detection precision/recall because that requires
human-drawn text-region annotations in addition to the transcript.
