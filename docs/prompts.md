# Local VLM Prompts

The app includes Ollama and `llama.cpp` vision adapters, plus an optional PaddleOCR-VL document parser. The deterministic lightweight OCR/extraction path does not require a VLM.

## PaddleOCR-VL

PaddleOCR 3.5.0 exposes `PaddleOCRVL` and the `doc_parser` CLI described in the official PaddleOCR-VL docs. In this app it is intentionally exposed through `POST /api/pages/{page_id}/document/parse` instead of being run for every upload.

Useful local settings:

```bash
PADDLE_OCR_VL_BACKEND=
PADDLE_OCR_VL_MAX_PIXELS=1000000
PADDLE_OCR_VL_MAX_NEW_TOKENS=1024
PADDLE_OCR_VL_USE_LAYOUT_DETECTION=false
```

Leave `PADDLE_OCR_VL_BACKEND` empty for PaddleOCR-VL's local PaddlePaddle path. For a separate server workflow, set `PADDLE_OCR_VL_BACKEND=mlx-vlm-server` or `PADDLE_OCR_VL_BACKEND=llama-cpp-server` and point `PADDLE_OCR_VL_SERVER_URL` at the OpenAI-compatible `/v1` server URL.

## Vocabulary Row

System:

```text
You extract Japanese study vocabulary from a cropped workbook row. Return only JSON matching the schema. Do not invent missing data. Every field must be supported by visible text or OCR tokens. If uncertain, set needs_review=true and add a warning.
```

Payload:

```json
{
  "task": "extract_vocab_row",
  "ocr_tokens": [],
  "expected_fields": ["surface", "reading", "meaning_ko"]
}
```

## Multiple Choice Question

System:

```text
You extract a Japanese multiple-choice vocabulary question from a cropped workbook region. Return only JSON. Do not guess the correct answer unless it is present in the supplied answer map. If the target underline is unclear, set needs_review=true.
```

Payload:

```json
{
  "task": "extract_mcq_question",
  "ocr_tokens": [],
  "answer_map": {"1": 2},
  "expected_fields": ["sentence", "target", "choices", "correct_choice_no"]
}
```
