from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from app.vision import paddle_ocr_vl
from app.vision.paddle_ocr_vl import PaddleOcrVlDocumentParser, _blocks_from_payload, _normalized_vl_backend
from app.models.schemas import DocumentParseBlock, DocumentParseResult
from app.ocr.engines import normalize_ocr_engine, tokens_from_document_parse


def test_blocks_from_paddle_ocr_vl_payload() -> None:
    payload = {
        "parsing_res_list": [
            {
                "block_label": "text",
                "block_content": "学校 がっこう 학교",
                "block_bbox": [10, 20, 200, 60],
                "block_order": 1,
            }
        ]
    }

    blocks = _blocks_from_payload(payload)

    assert len(blocks) == 1
    assert blocks[0].label == "text"
    assert blocks[0].content == "学校 がっこう 학교"
    assert blocks[0].bbox == [10, 20, 200, 60]
    assert blocks[0].order == 1


def test_paddle_ocr_vl_blocks_convert_to_normalized_tokens() -> None:
    result = DocumentParseResult(
        page_id="page-vl",
        provider="paddleocr_vl",
        source_image_path="/tmp/page.png",
        backend="fake",
        block_count=1,
        blocks=[
            DocumentParseBlock(
                label="text",
                content="2 まいにち あたらしい かんじを いつつ おぼえます。\n1 新しい 2 新しい 3 新い 4 新い",
                bbox=[10, 20, 410, 90],
                order=1,
            )
        ],
    )

    tokens, warnings = tokens_from_document_parse(result)

    assert warnings == []
    assert [token.source for token in tokens] == ["paddleocr_vl"] * len(tokens)
    assert "まいにち" in [token.text for token in tokens]
    assert tokens[0].bbox[0] == 10


def test_paddle_ocr_vl_markdown_without_boxes_uses_synthetic_token_boxes() -> None:
    result = DocumentParseResult(
        page_id="page-vl",
        provider="paddleocr_vl",
        source_image_path="/tmp/page.png",
        backend="fake",
        block_count=0,
        markdown_text="学校 がっこう 학교",
    )

    tokens, warnings = tokens_from_document_parse(result)

    assert [token.text for token in tokens] == ["学校", "がっこう", "학교"]
    assert warnings == [
        "PaddleOCR-VL output was converted to synthetic OCR boxes; use it for text quality comparison, not precise evidence geometry."
    ]


def test_normalize_ocr_engine_aliases() -> None:
    assert normalize_ocr_engine("base") == "paddleocr"
    assert normalize_ocr_engine("vl") == "paddleocr_vl"
    with pytest.raises(ValueError, match="Unsupported OCR engine"):
        normalize_ocr_engine("mystery")


def test_native_backend_alias_uses_local_paddlepaddle_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePaddleOCRVL:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCRVL=FakePaddleOCRVL))
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_BACKEND", "native")
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_SERVER_URL", "")
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_API_MODEL_NAME", "")
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_API_KEY", "")

    PaddleOcrVlDocumentParser()

    assert captured["vl_rec_backend"] is None


def test_server_backend_passes_openai_compatible_service_settings(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePaddleOCRVL:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCRVL=FakePaddleOCRVL))
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_BACKEND", "llama-cpp-server")
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_SERVER_URL", "http://localhost:8080/v1")
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_API_MODEL_NAME", "PaddlePaddle/PaddleOCR-VL-1.5")
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_API_KEY", "secret")

    PaddleOcrVlDocumentParser()

    assert captured["vl_rec_backend"] == "llama-cpp-server"
    assert captured["vl_rec_server_url"] == "http://localhost:8080/v1"
    assert captured["vl_rec_api_model_name"] == "PaddlePaddle/PaddleOCR-VL-1.5"
    assert captured["vl_rec_api_key"] == "secret"


def test_pipeline_creation_dependency_error_is_actionable(monkeypatch) -> None:
    class BrokenPaddleOCRVL:
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("A dependency error occurred during pipeline creation.")

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCRVL=BrokenPaddleOCRVL))
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_BACKEND", "")
    monkeypatch.setattr(paddle_ocr_vl, "PADDLE_OCR_VL_SERVER_URL", "")

    with pytest.raises(RuntimeError, match="uv sync --group dev --extra ocr-vl"):
        PaddleOcrVlDocumentParser()


def test_rejects_unsupported_vl_backend_before_pipeline_creation() -> None:
    with pytest.raises(RuntimeError, match="Unsupported PADDLE_OCR_VL_BACKEND"):
        _normalized_vl_backend("native-but-not-really")
