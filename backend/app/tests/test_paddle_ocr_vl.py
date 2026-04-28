from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from app.vision import paddle_ocr_vl
from app.vision.paddle_ocr_vl import PaddleOcrVlDocumentParser, _blocks_from_payload, _normalized_vl_backend


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
