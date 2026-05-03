from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from app.vision import paddle_ocr_vl
from app.ocr import engines
from app.vision.paddle_ocr_vl import (
    PaddleOcrVlDocumentParser,
    _blocks_from_payload,
    _pipeline_creation_help,
    _normalized_vl_backend,
)
from app.models.schemas import DocumentParseBlock, DocumentParseResult
from app.ocr.engines import normalize_ocr_engine, run_ocr_engine, tokens_from_document_parse
from app.ocr.providers import make_token


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
        source_image_path="test-fixtures/page.png",
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
        source_image_path="test-fixtures/page.png",
        backend="fake",
        block_count=0,
        markdown_text="学校 がっこう 학교",
    )

    tokens, warnings = tokens_from_document_parse(result)

    assert [token.text for token in tokens] == ["学校", "がっこう", "학교"]
    assert warnings == [
        "PaddleOCR-VL output was converted to synthetic OCR boxes; use it for text quality comparison, not precise evidence geometry."
    ]


def test_paddle_ocr_vl_processing_uses_base_word_geometry(monkeypatch, tmp_path) -> None:
    parsed = DocumentParseResult(
        page_id="page-vl",
        provider="paddleocr_vl",
        source_image_path=str(tmp_path / "page.png"),
        backend="fake",
        block_count=1,
        blocks=[
            DocumentParseBlock(
                label="text",
                content="2 \\underline{\\text{あたらしい}}",
                bbox=[20, 1100, 900, 1150],
                order=1,
            )
        ],
    )
    base_token = make_token("page-vl", "あたらしい", [120, 420, 210, 450], 0.98, "paddleocr")

    class FakeParser:
        def parse(self, image_path, page_id):
            return parsed

    monkeypatch.setattr(engines, "get_paddle_ocr_vl_parser", lambda: FakeParser())
    monkeypatch.setattr(engines, "recognize_with_warnings", lambda image_path, page_id: ([base_token], []))

    result = run_ocr_engine(tmp_path / "page.png", "page-vl", "paddleocr_vl")

    aligned_token = next(token for token in result.tokens if "underline" in token.text)
    assert aligned_token.source == "paddleocr_vl"
    assert aligned_token.bbox == base_token.bbox
    assert result.evidence_tokens == result.tokens
    assert result.document_parse == parsed
    assert any("visual evidence uses PaddleOCR word boxes" in warning for warning in result.warnings)
    assert any("tokens keep VL text" in warning for warning in result.warnings)


def test_paddle_ocr_vl_processing_falls_back_to_vl_boxes_when_base_geometry_missing(monkeypatch, tmp_path) -> None:
    parsed = DocumentParseResult(
        page_id="page-vl",
        provider="paddleocr_vl",
        source_image_path=str(tmp_path / "page.png"),
        backend="fake",
        block_count=1,
        blocks=[DocumentParseBlock(label="text", content="学校", bbox=[10, 20, 80, 44], order=1)],
    )

    class FakeParser:
        def parse(self, image_path, page_id):
            return parsed

    monkeypatch.setattr(engines, "get_paddle_ocr_vl_parser", lambda: FakeParser())
    monkeypatch.setattr(engines, "recognize_with_warnings", lambda image_path, page_id: ([], ["base unavailable"]))

    result = run_ocr_engine(tmp_path / "page.png", "page-vl", "paddleocr_vl")

    assert [token.text for token in result.tokens] == ["学校"]
    assert result.tokens[0].source == "paddleocr_vl"
    assert result.evidence_tokens == result.tokens
    assert "base unavailable" in result.warnings
    assert any("using PaddleOCR-VL block-derived boxes" in warning for warning in result.warnings)


def test_paddle_ocr_vl_processing_keeps_visual_boxes_on_real_geometry(monkeypatch, tmp_path) -> None:
    parsed = DocumentParseResult(
        page_id="page-vl",
        provider="paddleocr_vl",
        source_image_path=str(tmp_path / "page.png"),
        backend="fake",
        block_count=0,
        markdown_text="$\n2 \\underline{\\text{あたらしい}}\n1 新しい 2 新しい",
    )
    geometry_tokens = [
        make_token("page-vl", "2", [10, 100, 20, 120], 0.99, "paddleocr"),
        make_token("page-vl", "あたらしい", [30, 100, 140, 120], 0.98, "paddleocr"),
        make_token("page-vl", "新しい", [30, 140, 90, 160], 0.97, "paddleocr"),
        make_token("page-vl", "新しい", [180, 140, 240, 160], 0.96, "paddleocr"),
    ]

    class FakeParser:
        def parse(self, image_path, page_id):
            return parsed

    monkeypatch.setattr(engines, "get_paddle_ocr_vl_parser", lambda: FakeParser())
    monkeypatch.setattr(engines, "recognize_with_warnings", lambda image_path, page_id: (geometry_tokens, []))

    result = run_ocr_engine(tmp_path / "page.png", "page-vl", "paddleocr_vl")

    assert all(token.bbox in [geometry.bbox for geometry in geometry_tokens] for token in result.tokens)
    assert "$" not in [token.text for token in result.tokens]
    assert result.evidence_tokens is not None
    assert len(result.evidence_tokens) >= len(result.tokens)
    assert any(token.text == "\\underline{\\text{あたらしい}}" and token.bbox == [30, 100, 140, 120] for token in result.tokens)


def test_paddle_ocr_vl_processing_retains_unmatched_geometry_as_visual_evidence(monkeypatch, tmp_path) -> None:
    parsed = DocumentParseResult(
        page_id="page-vl",
        provider="paddleocr_vl",
        source_image_path=str(tmp_path / "page.png"),
        backend="fake",
        block_count=1,
        blocks=[DocumentParseBlock(label="text", content="学校", bbox=[10, 20, 120, 50], order=1)],
    )
    geometry_tokens = [
        make_token("page-vl", "学校", [10, 20, 70, 45], 0.98, "paddleocr"),
        make_token("page-vl", "base-only", [80, 20, 150, 45], 0.99, "paddleocr"),
    ]

    class FakeParser:
        def parse(self, image_path, page_id):
            return parsed

    monkeypatch.setattr(engines, "get_paddle_ocr_vl_parser", lambda: FakeParser())
    monkeypatch.setattr(engines, "recognize_with_warnings", lambda image_path, page_id: (geometry_tokens, []))

    result = run_ocr_engine(tmp_path / "page.png", "page-vl", "paddleocr_vl")

    assert [token.text for token in result.tokens] == ["学校"]
    assert all(token.source == "paddleocr_vl" for token in result.tokens)
    assert result.evidence_tokens is not None
    assert [token.text for token in result.evidence_tokens] == ["学校", "base-only"]
    assert result.evidence_tokens[1].source == "paddleocr"
    assert any("unmatched PaddleOCR geometry tokens are retained" in warning for warning in result.warnings)


def test_paddle_ocr_vl_processing_keeps_vl_text_without_base_geometry_match(monkeypatch, tmp_path) -> None:
    parsed = DocumentParseResult(
        page_id="page-vl",
        provider="paddleocr_vl",
        source_image_path=str(tmp_path / "page.png"),
        backend="fake",
        block_count=1,
        blocks=[DocumentParseBlock(label="text", content="学校 VLだけ", bbox=[10, 20, 160, 50], order=1)],
    )
    geometry_tokens = [make_token("page-vl", "学校", [10, 20, 70, 45], 0.98, "paddleocr")]

    class FakeParser:
        def parse(self, image_path, page_id):
            return parsed

    monkeypatch.setattr(engines, "get_paddle_ocr_vl_parser", lambda: FakeParser())
    monkeypatch.setattr(engines, "recognize_with_warnings", lambda image_path, page_id: (geometry_tokens, []))

    result = run_ocr_engine(tmp_path / "page.png", "page-vl", "paddleocr_vl")

    assert [token.text for token in result.tokens] == ["学校", "VLだけ"]
    assert result.tokens[0].bbox == geometry_tokens[0].bbox
    assert result.tokens[1].source == "paddleocr_vl"
    assert result.evidence_tokens is not None
    assert "VLだけ" in [token.text for token in result.evidence_tokens]


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

    with pytest.raises(RuntimeError, match="uv sync --python 3.12 --group dev --extra ocr --extra ocr-vl"):
        PaddleOcrVlDocumentParser()


def test_pipeline_creation_error_lists_missing_paddlex_ocr_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(paddle_ocr_vl, "_missing_paddlex_ocr_dependencies", lambda: ["einops", "tokenizers"])
    error = RuntimeError("A dependency error occurred during pipeline creation.")
    error.__cause__ = RuntimeError("PaddleX OCR extras are missing")

    message = _pipeline_creation_help("", error)

    assert "Missing PaddleX OCR dependencies: einops, tokenizers." in message
    assert "Original error: PaddleX OCR extras are missing" in message


def test_dev_backend_script_keeps_ocr_vl_dependencies_installed() -> None:
    script = Path(__file__).resolve().parents[3] / "scripts" / "dev-backend.sh"
    text = script.read_text(encoding="utf-8")

    assert "--no-build" not in text
    assert '--python "$backend_python"' in text
    assert "--extra ocr" in text
    assert "--extra ocr-vl" in text
    assert "ANKI_MAKER_REQUIRE_LOCAL_OCR" in text
    assert "BACKEND_HOST" in text
    assert "BACKEND_PORT" in text


def test_backend_python_version_is_pinned_for_paddle_wheels() -> None:
    python_version = Path(__file__).resolve().parents[2] / ".python-version"

    assert python_version.read_text(encoding="utf-8").strip() == "3.12"


def test_dev_backend_script_falls_back_to_base_ocr_when_vl_install_is_unavailable(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "dev-backend.sh"
    calls_path = tmp_path / "uv-calls.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "$FAKE_UV_CALLS"
if [[ "$1" == "sync" && "$*" == *"--extra ocr-vl"* ]]; then
  echo 'error: Distribution `paddlepaddle==3.3.1` cannot be installed because it has no binary distribution' >&2
  exit 1
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = {
        **os.environ,
        "FAKE_UV_CALLS": str(calls_path),
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
    }

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "sync --python 3.12 --group dev --extra ocr --extra ocr-vl" in calls
    assert "sync --python 3.12 --group dev --extra ocr\n" in calls
    assert "run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000" in calls
    assert "Full local OCR dependencies could not be installed" in result.stdout
    assert "Started with base PaddleOCR support only" in result.stdout


def test_dev_backend_script_can_start_without_ocr_when_base_paddle_is_unavailable(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "dev-backend.sh"
    calls_path = tmp_path / "uv-calls.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "$FAKE_UV_CALLS"
if [[ "$1" == "sync" && "$*" == *"--extra ocr"* ]]; then
  echo "error: Distribution paddlepaddle has no binary distribution" >&2
  exit 1
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = {
        **os.environ,
        "FAKE_UV_CALLS": str(calls_path),
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
    }

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_path.read_text(encoding="utf-8")
    assert "sync --python 3.12 --group dev --extra ocr --extra ocr-vl" in calls
    assert "sync --python 3.12 --group dev --extra ocr\n" in calls
    assert "sync --python 3.12 --group dev\n" in calls
    assert "Base PaddleOCR dependencies also could not be installed" in result.stdout


def test_dev_backend_script_can_require_local_ocr_install(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "dev-backend.sh"
    calls_path = tmp_path / "uv-calls.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "$FAKE_UV_CALLS"
echo "error: Distribution paddlepaddle has no binary distribution" >&2
exit 1
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = {
        **os.environ,
        "ANKI_MAKER_REQUIRE_LOCAL_OCR": "1",
        "FAKE_UV_CALLS": str(calls_path),
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
    }

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 1
    calls = calls_path.read_text(encoding="utf-8")
    assert "sync --python 3.12 --group dev --extra ocr --extra ocr-vl" in calls
    assert "sync --python 3.12 --group dev\n" not in calls


def test_rejects_unsupported_vl_backend_before_pipeline_creation() -> None:
    with pytest.raises(RuntimeError, match="Unsupported PADDLE_OCR_VL_BACKEND"):
        _normalized_vl_backend("native-but-not-really")
