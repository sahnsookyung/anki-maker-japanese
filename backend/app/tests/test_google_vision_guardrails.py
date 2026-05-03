from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.ocr import google_vision_provider
from app.ocr.google_vision_provider import GoogleVisionOcrProvider
from app.ocr.providers import make_token


def test_google_vision_returns_cached_tokens_without_cloud_call(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image")
    monkeypatch.setattr(google_vision_provider, "OCR_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(google_vision_provider, "GOOGLE_VISION_CACHE_ENABLED", True)
    image_sha = google_vision_provider._sha256_file(image_path)
    google_vision_provider._write_cached_tokens(image_sha, [make_token("old-page", "学校", [1, 2, 3, 4], 0.9, "google_vision")])
    provider = GoogleVisionOcrProvider()
    def fail_client_initialization():
        raise AssertionError("cached results must not initialize the Google client")

    monkeypatch.setattr(provider, "_client_and_vision", fail_client_initialization)

    tokens = provider.recognize(image_path, "new-page")

    assert [(token.page_id, token.text, token.bbox) for token in tokens] == [("new-page", "学校", [1.0, 2.0, 3.0, 4.0])]


def test_google_vision_requires_explicit_cloud_opt_in_when_not_cached(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image")
    monkeypatch.setattr(google_vision_provider, "OCR_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(google_vision_provider, "GOOGLE_VISION_CACHE_ENABLED", True)
    monkeypatch.setattr(google_vision_provider, "GOOGLE_VISION_ALLOW_CLOUD", False)
    provider = GoogleVisionOcrProvider()

    with pytest.raises(RuntimeError, match="GOOGLE_VISION_ALLOW_CLOUD=true"):
        provider.recognize(image_path, "page")


def test_google_vision_monthly_ledger_blocks_over_cap(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(google_vision_provider, "USAGE_DIR", tmp_path / "usage")
    monkeypatch.setattr(google_vision_provider, "GOOGLE_VISION_MONTHLY_CAP", 1)
    month = google_vision_provider._current_month()
    ledger_path = google_vision_provider._ledger_path()
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(json.dumps({month: {"units": 1, "image_sha256": ["seen"], "requests": []}}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="monthly cap"):
        google_vision_provider._assert_monthly_quota_available()


def test_google_vision_cloud_call_records_usage_and_cache(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image")
    monkeypatch.setattr(google_vision_provider, "OCR_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(google_vision_provider, "USAGE_DIR", tmp_path / "usage")
    monkeypatch.setattr(google_vision_provider, "GOOGLE_VISION_CACHE_ENABLED", True)
    monkeypatch.setattr(google_vision_provider, "GOOGLE_VISION_ALLOW_CLOUD", True)
    monkeypatch.setattr(google_vision_provider, "GOOGLE_VISION_MONTHLY_CAP", 1000)
    calls: list[bytes] = []

    class FakeClient:
        def document_text_detection(self, image):
            calls.append(image.content)
            word = SimpleNamespace(
                symbols=[SimpleNamespace(text="学"), SimpleNamespace(text="校")],
                bounding_box=SimpleNamespace(
                    vertices=[
                        SimpleNamespace(x=1, y=2),
                        SimpleNamespace(x=11, y=2),
                        SimpleNamespace(x=11, y=12),
                        SimpleNamespace(x=1, y=12),
                    ]
                ),
                confidence=0.87,
            )
            return SimpleNamespace(
                error=SimpleNamespace(message=""),
                full_text_annotation=SimpleNamespace(
                    pages=[
                        SimpleNamespace(
                            blocks=[
                                SimpleNamespace(
                                    paragraphs=[
                                        SimpleNamespace(words=[word]),
                                    ]
                                )
                            ]
                        )
                    ]
                ),
            )

    provider = GoogleVisionOcrProvider()
    monkeypatch.setattr(provider, "_client_and_vision", lambda: (_fake_vision_module(), FakeClient()))

    tokens = provider.recognize(image_path, "page")
    cached_tokens = provider.recognize(image_path, "page-2")

    assert [(token.text, token.bbox, token.confidence) for token in tokens] == [("学校", [1.0, 2.0, 11.0, 12.0], 0.87)]
    assert [(token.page_id, token.text) for token in cached_tokens] == [("page-2", "学校")]
    assert calls == [b"image"]
    ledger = json.loads(google_vision_provider._ledger_path().read_text(encoding="utf-8"))
    assert ledger[google_vision_provider._current_month()]["units"] == 1


def test_google_vision_counts_each_uncached_successful_cloud_call(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image")
    monkeypatch.setattr(google_vision_provider, "OCR_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(google_vision_provider, "USAGE_DIR", tmp_path / "usage")
    monkeypatch.setattr(google_vision_provider, "GOOGLE_VISION_CACHE_ENABLED", False)
    monkeypatch.setattr(google_vision_provider, "GOOGLE_VISION_ALLOW_CLOUD", True)
    monkeypatch.setattr(google_vision_provider, "GOOGLE_VISION_MONTHLY_CAP", 1000)

    class FakeClient:
        def document_text_detection(self, image):
            return SimpleNamespace(error=SimpleNamespace(message=""), full_text_annotation=SimpleNamespace(pages=[]))

    provider = GoogleVisionOcrProvider()
    monkeypatch.setattr(provider, "_client_and_vision", lambda: (_fake_vision_module(), FakeClient()))

    provider.recognize(image_path, "page")
    provider.recognize(image_path, "page")

    ledger = json.loads(google_vision_provider._ledger_path().read_text(encoding="utf-8"))
    assert ledger[google_vision_provider._current_month()]["units"] == 2
    assert ledger[google_vision_provider._current_month()]["image_sha256"] == [google_vision_provider._sha256_file(image_path)]


def test_google_vision_disabled_api_error_is_actionable(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image")
    monkeypatch.setattr(google_vision_provider, "OCR_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(google_vision_provider, "USAGE_DIR", tmp_path / "usage")
    monkeypatch.setattr(google_vision_provider, "GOOGLE_VISION_CACHE_ENABLED", False)
    monkeypatch.setattr(google_vision_provider, "GOOGLE_VISION_ALLOW_CLOUD", True)

    class FakeClient:
        def document_text_detection(self, image):
            raise RuntimeError(
                "403 Cloud Vision API has not been used in project 123 before or it is disabled. "
                'reason: "SERVICE_DISABLED"'
            )

    provider = GoogleVisionOcrProvider()
    monkeypatch.setattr(provider, "_client_and_vision", lambda: (_fake_vision_module(), FakeClient()))

    with pytest.raises(RuntimeError, match="Enable the Cloud Vision API"):
        provider.recognize(image_path, "page")


def test_google_vision_client_options_include_optional_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(google_vision_provider, "GOOGLE_VISION_API_ENDPOINT", "us-vision.googleapis.com")

    assert google_vision_provider._client_options() == {"client_options": {"api_endpoint": "us-vision.googleapis.com"}}

    monkeypatch.setattr(google_vision_provider, "GOOGLE_VISION_API_ENDPOINT", "")
    assert google_vision_provider._client_options() == {}


def _fake_vision_module() -> SimpleNamespace:
    def image(content):
        return SimpleNamespace(content=content)

    return SimpleNamespace(Image=image)
