from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.core.config import OCR_PROVIDER, OCR_PROVIDER_CACHE_ENABLED
from app.models.schemas import OcrToken
from app.ocr.providers import OcrProvider


def get_ocr_provider(provider_name: str | None = None) -> OcrProvider:
    provider_name = provider_name or OCR_PROVIDER
    if OCR_PROVIDER_CACHE_ENABLED:
        return _build_ocr_provider_cached(provider_name)
    return _build_ocr_provider_uncached(provider_name)


@lru_cache(maxsize=8)
def _build_ocr_provider_cached(provider_name: str) -> OcrProvider:
    return _build_ocr_provider_uncached(provider_name)


def _build_ocr_provider_uncached(provider_name: str) -> OcrProvider:
    provider_name = provider_name or OCR_PROVIDER
    errors: list[str] = []
    if provider_name in ("auto", "paddle"):
        try:
            from app.ocr.paddle_provider import PaddleOcrProvider

            return PaddleOcrProvider()
        except Exception as exc:
            errors.append(f"PaddleOCR unavailable: {exc}")
            if provider_name == "paddle":
                raise RuntimeError("; ".join(errors))

    if provider_name == "paddle_korean":
        try:
            from app.ocr.paddle_provider import PaddleKoreanOcrProvider

            return PaddleKoreanOcrProvider()
        except Exception as exc:
            errors.append(f"PaddleOCR Korean unavailable: {exc}")
            raise RuntimeError("; ".join(errors))

    if provider_name in ("auto", "tesseract"):
        try:
            from app.ocr.tesseract_provider import TesseractOcrProvider

            return TesseractOcrProvider()
        except Exception as exc:
            errors.append(f"Tesseract unavailable: {exc}")
            if provider_name == "tesseract":
                raise RuntimeError("; ".join(errors))

    if provider_name == "google_vision":
        try:
            from app.ocr.google_vision_provider import GoogleVisionOcrProvider

            return GoogleVisionOcrProvider()
        except Exception as exc:
            errors.append(f"Google Cloud Vision unavailable: {exc}")
            if provider_name == "google_vision":
                raise RuntimeError("; ".join(errors))

    raise RuntimeError("; ".join(errors) or "No OCR provider configured")


def recognize_with_warnings(image_path: Path, page_id: str) -> tuple[list[OcrToken], list[str]]:
    return recognize_with_provider(image_path, page_id, OCR_PROVIDER)


def recognize_with_provider(image_path: Path, page_id: str, provider_name: str) -> tuple[list[OcrToken], list[str]]:
    try:
        provider = get_ocr_provider(provider_name)
        tokens = provider.recognize(image_path, page_id)
        warnings: list[str] = []
        if not tokens:
            warnings.append(f"{provider.name} returned no OCR tokens; review page manually or install Japanese OCR support.")
        return tokens, warnings
    except Exception as exc:
        return [], [f"OCR failed: {exc}"]
