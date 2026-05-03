from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

from app.core.config import (
    GOOGLE_VISION_ALLOW_CLOUD,
    GOOGLE_VISION_API_ENDPOINT,
    GOOGLE_VISION_CACHE_ENABLED,
    GOOGLE_VISION_MONTHLY_CAP,
    OCR_CACHE_DIR,
    USAGE_DIR,
)
from app.models.schemas import OcrToken
from app.ocr.providers import make_token


class GoogleVisionOcrProvider:
    name = "google_vision"

    def __init__(self) -> None:
        self._vision = None
        self._client = None

    def _client_and_vision(self):
        try:
            from google.cloud import vision
        except Exception as exc:
            raise RuntimeError("google-cloud-vision is not installed. Run `uv sync --extra cloud` in backend/.") from exc
        if self._client is None:
            self._vision = vision
            self._client = vision.ImageAnnotatorClient(**_client_options())
        return self._vision, self._client

    def recognize(self, image_path: Path, page_id: str) -> list[OcrToken]:
        image_sha = _sha256_file(image_path)
        cached = _read_cached_tokens(image_sha, page_id) if GOOGLE_VISION_CACHE_ENABLED else None
        if cached is not None:
            return cached
        if not GOOGLE_VISION_ALLOW_CLOUD:
            raise RuntimeError(
                "Google Vision cloud calls are disabled. Set GOOGLE_VISION_ALLOW_CLOUD=true to use the free-tier "
                "comparison path, and configure GOOGLE_APPLICATION_CREDENTIALS for your service account."
            )
        _assert_monthly_quota_available(image_sha)
        vision, client = self._client_and_vision()
        image = vision.Image(content=image_path.read_bytes())
        try:
            response = client.document_text_detection(image=image)
        except Exception as exc:
            raise RuntimeError(_friendly_google_vision_error(exc)) from exc
        if response.error.message:
            raise RuntimeError(_friendly_google_vision_error(response.error.message))

        tokens: list[OcrToken] = []
        annotation = response.full_text_annotation
        for page in annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        text = "".join(symbol.text for symbol in word.symbols).strip()
                        if not text:
                            continue
                        vertices = word.bounding_box.vertices
                        xs = [float(vertex.x or 0) for vertex in vertices]
                        ys = [float(vertex.y or 0) for vertex in vertices]
                        confidence = float(getattr(word, "confidence", 0.0) or 0.0)
                        tokens.append(make_token(page_id, text, [min(xs), min(ys), max(xs), max(ys)], confidence, self.name))
        if GOOGLE_VISION_CACHE_ENABLED:
            _write_cached_tokens(image_sha, tokens)
        _record_usage(image_sha)
        return tokens


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(image_sha: str) -> Path:
    return OCR_CACHE_DIR / "google_vision" / f"{image_sha}.json"


def _read_cached_tokens(image_sha: str, page_id: str) -> list[OcrToken] | None:
    path = _cache_path(image_sha)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return [
        make_token(
            page_id=page_id,
            text=str(item.get("text") or ""),
            bbox=[float(value) for value in item.get("bbox", [0, 0, 1, 1])],
            confidence=float(item.get("confidence") or 0.0),
            source="google_vision",
        )
        for item in payload.get("tokens", [])
        if str(item.get("text") or "").strip()
    ]


def _write_cached_tokens(image_sha: str, tokens: list[OcrToken]) -> None:
    path = _cache_path(image_sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_sha256": image_sha,
        "provider": "google_vision",
        "tokens": [
            {
                "text": token.text,
                "bbox": token.bbox,
                "confidence": token.confidence,
            }
            for token in tokens
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _friendly_google_vision_error(error: object) -> str:
    message = str(error).strip()
    first_line = message.splitlines()[0] if message else "unknown error"
    if "SERVICE_DISABLED" in message or "Cloud Vision API has not been used" in message:
        return (
            "Google Vision API is disabled for the configured project. Enable the Cloud Vision API in Google Cloud "
            "Console, wait a few minutes for propagation, then retry."
        )
    if "GOOGLE_APPLICATION_CREDENTIALS" in message or "DefaultCredentialsError" in message:
        return (
            "Google Vision credentials are not configured. Set GOOGLE_APPLICATION_CREDENTIALS to a service-account "
            "JSON file or configure Application Default Credentials."
        )
    if "403" in first_line or "Permission" in first_line:
        return f"Google Vision request was rejected by Google Cloud permissions: {first_line}"
    return f"Google Vision OCR request failed: {first_line}"


def _client_options() -> dict:
    if not GOOGLE_VISION_API_ENDPOINT:
        return {}
    return {"client_options": {"api_endpoint": GOOGLE_VISION_API_ENDPOINT}}


def _ledger_path() -> Path:
    return USAGE_DIR / "google_vision_usage.json"


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _read_ledger() -> dict:
    path = _ledger_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _assert_monthly_quota_available(image_sha: str) -> None:
    ledger = _read_ledger()
    month = _current_month()
    month_payload = ledger.get(month, {})
    used = int(month_payload.get("units", 0))
    if used + 1 > GOOGLE_VISION_MONTHLY_CAP:
        raise RuntimeError(
            f"Google Vision monthly cap would be exceeded ({used + 1}/{GOOGLE_VISION_MONTHLY_CAP}). "
            "Use cached results, raise GOOGLE_VISION_MONTHLY_CAP, or wait for the next month."
        )


def _record_usage(image_sha: str) -> None:
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = _read_ledger()
    month = _current_month()
    month_payload = ledger.setdefault(month, {"units": 0, "image_sha256": [], "requests": []})
    month_payload["units"] = int(month_payload.get("units", 0)) + 1
    if image_sha not in set(month_payload.get("image_sha256", [])):
        month_payload.setdefault("image_sha256", []).append(image_sha)
    month_payload.setdefault("requests", []).append({"image_sha256": image_sha, "at": datetime.now(timezone.utc).isoformat()})
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
