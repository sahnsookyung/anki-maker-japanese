from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


BASELINE_MODEL_PROFILE = "jp_v3_mobile_current"
DEFAULT_EXTRACTION_VARIANT = "baseline_current"

LOCAL_MODEL_PROFILES = {
    BASELINE_MODEL_PROFILE,
    "jp_v5_mobile_general",
    "jp_v5_server_general",
    "jp_lang_auto",
}
OPTIONAL_MODEL_PROFILES = {"google_vision", "ko_v5_current"}
SUPPORTED_MODEL_PROFILES = LOCAL_MODEL_PROFILES | OPTIONAL_MODEL_PROFILES
SUPPORTED_EXTRACTION_VARIANTS = {
    DEFAULT_EXTRACTION_VARIANT,
    "line_graph_v1",
    "table_graph_v1",
    "ranked_rows_v1",
    "crop_confirm_v1",
    "provider_agreement_v1",
}
EXTRACTION_VARIANT_ORDER = [
    DEFAULT_EXTRACTION_VARIANT,
    "line_graph_v1",
    "table_graph_v1",
    "ranked_rows_v1",
    "crop_confirm_v1",
    "provider_agreement_v1",
]

EXTRACTION_VARIANT_DESCRIPTIONS = {
    DEFAULT_EXTRACTION_VARIANT: (
        "Frozen current extractor",
        "Control path for current production extraction.",
    ),
    "line_graph_v1": (
        "Line graph experiment",
        "Scores OCR reading-order lines and line-level relationships without changing the production default.",
    ),
    "table_graph_v1": (
        "Table graph experiment",
        "Adds table-cell and row/field hypothesis metrics for vocab/table pages.",
    ),
    "ranked_rows_v1": (
        "Ranked row experiment",
        "Reports row-hypothesis evidence quality for future ranked vocab extraction.",
    ),
    "crop_confirm_v1": (
        "Crop-confirm experiment",
        "Reserved for limited uncertain-field crop confirmation; not promoted without resource evidence.",
    ),
    "provider_agreement_v1": (
        "Provider agreement diagnostic",
        "Diagnostic review signal comparing providers; never an automatic extraction decision.",
    ),
}


@dataclass(frozen=True)
class OcrModelProfile:
    id: str
    label: str
    budget: str
    provider: str
    env: dict[str, str]
    description: str
    creates_candidates: bool = True

    def manifest(
        self,
        *,
        engine: str,
        extraction_variant: str,
        cache_hit: bool | None = None,
        cache_key: str | None = None,
        preprocessing_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        package_versions = _package_versions(["paddleocr", "paddlepaddle", "paddlex", "google-cloud-vision"])
        model_cache_paths = _model_cache_paths(self.env)
        return {
            "schema_version": 1,
            "profile_id": self.id,
            "label": self.label,
            "budget": self.budget,
            "provider": self.provider,
            "engine": engine,
            "extraction_variant": extraction_variant,
            "creates_candidates": self.creates_candidates,
            "env": dict(sorted(self.env.items())),
            "env_fingerprint": _fingerprint(self.env),
            "model_config": _model_config(self.env),
            "language_config": _language_config(self.env),
            "preprocessing_config": preprocessing_config or {},
            "model_cache_paths": model_cache_paths,
            "package_versions": package_versions,
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "device": _device_info(),
            },
            "cache": {
                "hit": cache_hit,
                "key": cache_key,
                "model_cache_hit": _all_model_paths_exist(model_cache_paths),
                "key_components": [
                    "original_image_sha256",
                    "preprocessing",
                    "profile_id",
                    "env_fingerprint",
                    "package_versions",
                    "model_cache_fingerprints",
                    "engine",
                    "extraction_variant",
                ],
            },
        }


def resolve_ocr_model_profile(profile_id: str | None = None) -> OcrModelProfile:
    normalized = normalize_model_profile(profile_id)
    if normalized == BASELINE_MODEL_PROFILE:
        return OcrModelProfile(
            id=BASELINE_MODEL_PROFILE,
            label="Japanese PP-OCRv3 mobile + Korean PP-OCRv5",
            budget="safe_local",
            provider="paddle",
            env={
                "PADDLE_OCR_USE_LANGUAGE_PROFILE": "false",
                "PADDLE_OCR_TEXT_DETECTION_MODEL_NAME": "PP-OCRv3_mobile_det",
                "PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME": "japan_PP-OCRv3_mobile_rec",
                "PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME": "PP-OCRv5_mobile_det",
                "PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME": "korean_PP-OCRv5_mobile_rec",
            },
            description="Frozen production control profile.",
        )
    if normalized == "jp_v5_mobile_general":
        return OcrModelProfile(
            id=normalized,
            label="PP-OCRv5 mobile general Japanese test",
            budget="safe_local",
            provider="paddle",
            env={
                "PADDLE_OCR_USE_LANGUAGE_PROFILE": "false",
                "PADDLE_OCR_TEXT_DETECTION_MODEL_NAME": "PP-OCRv5_mobile_det",
                "PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME": "PP-OCRv5_mobile_rec",
                "PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME": "PP-OCRv5_mobile_det",
                "PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME": "korean_PP-OCRv5_mobile_rec",
            },
            description="Newer mobile PP-OCRv5 detector/recognizer comparison.",
        )
    if normalized == "jp_v5_server_general":
        return OcrModelProfile(
            id=normalized,
            label="PP-OCRv5 server general Japanese test",
            budget="heavy_local",
            provider="paddle",
            env={
                "PADDLE_OCR_USE_LANGUAGE_PROFILE": "false",
                "PADDLE_OCR_TEXT_DETECTION_MODEL_NAME": "PP-OCRv5_server_det",
                "PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME": "PP-OCRv5_server_rec",
                "PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME": "PP-OCRv5_mobile_det",
                "PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME": "korean_PP-OCRv5_mobile_rec",
            },
            description="Heavier PP-OCRv5 server comparison; never promoted without resource evidence.",
        )
    if normalized == "jp_lang_auto":
        return OcrModelProfile(
            id=normalized,
            label='PaddleOCR lang="japan" profile',
            budget="heavy_local",
            provider="paddle",
            env={
                "PADDLE_OCR_USE_LANGUAGE_PROFILE": "true",
                "PADDLE_OCR_LANG": "japan",
                "PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME": "PP-OCRv5_mobile_det",
                "PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME": "korean_PP-OCRv5_mobile_rec",
            },
            description="Paddle language-profile comparison; may resolve to server-class models locally.",
        )
    if normalized == "ko_v5_current":
        return OcrModelProfile(
            id=normalized,
            label="Korean PP-OCRv5 diagnostic profile",
            budget="safe_local",
            provider="paddle_korean",
            env={
                "PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME": "PP-OCRv5_mobile_det",
                "PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME": "korean_PP-OCRv5_mobile_rec",
            },
            description="Korean gloss OCR diagnostic; not a standalone candidate-generation profile.",
            creates_candidates=False,
        )
    if normalized == "google_vision":
        return OcrModelProfile(
            id=normalized,
            label="Google Vision OCR diagnostic profile",
            budget="cloud_optional",
            provider="google_vision",
            env={},
            description="Optional cloud OCR comparison; independent from default extraction.",
            creates_candidates=False,
        )
    raise ValueError(f"Unsupported OCR model profile {profile_id!r}.")


def normalize_model_profile(profile_id: str | None = None) -> str:
    normalized = (profile_id or BASELINE_MODEL_PROFILE).strip().lower().replace("-", "_")
    aliases = {
        "default": BASELINE_MODEL_PROFILE,
        "baseline": BASELINE_MODEL_PROFILE,
        "current": BASELINE_MODEL_PROFILE,
        "safe_local": BASELINE_MODEL_PROFILE,
        "jp_v5_mobile": "jp_v5_mobile_general",
        "ppocrv5_mobile": "jp_v5_mobile_general",
        "jp_v5_server": "jp_v5_server_general",
        "ppocrv5_server": "jp_v5_server_general",
        "lang_japan": "jp_lang_auto",
        "japan_auto": "jp_lang_auto",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_MODEL_PROFILES:
        raise ValueError(f"Unsupported OCR model profile {profile_id!r}. Use one of: {', '.join(sorted(SUPPORTED_MODEL_PROFILES))}.")
    return normalized


def normalize_extraction_variant(variant: str | None = None) -> str:
    normalized = (variant or DEFAULT_EXTRACTION_VARIANT).strip().lower().replace("-", "_")
    aliases = {"baseline": DEFAULT_EXTRACTION_VARIANT, "current": DEFAULT_EXTRACTION_VARIANT, "default": DEFAULT_EXTRACTION_VARIANT}
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_EXTRACTION_VARIANTS:
        raise ValueError(
            f"Unsupported extraction variant {variant!r}. Use one of: {', '.join(sorted(SUPPORTED_EXTRACTION_VARIANTS))}."
        )
    return normalized


def profile_env_overrides(profile_id: str | None) -> dict[str, str]:
    return resolve_ocr_model_profile(profile_id).env


def available_profile_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": profile.id,
            "label": profile.label,
            "budget": profile.budget,
            "provider": profile.provider,
            "creates_candidates": profile.creates_candidates,
            "description": profile.description,
        }
        for profile in (resolve_ocr_model_profile(profile_id) for profile_id in sorted(SUPPORTED_MODEL_PROFILES))
    ]


def available_variant_payload() -> list[dict[str, str]]:
    return [
        {
            "id": variant_id,
            "label": EXTRACTION_VARIANT_DESCRIPTIONS[variant_id][0],
            "description": EXTRACTION_VARIANT_DESCRIPTIONS[variant_id][1],
        }
        for variant_id in EXTRACTION_VARIANT_ORDER
    ]


def _package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _model_config(env: dict[str, str]) -> dict[str, str | None]:
    return {
        "japanese_detection_model": env.get("PADDLE_OCR_TEXT_DETECTION_MODEL_NAME"),
        "japanese_recognition_model": env.get("PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME"),
        "korean_detection_model": env.get("PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME"),
        "korean_recognition_model": env.get("PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME"),
    }


def _language_config(env: dict[str, str]) -> dict[str, str | bool | None]:
    use_language_profile = env.get("PADDLE_OCR_USE_LANGUAGE_PROFILE")
    return {
        "use_language_profile": str(use_language_profile).lower() in {"1", "true", "yes", "on"},
        "lang": env.get("PADDLE_OCR_LANG"),
    }


def _device_info() -> dict[str, str | None]:
    return {
        "paddle_device": os.getenv("PADDLE_DEVICE") or None,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES") or None,
        "paddle_place": os.getenv("FLAGS_selected_gpus") or None,
    }


def _fingerprint(value: dict[str, str]) -> str:
    payload = "\n".join(f"{key}={value[key]}" for key in sorted(value))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _model_cache_paths(env: dict[str, str]) -> dict[str, dict[str, str | bool | None]]:
    names = {
        key: value
        for key, value in env.items()
        if key.endswith("_MODEL_NAME") and value
    }
    roots = [Path.home() / ".paddlex" / "official_models", Path.home() / ".paddleocr" / "whl"]
    paths: dict[str, dict[str, str | bool | None]] = {}
    for key, model_name in names.items():
        candidates = [root / model_name for root in roots]
        existing = next((candidate for candidate in candidates if candidate.exists()), None)
        fingerprint = _directory_fingerprint(existing) if existing else None
        paths[key] = {
            "model_name": model_name,
            "path": str(existing) if existing else None,
            "exists": existing is not None,
            "fingerprint": fingerprint,
            "hash": fingerprint,
            "hash_strategy": "directory-file-stat-list" if existing else None,
        }
    return paths


def _all_model_paths_exist(model_cache_paths: dict[str, dict[str, str | bool | None]]) -> bool:
    if not model_cache_paths:
        return False
    return all(bool(value.get("exists")) for value in model_cache_paths.values())


def _directory_fingerprint(path: Path) -> str | None:
    try:
        stats = sorted(
            (item.name, item.stat().st_size, int(item.stat().st_mtime))
            for item in path.iterdir()
            if item.is_file()
        )
    except OSError:
        return None
    payload = json.dumps(stats, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
