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
DEFAULT_KOREAN_PROFILE = "ko_v5_current"
KOREAN_MODEL_PAIR_PROFILE = "ko_v5_det_v5_rec"
KOREAN_LANG_AUTO_PROFILE = "ko_lang_auto"
DEFAULT_EXTRACTION_VARIANT = "baseline_current"

MODEL_PAIR_PROFILES = [
    "jp_v3_det_v3_rec",
    "jp_v3_det_v5_rec",
    "jp_v5_det_v3_rec",
    "jp_v5_det_v5_rec",
]
MODEL_PROFILE_CACHE_ALIASES = {
    "jp_v3_det_v3_rec": BASELINE_MODEL_PROFILE,
    "jp_v5_det_v5_rec": "jp_v5_mobile_general",
}
KOREAN_PROFILE_CACHE_ALIASES = {
    KOREAN_MODEL_PAIR_PROFILE: DEFAULT_KOREAN_PROFILE,
}
LOCAL_MODEL_PROFILES = {
    BASELINE_MODEL_PROFILE,
    *MODEL_PAIR_PROFILES,
    "jp_v5_mobile_general",
    "jp_v5_server_general",
    "jp_lang_auto",
}
SUPPORTED_KOREAN_PROFILES = {DEFAULT_KOREAN_PROFILE, KOREAN_MODEL_PAIR_PROFILE, KOREAN_LANG_AUTO_PROFILE}
OPTIONAL_MODEL_PROFILES = {"google_vision", DEFAULT_KOREAN_PROFILE}
SUPPORTED_MODEL_PROFILES = LOCAL_MODEL_PROFILES | OPTIONAL_MODEL_PROFILES
SUPPORTED_EXTRACTION_VARIANTS = {
    DEFAULT_EXTRACTION_VARIANT,
    "line_graph_v1",
    "table_graph_v1",
    "ranked_rows_v1",
    "crop_confirm_v1",
    "provider_agreement_v1",
    "v5_token_split_v1",
    "v5_vocab_rows_v1",
    "ko_alignment_v1",
    "v5_mcq_v1",
    "v5_token_split_plus_vocab_rows_v1",
    "v5_vocab_rows_plus_ko_alignment_v1",
    "v5_token_split_plus_mcq_v1",
    "v5_full_adapted_v1",
    "ko_crop_confirm_v1",
    "ko_region_columns_v1",
    "ko_consensus_v1",
    "mcq_source_rebuild_v1",
    "mcq_choice_band_ocr_v1",
    "accuracy_recovery_v1",
    "residual_diagnostics_v1",
    "jp_region_columns_v1",
    "ko_residual_glyph_v1",
    "mcq_prompt_line_ocr_v1",
    "mcq_choice_glyph_v1",
    "accuracy_recovery_v2",
}
EXTRACTION_VARIANT_ORDER = [
    DEFAULT_EXTRACTION_VARIANT,
    "line_graph_v1",
    "table_graph_v1",
    "ranked_rows_v1",
    "crop_confirm_v1",
    "provider_agreement_v1",
    "v5_token_split_v1",
    "v5_vocab_rows_v1",
    "ko_alignment_v1",
    "v5_mcq_v1",
    "v5_token_split_plus_vocab_rows_v1",
    "v5_vocab_rows_plus_ko_alignment_v1",
    "v5_token_split_plus_mcq_v1",
    "v5_full_adapted_v1",
    "ko_crop_confirm_v1",
    "ko_region_columns_v1",
    "ko_consensus_v1",
    "mcq_source_rebuild_v1",
    "mcq_choice_band_ocr_v1",
    "accuracy_recovery_v1",
    "residual_diagnostics_v1",
    "jp_region_columns_v1",
    "ko_residual_glyph_v1",
    "mcq_prompt_line_ocr_v1",
    "mcq_choice_glyph_v1",
    "accuracy_recovery_v2",
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
    "v5_token_split_v1": (
        "v5 token split",
        "Splits merged PP-OCRv5 vocab tokens into surface/reading field evidence with derived bboxes and raw-token provenance.",
    ),
    "v5_vocab_rows_v1": (
        "v5 vocab rows",
        "Reports v5-aware row-scoring diagnostics; candidate replacement stays gated off until benchmark evidence clears the guard.",
    ),
    "ko_alignment_v1": (
        "Korean alignment",
        "Reports Korean gloss pairing, raw recall, unpaired Hangul, stale evidence, and bbox alignment without changing candidate rows.",
    ),
    "v5_mcq_v1": (
        "v5 MCQ recovery",
        "Recovers PP-OCRv5 MCQ answer-strip and choice parsing failures without changing the safe default.",
    ),
    "v5_token_split_plus_vocab_rows_v1": (
        "v5 split + rows",
        "Combines merged-token splitting with guarded v5-aware row diagnostics.",
    ),
    "v5_vocab_rows_plus_ko_alignment_v1": (
        "v5 rows + Korean",
        "Combines guarded v5-aware row diagnostics with Korean gloss-alignment diagnostics.",
    ),
    "v5_token_split_plus_mcq_v1": (
        "v5 split + MCQ",
        "Combines merged-token splitting for vocab pages with v5 MCQ recovery.",
    ),
    "v5_full_adapted_v1": (
        "v5 full adapted",
        "Combines safe candidate changes with guarded row/Korean diagnostics and MCQ recovery.",
    ),
    "ko_crop_confirm_v1": (
        "Korean crop recovery",
        "Reruns OCR on uncertain Korean meaning fields and accepts only live crop OCR evidence that passes recovery guards.",
    ),
    "ko_region_columns_v1": (
        "Korean column recovery",
        "Reruns OCR on Korean meaning column bands and aligns recovered Hangul tokens back to vocab rows.",
    ),
    "ko_consensus_v1": (
        "Korean consensus recovery",
        "Combines full-page, crop, and region OCR for guarded Korean meaning recovery.",
    ),
    "mcq_source_rebuild_v1": (
        "MCQ source rebuild",
        "Separates raw OCR-backed MCQ source fields from semantic answer fields and rebuilds strict source evidence.",
    ),
    "mcq_choice_band_ocr_v1": (
        "MCQ choice-band OCR",
        "Runs bounded OCR over MCQ choice and answer-strip bands for strict source-field recovery.",
    ),
    "accuracy_recovery_v1": (
        "Accuracy recovery",
        "Combines Korean crop/region/consensus recovery with MCQ source-field repair.",
    ),
    "residual_diagnostics_v1": (
        "Residual diagnostics (diagnostic-only)",
        "Benchmark-only residual miss diagnostics; writes traces/artifacts without mutating candidates.",
    ),
    "jp_region_columns_v1": (
        "Japanese region recovery",
        "Reruns OCR on Japanese vocab surface/reading row regions and may add complete OCR-backed missing rows.",
    ),
    "ko_residual_glyph_v1": (
        "Korean residual glyph recovery",
        "Uses guarded local crop/glyph evidence to repair weak residual Korean meaning OCR.",
    ),
    "mcq_prompt_line_ocr_v1": (
        "MCQ prompt-line OCR",
        "Reruns OCR on clipped MCQ prompt-line bands to repair strict source sentences.",
    ),
    "mcq_choice_glyph_v1": (
        "MCQ choice glyph recovery",
        "Uses isolated choice crops to repair strict spelling-MCQ source choices.",
    ),
    "accuracy_recovery_v2": (
        "Accuracy recovery v2",
        "Combines accuracy recovery v1 with Japanese region, Korean residual glyph, and MCQ prompt/choice recovery.",
    ),
}

EXTRACTION_VARIANT_COMPONENTS = {
    DEFAULT_EXTRACTION_VARIANT: frozenset({DEFAULT_EXTRACTION_VARIANT}),
    "line_graph_v1": frozenset({"line_graph_v1"}),
    "table_graph_v1": frozenset({"table_graph_v1"}),
    "ranked_rows_v1": frozenset({"ranked_rows_v1"}),
    "crop_confirm_v1": frozenset({"crop_confirm_v1"}),
    "provider_agreement_v1": frozenset({"provider_agreement_v1"}),
    "v5_token_split_v1": frozenset({"v5_token_split_v1"}),
    "v5_vocab_rows_v1": frozenset({"v5_vocab_rows_v1"}),
    "ko_alignment_v1": frozenset({"ko_alignment_v1"}),
    "v5_mcq_v1": frozenset({"v5_mcq_v1"}),
    "v5_token_split_plus_vocab_rows_v1": frozenset({"v5_token_split_v1", "v5_vocab_rows_v1"}),
    "v5_vocab_rows_plus_ko_alignment_v1": frozenset({"v5_vocab_rows_v1", "ko_alignment_v1"}),
    "v5_token_split_plus_mcq_v1": frozenset({"v5_token_split_v1", "v5_mcq_v1"}),
    "v5_full_adapted_v1": frozenset({"v5_token_split_v1", "v5_vocab_rows_v1", "ko_alignment_v1", "v5_mcq_v1"}),
    "ko_crop_confirm_v1": frozenset({"v5_token_split_v1", "ko_crop_confirm_v1"}),
    "ko_region_columns_v1": frozenset({"v5_token_split_v1", "ko_region_columns_v1"}),
    "ko_consensus_v1": frozenset({"v5_token_split_v1", "ko_crop_confirm_v1", "ko_region_columns_v1", "ko_consensus_v1"}),
    "mcq_source_rebuild_v1": frozenset({"v5_mcq_v1", "mcq_source_rebuild_v1"}),
    "mcq_choice_band_ocr_v1": frozenset({"v5_mcq_v1", "mcq_source_rebuild_v1", "mcq_choice_band_ocr_v1"}),
    "accuracy_recovery_v1": frozenset(
        {
            "v5_token_split_v1",
            "v5_vocab_rows_v1",
            "ko_alignment_v1",
            "v5_mcq_v1",
            "ko_crop_confirm_v1",
            "ko_region_columns_v1",
            "ko_consensus_v1",
            "mcq_source_rebuild_v1",
            "mcq_choice_band_ocr_v1",
        }
    ),
    "residual_diagnostics_v1": frozenset({"residual_diagnostics_v1"}),
    "jp_region_columns_v1": frozenset({"v5_token_split_v1", "v5_vocab_rows_v1", "jp_region_columns_v1"}),
    "ko_residual_glyph_v1": frozenset(
        {"v5_token_split_v1", "ko_crop_confirm_v1", "ko_region_columns_v1", "ko_consensus_v1", "ko_residual_glyph_v1"}
    ),
    "mcq_prompt_line_ocr_v1": frozenset({"v5_mcq_v1", "mcq_source_rebuild_v1", "mcq_prompt_line_ocr_v1"}),
    "mcq_choice_glyph_v1": frozenset(
        {"v5_mcq_v1", "mcq_source_rebuild_v1", "mcq_choice_band_ocr_v1", "mcq_choice_glyph_v1"}
    ),
    "accuracy_recovery_v2": frozenset(
        {
            "v5_token_split_v1",
            "v5_vocab_rows_v1",
            "ko_alignment_v1",
            "v5_mcq_v1",
            "ko_crop_confirm_v1",
            "ko_region_columns_v1",
            "ko_consensus_v1",
            "mcq_source_rebuild_v1",
            "mcq_choice_band_ocr_v1",
            "jp_region_columns_v1",
            "ko_residual_glyph_v1",
            "mcq_prompt_line_ocr_v1",
            "mcq_choice_glyph_v1",
        }
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
        korean_profile: "OcrModelProfile | None" = None,
        cache_hit: bool | None = None,
        cache_key: str | None = None,
        preprocessing_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        package_versions = _package_versions(["paddleocr", "paddlepaddle", "paddlex", "google-cloud-vision"])
        model_cache_paths = _model_cache_paths(self.env)
        korean_profile_payload = _korean_profile_payload(korean_profile)
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
            "korean_profile": korean_profile_payload,
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
                    "korean_profile_id",
                    "korean_env_fingerprint",
                    "engine",
                ],
            },
        }


def resolve_ocr_model_profile(profile_id: str | None = None) -> OcrModelProfile:
    normalized = normalize_model_profile(profile_id)
    if normalized in {BASELINE_MODEL_PROFILE, "jp_v3_det_v3_rec"}:
        return OcrModelProfile(
            id=normalized,
            label=(
                "PP-OCRv3 mobile det + Japanese PP-OCRv3 mobile rec"
                if normalized == "jp_v3_det_v3_rec"
                else "Japanese PP-OCRv3 mobile + Korean PP-OCRv5"
            ),
            budget="safe_local",
            provider="paddle",
            env={
                "PADDLE_OCR_USE_LANGUAGE_PROFILE": "false",
                "PADDLE_OCR_TEXT_DETECTION_MODEL_NAME": "PP-OCRv3_mobile_det",
                "PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME": "japan_PP-OCRv3_mobile_rec",
                "PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME": "PP-OCRv5_mobile_det",
                "PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME": "korean_PP-OCRv5_mobile_rec",
            },
            description="Frozen production control profile." if normalized == BASELINE_MODEL_PROFILE else "Explicit model-pair alias for the frozen production control.",
        )
    if normalized == "jp_v3_det_v5_rec":
        return OcrModelProfile(
            id=normalized,
            label="PP-OCRv3 mobile det + PP-OCRv5 mobile rec",
            budget="safe_local",
            provider="paddle",
            env={
                "PADDLE_OCR_USE_LANGUAGE_PROFILE": "false",
                "PADDLE_OCR_TEXT_DETECTION_MODEL_NAME": "PP-OCRv3_mobile_det",
                "PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME": "PP-OCRv5_mobile_rec",
                "PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME": "PP-OCRv5_mobile_det",
                "PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME": "korean_PP-OCRv5_mobile_rec",
            },
            description="Hybrid control: keep the stable v3 detector and test the v5 recognizer.",
        )
    if normalized == "jp_v5_det_v3_rec":
        return OcrModelProfile(
            id=normalized,
            label="PP-OCRv5 mobile det + Japanese PP-OCRv3 mobile rec",
            budget="safe_local",
            provider="paddle",
            env={
                "PADDLE_OCR_USE_LANGUAGE_PROFILE": "false",
                "PADDLE_OCR_TEXT_DETECTION_MODEL_NAME": "PP-OCRv5_mobile_det",
                "PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME": "japan_PP-OCRv3_mobile_rec",
                "PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME": "PP-OCRv5_mobile_det",
                "PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME": "korean_PP-OCRv5_mobile_rec",
            },
            description="Hybrid control: test the v5 detector while keeping the current Japanese v3 recognizer.",
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
    if normalized == "jp_v5_det_v5_rec":
        profile = resolve_ocr_model_profile("jp_v5_mobile_general")
        return OcrModelProfile(
            id=normalized,
            label="PP-OCRv5 mobile det + PP-OCRv5 mobile rec",
            budget=profile.budget,
            provider=profile.provider,
            env=profile.env,
            description="Explicit latest local model-pair alias for jp_v5_mobile_general.",
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


def resolve_korean_ocr_profile(profile_id: str | None = None) -> OcrModelProfile:
    normalized = normalize_korean_profile(profile_id)
    if normalized in {DEFAULT_KOREAN_PROFILE, KOREAN_MODEL_PAIR_PROFILE}:
        return OcrModelProfile(
            id=normalized,
            label=(
                "Korean PP-OCRv5 mobile det + Korean PP-OCRv5 mobile rec"
                if normalized == KOREAN_MODEL_PAIR_PROFILE
                else "Korean PP-OCRv5 mobile det/rec"
            ),
            budget="safe_local",
            provider="paddle_korean",
            env={
                "PADDLE_OCR_KOREAN_USE_LANGUAGE_PROFILE": "false",
                "PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME": "PP-OCRv5_mobile_det",
                "PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME": "korean_PP-OCRv5_mobile_rec",
            },
            description=(
                "Explicit model-pair alias for the default Korean OCR pass."
                if normalized == KOREAN_MODEL_PAIR_PROFILE
                else "Default Korean OCR pass used by the two-pass vocab pipeline."
            ),
            creates_candidates=False,
        )
    if normalized == KOREAN_LANG_AUTO_PROFILE:
        return OcrModelProfile(
            id=normalized,
            label='PaddleOCR lang="korean" profile',
            budget="heavy_local",
            provider="paddle_korean",
            env={
                "PADDLE_OCR_KOREAN_USE_LANGUAGE_PROFILE": "true",
                "PADDLE_OCR_KOREAN_LANG": "korean",
                "PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME": "",
                "PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME": "",
            },
            description="Optional Korean language-profile comparison for gloss OCR; may resolve to heavier local models.",
            creates_candidates=False,
        )
    raise ValueError(f"Unsupported Korean OCR profile {profile_id!r}.")


def normalize_model_profile(profile_id: str | None = None) -> str:
    normalized = (profile_id or BASELINE_MODEL_PROFILE).strip().lower().replace("-", "_")
    aliases = {
        "default": BASELINE_MODEL_PROFILE,
        "baseline": BASELINE_MODEL_PROFILE,
        "current": BASELINE_MODEL_PROFILE,
        "safe_local": BASELINE_MODEL_PROFILE,
        "jp_v3_v3": "jp_v3_det_v3_rec",
        "jp_v3_det_jp_v3_rec": "jp_v3_det_v3_rec",
        "jp_v3_v5": "jp_v3_det_v5_rec",
        "jp_v3_det_jp_v5_rec": "jp_v3_det_v5_rec",
        "jp_v5_v3": "jp_v5_det_v3_rec",
        "jp_v5_det_jp_v3_rec": "jp_v5_det_v3_rec",
        "jp_v5_v5": "jp_v5_det_v5_rec",
        "jp_v5_det_jp_v5_rec": "jp_v5_det_v5_rec",
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


def normalize_korean_profile(profile_id: str | None = None) -> str:
    normalized = (profile_id or DEFAULT_KOREAN_PROFILE).strip().lower().replace("-", "_")
    aliases = {
        "default": DEFAULT_KOREAN_PROFILE,
        "baseline": DEFAULT_KOREAN_PROFILE,
        "current": DEFAULT_KOREAN_PROFILE,
        "ko_v5": DEFAULT_KOREAN_PROFILE,
        "korean_v5": DEFAULT_KOREAN_PROFILE,
        "ko_v5_v5": KOREAN_MODEL_PAIR_PROFILE,
        "korean_v5_v5": KOREAN_MODEL_PAIR_PROFILE,
        "ko_v5_mobile": KOREAN_MODEL_PAIR_PROFILE,
        "lang_korean": KOREAN_LANG_AUTO_PROFILE,
        "korean_auto": KOREAN_LANG_AUTO_PROFILE,
        "ko_auto": KOREAN_LANG_AUTO_PROFILE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_KOREAN_PROFILES:
        raise ValueError(f"Unsupported Korean OCR profile {profile_id!r}. Use one of: {', '.join(sorted(SUPPORTED_KOREAN_PROFILES))}.")
    return normalized


def normalize_extraction_variant(variant: str | None = None) -> str:
    normalized = (variant or DEFAULT_EXTRACTION_VARIANT).strip().lower().replace("-", "_")
    normalized = normalized.replace(" ", "")
    aliases = {
        "baseline": DEFAULT_EXTRACTION_VARIANT,
        "current": DEFAULT_EXTRACTION_VARIANT,
        "default": DEFAULT_EXTRACTION_VARIANT,
        "v5_token_split_v1+v5_vocab_rows_v1": "v5_token_split_plus_vocab_rows_v1",
        "v5_token_split_v1_plus_v5_vocab_rows_v1": "v5_token_split_plus_vocab_rows_v1",
        "v5_vocab_rows_v1+ko_alignment_v1": "v5_vocab_rows_plus_ko_alignment_v1",
        "v5_vocab_rows_v1_plus_ko_alignment_v1": "v5_vocab_rows_plus_ko_alignment_v1",
        "v5_token_split_v1+v5_mcq_v1": "v5_token_split_plus_mcq_v1",
        "v5_token_split_v1_plus_v5_mcq_v1": "v5_token_split_plus_mcq_v1",
        "v5_full": "v5_full_adapted_v1",
        "accuracy_recovery": "accuracy_recovery_v1",
        "ko_crop": "ko_crop_confirm_v1",
        "ko_region": "ko_region_columns_v1",
        "ko_consensus": "ko_consensus_v1",
        "mcq_source": "mcq_source_rebuild_v1",
        "mcq_choice_band": "mcq_choice_band_ocr_v1",
        "residual_diagnostics": "residual_diagnostics_v1",
        "jp_region": "jp_region_columns_v1",
        "japanese_region": "jp_region_columns_v1",
        "ko_residual": "ko_residual_glyph_v1",
        "korean_residual": "ko_residual_glyph_v1",
        "mcq_prompt": "mcq_prompt_line_ocr_v1",
        "mcq_prompt_line": "mcq_prompt_line_ocr_v1",
        "mcq_choice_glyph": "mcq_choice_glyph_v1",
        "accuracy_recovery_v2": "accuracy_recovery_v2",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_EXTRACTION_VARIANTS:
        raise ValueError(
            f"Unsupported extraction variant {variant!r}. Use one of: {', '.join(sorted(SUPPORTED_EXTRACTION_VARIANTS))}."
        )
    return normalized


def extraction_variant_components(variant: str | None = None) -> frozenset[str]:
    normalized = normalize_extraction_variant(variant)
    return EXTRACTION_VARIANT_COMPONENTS[normalized]


def cache_model_profile_id(profile_id: str | None = None) -> str:
    normalized = normalize_model_profile(profile_id)
    return MODEL_PROFILE_CACHE_ALIASES.get(normalized, normalized)


def cache_korean_profile_id(profile_id: str | None = None) -> str:
    normalized = normalize_korean_profile(profile_id)
    return KOREAN_PROFILE_CACHE_ALIASES.get(normalized, normalized)


def profile_env_overrides(profile_id: str | None, korean_profile_id: str | None = None) -> dict[str, str]:
    return combined_profile_env_overrides(profile_id, korean_profile_id)


def combined_profile_env_overrides(profile_id: str | None, korean_profile_id: str | None = None) -> dict[str, str]:
    env = dict(resolve_ocr_model_profile(profile_id).env)
    env.update(resolve_korean_ocr_profile(korean_profile_id).env)
    return env


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


def available_korean_profile_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": profile.id,
            "label": profile.label,
            "budget": profile.budget,
            "provider": profile.provider,
            "creates_candidates": profile.creates_candidates,
            "description": profile.description,
        }
        for profile in (resolve_korean_ocr_profile(profile_id) for profile_id in sorted(SUPPORTED_KOREAN_PROFILES))
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
    use_korean_language_profile = env.get("PADDLE_OCR_KOREAN_USE_LANGUAGE_PROFILE")
    return {
        "use_language_profile": str(use_language_profile).lower() in {"1", "true", "yes", "on"},
        "lang": env.get("PADDLE_OCR_LANG"),
        "korean_use_language_profile": str(use_korean_language_profile).lower() in {"1", "true", "yes", "on"},
        "korean_lang": env.get("PADDLE_OCR_KOREAN_LANG"),
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


def _korean_profile_payload(profile: OcrModelProfile | None) -> dict[str, Any]:
    profile = profile or resolve_korean_ocr_profile(DEFAULT_KOREAN_PROFILE)
    return {
        "profile_id": profile.id,
        "label": profile.label,
        "budget": profile.budget,
        "provider": profile.provider,
        "env": dict(sorted(profile.env.items())),
        "env_fingerprint": _fingerprint(profile.env),
        "model_config": _model_config(profile.env),
        "language_config": _language_config(profile.env),
        "model_cache_paths": _model_cache_paths(profile.env),
    }


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
