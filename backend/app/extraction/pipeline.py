from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from app.core.config import (
    DICTIONARY_PATH,
    OCR_COMPARE_PROVIDER,
    OCR_CROP_CONFIRM_MAX_FIELDS,
    OCR_RECOVERY_MAX_FIELDS,
    OCR_RECOVERY_MAX_REGIONS,
    PREPROCESS_MAX_SIDE_LEN,
    PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME,
    PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME,
    PADDLE_OCR_TEXT_DETECTION_MODEL_NAME,
    PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME,
    PROCESSED_DIR,
    VLM_CLEANUP_ENABLED,
    VOCAB_DUAL_OCR_ENABLED,
)
from app.core.ids import new_id
from app.core.images import preprocess_image
from app.core.script import script_summary
from app.db import database
from app.extraction.answer_strip import parse_answer_strip, parse_answer_strip_text, parse_answer_strip_v5
from app.extraction.cards import mcq_cards, vocab_cards
from app.extraction.classifier import classify_page
from app.extraction.document_graph import graph_from_document_parse, graph_from_tokens, graph_with_card_hypotheses
from app.extraction.mcq import extract_mcq_items
from app.extraction.vocab import extract_jp_ko_meaning_items, extract_vocab_items, extract_vocab_items_dual_ocr, vocab_alignment_diagnostics
from app.extraction.vl_document import extract_from_document_parse
from app.extraction.vlm_cleanup import cleanup_mcq_items, cleanup_vocab_items
from app.models.schemas import CardCandidate, DocumentParseResult, OcrRun, OcrToken, Page, ProcessResult
from app.ocr.comparison import compare_ocr_tokens
from app.ocr.crop_worker import CropOcrError, crop_ocr_worker
from app.ocr.engines import OcrEngineResult, PADDLEOCR_ENGINE, PADDLEOCR_VL_ENGINE, run_ocr_engine
from app.ocr.profiles import (
    DEFAULT_KOREAN_PROFILE,
    DEFAULT_EXTRACTION_VARIANT,
    cache_korean_profile_id,
    cache_model_profile_id,
    extraction_variant_components,
    resolve_korean_ocr_profile,
    resolve_ocr_model_profile,
    normalize_extraction_variant,
)
from app.ocr.service import recognize_with_provider
from app.validation.dictionary import DictionaryValidator


@dataclass(frozen=True)
class CachedOcrPayload:
    engine_result: OcrEngineResult
    korean_tokens: list[OcrToken]
    source_run_id: str


def process_page(
    page: Page,
    engine: str = PADDLEOCR_ENGINE,
    *,
    model_profile: str | None = None,
    korean_profile: str | None = None,
    extraction_variant: str = DEFAULT_EXTRACTION_VARIANT,
) -> ProcessResult:
    database.upsert_page(page)
    normalized_variant = normalize_extraction_variant(extraction_variant)
    variant_components = extraction_variant_components(normalized_variant)
    profile = resolve_ocr_model_profile(model_profile)
    korean_ocr_profile = resolve_korean_ocr_profile(korean_profile)
    original_path = Path(page.original_image_path)
    processed_path = PROCESSED_DIR / f"{page.id}.png"
    preprocess = preprocess_image(original_path, processed_path)
    image_sha = _sha256_file(original_path)
    transform = _transform_metadata(
        preprocess.width,
        preprocess.height,
        preprocess.warnings,
        original_path=original_path,
        processed_path=processed_path,
        original_width=getattr(preprocess, "original_width", None),
        original_height=getattr(preprocess, "original_height", None),
    )
    preprocessing_config = _preprocessing_config(preprocess.width, preprocess.height, preprocess.warnings, transform)
    preprocessing_hash = _preprocessing_hash(preprocessing_config)
    profile_manifest = profile.manifest(
        engine=engine,
        extraction_variant=normalized_variant,
        korean_profile=korean_ocr_profile,
        preprocessing_config=preprocessing_config,
    )
    cache_key = _ocr_cache_key(
        image_sha=image_sha,
        preprocessing_hash=preprocessing_hash,
        profile_manifest=profile_manifest,
        engine=engine,
        extraction_variant=normalized_variant,
    )
    cached_run = database.find_succeeded_run_by_cache_key(None, engine, image_sha, cache_key)
    cached_payload = _cached_ocr_payload(cached_run, page.id, processed_path, engine) if cached_run else None
    profile_manifest = profile.manifest(
        engine=engine,
        extraction_variant=normalized_variant,
        korean_profile=korean_ocr_profile,
        cache_hit=bool(cached_payload),
        cache_key=cache_key,
        preprocessing_config=preprocessing_config,
    )
    run = database.start_ocr_run(
        page.id,
        engine,
        image_sha256=image_sha,
        processed_image_path=str(processed_path),
        preprocessing={
            "processed_width": preprocess.width,
            "processed_height": preprocess.height,
            "warnings": preprocess.warnings,
            "hash": preprocessing_hash,
            "transform": transform,
            "cache_key": cache_key,
        },
        provider_config=_provider_config(engine, profile_manifest, normalized_variant),
    )
    try:
        engine_result = cached_payload.engine_result if cached_payload else run_ocr_engine(processed_path, page.id, engine)
        if engine_result.engine == PADDLEOCR_VL_ENGINE:
            if not engine_result.document_parse:
                raise RuntimeError("PaddleOCR-VL returned no document parse result.")
            return _process_document_parse_result(
                page=page,
                run_id=run.id,
                processed_path=processed_path,
                preprocess_width=preprocess.width,
                preprocess_height=preprocess.height,
                preprocess_warnings=preprocess.warnings,
                ocr_warnings=engine_result.warnings,
                document_parse=engine_result.document_parse,
                model_profile=model_profile,
                korean_profile=korean_ocr_profile.id,
                extraction_variant=normalized_variant,
                transform=transform,
                profile_manifest=profile_manifest,
                cache_source_run_id=cached_payload.source_run_id if cached_payload else None,
            )
        tokens = engine_result.tokens
        evidence_tokens = engine_result.evidence_tokens or tokens
        document_graph = graph_from_tokens(page.id, evidence_tokens, source=engine_result.engine, transform=transform)
        ocr_warnings = list(engine_result.warnings)
        page_type, page_confidence, _features = classify_page(tokens, preprocess.height)
        if "v5_mcq_v1" in variant_components:
            page_type, page_confidence = _recover_v5_mcq_page_type(tokens, page_type, page_confidence)
        answer_map = parse_answer_strip(tokens, preprocess.height)
        if "v5_mcq_v1" in variant_components:
            answer_map = parse_answer_strip_v5(tokens, preprocess.height, answer_map)

        validator = DictionaryValidator(DICTIONARY_PATH)
        cards: list[CardCandidate] = []
        all_tokens = list(evidence_tokens)
        recovery_diagnostics: dict[str, object] = {}
        if page_type in {"vocab_table", "jp_ko_meaning_vocab"}:
            meaning_only_vocab = page_type == "jp_ko_meaning_vocab"
            if VOCAB_DUAL_OCR_ENABLED and not any(token.source == PADDLEOCR_VL_ENGINE for token in tokens):
                if cached_payload and cached_payload.korean_tokens:
                    korean_tokens = cached_payload.korean_tokens
                    korean_ocr_warnings = []
                else:
                    korean_tokens, korean_ocr_warnings = recognize_with_provider(processed_path, page.id, "paddle_korean")
                all_tokens.extend(korean_tokens)
                document_graph = graph_from_tokens(page.id, all_tokens, source=engine_result.engine, transform=transform)
                ocr_warnings.extend(korean_ocr_warnings)
                if meaning_only_vocab:
                    items = extract_jp_ko_meaning_items(tokens, korean_tokens, validator)
                else:
                    items = extract_vocab_items_dual_ocr(tokens, korean_tokens, validator, extraction_variant=normalized_variant)
                if not items:
                    items = extract_jp_ko_meaning_items(tokens, [], validator) if meaning_only_vocab else extract_vocab_items(tokens, validator)
            else:
                items = extract_jp_ko_meaning_items(tokens, [], validator) if meaning_only_vocab else extract_vocab_items(tokens, validator)
            vlm_warnings: list[str] = []
            if VLM_CLEANUP_ENABLED and items:
                items, vlm_warnings = cleanup_vocab_items(processed_path, items, tokens, validator)
            if _uses_korean_recovery(variant_components) and items:
                items, recovered_tokens, recovery_diagnostics = _recover_korean_vocab_items(
                    items,
                    all_tokens,
                    processed_path,
                    page.id,
                    preprocess.width,
                    preprocess.height,
                    preprocessing_hash,
                    profile.id,
                    korean_ocr_profile.id,
                    variant_components,
                )
                if recovered_tokens:
                    all_tokens.extend(recovered_tokens)
                    document_graph = graph_from_tokens(page.id, all_tokens, source=engine_result.engine, transform=transform)
            if _uses_v2_vocab_recovery(variant_components) and items and not meaning_only_vocab:
                items, recovered_tokens, v2_vocab_diagnostics = _recover_v2_vocab_items(
                    items,
                    all_tokens,
                    processed_path,
                    page.id,
                    preprocess.width,
                    preprocess.height,
                    preprocessing_hash,
                    profile.id,
                    korean_ocr_profile.id,
                    variant_components,
                    validator,
                )
                recovery_diagnostics = _append_recovery_diagnostics(
                    recovery_diagnostics,
                    "v2_vocab_recovery",
                    v2_vocab_diagnostics,
                )
                if recovered_tokens:
                    all_tokens.extend(recovered_tokens)
                    document_graph = graph_from_tokens(page.id, all_tokens, source=engine_result.engine, transform=transform)
            for item in items:
                cards.extend(vocab_cards(page.id, item))
        elif page_type in {"reading_mcq", "spelling_mcq"}:
            items = extract_mcq_items(tokens, answer_map, page_type, extraction_variant=normalized_variant)
            vlm_warnings = []
            if VLM_CLEANUP_ENABLED and items:
                items, vlm_warnings = cleanup_mcq_items(processed_path, items, tokens, answer_map)
            if _uses_mcq_recovery(variant_components) and items:
                items, recovered_tokens, recovery_diagnostics = _recover_mcq_source_items(
                    items,
                    processed_path,
                    page.id,
                    preprocess.width,
                    preprocess.height,
                    preprocessing_hash,
                    profile.id,
                    korean_ocr_profile.id,
                    variant_components,
                )
                if recovered_tokens:
                    all_tokens.extend(recovered_tokens)
                    document_graph = graph_from_tokens(page.id, all_tokens, source=engine_result.engine, transform=transform)
            if _uses_v2_mcq_recovery(variant_components) and items:
                items, recovered_tokens, v2_mcq_diagnostics = _recover_v2_mcq_source_items(
                    items,
                    processed_path,
                    page.id,
                    preprocess.width,
                    preprocess.height,
                    preprocessing_hash,
                    profile.id,
                    korean_ocr_profile.id,
                    variant_components,
                )
                recovery_diagnostics = _append_recovery_diagnostics(
                    recovery_diagnostics,
                    "v2_mcq_source_recovery",
                    v2_mcq_diagnostics,
                )
                if recovered_tokens:
                    all_tokens.extend(recovered_tokens)
                    document_graph = graph_from_tokens(page.id, all_tokens, source=engine_result.engine, transform=transform)
            for item in items:
                cards.extend(mcq_cards(page.id, item))
        else:
            vlm_warnings = []
            page_type = "unknown_review_required"

        warnings = [*preprocess.warnings, *ocr_warnings, *vlm_warnings]
        if engine_result.engine == PADDLEOCR_VL_ENGINE:
            warnings.append("Processed with PaddleOCR-VL; verify output against visual evidence before approval.")
        if not cards:
            warnings.append("No card candidates were generated; inspect OCR overlay or enable optional VLM cleanup.")
        document_graph = graph_with_card_hypotheses(document_graph, cards)
        review_metrics = _review_quality_metrics(cards, document_graph.model_dump()["metrics"])
        variant_diagnostics = _extraction_variant_diagnostics(
            normalized_variant,
            cards,
            processed_path,
            page.id,
            preprocess.width,
            preprocess.height,
            all_tokens,
        )
        if recovery_diagnostics:
            variant_diagnostics["recovery"] = recovery_diagnostics
        variant_metrics = _extraction_variant_metrics(
            normalized_variant,
            cards,
            document_graph.model_dump(),
            variant_diagnostics=variant_diagnostics,
        )

        processed_page = Page(
            id=page.id,
            original_image_path=page.original_image_path,
            upload_name=page.upload_name,
            display_name=page.display_name,
            processed_image_path=str(processed_path),
            active_ocr_run_id=page.active_ocr_run_id,
            page_type=page_type,
            page_type_confidence=page_confidence,
            image_width=preprocess.width,
            image_height=preprocess.height,
            warnings=warnings,
            created_at=page.created_at,
        )
        database.upsert_page(processed_page)
        database.replace_tokens(page.id, all_tokens, run.id)
        database.replace_cards(page.id, cards, run.id)
        summary = script_summary([token.text for token in all_tokens])
        completed_run = database.complete_ocr_run(
            run.id,
            warnings=warnings,
            metrics={
                "token_count": len(all_tokens),
                "card_count": len(cards),
                "page_type": page_type,
                "page_type_confidence": page_confidence,
                "script_summary": summary,
                "answer_map_size": len(answer_map),
                "vlm_cleanup_enabled": VLM_CLEANUP_ENABLED,
                "vocab_dual_ocr_enabled": VOCAB_DUAL_OCR_ENABLED,
                "model_profile": profile_manifest,
                "korean_profile": korean_ocr_profile.id,
                "extraction_variant": normalized_variant,
                "cache_source_run_id": cached_payload.source_run_id if cached_payload else None,
                "document_graph": document_graph.model_dump(),
                "extraction_variant_metrics": variant_metrics,
                **review_metrics,
            },
            processed_image_path=str(processed_path),
            image_width=preprocess.width,
            image_height=preprocess.height,
        )
        return ProcessResult(
            page=processed_page.model_copy(
                update={
                    "active_ocr_run_id": run.id,
                    "active_ocr_engine": completed_run.engine if completed_run else engine,
                    "active_ocr_completed_at": completed_run.completed_at if completed_run else None,
                    "active_ocr_duration_ms": completed_run.duration_ms if completed_run else None,
                }
            ),
            tokens=all_tokens,
            cards=[card.model_copy(update={"run_id": run.id}) for card in cards],
            script_summary=summary,
            answer_map=answer_map,
            ocr_run=completed_run,
        )
    except Exception as exc:
        database.fail_ocr_run(run.id, str(exc), warnings=preprocess.warnings)
        raise


def _process_document_parse_result(
    *,
    page: Page,
    run_id: str,
    processed_path: Path,
    preprocess_width: int,
    preprocess_height: int,
    preprocess_warnings: list[str],
    ocr_warnings: list[str],
    document_parse: DocumentParseResult,
    model_profile: str | None = None,
    korean_profile: str | None = None,
    extraction_variant: str = DEFAULT_EXTRACTION_VARIANT,
    transform: dict[str, object] | None = None,
    profile_manifest: dict[str, object] | None = None,
    cache_source_run_id: str | None = None,
) -> ProcessResult:
    normalized_variant = normalize_extraction_variant(extraction_variant)
    profile = resolve_ocr_model_profile(model_profile)
    korean_ocr_profile = resolve_korean_ocr_profile(korean_profile)
    if profile_manifest is None:
        profile_manifest = profile.manifest(
            engine=PADDLEOCR_VL_ENGINE,
            extraction_variant=normalized_variant,
            korean_profile=korean_ocr_profile,
        )
    validator = DictionaryValidator(DICTIONARY_PATH)
    extraction = extract_from_document_parse(document_parse, validator)
    document_graph = graph_from_document_parse(document_parse, transform=transform)
    cards: list[CardCandidate] = []
    if extraction.page_type in {"vocab_table", "jp_ko_meaning_vocab"}:
        for item in extraction.items:
            cards.extend(vocab_cards(page.id, item))
    elif extraction.page_type in {"reading_mcq", "spelling_mcq"}:
        for item in extraction.items:
            cards.extend(mcq_cards(page.id, item))

    block_texts = [block.content for block in document_parse.blocks if block.content] or [document_parse.markdown_text]
    summary = script_summary(block_texts)
    warnings = [
        *preprocess_warnings,
        *ocr_warnings,
        *extraction.warnings,
        "Processed with PaddleOCR-VL document parsing; visual evidence is block-level.",
    ]
    if not cards:
        warnings.append("No card candidates were generated from PaddleOCR-VL document blocks.")
    document_graph = graph_with_card_hypotheses(document_graph, cards)
    review_metrics = _review_quality_metrics(cards, document_graph.model_dump()["metrics"])
    variant_diagnostics = _extraction_variant_diagnostics(
        normalized_variant,
        cards,
        processed_path,
        page.id,
        preprocess_width,
        preprocess_height,
        [],
    )
    variant_metrics = _extraction_variant_metrics(
        normalized_variant,
        cards,
        document_graph.model_dump(),
        variant_diagnostics=variant_diagnostics,
    )

    processed_page = Page(
        id=page.id,
        original_image_path=page.original_image_path,
        upload_name=page.upload_name,
        display_name=page.display_name,
        processed_image_path=str(processed_path),
        active_ocr_run_id=page.active_ocr_run_id,
        page_type=extraction.page_type,
        page_type_confidence=extraction.page_type_confidence,
        image_width=preprocess_width,
        image_height=preprocess_height,
        warnings=_unique_warnings(warnings),
        created_at=page.created_at,
    )
    database.upsert_page(processed_page)
    database.replace_tokens(page.id, [], run_id)
    database.replace_cards(page.id, cards, run_id)
    completed_run = database.complete_ocr_run(
        run_id,
        warnings=processed_page.warnings,
        metrics={
            "token_count": 0,
            "document_block_count": len(document_parse.blocks),
            "card_count": len(cards),
            "page_type": extraction.page_type,
            "page_type_confidence": extraction.page_type_confidence,
            "script_summary": summary,
            "answer_map_size": len(extraction.answer_map),
            "vlm_cleanup_enabled": VLM_CLEANUP_ENABLED,
            "vocab_dual_ocr_enabled": VOCAB_DUAL_OCR_ENABLED,
            "document_parse": document_parse.model_dump(mode="json"),
            "model_profile": profile_manifest,
            "korean_profile": korean_ocr_profile.id,
            "extraction_variant": normalized_variant,
            "cache_source_run_id": cache_source_run_id,
            "document_graph": document_graph.model_dump(),
            "extraction_variant_metrics": variant_metrics,
            **review_metrics,
        },
        processed_image_path=str(processed_path),
        image_width=preprocess_width,
        image_height=preprocess_height,
    )
    return ProcessResult(
        page=processed_page.model_copy(
            update={
                "active_ocr_run_id": run_id,
                "active_ocr_engine": completed_run.engine if completed_run else PADDLEOCR_VL_ENGINE,
                "active_ocr_completed_at": completed_run.completed_at if completed_run else None,
                "active_ocr_duration_ms": completed_run.duration_ms if completed_run else None,
            }
        ),
        tokens=[],
        cards=[card.model_copy(update={"run_id": run_id}) for card in cards],
        script_summary=summary,
        answer_map=extraction.answer_map,
        ocr_run=completed_run,
        document_parse=document_parse,
    )


def _unique_warnings(warnings: list[str]) -> list[str]:
    return list(dict.fromkeys(warning for warning in warnings if warning))


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _provider_config(engine: str, profile_manifest: dict[str, object], extraction_variant: str) -> dict[str, object]:
    cache = profile_manifest.get("cache") if isinstance(profile_manifest.get("cache"), dict) else {}
    return {
        "engine": engine,
        "japanese_detection_model": PADDLE_OCR_TEXT_DETECTION_MODEL_NAME,
        "japanese_recognition_model": PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME,
        "korean_detection_model": PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME,
        "korean_recognition_model": PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME,
        "cache_key": cache.get("key"),
        "vocab_dual_ocr_enabled": VOCAB_DUAL_OCR_ENABLED,
        "vlm_cleanup_enabled": VLM_CLEANUP_ENABLED,
        "model_profile": profile_manifest,
        "extraction_variant": extraction_variant,
        "full_page_cache_write": True,
        "full_page_cache_token_sources": [PADDLEOCR_ENGINE, "paddleocr_korean"],
    }


def _cached_ocr_payload(
    cached_run: OcrRun | None,
    page_id: str,
    processed_path: Path,
    engine: str,
) -> CachedOcrPayload | None:
    if not cached_run or cached_run.status != "succeeded":
        return None
    if engine == PADDLEOCR_VL_ENGINE:
        document_parse = database.get_document_parse_for_run(cached_run.id)
        if not document_parse:
            return None
        cloned_parse = document_parse.model_copy(
            update={
                "page_id": page_id,
                "source_image_path": str(processed_path),
            }
        )
        return CachedOcrPayload(
            engine_result=OcrEngineResult(
                engine=PADDLEOCR_VL_ENGINE,
                tokens=[],
                evidence_tokens=[],
                document_parse=cloned_parse,
                warnings=[],
            ),
            korean_tokens=[],
            source_run_id=cached_run.id,
        )

    if engine != PADDLEOCR_ENGINE:
        return None
    cached_tokens = database.get_tokens(cached_run.page_id, cached_run.id)
    if not cached_tokens:
        return None
    cached_tokens = _full_page_cache_tokens(cached_tokens)
    if not cached_tokens:
        return None
    cloned_tokens = _clone_tokens_for_page(cached_tokens, page_id)
    japanese_tokens = [token for token in cloned_tokens if token.source != "paddleocr_korean"]
    korean_tokens = [token for token in cloned_tokens if token.source == "paddleocr_korean"]
    if not japanese_tokens:
        return None
    return CachedOcrPayload(
        engine_result=OcrEngineResult(
            engine=PADDLEOCR_ENGINE,
            tokens=japanese_tokens,
            evidence_tokens=japanese_tokens,
            warnings=[],
        ),
        korean_tokens=korean_tokens,
        source_run_id=cached_run.id,
    )


def _clone_tokens_for_page(tokens: list[OcrToken], page_id: str) -> list[OcrToken]:
    return [token.model_copy(update={"id": new_id("tok"), "page_id": page_id}) for token in tokens]


def _full_page_cache_tokens(tokens: list[OcrToken]) -> list[OcrToken]:
    return [
        token
        for token in tokens
        if token.source in {PADDLEOCR_ENGINE, "paddleocr_korean"}
    ]


def _transform_metadata(
    width: int,
    height: int,
    warnings: list[str],
    *,
    original_path: Path | None = None,
    processed_path: Path | None = None,
    original_width: int | None = None,
    original_height: int | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "coordinate_space": "processed_image",
        "bbox_space": "processed_image",
        "original_width": original_width,
        "original_height": original_height,
        "processed_width": width,
        "processed_height": height,
        "original_image_path": str(original_path) if original_path else None,
        "processed_image_path": str(processed_path) if processed_path else None,
        "original_to_processed": {
            "pipeline": [
                "exif_transpose",
                "optional_perspective_crop",
                "optional_resize",
                "autocontrast",
                "sharpen",
            ],
            "invertible": False,
            "note": "OCR bboxes are stored against processed_image_path; rerun OCR if preprocessing output geometry changes.",
        },
        "warnings": warnings,
    }


def _preprocessing_config(width: int, height: int, warnings: list[str], transform: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "preprocess_max_side_len": PREPROCESS_MAX_SIDE_LEN,
        "pipeline": transform.get("original_to_processed", {}).get("pipeline") if isinstance(transform.get("original_to_processed"), dict) else [],
        "processed_width": width,
        "processed_height": height,
        "warnings": warnings,
        "transform": transform,
    }


def _preprocessing_hash(preprocessing_config: dict[str, object]) -> str:
    payload = json.dumps(_cacheable_preprocessing_config(preprocessing_config), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _ocr_cache_key(
    *,
    image_sha: str | None,
    preprocessing_hash: str,
    profile_manifest: dict[str, object],
    engine: str,
    extraction_variant: str,
) -> str:
    # Extraction variants rerun candidate generation from the same OCR payload.
    # Keep the OCR cache scoped to image/preprocessing/provider/model inputs only.
    del extraction_variant
    payload = {
        "schema_version": 1,
        "image_sha256": image_sha,
        "preprocessing_hash": preprocessing_hash,
        "engine": engine,
        "profile_id": cache_model_profile_id(str(profile_manifest.get("profile_id") or "")),
        "provider": profile_manifest.get("provider"),
        "env_fingerprint": profile_manifest.get("env_fingerprint"),
        "model_config": profile_manifest.get("model_config"),
        "language_config": profile_manifest.get("language_config"),
        "preprocessing_config": _cacheable_preprocessing_config(
            profile_manifest.get("preprocessing_config") if isinstance(profile_manifest.get("preprocessing_config"), dict) else {}
        ),
        "package_versions": profile_manifest.get("package_versions"),
        "model_cache_paths": profile_manifest.get("model_cache_paths"),
        "korean_profile": _cacheable_korean_profile(profile_manifest.get("korean_profile")),
    }
    encoded = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _cacheable_korean_profile(value: object) -> object:
    if not isinstance(value, dict):
        return value
    payload = dict(value)
    payload["profile_id"] = cache_korean_profile_id(str(payload.get("profile_id") or ""))
    payload.pop("label", None)
    return payload


def _cacheable_preprocessing_config(preprocessing_config: dict[str, object]) -> dict[str, object]:
    encoded = json.dumps(preprocessing_config, ensure_ascii=False, sort_keys=True, default=str)
    clean = json.loads(encoded)
    transform = clean.get("transform")
    if isinstance(transform, dict):
        transform.pop("original_image_path", None)
        transform.pop("processed_image_path", None)
    return clean


def _review_quality_metrics(cards: list[CardCandidate], graph_metrics: dict[str, object]) -> dict[str, object]:
    red_count = sum(1 for card in cards if card.review_state == "red")
    manual_review_count = sum(1 for card in cards if card.review_state == "yellow" or card.warnings)
    exportable_count = sum(1 for card in cards if card.review_state != "red")
    return {
        "candidate_recall_count": len(cards),
        "exportable_candidate_count": exportable_count,
        "manual_review_count": manual_review_count,
        "review_blocked_count": red_count,
        "evidence_alignment_score": graph_metrics.get("evidence_alignment_score", 0.0),
    }


def _extraction_variant_metrics(
    variant: str,
    cards: list[CardCandidate],
    graph_payload: dict[str, object],
    *,
    variant_diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    variant_diagnostics = variant_diagnostics or {}
    components = extraction_variant_components(variant)
    rows = graph_payload.get("row_hypotheses") if isinstance(graph_payload.get("row_hypotheses"), list) else []
    fields = graph_payload.get("field_hypotheses") if isinstance(graph_payload.get("field_hypotheses"), list) else []
    row_alignment_diagnostics = bool(components & {"v5_vocab_rows_v1", "ko_alignment_v1"})
    base = {
        "schema_version": 1,
        "variant": variant,
        "components": sorted(components),
        "candidate_mutation": bool(
            {
                "v5_token_split_v1",
                "v5_mcq_v1",
                *KOREAN_RECOVERY_COMPONENTS,
                *MCQ_RECOVERY_COMPONENTS,
                *JAPANESE_RECOVERY_COMPONENTS,
                *KOREAN_RESIDUAL_COMPONENTS,
                *MCQ_PROMPT_LINE_COMPONENTS,
                *MCQ_CHOICE_GLYPH_COMPONENTS,
            }
            & components
        ),
        "diagnostic_only": variant in {"provider_agreement_v1", "residual_diagnostics_v1"} or bool(
            components and components <= {"v5_vocab_rows_v1", "ko_alignment_v1", "residual_diagnostics_v1"}
        ),
        "row_alignment_candidate_replacement": "guarded_off" if row_alignment_diagnostics else "not_applicable",
        "row_hypothesis_count": len(rows),
        "field_hypothesis_count": len(fields),
    }
    if variant == DEFAULT_EXTRACTION_VARIANT:
        return base
    if variant == "line_graph_v1":
        line_nodes = graph_payload.get("line_nodes") if isinstance(graph_payload.get("line_nodes"), list) else []
        return {
            **base,
            "line_count": len(line_nodes),
            "reading_order_edges": max(0, len(line_nodes) - 1),
        }
    if variant == "table_graph_v1":
        table_cells = graph_payload.get("table_cells") if isinstance(graph_payload.get("table_cells"), list) else []
        selection_marks = graph_payload.get("selection_marks") if isinstance(graph_payload.get("selection_marks"), list) else []
        return {
            **base,
            "table_cell_count": len(table_cells),
            "selection_mark_count": len(selection_marks),
            "section_header_count": _section_header_count(table_cells),
        }
    if variant == "ranked_rows_v1":
        ranked_rows = _ranked_row_metrics(cards)
        return {
            **base,
            "ranked_rows": ranked_rows,
            "top_row_score": ranked_rows[0]["score"] if ranked_rows else None,
        }
    if variant == "crop_confirm_v1":
        return {
            **base,
            "uncertain_fields": _uncertain_fields(cards),
            "crop_confirmation_mode": "limited_crop_ocr_review_only",
            "crop_confirmation": variant_diagnostics.get("crop_confirmation", {}),
        }
    if variant == "provider_agreement_v1":
        return {
            **base,
            "agreement_source": "external_compare_route",
            "automatic_extraction_decision": False,
            "provider_agreement": variant_diagnostics.get("provider_agreement", {}),
        }
    if components & {"v5_token_split_v1", "v5_vocab_rows_v1", "v5_mcq_v1"}:
        return {
            **base,
            "token_split_enabled": "v5_token_split_v1" in components,
            "vocab_rows_enabled": "v5_vocab_rows_v1" in components,
            "korean_alignment_enabled": "ko_alignment_v1" in components,
            "mcq_recovery_enabled": "v5_mcq_v1" in components,
            "korean_recovery_enabled": bool(KOREAN_RECOVERY_COMPONENTS & components),
            "mcq_source_recovery_enabled": bool(MCQ_RECOVERY_COMPONENTS & components),
            "japanese_recovery_enabled": bool(JAPANESE_RECOVERY_COMPONENTS & components),
            "korean_residual_glyph_enabled": bool(KOREAN_RESIDUAL_COMPONENTS & components),
            "mcq_prompt_line_recovery_enabled": bool(MCQ_PROMPT_LINE_COMPONENTS & components),
            "mcq_choice_glyph_recovery_enabled": bool(MCQ_CHOICE_GLYPH_COMPONENTS & components),
            "vocab_rows_candidate_replacement": "guarded_off" if "v5_vocab_rows_v1" in components else "not_applicable",
            "vocab_alignment": variant_diagnostics.get("vocab_alignment", {}),
            "korean_alignment": variant_diagnostics.get("korean_alignment", {}),
            "recovery": variant_diagnostics.get("recovery", {}),
        }
    if "ko_alignment_v1" in components:
        return {
            **base,
            "korean_alignment_enabled": True,
            "vocab_alignment": variant_diagnostics.get("vocab_alignment", {}),
            "korean_alignment": variant_diagnostics.get("korean_alignment", {}),
            "recovery": variant_diagnostics.get("recovery", {}),
        }
    if (
        KOREAN_RECOVERY_COMPONENTS & components
        or MCQ_RECOVERY_COMPONENTS & components
        or JAPANESE_RECOVERY_COMPONENTS & components
        or KOREAN_RESIDUAL_COMPONENTS & components
        or MCQ_PROMPT_LINE_COMPONENTS & components
        or MCQ_CHOICE_GLYPH_COMPONENTS & components
    ):
        return {
            **base,
            "korean_recovery_enabled": bool(KOREAN_RECOVERY_COMPONENTS & components),
            "mcq_source_recovery_enabled": bool(MCQ_RECOVERY_COMPONENTS & components),
            "japanese_recovery_enabled": bool(JAPANESE_RECOVERY_COMPONENTS & components),
            "korean_residual_glyph_enabled": bool(KOREAN_RESIDUAL_COMPONENTS & components),
            "mcq_prompt_line_recovery_enabled": bool(MCQ_PROMPT_LINE_COMPONENTS & components),
            "mcq_choice_glyph_recovery_enabled": bool(MCQ_CHOICE_GLYPH_COMPONENTS & components),
            "recovery": variant_diagnostics.get("recovery", {}),
        }
    return base


def _extraction_variant_diagnostics(
    variant: str,
    cards: list[CardCandidate],
    image_path: Path,
    page_id: str,
    page_width: int | None,
    page_height: int | None,
    tokens: list[OcrToken],
) -> dict[str, object]:
    if variant == "crop_confirm_v1":
        return {
            "crop_confirmation": _crop_confirmation_diagnostics(
                cards,
                image_path,
                page_id,
                page_width,
                page_height,
            )
        }
    if variant == "provider_agreement_v1":
        return {"provider_agreement": _provider_agreement_diagnostics(image_path, page_id, tokens)}
    components = extraction_variant_components(variant)
    diagnostics: dict[str, object] = {}
    if components & {"v5_vocab_rows_v1", "ko_alignment_v1"} and any(card.source_type == "vocab_item" for card in cards):
        japanese_tokens = [token for token in tokens if token.source != "paddleocr_korean"]
        korean_tokens = [token for token in tokens if token.source == "paddleocr_korean"]
        diagnostics["vocab_alignment"] = vocab_alignment_diagnostics(japanese_tokens, korean_tokens, variant)
    if "ko_alignment_v1" in components:
        diagnostics["korean_alignment"] = _korean_alignment_diagnostics(cards, tokens)
    return diagnostics


def _korean_alignment_diagnostics(cards: list[CardCandidate], tokens: list[OcrToken]) -> dict[str, object]:
    korean_tokens = [token for token in tokens if token.source == "paddleocr_korean" or _has_hangul_text(token.text)]
    paired_token_ids = set()
    stale_evidence = 0
    boxed_meaning_fields = 0
    vocab_cards_seen = 0
    for card in cards:
        if card.source_type != "vocab_item":
            continue
        vocab_cards_seen += 1
        evidence = card.source.get("field_evidence")
        meaning_evidence = evidence.get("meaning_ko") if isinstance(evidence, dict) else None
        if not isinstance(meaning_evidence, dict):
            stale_evidence += 1
            continue
        token_ids = meaning_evidence.get("token_ids")
        if isinstance(token_ids, list):
            paired_token_ids.update(token_id for token_id in token_ids if isinstance(token_id, str))
        bbox = meaning_evidence.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            boxed_meaning_fields += 1
        if meaning_evidence.get("provenance") == "ocr" and not token_ids:
            stale_evidence += 1
    korean_token_ids = {token.id for token in korean_tokens}
    paired_korean_ids = paired_token_ids & korean_token_ids
    unpaired = [token.text for token in korean_tokens if token.id not in paired_korean_ids]
    return {
        "schema_version": 1,
        "korean_token_count": len(korean_tokens),
        "paired_korean_token_count": len(paired_korean_ids),
        "paired_meaning_recall": round(len(paired_korean_ids) / len(korean_tokens), 4) if korean_tokens else 0.0,
        "vocab_card_count": vocab_cards_seen,
        "meaning_bbox_alignment": round(boxed_meaning_fields / vocab_cards_seen, 4) if vocab_cards_seen else 0.0,
        "stale_evidence_count": stale_evidence,
        "unpaired_hangul_token_count": len(unpaired),
        "unpaired_hangul_sample": unpaired[:20],
    }


def _has_hangul_text(text: str) -> bool:
    return any(0xAC00 <= ord(char) <= 0xD7AF for char in text)


KOREAN_RECOVERY_COMPONENTS = {"ko_crop_confirm_v1", "ko_region_columns_v1", "ko_consensus_v1"}
MCQ_RECOVERY_COMPONENTS = {"mcq_source_rebuild_v1", "mcq_choice_band_ocr_v1"}
JAPANESE_RECOVERY_COMPONENTS = {"jp_region_columns_v1"}
KOREAN_RESIDUAL_COMPONENTS = {"ko_residual_glyph_v1"}
MCQ_PROMPT_LINE_COMPONENTS = {"mcq_prompt_line_ocr_v1"}
MCQ_CHOICE_GLYPH_COMPONENTS = {"mcq_choice_glyph_v1"}


def _uses_korean_recovery(components: frozenset[str]) -> bool:
    return bool(KOREAN_RECOVERY_COMPONENTS & components)


def _uses_mcq_recovery(components: frozenset[str]) -> bool:
    return bool(MCQ_RECOVERY_COMPONENTS & components)


def _uses_v2_vocab_recovery(components: frozenset[str]) -> bool:
    return bool((JAPANESE_RECOVERY_COMPONENTS | KOREAN_RESIDUAL_COMPONENTS) & components)


def _uses_v2_mcq_recovery(components: frozenset[str]) -> bool:
    return bool((MCQ_PROMPT_LINE_COMPONENTS | MCQ_CHOICE_GLYPH_COMPONENTS) & components)


def _recover_korean_vocab_items(
    items: list[dict],
    all_tokens: list[OcrToken],
    image_path: Path,
    page_id: str,
    page_width: int,
    page_height: int,
    preprocessing_hash: str,
    profile_id: str,
    korean_profile_id: str,
    components: frozenset[str],
) -> tuple[list[dict], list[OcrToken], dict[str, object]]:
    recovered_tokens: list[OcrToken] = []
    attempts: list[dict[str, object]] = []
    counts = {
        "raw_ocr_absent": 0,
        "raw_present_wrong_text": 0,
        "raw_present_wrong_pairing": 0,
        "recovered_by_crop": 0,
        "recovered_by_region": 0,
        "recovered_by_numeric_unit": 0,
        "rejected_by_consensus": 0,
        "recovery_resource_cap": 0,
    }
    meaning_anchors = _meaning_anchor_bboxes(items)
    used_recovery_token_ids: set[str] = set()
    attempted_fields = 0
    for item in sorted(items, key=_korean_recovery_priority):
        if attempted_fields >= OCR_RECOVERY_MAX_FIELDS:
            counts["recovery_resource_cap"] += 1
            break
        bucket = _korean_uncertainty_bucket(item)
        if bucket is None:
            continue
        if not str(item.get("surface") or "").strip() or not str(item.get("reading") or "").strip():
            continue
        counts[bucket] = counts.get(bucket, 0) + 1
        attempted_fields += 1
        item_attempt: dict[str, object] = {
            "source_id": item.get("id"),
            "surface": item.get("surface"),
            "reading": item.get("reading"),
            "current_meaning_ko": item.get("meaning_ko"),
            "bucket": bucket,
            "accepted": False,
            "candidates": [],
        }
        numeric_unit = _numeric_unit_recovery_candidate(item)
        if numeric_unit:
            text, confidence, token_ids, bbox = numeric_unit
            evidence = dict(_field_evidence_for(item, "meaning_ko") or {})
            evidence["text"] = text
            evidence["raw_text"] = item.get("meaning_ko", "")
            evidence["confidence"] = confidence
            evidence["token_ids"] = token_ids
            evidence["bbox"] = bbox
            evidence["provenance"] = evidence.get("provenance") or "ocr"
            evidence["normalization_strategy"] = "numeric_unit_completion_v1"
            item["meaning_ko"] = text
            field_evidence = item.get("field_evidence")
            if not isinstance(field_evidence, dict):
                field_evidence = {}
                item["field_evidence"] = field_evidence
            field_evidence["meaning_ko"] = evidence
            item["evidence_tokens"] = _unique_string_values([*item.get("evidence_tokens", []), *token_ids])
            item["meaning_bbox"] = bbox
            item["bbox"] = _bbox_union([item.get("bbox"), bbox])
            item["confidence"] = round(max(float(item.get("confidence") or 0.0), confidence), 3)
            item["warnings"] = _recovery_warnings(item.get("warnings", []))
            item["needs_review"] = bool(item["warnings"])
            item["recovery"] = {
                "accepted": True,
                "source": "ocr_numeric_unit",
                "provenance": evidence.get("provenance"),
                "diagnostic_bucket": bucket,
            }
            used_recovery_token_ids.update(token_ids)
            counts["recovered_by_numeric_unit"] += 1
            item_attempt["accepted"] = True
            item_attempt["accepted_source"] = "ocr_numeric_unit"
            item_attempt["accepted_text"] = text
            item_attempt["candidates"] = [
                {
                    "source": "ocr_numeric_unit",
                    "text": text,
                    "confidence": confidence,
                    "token_count": len(token_ids),
                    "selected_token_ids": token_ids,
                    "warnings": [],
                }
            ]
            attempts.append(item_attempt)
            continue
        candidates: list[tuple[str, object]] = []
        if "ko_crop_confirm_v1" in components:
            crop_bbox = _meaning_recovery_bbox(item, meaning_anchors, page_width, page_height, prefer_existing=True)
            if crop_bbox:
                result = _safe_recognize_region(
                    image_path=image_path,
                    page_id=page_id,
                    region_id=str(item.get("id") or f"row_{attempted_fields}"),
                    field="meaning_ko",
                    bbox=crop_bbox,
                    page_width=page_width,
                    page_height=page_height,
                    preprocessing_hash=preprocessing_hash,
                    strategy="korean_field_crop",
                    profile_id=profile_id,
                    korean_profile_id=korean_profile_id,
                    provenance="crop_ocr",
                )
                if result:
                    candidates.append(("crop", result))
        if "ko_region_columns_v1" in components:
            region_bbox = _meaning_recovery_bbox(item, meaning_anchors, page_width, page_height, prefer_existing=False)
            if region_bbox:
                result = _safe_recognize_region(
                    image_path=image_path,
                    page_id=page_id,
                    region_id=str(item.get("id") or f"row_{attempted_fields}"),
                    field="korean_region",
                    bbox=region_bbox,
                    page_width=page_width,
                    page_height=page_height,
                    preprocessing_hash=preprocessing_hash,
                    strategy="korean_column_region",
                    profile_id=profile_id,
                    korean_profile_id=korean_profile_id,
                    provenance="region_ocr",
                    provider="paddle_korean",
                )
                if result:
                    candidates.append(("region", result))
        accepted = _select_korean_recovery_candidate(item, candidates, used_recovery_token_ids)
        item_attempt["candidates"] = [_recovery_candidate_summary(source, result, item) for source, result in candidates]
        if accepted:
            source, result, text, confidence, token_ids, bbox = accepted
            evidence = dict(result.field_evidence)
            evidence["text"] = text
            evidence["raw_text"] = result.text
            evidence["confidence"] = confidence
            evidence["token_ids"] = token_ids
            evidence["bbox"] = bbox
            item["meaning_ko"] = text
            field_evidence = item.get("field_evidence")
            if not isinstance(field_evidence, dict):
                field_evidence = {}
                item["field_evidence"] = field_evidence
            field_evidence["meaning_ko"] = evidence
            item["evidence_tokens"] = _unique_string_values([*item.get("evidence_tokens", []), *token_ids])
            item["meaning_bbox"] = bbox
            item["bbox"] = _bbox_union([item.get("bbox"), bbox])
            item["confidence"] = round(max(float(item.get("confidence") or 0.0), confidence), 3)
            item["warnings"] = _recovery_warnings(item.get("warnings", []))
            item["needs_review"] = bool(item["warnings"])
            item["recovery"] = {
                "accepted": True,
                "source": source,
                "provenance": evidence.get("provenance"),
                "diagnostic_bucket": bucket,
                "cache": result.cache,
            }
            recovered_tokens.extend(result.tokens)
            used_recovery_token_ids.update(token_ids)
            counts["recovered_by_crop" if source == "crop" else "recovered_by_region"] += 1
            item_attempt["accepted"] = True
            item_attempt["accepted_source"] = source
            item_attempt["accepted_text"] = text
        elif candidates:
            counts["rejected_by_consensus"] += 1
            item["recovery"] = {"accepted": False, "diagnostic_bucket": bucket, "reason": "consensus_rejected"}
        attempts.append(item_attempt)
    return items, recovered_tokens, {
        "schema_version": 1,
        "kind": "korean_recovery",
        "attempted": attempted_fields,
        "accepted": counts["recovered_by_crop"] + counts["recovered_by_region"] + counts["recovered_by_numeric_unit"],
        "limit": OCR_RECOVERY_MAX_FIELDS,
        "counts": {key: value for key, value in counts.items() if value},
        "attempts": attempts,
    }


def _append_recovery_diagnostics(current: dict[str, object], component: str, payload: dict[str, object]) -> dict[str, object]:
    if not payload:
        return current
    if not current:
        current = {}
    if current.get("schema_version") == 2 and isinstance(current.get("components"), dict):
        combined = dict(current)
        components = dict(combined.get("components") if isinstance(combined.get("components"), dict) else {})
    else:
        combined = {
            "schema_version": 2,
            "kind": "accuracy_recovery_v2",
            "attempted": 0,
            "accepted": 0,
            "counts": {},
            "components": {},
            "resource_caps": {},
            "cache": {},
        }
        components = {}
        if current:
            previous_key = str(current.get("kind") or "recovery")
            components[previous_key] = current
    nested_components = payload.get("components") if isinstance(payload.get("components"), dict) else {}
    if payload.get("schema_version") == 2 and nested_components:
        for nested_component, nested_payload in nested_components.items():
            _store_recovery_component(components, str(nested_component), nested_payload)
    else:
        _store_recovery_component(components, component, payload)
    counts: dict[str, int] = {}
    resource_caps: dict[str, int] = {}
    cache = {"hits": 0, "misses": 0}
    attempted = 0
    accepted = 0
    for value in components.values():
        if not isinstance(value, dict):
            continue
        attempted += int(value.get("attempted") or 0)
        accepted += int(value.get("accepted") or 0)
        component_counts = value.get("counts") if isinstance(value.get("counts"), dict) else {}
        for key, count in component_counts.items():
            if isinstance(count, int):
                counts[str(key)] = counts.get(str(key), 0) + count
                if "resource_cap" in str(key) and count:
                    resource_caps[str(key)] = resource_caps.get(str(key), 0) + count
        _accumulate_recovery_cache_counts(value, cache)
    combined["components"] = components
    combined["attempted"] = attempted
    combined["accepted"] = accepted
    combined["counts"] = counts
    combined["resource_caps"] = resource_caps
    combined["cache"] = cache
    return combined


def _store_recovery_component(components: dict[str, object], component: str, payload: object) -> None:
    key = "korean_recovery" if component == "korean_residual_glyph_recovery" else component
    if key in components and isinstance(components[key], dict) and isinstance(payload, dict):
        components[key] = _merge_recovery_component_payload(key, components[key], payload)
    else:
        components[key] = payload


def _merge_recovery_component_payload(kind: str, left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    merged = dict(left)
    merged["kind"] = kind
    merged["attempted"] = int(left.get("attempted") or 0) + int(right.get("attempted") or 0)
    merged["accepted"] = int(left.get("accepted") or 0) + int(right.get("accepted") or 0)
    counts: dict[str, int] = {}
    for payload in (left, right):
        payload_counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
        for count_key, count_value in payload_counts.items():
            if isinstance(count_value, int):
                counts[str(count_key)] = counts.get(str(count_key), 0) + count_value
    if counts:
        merged["counts"] = counts
    attempts: list[object] = []
    for payload in (left, right):
        payload_attempts = payload.get("attempts") if isinstance(payload.get("attempts"), list) else []
        attempts.extend(payload_attempts)
    if attempts:
        merged["attempts"] = attempts
    return merged


def _accumulate_recovery_cache_counts(payload: dict[str, object], cache: dict[str, int]) -> None:
    attempts = payload.get("attempts") if isinstance(payload.get("attempts"), list) else []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        _accumulate_attempt_cache(attempt, cache)
        candidates = attempt.get("candidates") if isinstance(attempt.get("candidates"), list) else []
        for candidate in candidates:
            if isinstance(candidate, dict):
                _accumulate_attempt_cache(candidate, cache)
    components = payload.get("components") if isinstance(payload.get("components"), dict) else {}
    for component_payload in components.values():
        if isinstance(component_payload, dict):
            _accumulate_recovery_cache_counts(component_payload, cache)


def _accumulate_attempt_cache(payload: dict[str, object], cache: dict[str, int]) -> None:
    cache_payload = payload.get("cache") if isinstance(payload.get("cache"), dict) else {}
    if cache_payload.get("hit") is True:
        cache["hits"] = cache.get("hits", 0) + 1
    elif cache_payload.get("hit") is False:
        cache["misses"] = cache.get("misses", 0) + 1


def _recover_v2_vocab_items(
    items: list[dict],
    all_tokens: list[OcrToken],
    image_path: Path,
    page_id: str,
    page_width: int,
    page_height: int,
    preprocessing_hash: str,
    profile_id: str,
    korean_profile_id: str,
    components: frozenset[str],
    validator: DictionaryValidator,
) -> tuple[list[dict], list[OcrToken], dict[str, object]]:
    recovered_tokens: list[OcrToken] = []
    diagnostics: dict[str, object] = {}
    if "jp_region_columns_v1" in components:
        items, tokens, jp_diagnostics = _recover_japanese_vocab_items(
            items,
            all_tokens,
            image_path,
            page_id,
            page_width,
            page_height,
            preprocessing_hash,
            profile_id,
            korean_profile_id,
            validator,
        )
        recovered_tokens.extend(tokens)
        diagnostics = _append_recovery_diagnostics(diagnostics, "japanese_vocab_recovery", jp_diagnostics)
        all_tokens = [*all_tokens, *tokens]
    if "ko_residual_glyph_v1" in components:
        items, tokens, ko_diagnostics = _recover_korean_residual_glyph_items(
            items,
            all_tokens,
            image_path,
            page_id,
            page_width,
            page_height,
            preprocessing_hash,
            profile_id,
            korean_profile_id,
        )
        recovered_tokens.extend(tokens)
        diagnostics = _append_recovery_diagnostics(diagnostics, "korean_residual_glyph_recovery", ko_diagnostics)
    return items, recovered_tokens, diagnostics


def _recover_japanese_vocab_items(
    items: list[dict],
    all_tokens: list[OcrToken],
    image_path: Path,
    page_id: str,
    page_width: int,
    page_height: int,
    preprocessing_hash: str,
    profile_id: str,
    korean_profile_id: str,
    validator: DictionaryValidator,
) -> tuple[list[dict], list[OcrToken], dict[str, object]]:
    recovered_tokens: list[OcrToken] = []
    attempts: list[dict[str, object]] = []
    counts = {
        "jp_recovered_surface": 0,
        "jp_recovered_reading": 0,
        "jp_recovered_missing_row": 0,
        "jp_rejected_low_confidence": 0,
        "jp_rejected_no_tokens": 0,
        "jp_rejected_incomplete_row": 0,
        "jp_resource_cap": 0,
    }
    attempted = 0
    used_token_ids = _used_item_token_ids(items)
    for item in items:
        for field in ("surface", "reading"):
            if attempted >= OCR_RECOVERY_MAX_FIELDS:
                counts["jp_resource_cap"] += 1
                break
            if not _jp_field_needs_recovery(item, field):
                continue
            bbox = _japanese_field_recovery_bbox(item, field, page_width, page_height)
            if not bbox:
                continue
            attempted += 1
            result = _safe_recognize_region(
                image_path=image_path,
                page_id=page_id,
                region_id=str(item.get("id") or f"jp_{field}_{attempted}"),
                field=field,
                bbox=bbox,
                page_width=page_width,
                page_height=page_height,
                preprocessing_hash=preprocessing_hash,
                strategy=f"japanese_{field}_region",
                profile_id=profile_id,
                korean_profile_id=korean_profile_id,
                provenance="jp_region_ocr",
            )
            attempt = {"source_id": item.get("id"), "field": field, "bbox": bbox, "accepted": False}
            if not result or not getattr(result, "tokens", []):
                counts["jp_rejected_no_tokens"] += 1
                attempts.append(attempt)
                continue
            selection = _select_japanese_region_evidence(result, field, used_token_ids)
            attempt["text"] = getattr(result, "text", "")
            attempt["cache"] = getattr(result, "cache", {})
            if not selection:
                counts["jp_rejected_low_confidence"] += 1
                attempts.append(attempt)
                continue
            text, confidence, token_ids, evidence_bbox = selection
            _apply_vocab_field_recovery(item, field, text, confidence, token_ids, evidence_bbox, "jp_region_ocr")
            recovered_tokens.extend(result.tokens)
            used_token_ids.update(token_ids)
            counts[f"jp_recovered_{field}"] += 1
            attempt["accepted"] = True
            attempt["accepted_text"] = text
            attempts.append(attempt)
        if attempted >= OCR_RECOVERY_MAX_FIELDS:
            break
    new_items, new_tokens, missing_diagnostics = _recover_missing_vocab_rows_from_unpaired_tokens(
        items,
        all_tokens,
        used_token_ids,
        page_id,
        validator,
        image_path=image_path,
        page_width=page_width,
        page_height=page_height,
        preprocessing_hash=preprocessing_hash,
        profile_id=profile_id,
        korean_profile_id=korean_profile_id,
    )
    if new_items:
        items.extend(new_items)
        items.sort(key=_vocab_item_workbook_order_key)
        recovered_tokens.extend(new_tokens)
        counts["jp_recovered_missing_row"] += len(new_items)
    elif missing_diagnostics.get("attempted"):
        counts["jp_rejected_incomplete_row"] += int(missing_diagnostics.get("rejected") or 0)
    attempts.extend(missing_diagnostics.get("attempts") if isinstance(missing_diagnostics.get("attempts"), list) else [])
    return items, recovered_tokens, {
        "schema_version": 1,
        "kind": "japanese_vocab_recovery",
        "attempted": attempted + int(missing_diagnostics.get("attempted") or 0),
        "accepted": counts["jp_recovered_surface"] + counts["jp_recovered_reading"] + counts["jp_recovered_missing_row"],
        "counts": {key: value for key, value in counts.items() if value},
        "attempts": attempts,
    }


def _jp_field_needs_recovery(item: dict, field: str) -> bool:
    value = str(item.get(field) or "").strip()
    if not value:
        return True
    evidence = _field_evidence_for(item, field)
    if field == "surface":
        return not _has_japanese_text(value)
    return not _reading_like_text(value)


def _japanese_field_recovery_bbox(item: dict, field: str, page_width: int, page_height: int) -> list[float] | None:
    existing = _evidence_bbox(item, field)
    if existing:
        return _expanded_bbox(existing, page_width, page_height, x_pad=8.0, y_pad=5.0)
    row_bbox = _valid_bbox(item.get("row_bbox")) or _valid_bbox(item.get("bbox"))
    if not row_bbox:
        return None
    surface_bbox = _evidence_bbox(item, "surface")
    reading_bbox = _evidence_bbox(item, "reading")
    if field == "surface" and reading_bbox:
        width = max(35.0, reading_bbox[2] - reading_bbox[0])
        return _clip_bbox([reading_bbox[0] - width - 12.0, row_bbox[1] - 5.0, reading_bbox[0] - 2.0, row_bbox[3] + 5.0], page_width, page_height)
    if field == "reading" and surface_bbox:
        width = max(45.0, surface_bbox[2] - surface_bbox[0])
        return _clip_bbox([surface_bbox[2] + 2.0, row_bbox[1] - 5.0, surface_bbox[2] + width + 14.0, row_bbox[3] + 5.0], page_width, page_height)
    row_width = row_bbox[2] - row_bbox[0]
    if field == "surface":
        return _clip_bbox([row_bbox[0], row_bbox[1] - 5.0, row_bbox[0] + row_width * 0.36, row_bbox[3] + 5.0], page_width, page_height)
    return _clip_bbox([row_bbox[0] + row_width * 0.22, row_bbox[1] - 5.0, row_bbox[0] + row_width * 0.62, row_bbox[3] + 5.0], page_width, page_height)


def _select_japanese_region_evidence(result: object, field: str, used_token_ids: set[str]) -> tuple[str, float, list[str], list[float]] | None:
    candidates: list[tuple[float, str, list[str], list[float]]] = []
    combined_text = _clean_japanese_field_text(str(getattr(result, "text", "") or ""), field)
    combined_token_ids = [token.id for token in getattr(result, "tokens", []) if token.id not in used_token_ids]
    combined_bbox = _bbox_union([token.bbox for token in getattr(result, "tokens", []) if token.id not in used_token_ids]) or list(getattr(result, "bbox", []))
    combined_confidence = max(
        [float(getattr(token, "confidence", 0.0) or 0.0) for token in getattr(result, "tokens", []) if token.id in combined_token_ids]
        or [float(getattr(result, "confidence", 0.0) or 0.0)]
    )
    if combined_text and combined_token_ids and combined_confidence >= 0.72 and _valid_bbox(combined_bbox):
        candidates.append((combined_confidence, combined_text, combined_token_ids, [float(value) for value in combined_bbox]))
    for token in getattr(result, "tokens", []):
        if token.id in used_token_ids:
            continue
        text = _clean_japanese_field_text(str(token.text or ""), field)
        if not text:
            continue
        confidence = float(getattr(token, "confidence", 0.0) or 0.0)
        if confidence < 0.72:
            continue
        candidates.append((confidence, text, [token.id], list(token.bbox)))
    if not candidates:
        return None
    confidence, text, token_ids, bbox = max(candidates, key=lambda value: (len(value[1]), value[0]))
    return text, confidence, token_ids, bbox


def _clean_japanese_field_text(text: str, field: str) -> str:
    text = re.sub(r"^[□☐▢口日回ロ\s]+", "", text.strip())
    if field == "reading":
        runs = re.findall(r"[ぁ-ゖー]+", text)
        return max(runs, key=len) if runs else ""
    normalized = re.sub(r"(?<=[ァ-ヺー])[-‐‑‒–—―](?=[ァ-ヺー])", "ー", text)
    katakana_runs = re.findall(r"[ァ-ヺー]{3,}", normalized)
    if katakana_runs:
        return max(katakana_runs, key=len)
    chars = [char for char in text if _is_japanese_char(char)]
    cleaned = "".join(chars)
    if not cleaned:
        return ""
    return cleaned


def _apply_vocab_field_recovery(
    item: dict,
    field: str,
    text: str,
    confidence: float,
    token_ids: list[str],
    bbox: list[float],
    provenance: str,
) -> None:
    item[field] = text
    field_evidence = item.get("field_evidence")
    if not isinstance(field_evidence, dict):
        field_evidence = {}
        item["field_evidence"] = field_evidence
    field_evidence[field] = {
        "text": text,
        "bbox": bbox,
        "token_ids": token_ids,
        "confidence": confidence,
        "provenance": provenance,
    }
    item[f"{field}_bbox"] = bbox
    item["bbox"] = _bbox_union([item.get("bbox"), bbox])
    item["evidence_tokens"] = _unique_string_values([*item.get("evidence_tokens", []), *token_ids])
    item["confidence"] = round(max(float(item.get("confidence") or 0.0), confidence), 3)


def _recover_missing_vocab_rows_from_unpaired_tokens(
    items: list[dict],
    all_tokens: list[OcrToken],
    used_token_ids: set[str],
    page_id: str,
    validator: DictionaryValidator,
    *,
    image_path: Path | None = None,
    page_width: int | None = None,
    page_height: int | None = None,
    preprocessing_hash: str = "",
    profile_id: str = "",
    korean_profile_id: str = "",
) -> tuple[list[dict], list[OcrToken], dict[str, object]]:
    unpaired = [token for token in all_tokens if token.id not in used_token_ids and _token_has_vocab_text(token)]
    lines = _line_groups_for_tokens(unpaired, tolerance=20.0)
    new_items: list[dict] = []
    recovered_tokens: list[OcrToken] = []
    attempts: list[dict[str, object]] = []
    split_x = _vocab_recovery_split_x([token for token in all_tokens if token.source != "paddleocr_korean"])
    existing_rows = [
        (_valid_bbox(item.get("row_bbox")) or _valid_bbox(item.get("bbox")), str(item.get("column") or _column_for_item(item, split_x)))
        for item in items
    ]
    templates = _vocab_column_field_templates(items, split_x)
    for line in lines:
        for column, side_tokens in _split_vocab_line_by_column(line, split_x).items():
            if not side_tokens:
                continue
            if any(
                bbox and existing_column == column and _row_overlaps_existing(side_tokens, bbox)
                for bbox, existing_column in existing_rows
            ):
                continue
            item = _vocab_item_from_line_tokens(side_tokens, page_id, validator)
            row_tokens: list[OcrToken] = []
            attempt: dict[str, object] = {
                "kind": "missing_row_candidate",
                "column": column,
                "token_ids": [token.id for token in side_tokens],
                "accepted": bool(item),
            }
            if not item and image_path and page_width and page_height:
                item, row_tokens, region_attempt = _vocab_item_from_missing_row_regions(
                    side_tokens,
                    column,
                    templates,
                    page_id,
                    validator,
                    image_path=image_path,
                    page_width=page_width,
                    page_height=page_height,
                    preprocessing_hash=preprocessing_hash,
                    profile_id=profile_id,
                    korean_profile_id=korean_profile_id,
                    used_token_ids=used_token_ids,
                )
                attempt.update(region_attempt)
                attempt["accepted"] = bool(item)
            attempts.append(attempt)
            if not item:
                continue
            new_items.append(item)
            recovered_tokens.extend(row_tokens)
            used_token_ids.update(str(token_id) for token_id in item.get("evidence_tokens", []) if isinstance(token_id, str))
            existing_rows.append((_valid_bbox(item.get("row_bbox")) or _valid_bbox(item.get("bbox")), column))
    return new_items, recovered_tokens, {"attempted": len(attempts), "rejected": len(attempts) - len(new_items), "attempts": attempts}


def _vocab_recovery_split_x(tokens: list[OcrToken]) -> float:
    centers = sorted(
        (token.bbox[0] + token.bbox[2]) / 2.0
        for token in tokens
        if _has_japanese_text(token.text)
    )
    if not centers:
        return 0.0
    if len(centers) < 4:
        return _median_float(centers)
    gaps = [(right - left, left, right) for left, right in zip(centers, centers[1:])]
    min_x = centers[0]
    max_x = centers[-1]
    median_gap = _median_float([gap for gap, _left, _right in gaps]) or 1.0
    central_gaps = [
        (gap, left, right)
        for gap, left, right in gaps
        if gap > median_gap * 2.0 and min_x + (max_x - min_x) * 0.25 <= (left + right) / 2.0 <= min_x + (max_x - min_x) * 0.75
    ]
    if central_gaps:
        _gap, left, right = max(central_gaps, key=lambda value: value[0])
        return (left + right) / 2.0
    return _median_float(centers)


def _split_vocab_line_by_column(line: list[OcrToken], split_x: float) -> dict[str, list[OcrToken]]:
    columns = {"left": [], "right": []}
    for token in line:
        columns[_token_column_for_bbox(token.bbox, split_x)].append(token)
    return {column: sorted(tokens, key=lambda token: token.bbox[0]) for column, tokens in columns.items()}


def _token_column_for_bbox(bbox: list[float], split_x: float) -> str:
    return "left" if ((bbox[0] + bbox[2]) / 2.0) < split_x else "right"


def _vocab_column_field_templates(items: list[dict], split_x: float) -> dict[str, dict[str, tuple[float, float]]]:
    templates: dict[str, dict[str, tuple[float, float]]] = {"left": {}, "right": {}}
    for column in ("left", "right"):
        column_items = [item for item in items if str(item.get("column") or _column_for_item(item, split_x)) == column]
        for field in ("reading", "surface", "meaning_ko"):
            bboxes = [_evidence_bbox(item, field) for item in column_items]
            bboxes = [bbox for bbox in bboxes if bbox]
            if not bboxes:
                continue
            templates[column][field] = (
                _median_float([bbox[0] for bbox in bboxes]),
                _median_float([bbox[2] for bbox in bboxes]),
            )
    return templates


def _column_for_item(item: dict, split_x: float) -> str:
    bbox = _valid_bbox(item.get("row_bbox")) or _valid_bbox(item.get("bbox"))
    if not bbox:
        return "left"
    return _token_column_for_bbox(bbox, split_x)


def _vocab_item_from_missing_row_regions(
    line: list[OcrToken],
    column: str,
    templates: dict[str, dict[str, tuple[float, float]]],
    page_id: str,
    validator: DictionaryValidator,
    *,
    image_path: Path,
    page_width: int,
    page_height: int,
    preprocessing_hash: str,
    profile_id: str,
    korean_profile_id: str,
    used_token_ids: set[str],
) -> tuple[dict | None, list[OcrToken], dict[str, object]]:
    line_bbox = _bbox_union([token.bbox for token in line])
    if not line_bbox:
        return None, [], {"rejection_reason": "jp_rejected_incomplete_row"}
    row_bbox = _expanded_bbox(line_bbox, page_width, page_height, x_pad=10.0, y_pad=8.0) or line_bbox
    recovered_tokens: list[OcrToken] = []
    attempt: dict[str, object] = {
        "region_strategy": "japanese_missing_row",
        "crop_bbox": row_bbox,
        "ocr_candidates": [],
        "rejected_candidates": [],
    }

    surface = ""
    surface_confidence = 0.0
    surface_token_ids: list[str] = []
    surface_bbox: list[float] | None = None
    surface_region = _missing_row_field_bbox(templates, column, "surface", row_bbox, page_width, page_height)
    surface_result = _recognize_missing_japanese_field(
        image_path=image_path,
        page_id=page_id,
        region_id=f"missing_{column}_{int(row_bbox[1])}_surface",
        field="surface",
        bbox=surface_region,
        page_width=page_width,
        page_height=page_height,
        preprocessing_hash=preprocessing_hash,
        profile_id=profile_id,
        korean_profile_id=korean_profile_id,
        used_token_ids=used_token_ids,
    ) if surface_region else None
    if not surface_result:
        surface_result = _recognize_missing_japanese_field(
            image_path=image_path,
            page_id=page_id,
            region_id=f"missing_{column}_{int(row_bbox[1])}_surface_full",
            field="surface",
            bbox=row_bbox,
            page_width=page_width,
            page_height=page_height,
            preprocessing_hash=preprocessing_hash,
            profile_id=profile_id,
            korean_profile_id=korean_profile_id,
            used_token_ids=used_token_ids,
        )
    if surface_result:
        surface, surface_confidence, surface_token_ids, surface_bbox, surface_tokens, surface_summary = surface_result
        recovered_tokens.extend(surface_tokens)
        attempt["ocr_candidates"].append({"field": "surface", **surface_summary})

    reading_token = next((token for token in line if _reading_like_text(token.text)), None)
    reading = _clean_japanese_field_text(reading_token.text, "reading") if reading_token else ""
    reading_confidence = float(reading_token.confidence) if reading_token else 0.0
    reading_token_ids = [reading_token.id] if reading_token else []
    reading_bbox = list(reading_token.bbox) if reading_token else None
    if not reading:
        reading_region = _missing_row_field_bbox(templates, column, "reading", row_bbox, page_width, page_height)
        reading_result = _recognize_missing_japanese_field(
            image_path=image_path,
            page_id=page_id,
            region_id=f"missing_{column}_{int(row_bbox[1])}_reading",
            field="reading",
            bbox=reading_region or row_bbox,
            page_width=page_width,
            page_height=page_height,
            preprocessing_hash=preprocessing_hash,
            profile_id=profile_id,
            korean_profile_id=korean_profile_id,
            used_token_ids=used_token_ids,
        )
        if not reading_result and reading_region != row_bbox:
            reading_result = _recognize_missing_japanese_field(
                image_path=image_path,
                page_id=page_id,
                region_id=f"missing_{column}_{int(row_bbox[1])}_reading_full",
                field="reading",
                bbox=row_bbox,
                page_width=page_width,
                page_height=page_height,
                preprocessing_hash=preprocessing_hash,
                profile_id=profile_id,
                korean_profile_id=korean_profile_id,
                used_token_ids=used_token_ids,
            )
        if reading_result:
            reading, reading_confidence, reading_token_ids, reading_bbox, reading_tokens, reading_summary = reading_result
            recovered_tokens.extend(reading_tokens)
            attempt["ocr_candidates"].append({"field": "reading", **reading_summary})

    meaning_token = max((token for token in line if _has_hangul_text(token.text)), key=lambda token: token.confidence, default=None)
    meaning = _clean_recovered_korean(meaning_token.text) if meaning_token else ""
    meaning_confidence = float(meaning_token.confidence) if meaning_token else 0.0
    meaning_token_ids = [meaning_token.id] if meaning_token else []
    meaning_bbox = list(meaning_token.bbox) if meaning_token else None
    meaning_provenance = "ocr" if meaning_token else "ko_glyph_ocr"
    if surface and reading:
        meaning_region = _missing_row_field_bbox(templates, column, "meaning_ko", row_bbox, page_width, page_height) or row_bbox
        meaning_result = _safe_recognize_region(
            image_path=image_path,
            page_id=page_id,
            region_id=f"missing_{column}_{int(row_bbox[1])}_meaning",
            field="meaning_ko",
            bbox=_bbox_union([meaning_region, row_bbox]) or meaning_region,
            page_width=page_width,
            page_height=page_height,
            preprocessing_hash=preprocessing_hash,
            strategy="korean_missing_row_meaning",
            profile_id=profile_id,
            korean_profile_id=korean_profile_id,
            provenance="ko_glyph_ocr",
            provider="paddle_korean",
        )
        if meaning_result:
            selection = _selected_recovered_korean_evidence(meaning_result, {"meaning_ko": meaning})
            if selection:
                meaning, meaning_confidence, meaning_token_ids, meaning_bbox = selection
                meaning_provenance = "ko_glyph_ocr"
                recovered_tokens.extend(getattr(meaning_result, "tokens", []))
                attempt["ocr_candidates"].append(
                    {
                        "field": "meaning_ko",
                        "text": meaning,
                        "confidence": meaning_confidence,
                        "token_ids": meaning_token_ids,
                        "bbox": meaning_bbox,
                        "cache": getattr(meaning_result, "cache", {}),
                    }
                )

    if not surface or not reading or not meaning:
        attempt["rejected_candidates"].append(
            {
                "reason": "jp_rejected_incomplete_row",
                "surface": surface,
                "reading": reading,
                "meaning_ko": meaning,
            }
        )
        return None, recovered_tokens, attempt
    if surface_confidence < 0.72 or reading_confidence < 0.72 or meaning_confidence < 0.72:
        attempt["rejected_candidates"].append(
            {
                "reason": "jp_rejected_low_confidence",
                "surface_confidence": surface_confidence,
                "reading_confidence": reading_confidence,
                "meaning_confidence": meaning_confidence,
            }
        )
        return None, recovered_tokens, attempt

    _status, warnings = validator.validate_vocab(surface, reading)
    evidence_token_ids = _unique_string_values([*surface_token_ids, *reading_token_ids, *meaning_token_ids])
    item = {
        "id": new_id("vocab"),
        "type": "vocab_item",
        "surface": surface,
        "reading": reading,
        "meaning_ko": meaning,
        "field_evidence": {
            "surface": {
                "text": surface,
                "bbox": surface_bbox,
                "token_ids": surface_token_ids,
                "confidence": surface_confidence,
                "provenance": "jp_region_ocr",
                "region_strategy": "japanese_missing_row_surface",
            },
            "reading": {
                "text": reading,
                "bbox": reading_bbox,
                "token_ids": reading_token_ids,
                "confidence": reading_confidence,
                "provenance": "jp_region_ocr" if reading_token is None else "ocr",
                "region_strategy": "japanese_missing_row_reading" if reading_token is None else "full_page_ocr",
            },
            "meaning_ko": {
                "text": meaning,
                "bbox": meaning_bbox,
                "token_ids": meaning_token_ids,
                "confidence": meaning_confidence,
                "provenance": meaning_provenance,
            },
        },
        "surface_bbox": surface_bbox,
        "reading_bbox": reading_bbox,
        "meaning_bbox": meaning_bbox,
        "row_bbox": _bbox_union([surface_bbox, reading_bbox, meaning_bbox]) or row_bbox,
        "bbox": _bbox_union([surface_bbox, reading_bbox, meaning_bbox]) or row_bbox,
        "evidence_tokens": evidence_token_ids,
        "column": column,
        "confidence": round(min(surface_confidence, reading_confidence, meaning_confidence), 3),
        "needs_review": bool(warnings),
        "warnings": warnings,
        "page_id": page_id,
        "recovery": {"accepted": True, "source": "japanese_missing_row_regions", "provenance": "jp_region_ocr"},
    }
    attempt["accepted_text"] = {"surface": surface, "reading": reading, "meaning_ko": meaning}
    return item, recovered_tokens, attempt


def _missing_row_field_bbox(
    templates: dict[str, dict[str, tuple[float, float]]],
    column: str,
    field: str,
    row_bbox: list[float],
    page_width: int,
    page_height: int,
) -> list[float] | None:
    field_template = templates.get(column, {}).get(field)
    if not field_template:
        return row_bbox
    x1, x2 = field_template
    height = max(26.0, row_bbox[3] - row_bbox[1])
    return _clip_bbox([x1 - 18.0, row_bbox[1] - height * 0.18, x2 + 18.0, row_bbox[3] + height * 0.18], page_width, page_height)


def _recognize_missing_japanese_field(
    *,
    image_path: Path,
    page_id: str,
    region_id: str,
    field: str,
    bbox: list[float],
    page_width: int,
    page_height: int,
    preprocessing_hash: str,
    profile_id: str,
    korean_profile_id: str,
    used_token_ids: set[str],
) -> tuple[str, float, list[str], list[float], list[OcrToken], dict[str, object]] | None:
    result = _safe_recognize_region(
        image_path=image_path,
        page_id=page_id,
        region_id=region_id,
        field=field,
        bbox=bbox,
        page_width=page_width,
        page_height=page_height,
        preprocessing_hash=preprocessing_hash,
        strategy=f"japanese_missing_row_{field}",
        profile_id=profile_id,
        korean_profile_id=korean_profile_id,
        provenance="jp_region_ocr",
    )
    if not result:
        return None
    selection = _select_japanese_region_evidence(result, field, used_token_ids)
    if not selection:
        return None
    text, confidence, token_ids, evidence_bbox = selection
    summary = {
        "text": text,
        "confidence": confidence,
        "token_ids": token_ids,
        "bbox": evidence_bbox,
        "cache": getattr(result, "cache", {}),
        "raw_text": getattr(result, "text", ""),
    }
    return text, confidence, token_ids, evidence_bbox, list(getattr(result, "tokens", [])), summary


def _vocab_item_workbook_order_key(item: dict) -> tuple[float, float, str]:
    bbox = _valid_bbox(item.get("row_bbox")) or _valid_bbox(item.get("bbox")) or [0.0, 0.0, 0.0, 0.0]
    return (bbox[1], bbox[0], str(item.get("id") or ""))


def _token_has_vocab_text(token: OcrToken) -> bool:
    return _has_japanese_text(token.text) or _has_hangul_text(token.text)


def _line_groups_for_tokens(tokens: list[OcrToken], *, tolerance: float) -> list[list[OcrToken]]:
    lines: list[list[OcrToken]] = []
    for token in sorted(tokens, key=lambda value: ((value.bbox[1] + value.bbox[3]) / 2, value.bbox[0])):
        cy = (token.bbox[1] + token.bbox[3]) / 2
        for line in lines:
            line_cy = sum((item.bbox[1] + item.bbox[3]) / 2 for item in line) / len(line)
            if abs(cy - line_cy) <= tolerance:
                line.append(token)
                break
        else:
            lines.append([token])
    return [sorted(line, key=lambda token: token.bbox[0]) for line in lines]


def _row_overlaps_existing(line: list[OcrToken], bbox: list[float]) -> bool:
    line_bbox = _bbox_union([token.bbox for token in line])
    if not line_bbox:
        return False
    overlap = max(0.0, min(line_bbox[3], bbox[3]) - max(line_bbox[1], bbox[1]))
    return overlap / max(1.0, line_bbox[3] - line_bbox[1]) > 0.55


def _vocab_item_from_line_tokens(line: list[OcrToken], page_id: str, validator: DictionaryValidator) -> dict | None:
    japanese = [token for token in line if _has_japanese_text(token.text)]
    korean = [token for token in line if _has_hangul_text(token.text)]
    if len(japanese) < 2 or not korean:
        return None
    surface_token = _surface_token_from_line(japanese)
    reading_token = _reading_token_from_line(japanese, surface_token)
    meaning_token = max(korean, key=lambda token: (token.confidence, token.bbox[2] - token.bbox[0]))
    if not surface_token or not reading_token:
        return None
    surface = _clean_japanese_field_text(surface_token.text, "surface")
    reading = _clean_japanese_field_text(reading_token.text, "reading")
    meaning = _clean_recovered_korean(meaning_token.text)
    if not surface or not reading or not meaning:
        return None
    _status, warnings = validator.validate_vocab(surface, reading)
    token_ids = [surface_token.id, reading_token.id, meaning_token.id]
    confidence = min(surface_token.confidence, reading_token.confidence, meaning_token.confidence)
    return {
        "id": new_id("vocab"),
        "type": "vocab_item",
        "surface": surface,
        "reading": reading,
        "meaning_ko": meaning,
        "field_evidence": {
            "surface": {"text": surface, "bbox": surface_token.bbox, "token_ids": [surface_token.id], "confidence": surface_token.confidence, "provenance": "jp_region_ocr"},
            "reading": {"text": reading, "bbox": reading_token.bbox, "token_ids": [reading_token.id], "confidence": reading_token.confidence, "provenance": "jp_region_ocr"},
            "meaning_ko": {"text": meaning, "bbox": meaning_token.bbox, "token_ids": [meaning_token.id], "confidence": meaning_token.confidence, "provenance": meaning_token.source or "ocr"},
        },
        "surface_bbox": surface_token.bbox,
        "reading_bbox": reading_token.bbox,
        "meaning_bbox": meaning_token.bbox,
        "row_bbox": _bbox_union([token.bbox for token in line]),
        "bbox": _bbox_union([surface_token.bbox, reading_token.bbox, meaning_token.bbox]),
        "evidence_tokens": token_ids,
        "confidence": round(confidence, 3),
        "needs_review": bool(warnings),
        "warnings": warnings,
        "page_id": page_id,
    }


def _surface_token_from_line(tokens: list[OcrToken]) -> OcrToken | None:
    candidates = [token for token in tokens if _clean_japanese_field_text(token.text, "surface") and not _reading_like_text(token.text)]
    if candidates:
        return min(candidates, key=lambda token: token.bbox[0])
    katakana = [token for token in tokens if re.search(r"[ァ-ヺー]", token.text)]
    return min(katakana, key=lambda token: token.bbox[0]) if katakana else None


def _reading_token_from_line(tokens: list[OcrToken], surface_token: OcrToken | None) -> OcrToken | None:
    candidates = [token for token in tokens if token is not surface_token and _reading_like_text(token.text)]
    return min(candidates, key=lambda token: abs(token.bbox[0] - (surface_token.bbox[2] if surface_token else 0.0))) if candidates else None


def _recover_korean_residual_glyph_items(
    items: list[dict],
    all_tokens: list[OcrToken],
    image_path: Path,
    page_id: str,
    page_width: int,
    page_height: int,
    preprocessing_hash: str,
    profile_id: str,
    korean_profile_id: str,
) -> tuple[list[dict], list[OcrToken], dict[str, object]]:
    recovered_tokens: list[OcrToken] = []
    attempts: list[dict[str, object]] = []
    counts = {
        "ko_glyph_attempted": 0,
        "ko_glyph_accepted": 0,
        "ko_glyph_rejected_low_confidence": 0,
        "ko_glyph_rejected_no_hangul": 0,
        "ko_glyph_rejected_no_region": 0,
        "ko_glyph_rejected_cross_column": 0,
        "ko_glyph_rejected_duplicate_token": 0,
        "ko_glyph_rejected_weak_overlap": 0,
        "ko_glyph_rejected_template_unavailable": 0,
    }
    anchors = _meaning_anchor_bboxes(items)
    used_token_ids = _used_item_token_ids(items)
    for item in sorted(items, key=_korean_residual_priority):
        if counts["ko_glyph_attempted"] >= OCR_RECOVERY_MAX_FIELDS:
            break
        if _korean_uncertainty_bucket(item) is None:
            continue
        bbox = _meaning_recovery_bbox(item, anchors, page_width, page_height, prefer_existing=True) or _meaning_recovery_bbox(
            item,
            anchors,
            page_width,
            page_height,
            prefer_existing=False,
        )
        if not bbox:
            continue
        counts["ko_glyph_attempted"] += 1
        result = _safe_recognize_region(
            image_path=image_path,
            page_id=page_id,
            region_id=str(item.get("id") or f"ko_glyph_{counts['ko_glyph_attempted']}"),
            field="meaning_ko",
            bbox=bbox,
            page_width=page_width,
            page_height=page_height,
            preprocessing_hash=preprocessing_hash,
            strategy="korean_residual_glyph",
            profile_id=profile_id,
            korean_profile_id=korean_profile_id,
            provenance="ko_glyph_ocr",
            provider="paddle_korean",
        )
        original_meaning = item.get("meaning_ko")
        attempt: dict[str, object] = {"source_id": item.get("id"), "surface": item.get("surface"), "current": original_meaning, "bbox": bbox, "accepted": False}
        if not result:
            counts["ko_glyph_rejected_no_region"] += 1
            attempt["warnings"] = ["Region OCR unavailable; skipped residual glyph recovery."]
            attempts.append(attempt)
            continue
        else:
            attempt["text"] = getattr(result, "text", "")
            attempt["cache"] = getattr(result, "cache", {})
        selection = _selected_recovered_korean_evidence(result, item)
        if not selection:
            counts["ko_glyph_rejected_no_hangul"] += 1
            attempts.append(attempt)
            continue
        text, confidence, token_ids, evidence_bbox = selection
        if any(token_id in used_token_ids for token_id in token_ids):
            counts["ko_glyph_rejected_duplicate_token"] += 1
            attempts.append(attempt)
            continue
        if confidence < 0.74:
            counts["ko_glyph_rejected_low_confidence"] += 1
            attempts.append(attempt)
            continue
        _apply_vocab_field_recovery(item, "meaning_ko", text, confidence, token_ids, evidence_bbox, "ko_glyph_ocr")
        item["meaning_bbox"] = evidence_bbox
        item["recovery"] = {"accepted": True, "source": "ko_residual_glyph", "provenance": "ko_glyph_ocr", "cache": getattr(result, "cache", {})}
        if result:
            recovered_tokens.extend(result.tokens)
        used_token_ids.update(token_ids)
        counts["ko_glyph_accepted"] += 1
        attempt["accepted"] = True
        attempt["accepted_text"] = text
        attempts.append(attempt)
    return items, recovered_tokens, {
        "schema_version": 1,
        "kind": "korean_residual_glyph_recovery",
        "attempted": counts["ko_glyph_attempted"],
        "accepted": counts["ko_glyph_accepted"],
        "counts": {key: value for key, value in counts.items() if value},
        "attempts": attempts,
    }

def _korean_residual_priority(item: dict) -> tuple[int, float]:
    base_priority, confidence = _korean_recovery_priority(item)
    return (base_priority + 1, confidence)


def _recover_mcq_source_items(
    items: list[dict],
    image_path: Path,
    page_id: str,
    page_width: int,
    page_height: int,
    preprocessing_hash: str,
    profile_id: str,
    korean_profile_id: str,
    components: frozenset[str],
) -> tuple[list[dict], list[OcrToken], dict[str, object]]:
    recovered_tokens: list[OcrToken] = []
    attempts: list[dict[str, object]] = []
    accepted = 0
    for item in items:
        source_fields = item.get("source_fields")
        if not isinstance(source_fields, dict):
            source_fields = _strict_mcq_source_fields_from_item(item)
            item["source_fields"] = source_fields
        item.setdefault(
            "semantic_fields",
            {
                "sentence": item.get("sentence", ""),
                "target": item.get("target", ""),
                "choices": list(item.get("choices") if isinstance(item.get("choices"), list) else []),
                "correct_answer": item.get("correct_answer", ""),
                "correct_choice_no": item.get("correct_choice_no"),
                "answer_source": item.get("answer_source", ""),
            },
        )
    source_rebuild_accepted = _repair_mcq_source_fields(items) if "mcq_source_rebuild_v1" in components else 0
    if "mcq_choice_band_ocr_v1" in components and len(attempts) < OCR_RECOVERY_MAX_REGIONS:
        bbox = _mcq_answer_strip_bbox(items, page_width, page_height)
        result = _safe_recognize_region(
            image_path=image_path,
            page_id=page_id,
            region_id=f"{page_id}_answer_strip",
            field="answer_source",
            bbox=bbox,
            page_width=page_width,
            page_height=page_height,
            preprocessing_hash=preprocessing_hash,
            strategy="mcq_answer_strip",
            profile_id=profile_id,
            korean_profile_id=korean_profile_id,
            provenance="answer_strip_ocr",
            provider=None,
        )
        if not result:
            attempts.append(
                {
                    "kind": "answer_strip",
                    "region_id": f"{page_id}_answer_strip",
                    "skipped": True,
                    "reason": "region_ocr_unavailable",
                    "candidate_mutation": False,
                    "accepted": 0,
                }
            )
        else:
            image_answer_map, image_tokens, image_warnings = _parse_answer_strip_image(
                image_path=image_path,
                page_id=page_id,
                bbox=result.bbox,
            )
            recovered_tokens.extend(result.tokens)
            recovered_tokens.extend(image_tokens)
            answer_map = parse_answer_strip_v5(result.tokens, page_height, existing=parse_answer_strip_text(result.text))
            answer_map = _merge_answer_maps(answer_map, image_answer_map)
            evidence_tokens = [*result.tokens, *image_tokens]
            evidence_confidence = _mean_token_confidence(evidence_tokens) if evidence_tokens else result.confidence
            accepted = _apply_mcq_answer_strip_source_fields(
                items,
                answer_map,
                result,
                evidence_tokens=evidence_tokens,
                confidence=evidence_confidence,
            )
            attempts.append(
                {
                    "kind": "answer_strip",
                    "region_id": f"{page_id}_answer_strip",
                    "token_count": len(result.tokens) + len(image_tokens),
                    "text": result.text,
                    "parsed_answer_map": {str(key): value for key, value in sorted(answer_map.items())},
                    "image_answer_map": {str(key): value for key, value in sorted(image_answer_map.items())},
                    "cache": result.cache,
                    "candidate_mutation": accepted > 0,
                    "accepted": accepted,
                    "warnings": [*result.warnings, *image_warnings],
                }
            )
    for index, item in enumerate(items):
        source_fields = item.get("source_fields")
        if not isinstance(source_fields, dict):
            continue
        if "mcq_choice_band_ocr_v1" not in components or len(attempts) >= OCR_RECOVERY_MAX_REGIONS:
            continue
        if _mcq_source_complete(source_fields):
            continue
        bbox = _expanded_bbox(item.get("bbox"), page_width, page_height, x_pad=10.0, y_pad=8.0)
        if not bbox:
            continue
        result = _safe_recognize_region(
            image_path=image_path,
            page_id=page_id,
            region_id=str(item.get("id") or f"mcq_{index + 1}"),
            field="sentence",
            bbox=bbox,
            page_width=page_width,
            page_height=page_height,
            preprocessing_hash=preprocessing_hash,
            strategy="mcq_choice_band",
            profile_id=profile_id,
            korean_profile_id=korean_profile_id,
            provenance="region_ocr",
            provider=None,
        )
        if not result:
            continue
        recovered_tokens.extend(result.tokens)
        attempts.append(
            {
                "kind": "question_band",
                "source_id": item.get("id"),
                "question_no": item.get("question_no"),
                "token_count": len(result.tokens),
                "text": result.text,
                "cache": result.cache,
                "candidate_mutation": False,
            }
        )
    return items, recovered_tokens, {
        "schema_version": 1,
        "kind": "mcq_source_recovery",
        "source_rebuild_enabled": "mcq_source_rebuild_v1" in components,
        "choice_band_ocr_enabled": "mcq_choice_band_ocr_v1" in components,
        "attempted": len(attempts),
        "accepted": accepted + source_rebuild_accepted,
        "source_rebuild_accepted": source_rebuild_accepted,
        "attempts": attempts,
    }


def _recover_v2_mcq_source_items(
    items: list[dict],
    image_path: Path,
    page_id: str,
    page_width: int,
    page_height: int,
    preprocessing_hash: str,
    profile_id: str,
    korean_profile_id: str,
    components: frozenset[str],
) -> tuple[list[dict], list[OcrToken], dict[str, object]]:
    recovered_tokens: list[OcrToken] = []
    diagnostics: dict[str, object] = {}
    if "mcq_prompt_line_ocr_v1" in components:
        items, tokens, prompt_diagnostics = _recover_mcq_prompt_lines(
            items,
            image_path,
            page_id,
            page_width,
            page_height,
            preprocessing_hash,
            profile_id,
            korean_profile_id,
        )
        recovered_tokens.extend(tokens)
        diagnostics = _append_recovery_diagnostics(diagnostics, "mcq_prompt_line_recovery", prompt_diagnostics)
    if "mcq_choice_glyph_v1" in components:
        items, tokens, choice_diagnostics = _recover_mcq_choice_glyphs(
            items,
            image_path,
            page_id,
            page_width,
            page_height,
            preprocessing_hash,
            profile_id,
            korean_profile_id,
        )
        recovered_tokens.extend(tokens)
        diagnostics = _append_recovery_diagnostics(diagnostics, "mcq_choice_glyph_recovery", choice_diagnostics)
    return items, recovered_tokens, diagnostics


def _recover_mcq_prompt_lines(
    items: list[dict],
    image_path: Path,
    page_id: str,
    page_width: int,
    page_height: int,
    preprocessing_hash: str,
    profile_id: str,
    korean_profile_id: str,
) -> tuple[list[dict], list[OcrToken], dict[str, object]]:
    recovered_tokens: list[OcrToken] = []
    attempts: list[dict[str, object]] = []
    counts = {
        "prompt_line_attempted": 0,
        "prompt_line_accepted": 0,
        "prompt_line_rejected_low_confidence": 0,
        "prompt_line_rejected_contains_choices": 0,
        "prompt_line_rejected_missing_target": 0,
        "prompt_line_rejected_not_sentence_like": 0,
        "prompt_line_rejected_no_tokens": 0,
    }
    for index, item in enumerate(items):
        if item.get("question_type") != "reading_mcq":
            continue
        source_fields = item.get("source_fields")
        if not isinstance(source_fields, dict):
            source_fields = _strict_mcq_source_fields_from_item(item)
            item["source_fields"] = source_fields
        target = str(source_fields.get("target") or item.get("target") or "")
        if counts["prompt_line_attempted"] >= OCR_RECOVERY_MAX_REGIONS:
            continue
        bbox = _mcq_prompt_line_bbox(item, page_width, page_height)
        if not bbox:
            continue
        counts["prompt_line_attempted"] += 1
        result = _safe_recognize_region(
            image_path=image_path,
            page_id=page_id,
            region_id=str(item.get("id") or f"prompt_{index + 1}"),
            field="sentence",
            bbox=bbox,
            page_width=page_width,
            page_height=page_height,
            preprocessing_hash=preprocessing_hash,
            strategy="mcq_prompt_line",
            profile_id=profile_id,
            korean_profile_id=korean_profile_id,
            provenance="prompt_line_ocr",
        )
        attempt: dict[str, object] = {"source_id": item.get("id"), "question_no": item.get("question_no"), "bbox": bbox, "accepted": False}
        candidate = _clean_mcq_prompt_line_text(str(getattr(result, "text", "") or ""))
        repaired = _repair_mcq_prompt_sentence_v2(candidate, target)
        candidate = repaired if repaired != candidate else candidate
        attempt["text"] = candidate
        if result:
            attempt["cache"] = getattr(result, "cache", {})
        if _prompt_line_contains_choices(candidate):
            counts["prompt_line_rejected_contains_choices"] += 1
            attempts.append(attempt)
            continue
        tokens = list(getattr(result, "tokens", []) if result else [])
        if not tokens:
            counts["prompt_line_rejected_no_tokens"] += 1
            attempts.append(attempt)
            continue
        confidence = max(
            float(getattr(result, "confidence", 0.0) or 0.0),
            max([float(getattr(token, "confidence", 0.0) or 0.0) for token in tokens] or [0.0]),
        )
        if confidence < 0.72:
            counts["prompt_line_rejected_low_confidence"] += 1
            attempts.append(attempt)
            continue
        if not _japanese_sentence_like(candidate):
            counts["prompt_line_rejected_not_sentence_like"] += 1
            attempts.append(attempt)
            continue
        if target and target not in candidate and not _source_target_bbox_aligns_prompt(item):
            counts["prompt_line_rejected_missing_target"] += 1
            attempts.append(attempt)
            continue
        _set_mcq_source_field(item, "sentence", candidate, "prompt_line_ocr", result=result)
        recovered_tokens.extend(tokens)
        counts["prompt_line_accepted"] += 1
        attempt["accepted"] = True
        attempts.append(attempt)
    return items, recovered_tokens, {
        "schema_version": 1,
        "kind": "mcq_prompt_line_recovery",
        "attempted": counts["prompt_line_attempted"],
        "accepted": counts["prompt_line_accepted"],
        "counts": {key: value for key, value in counts.items() if value},
        "attempts": attempts,
    }


def _recover_mcq_choice_glyphs(
    items: list[dict],
    image_path: Path,
    page_id: str,
    page_width: int,
    page_height: int,
    preprocessing_hash: str,
    profile_id: str,
    korean_profile_id: str,
) -> tuple[list[dict], list[OcrToken], dict[str, object]]:
    recovered_tokens: list[OcrToken] = []
    attempts: list[dict[str, object]] = []
    counts = {
        "choice_glyph_attempted": 0,
        "choice_glyph_accepted": 0,
        "choice_glyph_rejected_incomplete_choices": 0,
        "choice_glyph_rejected_duplicate_unsupported": 0,
        "choice_glyph_rejected_low_confidence": 0,
        "choice_glyph_rejected_no_crop": 0,
        "choice_glyph_rejected_no_change": 0,
        "choice_glyph_rejected_semantic_contamination": 0,
    }
    for index, item in enumerate(items):
        if item.get("question_type") != "spelling_mcq":
            continue
        semantic_choices_before = list(item.get("choices") if isinstance(item.get("choices"), list) else [])
        semantic_answer_before = item.get("correct_answer")
        semantic_choice_no_before = item.get("correct_choice_no")
        source_fields = item.get("source_fields")
        if not isinstance(source_fields, dict):
            source_fields = _strict_mcq_source_fields_from_item(item)
            item["source_fields"] = source_fields
        choices = source_fields.get("choices")
        if not isinstance(choices, list) or len(choices) != 4:
            counts["choice_glyph_rejected_incomplete_choices"] += 1
            continue
        counts["choice_glyph_attempted"] += 1
        attempt: dict[str, object] = {"source_id": item.get("id"), "question_no": item.get("question_no"), "accepted": False, "raw_choices": choices}
        choice_bboxes = _mcq_choice_glyph_bboxes(item, page_width, page_height)
        if not choice_bboxes:
            counts["choice_glyph_rejected_no_crop"] += 1
            attempts.append(attempt)
            continue
        choice_results = []
        for choice_no, bbox in enumerate(choice_bboxes, start=1):
            if len(choice_results) >= OCR_RECOVERY_MAX_REGIONS * 4:
                break
            result = _safe_recognize_region(
                image_path=image_path,
                page_id=page_id,
                region_id=f"{item.get('id') or f'choice_glyph_{index + 1}'}_{choice_no}",
                field=f"choice_{choice_no}",
                bbox=bbox,
                page_width=page_width,
                page_height=page_height,
                preprocessing_hash=preprocessing_hash,
                strategy=f"mcq_choice_glyph_{choice_no}",
                profile_id=profile_id,
                korean_profile_id=korean_profile_id,
                provenance="choice_glyph_ocr",
            )
            choice_results.append(result)
        attempt["crops"] = [
            {
                "choice_no": choice_no,
                "bbox": bbox,
                "cache": getattr(result, "cache", {}) if result else {},
                "text": getattr(result, "text", "") if result else "",
            }
            for choice_no, (bbox, result) in enumerate(zip(choice_bboxes, choice_results), start=1)
        ]
        if len(choice_results) != 4 or any(not list(getattr(result, "tokens", []) if result else []) for result in choice_results):
            counts["choice_glyph_rejected_no_crop"] += 1
            attempts.append(attempt)
            continue
        repaired, token_groups, glyph_confidence = _choice_glyph_choices_from_results(choice_results)
        if len(repaired) != 4 or len(token_groups) != 4:
            counts["choice_glyph_rejected_incomplete_choices"] += 1
            attempts.append(attempt)
            continue
        if repaired == [str(choice) for choice in choices]:
            counts["choice_glyph_rejected_no_change"] += 1
            attempts.append(attempt)
            continue
        if len(set(repaired)) < 4 and not _choice_duplicate_text_is_crop_supported(repaired, token_groups):
            counts["choice_glyph_rejected_duplicate_unsupported"] += 1
            attempts.append(attempt)
            continue
        if glyph_confidence < 0.76:
            counts["choice_glyph_rejected_low_confidence"] += 1
            attempts.append(attempt)
            continue
        source_fields["choices"] = repaired
        _sync_source_correct_answer(source_fields)
        recovered_tokens.extend(token for group in token_groups for token in group)
        _mark_choice_glyph_evidence(item, repaired, choices, token_groups, glyph_confidence)
        field_evidence = item.get("field_evidence") if isinstance(item.get("field_evidence"), dict) else {}
        if isinstance(field_evidence, dict) and isinstance(field_evidence.get("choices"), dict):
            field_evidence["choices"]["provenance"] = "choice_glyph_ocr"
            field_evidence["choices"]["normalization_strategy"] = "mcq_choice_glyph_v1"
        if (
            semantic_choices_before != list(item.get("choices") if isinstance(item.get("choices"), list) else [])
            or semantic_answer_before != item.get("correct_answer")
            or semantic_choice_no_before != item.get("correct_choice_no")
        ):
            counts["choice_glyph_rejected_semantic_contamination"] += 1
            source_fields["choices"] = choices
            _sync_source_correct_answer(source_fields)
            attempts.append(attempt)
            continue
        counts["choice_glyph_accepted"] += 1
        attempt["accepted"] = True
        attempt["accepted_choices"] = repaired
        attempts.append(attempt)
    return items, recovered_tokens, {
        "schema_version": 1,
        "kind": "mcq_choice_glyph_recovery",
        "attempted": counts["choice_glyph_attempted"],
        "accepted": counts["choice_glyph_accepted"],
        "counts": {key: value for key, value in counts.items() if value},
        "attempts": attempts,
    }


def _repair_mcq_prompt_sentence_v2(sentence: str, target: str = "") -> str:
    del target
    repaired = sentence.strip()
    repaired = re.sub(r"^[円子日しよさい、な\s]+(?=(きょう|ともだち|まいにち|じぶん))", "", repaired)
    repaired = _normalize_prompt_line_visual_noise(repaired)
    repaired = _trim_after_first_japanese_sentence_predicate(repaired)
    if repaired and not repaired.endswith(("。", "？", "?", "！", "!")) and re.search(r"(です|ます|だ)$", repaired):
        repaired = f"{repaired}。"
    return repaired if repaired != sentence.strip() and _japanese_sentence_like(repaired) else sentence


def _normalize_prompt_line_visual_noise(text: str) -> str:
    text = re.sub(r"^[「]?皆(?=ょう[日目背])", "日", text)
    text = re.sub(r"(?<=[よょ]う)[目背]", "日", text)
    text = re.sub(r"妹(?=みです)", "休", text)
    text = re.sub(r"(?<=外)囲", "国", text)
    text = re.sub(r"きれ(?=です)", "きれい", text)
    return re.sub(r"[正店]{2,}$", "", text)


def _trim_after_first_japanese_sentence_predicate(text: str) -> str:
    match = re.search(r"(です|ます|(?<=い)だ)", text)
    if not match:
        return text
    trimmed = text[: match.end(1)]
    return trimmed if _japanese_sentence_like(trimmed) else text


def _drop_target_prefix_noise(text: str, target: str) -> str:
    if not target or target not in text:
        return text
    prefix, _, suffix = text.partition(target)
    particle_index = max(prefix.rfind("は"), prefix.rfind("が"), prefix.rfind("を"))
    if particle_index < 0 or particle_index == len(prefix) - 1:
        return text
    between = prefix[particle_index + 1 :]
    if between and not re.search(r"[ぁ-ん]", between):
        return f"{prefix[: particle_index + 1]}{target}{suffix}"
    return text


def _source_target_bbox_aligns_prompt(item: dict) -> bool:
    target_bbox = _evidence_bbox(item, "target")
    sentence_bbox = _evidence_bbox(item, "sentence") or _valid_bbox(item.get("bbox"))
    if not target_bbox or not sentence_bbox:
        return False
    return _bbox_iou(target_bbox, sentence_bbox) > 0.0 or (
        sentence_bbox[0] <= target_bbox[0] <= sentence_bbox[2]
        and sentence_bbox[1] - 8.0 <= target_bbox[1] <= sentence_bbox[3] + 8.0
    )


def _choice_glyph_choices_from_results(results: list[object | None]) -> tuple[list[str], list[list[OcrToken]], float]:
    choices: list[str] = []
    token_groups: list[list[OcrToken]] = []
    confidences: list[float] = []
    for result in results[:4]:
        result_tokens = list(getattr(result, "tokens", []) if result else [])
        if not result_tokens:
            return [], [], 0.0
        text = _clean_choice_glyph_result_text(str(getattr(result, "text", "") or " ".join(str(token.text) for token in result_tokens)))
        if not text or not _has_japanese_text(text):
            return [], [], 0.0
        result_confidence = max(
            [float(getattr(result, "confidence", 0.0) or 0.0), *[float(getattr(token, "confidence", 0.0) or 0.0) for token in result_tokens]]
        )
        choices.append(text)
        token_groups.append(result_tokens)
        confidences.append(min(0.92, result_confidence))
    return choices, token_groups, round(sum(confidences) / len(confidences), 3) if confidences else 0.0


def _clean_choice_glyph_result_text(text: str) -> str:
    text = re.sub(r"^[\s①②③④1-4.、:：)）(（-]+", "", text.strip())
    text = re.split(r"[\s①②③④]\s*", text, maxsplit=1)[0]
    text = re.sub(r"\s+", "", text)
    return text.strip("「」[]()（）")


def _choice_duplicate_text_is_crop_supported(choices: list[str], token_groups: list[list[OcrToken]]) -> bool:
    for index, choice in enumerate(choices):
        if choices.count(choice) <= 1:
            continue
        token_text = "".join(token.text for token in token_groups[index])
        if choice not in token_text:
            return False
    return True


def _mark_choice_glyph_evidence(
    item: dict,
    choices: list[str],
    raw_choices: list[object],
    token_groups: list[list[OcrToken]],
    confidence: float,
) -> None:
    field_evidence = item.get("field_evidence")
    if not isinstance(field_evidence, dict):
        field_evidence = {}
        item["field_evidence"] = field_evidence
    for index, tokens in enumerate(token_groups, start=1):
        bbox = _bbox_union([token.bbox for token in tokens])
        field_evidence[f"choice_{index}"] = {
            "bbox": bbox,
            "token_ids": [token.id for token in tokens],
            "text": choices[index - 1],
            "confidence": max([token.confidence for token in tokens], default=confidence),
            "provenance": "choice_glyph_ocr",
            "region_strategy": f"mcq_choice_glyph_{index}",
        }
    field_evidence["choices"] = {
        "bbox": _bbox_union([token.bbox for tokens in token_groups for token in tokens]),
        "token_ids": [token.id for tokens in token_groups for token in tokens],
        "text": json.dumps(choices, ensure_ascii=False),
        "raw_text": json.dumps(raw_choices, ensure_ascii=False),
        "confidence": confidence,
        "provenance": "choice_glyph_ocr",
        "normalization_strategy": "mcq_choice_glyph_v1",
        "region_strategy": "mcq_choice_glyph_v1",
    }


def _set_mcq_source_field(
    item: dict,
    field: str,
    text: object,
    provenance: str,
    *,
    result: object | None = None,
    raw_text: object | None = None,
) -> None:
    source_fields = item.get("source_fields")
    if not isinstance(source_fields, dict):
        source_fields = _strict_mcq_source_fields_from_item(item)
        item["source_fields"] = source_fields
    previous = source_fields.get(field)
    source_fields[field] = text
    field_evidence = item.get("field_evidence")
    if not isinstance(field_evidence, dict):
        field_evidence = {}
        item["field_evidence"] = field_evidence
    tokens = list(getattr(result, "tokens", []) if result else [])
    evidence = field_evidence.get(field) if isinstance(field_evidence.get(field), dict) else {}
    token_ids = [token.id for token in tokens] or evidence.get("token_ids", [])
    bbox = (_bbox_union([token.bbox for token in tokens]) or getattr(result, "bbox", None)) if result else evidence.get("bbox")
    field_evidence[field] = {
        **evidence,
        "text": text if isinstance(text, str) else json.dumps(text, ensure_ascii=False),
        "raw_text": raw_text if raw_text is not None else previous,
        "bbox": bbox,
        "token_ids": token_ids if isinstance(token_ids, list) else [],
        "confidence": getattr(result, "confidence", evidence.get("confidence", 0.0)) if result else evidence.get("confidence", 0.0),
        "provenance": provenance,
        "normalization_strategy": "mcq_prompt_line_ocr_v1",
    }


def _mcq_prompt_line_bbox(item: dict, page_width: int, page_height: int) -> list[float] | None:
    sentence_bbox = _evidence_bbox(item, "sentence")
    if sentence_bbox:
        return _expanded_bbox(sentence_bbox, page_width, page_height, x_pad=8.0, y_pad=5.0)
    item_bbox = _valid_bbox(item.get("bbox"))
    choice_bboxes = _mcq_choice_evidence_bboxes(item)
    if not item_bbox:
        return None
    bottom = min([bbox[1] for bbox in choice_bboxes], default=item_bbox[1] + (item_bbox[3] - item_bbox[1]) * 0.45)
    return _clip_bbox([item_bbox[0], item_bbox[1], item_bbox[2], bottom], page_width, page_height)


def _mcq_choice_glyph_region_bbox(item: dict, page_width: int, page_height: int) -> list[float] | None:
    choice_bboxes = _mcq_choice_evidence_bboxes(item)
    return _expanded_bbox(_bbox_union(choice_bboxes), page_width, page_height, x_pad=5.0, y_pad=5.0) if choice_bboxes else None


def _mcq_choice_glyph_bboxes(item: dict, page_width: int, page_height: int) -> list[list[float]] | None:
    choice_bboxes = _mcq_choice_evidence_bboxes(item)
    if len(choice_bboxes) == 4 and _choice_bboxes_are_isolated(choice_bboxes):
        return [_expanded_bbox(bbox, page_width, page_height, x_pad=4.0, y_pad=4.0) for bbox in choice_bboxes]
    item_bbox = _valid_bbox(item.get("bbox"))
    if not item_bbox:
        union = _bbox_union(choice_bboxes)
        item_bbox = _valid_bbox(union)
    if not item_bbox:
        evidence = item.get("field_evidence")
        choices_evidence = evidence.get("choices") if isinstance(evidence, dict) else None
        item_bbox = _valid_bbox(choices_evidence.get("bbox") if isinstance(choices_evidence, dict) else None)
    if not item_bbox:
        return None
    if choice_bboxes:
        tops = sorted(bbox[1] for bbox in choice_bboxes)
        bottoms = sorted(bbox[3] for bbox in choice_bboxes)
        top = max(0.0, tops[len(tops) // 2] - 6.0)
        bottom = min(float(page_height), bottoms[len(bottoms) // 2] + 8.0)
    else:
        height = item_bbox[3] - item_bbox[1]
        top = item_bbox[1] + height * 0.45
        bottom = item_bbox[3] + 6.0
    width = item_bbox[2] - item_bbox[0]
    if width <= 0 or bottom <= top:
        return None
    cells: list[list[float]] = []
    for index in range(4):
        left = item_bbox[0] + width * (index / 4.0)
        right = item_bbox[0] + width * ((index + 1) / 4.0)
        cells.append(_clip_bbox([left - 4.0, top, right + 4.0, bottom], page_width, page_height))
    return cells


def _choice_bboxes_are_isolated(bboxes: list[list[float]]) -> bool:
    if len(bboxes) != 4:
        return False
    centers = [((bbox[0] + bbox[2]) / 2.0) for bbox in bboxes]
    if centers != sorted(centers):
        return False
    union = _bbox_union(bboxes)
    if not union:
        return False
    union_width = max(1.0, union[2] - union[0])
    for bbox in bboxes:
        if bbox[2] - bbox[0] > union_width * 0.45:
            return False
    for left, right in zip(bboxes, bboxes[1:]):
        if _bbox_iou(left, right) > 0.15:
            return False
    return True


def _clean_mcq_prompt_line_text(text: str) -> str:
    text = re.sub(r"^[\s①-⑳.、;；:)）(（円日子-]+", "", text.strip())
    text = re.split(r"[①②③④]\s*", text, maxsplit=1)[0]
    text = re.sub(r"[-‐‑‒–—―]+", "", text)
    text = text.replace("」", "").replace("「", "").replace(";", "").replace("；", "")
    for suffix in ("です日", "ます日", "ました日"):
        if text.endswith(suffix):
            text = text[: -1]
            break
    if text and not text.endswith(("。", "？", "?", "！", "!")) and re.search(r"(です|ます|ました|でしょう|だ)$", text):
        text = f"{text}。"
    return text.strip()


def _prompt_line_contains_choices(text: str) -> bool:
    return bool(re.search(r"[①②③④].+[①②③④]", text) or re.search(r"\b[1-4]\s*[^。？?!]+[1-4]\s*", text))


def _japanese_sentence_like(text: str) -> bool:
    return bool(text and _has_japanese_text(text) and re.search(r"[ぁ-んァ-ヺ一-龯]", text) and len(text) >= 6)


def _mcq_answer_strip_bbox(items: list[dict], page_width: int, page_height: int) -> list[float]:
    choice_bottoms = [
        bbox[3]
        for item in items
        for bbox in _mcq_choice_evidence_bboxes(item)
    ]
    fallback_bottoms = [bbox[3] for item in items if (bbox := _valid_bbox(item.get("bbox")))]
    anchor_bottom = max(choice_bottoms or fallback_bottoms, default=page_height * 0.72)
    top = max(page_height * 0.76, anchor_bottom + 8.0)
    if page_height - top < 40.0:
        top = page_height * 0.76
    bottom = min(float(page_height), top + max(80.0, page_height * 0.055))
    return [page_width * 0.10, max(0.0, top), page_width * 0.96, bottom]


def _repair_mcq_source_fields(items: list[dict]) -> int:
    accepted = 0
    for item in items:
        source_fields = item.get("source_fields")
        if not isinstance(source_fields, dict):
            continue
        choices = source_fields.get("choices")
        if isinstance(choices, list):
            repaired_choices = _repair_mcq_source_choices(item, choices)
            if repaired_choices != choices:
                source_fields["choices"] = repaired_choices
                _sync_source_correct_answer(source_fields)
                _mark_source_rebuild_evidence(item, "choices", repaired_choices, choices)
                accepted += 1
        sentence = str(source_fields.get("sentence") or "")
        repaired_sentence = _repair_mcq_source_sentence(sentence)
        if repaired_sentence != sentence:
            source_fields["sentence"] = repaired_sentence
            _mark_source_rebuild_evidence(item, "sentence", repaired_sentence, sentence)
            accepted += 1
    return accepted


def _repair_mcq_source_choices(item: dict, choices: list[object]) -> list[object]:
    text_choices = [str(choice) for choice in choices]
    question_type = str(item.get("question_type") or "")
    if question_type == "reading_mcq" or _looks_like_reading_choice_set(text_choices):
        return _repair_reading_mcq_source_choices(text_choices)
    return _repair_spelling_mcq_source_choices(text_choices)


def _repair_reading_mcq_source_choices(choices: list[str]) -> list[str]:
    expanded: list[str] = []
    for choice in choices:
        runs = re.findall(r"[ぁ-んー]+", choice)
        if len(choices) < 4 and len(runs) >= 2 and len(choices) + len(runs) - 1 <= 4:
            expanded.extend(runs)
            continue
        cleaned = choice
        if re.search(r"[？?:：;；]", choice) and runs:
            cleaned = max(runs, key=len)
        expanded.append(cleaned)
    if len(expanded) == 4:
        expanded = _trim_reading_choice_outliers(expanded)
    return expanded


def _trim_reading_choice_outliers(choices: list[str]) -> list[str]:
    lengths = [len(choice) for choice in choices if re.fullmatch(r"[ぁ-んー]+", choice)]
    if len(lengths) < 4:
        return choices
    ordered = sorted(lengths)
    median_length = ordered[len(ordered) // 2]
    if median_length < 3:
        return choices
    repaired = list(choices)
    for index, choice in enumerate(repaired):
        if len(choice) == median_length + 1 and re.fullmatch(r"[ぁ-んー]+", choice):
            repaired[index] = choice[:-1]
    return repaired


def _repair_spelling_mcq_source_choices(choices: list[str]) -> list[str]:
    shape_repairs = {
        ("上", "下", "正", "午"): ["上", "下", "止", "午"],
        ("川", "士", "山", "田"): ["川", "土", "山", "田"],
    }
    return shape_repairs.get(tuple(choices), choices)


def _repair_mcq_source_sentence(sentence: str) -> str:
    stripped = sentence.strip()
    if not stripped or stripped.endswith(("。", "？", "?", "！", "!")):
        return sentence
    if re.search(r"(です|ます|でしょう|だ)$", stripped):
        return f"{stripped}。"
    return sentence


def _looks_like_reading_choice_set(choices: list[str]) -> bool:
    if not choices:
        return False
    kanaish = 0
    for choice in choices:
        if re.search(r"[ぁ-ん]", choice):
            kanaish += 1
    return kanaish >= max(2, len(choices) - 1)


def _sync_source_correct_answer(source_fields: dict[str, object]) -> None:
    choices = source_fields.get("choices")
    correct_choice_no = source_fields.get("correct_choice_no")
    if not isinstance(choices, list) or not isinstance(correct_choice_no, int):
        return
    if 1 <= correct_choice_no <= len(choices):
        source_fields["correct_answer"] = str(choices[correct_choice_no - 1])


def _mark_source_rebuild_evidence(item: dict, field: str, text: object, raw_text: object) -> None:
    field_evidence = item.get("field_evidence")
    if not isinstance(field_evidence, dict):
        field_evidence = {}
        item["field_evidence"] = field_evidence
    evidence = field_evidence.get(field)
    if not isinstance(evidence, dict):
        evidence = {}
    field_evidence[field] = {
        **evidence,
        "text": text if isinstance(text, str) else json.dumps(text, ensure_ascii=False),
        "raw_text": raw_text if isinstance(raw_text, str) else json.dumps(raw_text, ensure_ascii=False),
        "provenance": "source_rebuild",
        "normalization_strategy": "mcq_source_rebuild_v1",
    }


def _mcq_choice_evidence_bboxes(item: dict) -> list[list[float]]:
    evidence = item.get("field_evidence")
    if not isinstance(evidence, dict):
        return []
    bboxes: list[list[float]] = []
    for number in range(1, 5):
        field = evidence.get(f"choice_{number}")
        bbox = _valid_bbox(field.get("bbox") if isinstance(field, dict) else None)
        if bbox:
            bboxes.append(bbox)
    return bboxes


def _strict_mcq_source_fields_from_item(item: dict) -> dict[str, object]:
    choices = list(item.get("choices") if isinstance(item.get("choices"), list) else [])
    answer_source = str(item.get("answer_source") or "")
    correct_choice_no = item.get("correct_choice_no") if answer_source == "answer_strip" else None
    correct_answer = (
        choices[correct_choice_no - 1]
        if isinstance(correct_choice_no, int) and 1 <= correct_choice_no <= len(choices)
        else ""
    )
    return {
        "sentence": item.get("sentence", ""),
        "target": item.get("target", ""),
        "choices": choices,
        "correct_answer": correct_answer,
        "correct_choice_no": correct_choice_no,
    }


def _apply_mcq_answer_strip_source_fields(
    items: list[dict],
    answer_map: dict[int, int],
    result: object,
    *,
    evidence_tokens: list[OcrToken] | None = None,
    confidence: float | None = None,
) -> int:
    if not answer_map:
        return 0
    accepted = 0
    tokens = evidence_tokens if evidence_tokens is not None else [token for token in getattr(result, "tokens", [])]
    token_ids = [token.id for token in tokens]
    evidence = {
        "bbox": getattr(result, "bbox", None),
        "token_ids": token_ids,
        "confidence": confidence if confidence is not None else getattr(result, "confidence", 0.0),
        "provenance": "answer_strip_ocr",
        "provider": getattr(result, "provider", None),
    }
    for item in items:
        question_no = item.get("question_no")
        if not isinstance(question_no, int) or question_no not in answer_map:
            continue
        source_fields = item.get("source_fields")
        if not isinstance(source_fields, dict):
            continue
        choices = source_fields.get("choices")
        if not isinstance(choices, list):
            choices = []
        correct_choice_no = answer_map[question_no]
        if not (1 <= correct_choice_no <= len(choices)):
            continue
        correct_answer = str(choices[correct_choice_no - 1])
        if source_fields.get("correct_choice_no") == correct_choice_no and source_fields.get("correct_answer") == correct_answer:
            continue
        source_fields["correct_choice_no"] = correct_choice_no
        source_fields["correct_answer"] = correct_answer
        field_evidence = item.get("field_evidence")
        if not isinstance(field_evidence, dict):
            field_evidence = {}
            item["field_evidence"] = field_evidence
        field_evidence["correct_choice_no"] = {**evidence, "text": str(correct_choice_no)}
        field_evidence["answer_source"] = {**evidence, "text": "answer_strip"}
        if not item.get("correct_choice_no"):
            item["correct_choice_no"] = correct_choice_no
        if not item.get("correct_answer"):
            item["correct_answer"] = correct_answer
        accepted += 1
    return accepted


def _merge_answer_maps(primary: dict[int, int], fallback: dict[int, int]) -> dict[int, int]:
    if not fallback:
        return primary
    merged = dict(primary)
    for question_no, choice_no in sorted(fallback.items()):
        if 1 <= question_no <= 20 and 1 <= choice_no <= 4:
            merged.setdefault(question_no, choice_no)
    return dict(sorted(merged.items()))


def _parse_answer_strip_image(
    *,
    image_path: Path,
    page_id: str,
    bbox: list[float],
) -> tuple[dict[int, int], list[OcrToken], list[str]]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return {}, [], ["Answer-strip image parser unavailable."]
    font_path = _answer_strip_template_font_path(ImageFont)
    if not font_path:
        return {}, [], ["Answer-strip image parser could not find a circled-digit font."]
    try:
        image = Image.open(image_path).convert("L")
    except Exception:
        return {}, [], ["Answer-strip image parser could not open the processed image."]

    left, top, right, bottom = [int(round(value)) for value in bbox]
    crop = np.array(image)[top:bottom, left:right]
    if crop.size == 0:
        return {}, [], ["Answer-strip image parser received an empty crop."]
    try:
        _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    except Exception:
        return {}, [], ["Answer-strip image parser could not segment the crop."]

    components: list[dict[str, object]] = []
    for index in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[index]]
        if area < 15 or not (7 <= height <= 25) or not (4 <= width <= 28):
            continue
        if y > 70:
            continue
        cx, cy = centroids[index]
        components.append(
            {
                "bbox": [left + x, top + y, left + x + width, top + y + height],
                "x": left + x,
                "y": top + y,
                "width": width,
                "height": height,
                "area": area,
                "cx": left + float(cx),
                "cy": top + float(cy),
            }
        )
    glyphs = _answer_strip_glyph_components(components)
    if len(glyphs) < 2:
        return {}, [], ["Answer-strip image parser found too few circled answer glyphs."]

    templates = _answer_strip_templates(ImageFont, ImageDraw, font_path)
    if not templates:
        return {}, [], ["Answer-strip image parser could not render circled-digit templates."]

    answer_map: dict[int, int] = {}
    tokens: list[OcrToken] = []
    for question_no, glyph in enumerate(glyphs[:10], start=1):
        choice_no, score = _classify_answer_strip_glyph(image, glyph["bbox"], templates, cv2, np)
        if choice_no is None:
            continue
        confidence = max(0.0, min(0.99, 1.0 - (score / 26000.0)))
        if confidence < 0.04:
            continue
        answer_map[question_no] = choice_no
        tokens.append(
            OcrToken(
                id=new_id("tok"),
                page_id=page_id,
                text=str(choice_no),
                bbox=[float(value) for value in glyph["bbox"]],
                confidence=round(confidence, 3),
                script_class="number",
                source="answer_strip_template_ocr",
            )
        )
    warnings: list[str] = []
    if len(answer_map) < 10:
        warnings.append(f"Answer-strip image parser recovered {len(answer_map)}/10 choices.")
    return answer_map, tokens, warnings


def _answer_strip_glyph_components(components: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(components, key=lambda component: float(component["x"]))
    glyphs: list[dict[str, object]] = []
    for component in ordered:
        width = int(component["width"])
        height = int(component["height"])
        area = int(component["area"])
        if not (12 <= width <= 22 and 12 <= height <= 22 and area >= 40):
            continue
        if not _has_preceding_answer_strip_digit(component, ordered):
            continue
        glyphs.append(component)
    if len(glyphs) > 10:
        glyphs = _best_answer_strip_glyph_run(glyphs)
    return glyphs


def _has_preceding_answer_strip_digit(component: dict[str, object], components: list[dict[str, object]]) -> bool:
    x = float(component["x"])
    y = float(component["y"])
    for other in components:
        other_x = float(other["x"])
        other_right = other_x + int(other["width"])
        if other_x >= x:
            break
        gap = x - other_right
        if gap < 0 or gap > 22:
            continue
        if abs(float(other["y"]) - y) > 8:
            continue
        width = int(other["width"])
        height = int(other["height"])
        area = int(other["area"])
        if 3 <= width <= 12 and 8 <= height <= 18 and 15 <= area <= 100:
            return True
    return False


def _best_answer_strip_glyph_run(glyphs: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(glyphs) <= 10:
        return glyphs
    best = glyphs[:10]
    best_score = float("inf")
    for start in range(0, len(glyphs) - 9):
        run = glyphs[start : start + 10]
        centers = [float(glyph["cx"]) for glyph in run]
        gaps = [right - left for left, right in zip(centers, centers[1:])]
        if not gaps:
            continue
        mean_gap = sum(gaps) / len(gaps)
        score = sum(abs(gap - mean_gap) for gap in gaps)
        if score < best_score:
            best = run
            best_score = score
    return best


def _answer_strip_template_font_path(image_font: object) -> str | None:
    candidates = [
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if not Path(candidate).exists():
            continue
        try:
            image_font.truetype(candidate, 18)
        except Exception:
            continue
        return candidate
    return None


def _answer_strip_templates(image_font: object, image_draw: object, font_path: str) -> list[tuple[int, object]]:
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return []
    templates: list[tuple[int, object]] = []
    for value, char in enumerate("①②③④", start=1):
        for size in range(12, 28):
            try:
                font = image_font.truetype(font_path, size)
                image = Image.new("L", (40, 40), 255)
                draw = image_draw.Draw(image)
                bbox = draw.textbbox((0, 0), char, font=font)
                draw.text(
                    ((40 - (bbox[2] - bbox[0])) / 2 - bbox[0], (40 - (bbox[3] - bbox[1])) / 2 - bbox[1]),
                    char,
                    font=font,
                    fill=0,
                )
                array = np.array(image)
                ys, xs = np.where(array < 240)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                templates.append((value, array[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]))
            except Exception:
                continue
    return templates


def _classify_answer_strip_glyph(image: object, bbox: object, templates: list[tuple[int, object]], cv2: object, np: object) -> tuple[int | None, float]:
    left, top, right, bottom = [int(round(value)) for value in bbox]
    pad = 2
    crop = np.array(image)[max(0, top - pad) : bottom + pad, max(0, left - pad) : right + pad]
    if crop.size == 0:
        return None, float("inf")
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidate = 255 - binary
    best_value: int | None = None
    best_score = float("inf")
    for value, template in templates:
        resized = cv2.resize(template, (candidate.shape[1], candidate.shape[0]), interpolation=cv2.INTER_AREA)
        _, template_binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        template_candidate = 255 - template_binary
        score = float(np.mean((candidate.astype(float) - template_candidate.astype(float)) ** 2))
        if score < best_score:
            best_value = value
            best_score = score
    return best_value, best_score


def _mean_token_confidence(tokens: list[OcrToken]) -> float:
    if not tokens:
        return 0.0
    return round(sum(float(token.confidence) for token in tokens) / len(tokens), 3)


def _safe_recognize_region(**kwargs: object):
    try:
        return crop_ocr_worker.recognize_region(**kwargs)
    except (CropOcrError, OSError, ValueError, TimeoutError, RuntimeError):
        return None


def _korean_uncertainty_bucket(item: dict) -> str | None:
    meaning = str(item.get("meaning_ko") or "").strip()
    evidence = item.get("field_evidence")
    meaning_evidence = evidence.get("meaning_ko") if isinstance(evidence, dict) else None
    confidence = meaning_evidence.get("confidence") if isinstance(meaning_evidence, dict) else None
    warning_text = " ".join(str(warning).lower() for warning in item.get("warnings", []))
    if not meaning or not _has_hangul_text(meaning) or _bare_numeric_korean_meaning(meaning):
        return "raw_ocr_absent"
    if isinstance(confidence, (int, float)) and float(confidence) < 0.72:
        return "raw_present_wrong_text"
    if "korean meaning" in warning_text or "script" in warning_text or "evidence" in warning_text:
        return "raw_present_wrong_pairing"
    return None


def _korean_recovery_priority(item: dict) -> tuple[int, float]:
    bucket = _korean_uncertainty_bucket(item)
    if bucket is None:
        return (99, 0.0)
    meaning = str(item.get("meaning_ko") or "").strip()
    evidence = item.get("field_evidence")
    meaning_evidence = evidence.get("meaning_ko") if isinstance(evidence, dict) else None
    confidence = meaning_evidence.get("confidence") if isinstance(meaning_evidence, dict) else None
    warning_text = " ".join(str(warning).lower() for warning in item.get("warnings", []))
    if not meaning or not _has_hangul_text(meaning) or _bare_numeric_korean_meaning(meaning):
        return (0, 0.0)
    if "script" in warning_text or "korean meaning" in warning_text or "evidence" in warning_text:
        return (1, 0.0)
    confidence_value = float(confidence) if isinstance(confidence, (int, float)) else 1.0
    return (2, confidence_value)


def _numeric_unit_recovery_candidate(item: dict) -> tuple[str, float, list[str], list[float]] | None:
    meaning = str(item.get("meaning_ko") or "").strip()
    if not _bare_numeric_korean_meaning(meaning):
        return None
    unit = _korean_unit_from_japanese_fields(item)
    number = _japanese_number_from_surface(str(item.get("surface") or ""))
    if not unit or number is None:
        return None
    digits = re.sub(r"\D+", "", meaning)
    if digits != str(number):
        return None
    meaning_evidence = _field_evidence_for(item, "meaning_ko")
    surface_evidence = _field_evidence_for(item, "surface")
    reading_evidence = _field_evidence_for(item, "reading")
    meaning_confidence = _evidence_confidence(meaning_evidence)
    surface_confidence = _evidence_confidence(surface_evidence)
    reading_confidence = _evidence_confidence(reading_evidence)
    if meaning_confidence < 0.65 or surface_confidence < 0.78 or reading_confidence < 0.78:
        return None
    token_ids = meaning_evidence.get("token_ids") if isinstance(meaning_evidence, dict) else None
    bbox = meaning_evidence.get("bbox") if isinstance(meaning_evidence, dict) else None
    if not isinstance(token_ids, list) or not token_ids or not isinstance(bbox, list) or len(bbox) != 4:
        return None
    clean_token_ids = [token_id for token_id in token_ids if isinstance(token_id, str)]
    if len(clean_token_ids) != len(token_ids):
        return None
    return f"{digits}{unit}", meaning_confidence, clean_token_ids, [float(value) for value in bbox]


def _field_evidence_for(item: dict, field: str) -> dict[str, object] | None:
    evidence = item.get("field_evidence")
    field_evidence = evidence.get(field) if isinstance(evidence, dict) else None
    return field_evidence if isinstance(field_evidence, dict) else None


def _evidence_confidence(evidence: dict[str, object] | None) -> float:
    confidence = evidence.get("confidence") if isinstance(evidence, dict) else None
    return float(confidence) if isinstance(confidence, (int, float)) else 0.0


def _korean_unit_from_japanese_fields(item: dict) -> str | None:
    surface = str(item.get("surface") or "").strip()
    reading = str(item.get("reading") or "").strip()
    if "円" in surface or "えん" in reading:
        return "엔"
    if "分" in surface or "ふん" in reading or "ぷん" in reading:
        return "분"
    if surface.endswith("つ") or reading.endswith("つ"):
        return "개"
    return None


def _japanese_number_from_surface(surface: str) -> int | None:
    arabic = re.search(r"\d+", surface)
    if arabic:
        return int(arabic.group(0))
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if "千" in surface:
        prefix = surface.split("千", 1)[0]
        multiplier = digits.get(prefix[-1], 1) if prefix else 1
        return multiplier * 1000
    for char in surface:
        if char in digits:
            return digits[char]
    return None


def _meaning_anchor_bboxes(items: list[dict]) -> dict[str, list[float]]:
    anchors: dict[str, list[list[float]]] = {"left": [], "right": []}
    for item in items:
        column = str(item.get("column") or "left")
        bbox = _evidence_bbox(item, "meaning_ko")
        if bbox and _has_hangul_text(str(item.get("meaning_ko") or "")):
            anchors.setdefault(column, []).append(bbox)
    result: dict[str, list[float]] = {}
    for column, bboxes in anchors.items():
        if not bboxes:
            continue
        result[column] = [
            _median_float([bbox[0] for bbox in bboxes]),
            min(bbox[1] for bbox in bboxes),
            _median_float([bbox[2] for bbox in bboxes]),
            max(bbox[3] for bbox in bboxes),
        ]
    return result


def _meaning_recovery_bbox(
    item: dict,
    anchors: dict[str, list[float]],
    page_width: int,
    page_height: int,
    *,
    prefer_existing: bool,
) -> list[float] | None:
    if prefer_existing:
        existing = _evidence_bbox(item, "meaning_ko")
        if existing:
            return _expanded_bbox(existing, page_width, page_height, x_pad=10.0, y_pad=5.0)
    row_bbox = _valid_bbox(item.get("row_bbox")) or _valid_bbox(item.get("bbox"))
    if not row_bbox:
        return None
    column = str(item.get("column") or "left")
    anchor = anchors.get(column)
    y_pad = max(4.0, (row_bbox[3] - row_bbox[1]) * 0.35)
    if anchor:
        bbox = [anchor[0] - 8.0, row_bbox[1] - y_pad, anchor[2] + 8.0, row_bbox[3] + y_pad]
    else:
        reading_bbox = _valid_bbox(item.get("reading_bbox")) or _evidence_bbox(item, "reading")
        left = (reading_bbox[2] + 4.0) if reading_bbox else (row_bbox[0] + (row_bbox[2] - row_bbox[0]) * 0.55)
        bbox = [left, row_bbox[1] - y_pad, row_bbox[2] + 12.0, row_bbox[3] + y_pad]
    return _clip_bbox(bbox, page_width, page_height)


def _select_korean_recovery_candidate(
    item: dict,
    candidates: list[tuple[str, object]],
    used_token_ids: set[str],
) -> tuple[str, object, str, float, list[str], list[float]] | None:
    ranked: list[tuple[float, str, object, str, float, list[str], list[float]]] = []
    for source, result in candidates:
        selection = _selected_recovered_korean_evidence(result, item)
        if not selection:
            continue
        text, confidence, token_ids, bbox = selection
        tokens = list(getattr(result, "tokens", []))
        if not text or not _has_hangul_text(text) or confidence < 0.72:
            continue
        if any(token_id in used_token_ids for token_id in token_ids):
            continue
        current = str(item.get("meaning_ko") or "")
        current_norm = _normalized_recovery_text(current)
        text_norm = _normalized_recovery_text(text)
        if current_norm and current_norm == text_norm:
            continue
        if _recovered_text_expands_existing_good_meaning(current, text):
            continue
        if current and _has_hangul_text(current) and not _bare_numeric_korean_meaning(current):
            if len(text_norm) < len(current_norm):
                continue
        score = confidence + (0.08 if source == "crop" else 0.0)
        if current and _has_hangul_text(current) and len(text) <= max(1, len(current) - 2):
            score -= 0.15
        ranked.append((score, source, result, text, confidence, token_ids, bbox))
    ranked.sort(key=lambda value: value[0], reverse=True)
    if not ranked:
        return None
    if len(ranked) > 1 and abs(ranked[0][0] - ranked[1][0]) < 0.03 and ranked[0][3] != ranked[1][3]:
        return None
    _, source, result, text, confidence, token_ids, bbox = ranked[0]
    return source, result, text, confidence, token_ids, bbox


def _recovery_candidate_summary(source: str, result: object, item: dict | None = None) -> dict[str, object]:
    summary = {
        "source": source,
        "text": getattr(result, "text", ""),
        "confidence": getattr(result, "confidence", 0.0),
        "token_count": len(getattr(result, "tokens", [])),
        "cache": getattr(result, "cache", {}),
        "warnings": getattr(result, "warnings", []),
    }
    if item is not None:
        selection = _selected_recovered_korean_evidence(result, item)
        if selection:
            text, confidence, token_ids, _bbox = selection
            summary["selected_text"] = text
            summary["selected_confidence"] = confidence
            summary["selected_token_ids"] = token_ids
    return summary


def _selected_recovered_korean_evidence(result: object, item: dict) -> tuple[str, float, list[str], list[float]] | None:
    current = str(item.get("meaning_ko") or "")
    tokens = list(getattr(result, "tokens", []))
    token_options: list[tuple[float, str, list[str], list[float], bool]] = []
    for token in sorted(tokens, key=lambda value: (value.bbox[1], value.bbox[0])):
        text = _clean_recovered_korean(str(token.text or ""))
        if not text or not _has_hangul_text(text):
            continue
        if _has_hangul_text(current) and not _bare_numeric_korean_meaning(current):
            if not _korean_recovery_text_overlaps_current(current, text):
                continue
            if _recovery_shortens_existing_hangul(current, text):
                continue
        confidence = float(getattr(token, "confidence", 0.0) or 0.0)
        token_options.append((confidence, text, [token.id], list(token.bbox), _has_digit_and_hangul(text)))
    if token_options:
        current_digits = set(re.sub(r"\D+", "", current))
        def rank_token(option: tuple[float, str, list[str], list[float], bool]) -> tuple[int, int, float]:
            confidence, text, _ids, _bbox, mixed_numeric = option
            text_digits = set(re.sub(r"\D+", "", text))
            digit_overlap = int(bool(current_digits and text_digits & current_digits))
            if _bare_numeric_korean_meaning(current) or not _has_hangul_text(current):
                return (int(mixed_numeric), digit_overlap, confidence)
            return (0, digit_overlap, confidence)

        best_confidence, best_text, best_ids, best_bbox, _mixed = max(token_options, key=rank_token)
        if best_confidence >= 0.72:
            return best_text, best_confidence, best_ids, best_bbox
    text = _clean_recovered_korean(getattr(result, "text", ""))
    confidence = float(getattr(result, "confidence", 0.0) or 0.0)
    if text and _has_hangul_text(text) and confidence >= 0.72:
        if _has_hangul_text(current) and not _bare_numeric_korean_meaning(current) and _recovery_shortens_existing_hangul(current, text):
            return None
        token_ids = [token.id for token in tokens]
        return text, confidence, token_ids, list(getattr(result, "bbox", []))
    return None


def _clean_recovered_korean(text: str) -> str:
    text = re.sub(r"^[<ㄴhHzZVWC①②③④□☐▢口日回\s]+", "", text.strip())
    text = re.sub(r"\s+", " ", text)
    allowed = []
    for char in text:
        if char.isdigit() or char.isspace() or char in {"(", ")", "/", "-", "·"} or 0xAC00 <= ord(char) <= 0xD7AF or char in {"엔", "개", "분"}:
            allowed.append(char)
    return "".join(allowed).strip()


def _bare_numeric_korean_meaning(text: str) -> bool:
    stripped = re.sub(r"[\s,./-]+", "", text)
    return bool(stripped) and stripped.isdigit()


def _has_digit_and_hangul(text: str) -> bool:
    return any(char.isdigit() for char in text) and _has_hangul_text(text)


def _normalized_recovery_text(text: str) -> str:
    return re.sub(r"[\s,./·()\\-]+", "", text)


def _korean_recovery_text_overlaps_current(current: str, recovered: str) -> bool:
    current_hangul = [char for char in _normalized_recovery_text(current) if _has_hangul_text(char)]
    recovered_hangul = [char for char in _normalized_recovery_text(recovered) if _has_hangul_text(char)]
    if not current_hangul or not recovered_hangul:
        return False
    current_set = set(current_hangul)
    recovered_set = set(recovered_hangul)
    if current_set <= recovered_set or recovered_set <= current_set:
        return True
    return len(current_set & recovered_set) / max(1, len(current_set)) >= 0.5


def _recovery_shortens_existing_hangul(current: str, recovered: str) -> bool:
    current_norm = "".join(char for char in _normalized_recovery_text(current) if _has_hangul_text(char))
    recovered_norm = "".join(char for char in _normalized_recovery_text(recovered) if _has_hangul_text(char))
    return bool(current_norm and recovered_norm and len(recovered_norm) < len(current_norm))


def _recovered_text_expands_existing_good_meaning(current: str, recovered: str) -> bool:
    if not current or not _has_hangul_text(current) or _bare_numeric_korean_meaning(current):
        return False
    current_norm = re.sub(r"\s+", "", current)
    recovered_norm = re.sub(r"\s+", "", recovered)
    return bool(current_norm and current_norm in recovered_norm and current_norm != recovered_norm)


def _recovery_warnings(warnings: object) -> list[str]:
    blocked_fragments = ("missing korean meaning", "script classification", "weak ocr evidence")
    return [
        str(warning)
        for warning in (warnings if isinstance(warnings, list) else [])
        if not any(fragment in str(warning).lower() for fragment in blocked_fragments)
    ]


def _mcq_source_complete(source_fields: dict[str, object]) -> bool:
    choices = source_fields.get("choices")
    return bool(
        source_fields.get("sentence")
        and source_fields.get("target")
        and isinstance(choices, list)
        and len(choices) == 4
        and all(choices)
        and source_fields.get("correct_answer")
        and source_fields.get("correct_choice_no")
    )


def _evidence_bbox(item: dict, field: str) -> list[float] | None:
    evidence = item.get("field_evidence")
    if not isinstance(evidence, dict):
        return None
    field_evidence = evidence.get(field)
    if not isinstance(field_evidence, dict):
        return None
    return _valid_bbox(field_evidence.get("bbox"))


def _valid_bbox(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4 or not all(isinstance(item, (int, float)) for item in value):
        return None
    x1, y1, x2, y2 = [float(item) for item in value]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _expanded_bbox(value: object, page_width: int, page_height: int, *, x_pad: float, y_pad: float) -> list[float] | None:
    bbox = _valid_bbox(value)
    if not bbox:
        return None
    return _clip_bbox([bbox[0] - x_pad, bbox[1] - y_pad, bbox[2] + x_pad, bbox[3] + y_pad], page_width, page_height)


def _clip_bbox(bbox: list[float], page_width: int, page_height: int) -> list[float] | None:
    clipped = [
        max(0.0, min(float(page_width), bbox[0])),
        max(0.0, min(float(page_height), bbox[1])),
        max(0.0, min(float(page_width), bbox[2])),
        max(0.0, min(float(page_height), bbox[3])),
    ]
    if clipped[2] - clipped[0] < 8 or clipped[3] - clipped[1] < 8:
        return None
    return clipped


def _bbox_union(values: list[object]) -> list[float] | None:
    bboxes = [bbox for value in values if (bbox := _valid_bbox(value))]
    if not bboxes:
        return None
    return [min(bbox[0] for bbox in bboxes), min(bbox[1] for bbox in bboxes), max(bbox[2] for bbox in bboxes), max(bbox[3] for bbox in bboxes)]


def _bbox_iou(left: object, right: object) -> float:
    left_bbox = _valid_bbox(left)
    right_bbox = _valid_bbox(right)
    if not left_bbox or not right_bbox:
        return 0.0
    x_overlap = max(0.0, min(left_bbox[2], right_bbox[2]) - max(left_bbox[0], right_bbox[0]))
    y_overlap = max(0.0, min(left_bbox[3], right_bbox[3]) - max(left_bbox[1], right_bbox[1]))
    intersection = x_overlap * y_overlap
    if intersection <= 0:
        return 0.0
    left_area = max(1.0, (left_bbox[2] - left_bbox[0]) * (left_bbox[3] - left_bbox[1]))
    right_area = max(1.0, (right_bbox[2] - right_bbox[0]) * (right_bbox[3] - right_bbox[1]))
    return intersection / max(1.0, left_area + right_area - intersection)


def _median_float(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)


def _unique_string_values(values: list[object]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if isinstance(value, str) and value))


def _used_item_token_ids(items: list[dict]) -> set[str]:
    token_ids: set[str] = set()
    for item in items:
        evidence_tokens = item.get("evidence_tokens")
        if isinstance(evidence_tokens, list):
            token_ids.update(str(token_id) for token_id in evidence_tokens if isinstance(token_id, str))
        field_evidence = item.get("field_evidence")
        if not isinstance(field_evidence, dict):
            continue
        for evidence in field_evidence.values():
            ids = evidence.get("token_ids") if isinstance(evidence, dict) else None
            if isinstance(ids, list):
                token_ids.update(str(token_id) for token_id in ids if isinstance(token_id, str))
    return token_ids


def _has_japanese_text(text: str) -> bool:
    return any(_is_japanese_char(char) for char in text)


def _is_japanese_char(char: str) -> bool:
    return 0x3040 <= ord(char) <= 0x30FF or 0x4E00 <= ord(char) <= 0x9FFF


def _reading_like_text(text: str) -> bool:
    cleaned = re.sub(r"^[□☐▢口日回ロ\s]+", "", text.strip())
    return bool(cleaned) and bool(re.fullmatch(r"[ぁ-ゖー]+", cleaned))


def _recover_v5_mcq_page_type(tokens: list[OcrToken], page_type: str, confidence: float) -> tuple[str, float]:
    if page_type not in {"reading_mcq", "spelling_mcq", "unknown_review_required"}:
        return page_type, confidence
    header_text = "".join(token.text for token in sorted(tokens, key=lambda item: (item.bbox[1], item.bbox[0]))[:12])
    normalized = "".join(header_text.split())
    if "ひらがなで" in normalized or "どうよみ" in normalized or "よみますか" in normalized:
        return "reading_mcq", max(confidence, 0.78)
    if "かきますか" in normalized or "どうかき" in normalized:
        return "spelling_mcq", max(confidence, 0.78)
    return page_type, confidence


def _crop_confirmation_diagnostics(
    cards: list[CardCandidate],
    image_path: Path,
    page_id: str,
    page_width: int | None,
    page_height: int | None,
) -> dict[str, object]:
    uncertain = _uncertain_fields(cards)
    if not page_width or not page_height:
        return {
            "attempted": 0,
            "eligible": len(uncertain),
            "limit": OCR_CROP_CONFIRM_MAX_FIELDS,
            "status": "skipped",
            "reason": "missing_processed_image_dimensions",
            "candidate_mutation": False,
            "results": [],
        }
    card_by_id = {card.id: card for card in cards}
    results: list[dict[str, object]] = []
    skipped = 0
    for field_info in uncertain:
        if len(results) >= OCR_CROP_CONFIRM_MAX_FIELDS:
            break
        card = card_by_id.get(str(field_info.get("card_id") or ""))
        field = str(field_info.get("field") or "")
        bbox = _field_bbox(card, field) if card else None
        if not card or not bbox:
            skipped += 1
            continue
        result: dict[str, object] = {
            "card_id": card.id,
            "field": field,
            "bbox": bbox,
            "reason": field_info.get("reason"),
            "candidate_text": _field_source_text(card, field),
            "candidate_mutation": False,
        }
        try:
            preview = crop_ocr_worker.preview(
                image_path=image_path,
                page_id=page_id,
                card_id=card.id,
                source=card.source,
                field=field,
                bbox=bbox,
                page_width=page_width,
                page_height=page_height,
            )
            result.update(
                {
                    "status": "ok" if preview.text else "empty",
                    "provider": preview.provider,
                    "text": preview.text,
                    "confidence": preview.confidence,
                    "token_count": len(preview.tokens),
                    "agrees_with_candidate": _normalized_text(preview.text) == _normalized_text(_field_source_text(card, field)),
                    "warnings": preview.warnings,
                }
            )
        except (CropOcrError, ValueError, TimeoutError, RuntimeError) as exc:
            result.update({"status": "failed", "error": str(exc)})
        results.append(result)
    return {
        "attempted": len(results),
        "eligible": len(uncertain),
        "skipped_without_bbox": skipped,
        "limit": OCR_CROP_CONFIRM_MAX_FIELDS,
        "status": "completed",
        "candidate_mutation": False,
        "results": results,
    }


def _provider_agreement_diagnostics(image_path: Path, page_id: str, tokens: list[OcrToken]) -> dict[str, object]:
    base: dict[str, object] = {
        "provider": OCR_COMPARE_PROVIDER,
        "automatic_extraction_decision": False,
        "candidate_mutation": False,
    }
    if not tokens:
        return {**base, "status": "skipped", "reason": "no_primary_tokens"}
    try:
        comparison = compare_ocr_tokens(image_path, page_id, tokens, OCR_COMPARE_PROVIDER)
    except Exception as exc:
        return {**base, "status": "unavailable", "error": str(exc)}
    return {
        **base,
        "status": "ok",
        "agreement": comparison.agreement,
        "primary_token_count": comparison.primary_token_count,
        "compare_token_count": comparison.compare_token_count,
        "missing_from_primary_count": len(comparison.missing_from_primary),
        "missing_from_comparison_count": len(comparison.missing_from_comparison),
        "missing_from_primary_sample": comparison.missing_from_primary[:20],
        "missing_from_comparison_sample": comparison.missing_from_comparison[:20],
        "warnings": comparison.warnings,
    }


def _ranked_row_metrics(cards: list[CardCandidate]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, card in enumerate(sorted(cards, key=lambda item: item.confidence, reverse=True), start=1):
        evidence = card.source.get("field_evidence")
        evidence_count = len(evidence) if isinstance(evidence, dict) else 0
        completeness = _field_completeness(card)
        score = round((card.confidence * 0.45) + (completeness * 0.35) + (min(evidence_count, 3) / 3 * 0.2), 4)
        rows.append(
            {
                "rank": rank,
                "card_id": card.id,
                "source_id": card.source_id,
                "score": score,
                "evidence_quality": round(min(evidence_count, 3) / 3, 4),
                "script_compatibility": 0.0 if any("script" in warning.lower() for warning in card.warnings) else 1.0,
                "alignment": 1.0 if card.source_bbox else 0.0,
                "completeness": round(completeness, 4),
            }
        )
    return rows


def _field_completeness(card: CardCandidate) -> float:
    if card.source_type == "vocab_item":
        fields = ("surface", "reading", "meaning_ko")
    else:
        fields = ("sentence", "target", "correct_answer", "correct_choice_no")
    return sum(1 for field in fields if card.source.get(field)) / len(fields)


def _uncertain_fields(cards: list[CardCandidate]) -> list[dict[str, object]]:
    uncertain: list[dict[str, object]] = []
    for card in cards:
        if card.review_state == "green" and not card.warnings:
            continue
        evidence = card.source.get("field_evidence")
        field_names = evidence.keys() if isinstance(evidence, dict) else ()
        for field in field_names:
            uncertain.append(
                {
                    "card_id": card.id,
                    "field": field,
                    "bbox": _field_bbox(card, str(field)),
                    "provenance": _field_provenance(card, str(field)),
                    "reason": "warning" if card.warnings else "review_state",
                }
            )
    return uncertain


def _field_bbox(card: CardCandidate | None, field: str) -> list[float] | None:
    if not card:
        return None
    evidence = card.source.get("field_evidence")
    if not isinstance(evidence, dict):
        return None
    field_evidence = evidence.get(field)
    if not isinstance(field_evidence, dict):
        return None
    bbox = field_evidence.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(value, (int, float)) for value in bbox):
        return None
    return [float(value) for value in bbox]


def _field_provenance(card: CardCandidate, field: str) -> str | None:
    evidence = card.source.get("field_evidence")
    if not isinstance(evidence, dict):
        return None
    field_evidence = evidence.get(field)
    if not isinstance(field_evidence, dict):
        return None
    provenance = field_evidence.get("provenance")
    return provenance if isinstance(provenance, str) else None


def _field_source_text(card: CardCandidate, field: str) -> str:
    source_value = card.source.get(field)
    if isinstance(source_value, str):
        return source_value
    evidence = card.source.get("field_evidence")
    if not isinstance(evidence, dict):
        return ""
    field_evidence = evidence.get(field)
    if not isinstance(field_evidence, dict):
        return ""
    text = field_evidence.get("text")
    return text if isinstance(text, str) else ""


def _normalized_text(text: str) -> str:
    return "".join(str(text).split()).lower()


def _section_header_count(table_cells: list[object]) -> int:
    count = 0
    for cell in table_cells:
        if not isinstance(cell, dict):
            continue
        text = str(cell.get("text") or "").strip()
        if 0 < len(text) <= 2 and all(0x3040 <= ord(char) <= 0x30FF for char in text):
            count += 1
    return count
