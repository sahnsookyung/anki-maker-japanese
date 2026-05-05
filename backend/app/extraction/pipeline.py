from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from app.core.config import (
    DICTIONARY_PATH,
    OCR_COMPARE_PROVIDER,
    OCR_CROP_CONFIRM_MAX_FIELDS,
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
from app.extraction.answer_strip import parse_answer_strip
from app.extraction.cards import mcq_cards, vocab_cards
from app.extraction.classifier import classify_page
from app.extraction.document_graph import graph_from_document_parse, graph_from_tokens, graph_with_card_hypotheses
from app.extraction.mcq import extract_mcq_items
from app.extraction.vocab import extract_vocab_items, extract_vocab_items_dual_ocr
from app.extraction.vl_document import extract_from_document_parse
from app.extraction.vlm_cleanup import cleanup_mcq_items, cleanup_vocab_items
from app.models.schemas import CardCandidate, DocumentParseResult, OcrRun, OcrToken, Page, ProcessResult
from app.ocr.comparison import compare_ocr_tokens
from app.ocr.crop_worker import CropOcrError, crop_ocr_worker
from app.ocr.engines import OcrEngineResult, PADDLEOCR_ENGINE, PADDLEOCR_VL_ENGINE, run_ocr_engine
from app.ocr.profiles import (
    DEFAULT_EXTRACTION_VARIANT,
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
    extraction_variant: str = DEFAULT_EXTRACTION_VARIANT,
) -> ProcessResult:
    database.upsert_page(page)
    normalized_variant = normalize_extraction_variant(extraction_variant)
    profile = resolve_ocr_model_profile(model_profile)
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
        answer_map = parse_answer_strip(tokens, preprocess.height)

        validator = DictionaryValidator(DICTIONARY_PATH)
        cards: list[CardCandidate] = []
        all_tokens = list(evidence_tokens)
        if page_type == "vocab_table":
            if VOCAB_DUAL_OCR_ENABLED and not any(token.source == PADDLEOCR_VL_ENGINE for token in tokens):
                if cached_payload and cached_payload.korean_tokens:
                    korean_tokens = cached_payload.korean_tokens
                    korean_ocr_warnings = []
                else:
                    korean_tokens, korean_ocr_warnings = recognize_with_provider(processed_path, page.id, "paddle_korean")
                all_tokens.extend(korean_tokens)
                document_graph = graph_from_tokens(page.id, all_tokens, source=engine_result.engine, transform=transform)
                ocr_warnings.extend(korean_ocr_warnings)
                items = extract_vocab_items_dual_ocr(tokens, korean_tokens, validator)
                if not items:
                    items = extract_vocab_items(tokens, validator)
            else:
                items = extract_vocab_items(tokens, validator)
            vlm_warnings: list[str] = []
            if VLM_CLEANUP_ENABLED and items:
                items, vlm_warnings = cleanup_vocab_items(processed_path, items, tokens, validator)
            for item in items:
                cards.extend(vocab_cards(page.id, item))
        elif page_type in {"reading_mcq", "spelling_mcq"}:
            items = extract_mcq_items(tokens, answer_map, page_type)
            vlm_warnings = []
            if VLM_CLEANUP_ENABLED and items:
                items, vlm_warnings = cleanup_mcq_items(processed_path, items, tokens, answer_map)
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
    extraction_variant: str = DEFAULT_EXTRACTION_VARIANT,
    transform: dict[str, object] | None = None,
    profile_manifest: dict[str, object] | None = None,
    cache_source_run_id: str | None = None,
) -> ProcessResult:
    normalized_variant = normalize_extraction_variant(extraction_variant)
    profile = resolve_ocr_model_profile(model_profile)
    if profile_manifest is None:
        profile_manifest = profile.manifest(engine=PADDLEOCR_VL_ENGINE, extraction_variant=normalized_variant)
    validator = DictionaryValidator(DICTIONARY_PATH)
    extraction = extract_from_document_parse(document_parse, validator)
    document_graph = graph_from_document_parse(document_parse, transform=transform)
    cards: list[CardCandidate] = []
    if extraction.page_type == "vocab_table":
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
    payload = {
        "schema_version": 1,
        "image_sha256": image_sha,
        "preprocessing_hash": preprocessing_hash,
        "engine": engine,
        "extraction_variant": extraction_variant,
        "profile_id": profile_manifest.get("profile_id"),
        "provider": profile_manifest.get("provider"),
        "env_fingerprint": profile_manifest.get("env_fingerprint"),
        "model_config": profile_manifest.get("model_config"),
        "language_config": profile_manifest.get("language_config"),
        "preprocessing_config": _cacheable_preprocessing_config(
            profile_manifest.get("preprocessing_config") if isinstance(profile_manifest.get("preprocessing_config"), dict) else {}
        ),
        "package_versions": profile_manifest.get("package_versions"),
        "model_cache_paths": profile_manifest.get("model_cache_paths"),
    }
    encoded = repr(sorted(payload.items())).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


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
    rows = graph_payload.get("row_hypotheses") if isinstance(graph_payload.get("row_hypotheses"), list) else []
    fields = graph_payload.get("field_hypotheses") if isinstance(graph_payload.get("field_hypotheses"), list) else []
    base = {
        "schema_version": 1,
        "variant": variant,
        "candidate_mutation": False,
        "diagnostic_only": variant in {"provider_agreement_v1"},
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
    return {}


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
