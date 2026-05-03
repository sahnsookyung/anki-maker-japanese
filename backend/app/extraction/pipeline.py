from __future__ import annotations

import hashlib
from pathlib import Path

from app.core.config import (
    DICTIONARY_PATH,
    PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME,
    PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME,
    PADDLE_OCR_TEXT_DETECTION_MODEL_NAME,
    PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME,
    PROCESSED_DIR,
    VLM_CLEANUP_ENABLED,
    VOCAB_DUAL_OCR_ENABLED,
)
from app.core.images import preprocess_image
from app.core.script import script_summary
from app.db import database
from app.extraction.answer_strip import parse_answer_strip
from app.extraction.cards import mcq_cards, vocab_cards
from app.extraction.classifier import classify_page
from app.extraction.mcq import extract_mcq_items
from app.extraction.vocab import extract_vocab_items, extract_vocab_items_dual_ocr
from app.extraction.vlm_cleanup import cleanup_mcq_items, cleanup_vocab_items
from app.models.schemas import CardCandidate, Page, ProcessResult
from app.ocr.engines import PADDLEOCR_ENGINE, PADDLEOCR_VL_ENGINE, run_ocr_engine
from app.ocr.service import recognize_with_provider
from app.validation.dictionary import DictionaryValidator


def process_page(page: Page, engine: str = PADDLEOCR_ENGINE) -> ProcessResult:
    database.upsert_page(page)
    original_path = Path(page.original_image_path)
    processed_path = PROCESSED_DIR / f"{page.id}.png"
    preprocess = preprocess_image(original_path, processed_path)
    run = database.start_ocr_run(
        page.id,
        engine,
        image_sha256=_sha256_file(original_path),
        processed_image_path=str(processed_path),
        preprocessing={
            "processed_width": preprocess.width,
            "processed_height": preprocess.height,
            "warnings": preprocess.warnings,
        },
        provider_config=_provider_config(engine),
    )
    try:
        engine_result = run_ocr_engine(processed_path, page.id, engine)
        tokens = engine_result.tokens
        evidence_tokens = engine_result.evidence_tokens or tokens
        ocr_warnings = list(engine_result.warnings)
        page_type, page_confidence, _features = classify_page(tokens, preprocess.height)
        answer_map = parse_answer_strip(tokens, preprocess.height)

        validator = DictionaryValidator(DICTIONARY_PATH)
        cards: list[CardCandidate] = []
        all_tokens = list(evidence_tokens)
        if page_type == "vocab_table":
            if VOCAB_DUAL_OCR_ENABLED and not any(token.source == PADDLEOCR_VL_ENGINE for token in tokens):
                korean_tokens, korean_ocr_warnings = recognize_with_provider(processed_path, page.id, "paddle_korean")
                all_tokens.extend(korean_tokens)
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


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _provider_config(engine: str) -> dict[str, object]:
    return {
        "engine": engine,
        "japanese_detection_model": PADDLE_OCR_TEXT_DETECTION_MODEL_NAME,
        "japanese_recognition_model": PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME,
        "korean_detection_model": PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME,
        "korean_recognition_model": PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME,
        "vocab_dual_ocr_enabled": VOCAB_DUAL_OCR_ENABLED,
        "vlm_cleanup_enabled": VLM_CLEANUP_ENABLED,
    }
