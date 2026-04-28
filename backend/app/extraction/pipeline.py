from __future__ import annotations

from pathlib import Path

from app.core.config import DICTIONARY_PATH, PROCESSED_DIR, VLM_CLEANUP_ENABLED, VOCAB_DUAL_OCR_ENABLED
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
    original_path = Path(page.original_image_path)
    processed_path = PROCESSED_DIR / f"{page.id}.png"
    preprocess = preprocess_image(original_path, processed_path)
    engine_result = run_ocr_engine(processed_path, page.id, engine)
    tokens = engine_result.tokens
    ocr_warnings = list(engine_result.warnings)
    page_type, page_confidence, _features = classify_page(tokens, preprocess.height)
    answer_map = parse_answer_strip(tokens, preprocess.height)

    validator = DictionaryValidator(DICTIONARY_PATH)
    cards: list[CardCandidate] = []
    all_tokens = list(tokens)
    if page_type == "vocab_table":
        if VOCAB_DUAL_OCR_ENABLED and engine_result.engine == PADDLEOCR_ENGINE:
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
        page_type=page_type,
        page_type_confidence=page_confidence,
        image_width=preprocess.width,
        image_height=preprocess.height,
        warnings=warnings,
        created_at=page.created_at,
    )
    database.upsert_page(processed_page)
    database.replace_tokens(page.id, all_tokens)
    database.replace_cards(page.id, cards)
    return ProcessResult(
        page=processed_page,
        tokens=all_tokens,
        cards=cards,
        script_summary=script_summary([token.text for token in all_tokens]),
        answer_map=answer_map,
    )
