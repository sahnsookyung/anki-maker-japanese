from __future__ import annotations

from collections import Counter
from pathlib import Path
import unicodedata

from app.models.schemas import OcrComparison, OcrToken
from app.ocr.service import recognize_with_provider


def compare_ocr_tokens(
    image_path: Path,
    page_id: str,
    primary_tokens: list[OcrToken],
    compare_provider: str,
) -> OcrComparison:
    compare_tokens, warnings = recognize_with_provider(image_path, page_id, compare_provider)
    primary_texts = [_norm(token.text) for token in primary_tokens if token.text.strip()]
    compare_texts = [_norm(token.text) for token in compare_tokens if token.text.strip()]
    primary_counts = Counter(primary_texts)
    compare_counts = Counter(compare_texts)
    missing_from_primary = list((compare_counts - primary_counts).elements())
    missing_from_comparison = list((primary_counts - compare_counts).elements())
    overlap = sum((primary_counts & compare_counts).values())
    total = sum((primary_counts | compare_counts).values()) or 1
    agreement = overlap / total
    return OcrComparison(
        primary_provider=_provider_name(primary_tokens),
        compare_provider=compare_provider,
        primary_token_count=len(primary_tokens),
        compare_token_count=len(compare_tokens),
        agreement=round(agreement, 3),
        missing_from_primary=missing_from_primary[:100],
        missing_from_comparison=missing_from_comparison[:100],
        compare_tokens=compare_tokens,
        warnings=warnings,
    )


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().lower()


def _provider_name(tokens: list[OcrToken]) -> str:
    providers = {token.source for token in tokens}
    if len(providers) == 1:
        return next(iter(providers))
    if not providers:
        return "none"
    return "mixed"
