from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from app.core.ids import new_id
from app.core.script import classify_script
from app.extraction.field_evidence import static_evidence, token_evidence
from app.extraction.geometry import group_tokens_by_line, text_of, union_bbox
from app.models.schemas import OcrToken
from app.ocr.profiles import DEFAULT_EXTRACTION_VARIANT, extraction_variant_components
from app.validation.dictionary import DictionaryValidator


JP_CLASSES = {"hiragana", "katakana", "kanji", "mixed"}
KANA_PREFIX_RE = re.compile(r"^[ぁ-ゖァ-ヺー]+")
SURFACE_LEADING_NOISE_RE = re.compile(r"^[□☐▢口日回]+")
READING_LEADING_NOISE_RE = re.compile(r"^[□☐▢口日回ロ]+")
KO_LEADING_NOISE_RE = re.compile(r"^[<ㄴhHzZVWC①②③④□☐▢口日回\s]+")
ROW_ALIGNMENT_DIAGNOSTIC_COMPONENTS = {"v5_vocab_rows_v1", "ko_alignment_v1"}
EMBEDDED_READING_SURFACE_DELIMITERS = {"□", "☐", "▢", "口", "日", "回"}

WARNING_MESSAGES = {
    "MISSING_SURFACE": "Missing surface from OCR evidence.",
    "MISSING_READING": "Missing reading from OCR evidence.",
    "MISSING_KOREAN_MEANING": "Missing Korean meaning from OCR evidence.",
    "WEAK_OCR_EVIDENCE": "Weak OCR evidence; verify this row manually.",
    "SCRIPT_MISMATCH": "OCR script classification does not match the expected field.",
    "DICTIONARY_MISMATCH": "Surface-reading pair was not found in the local dictionary.",
}


@dataclass(frozen=True)
class VocabLayoutRow:
    surface_token: OcrToken | None
    reading_token: OcrToken | None
    meaning_token: OcrToken | None
    surface: str
    reading: str
    meaning_ko: str
    surface_bbox: list[float] | None
    reading_bbox: list[float] | None
    meaning_bbox: list[float] | None
    row_bbox: list[float]
    column: str
    section: str
    layout_confidence: float
    warning_codes: list[str]


def extract_vocab_items(tokens: list[OcrToken], validator: DictionaryValidator | None = None) -> list[dict]:
    validator = validator or DictionaryValidator()
    items: list[dict] = []
    for line in group_tokens_by_line(tokens):
        content = [token for token in line if token.script_class != "punctuation" and token.text not in {"□", "☐", "▢"}]
        if len(content) < 2:
            continue
        hangul = [token for token in content if token.script_class == "hangul" or _has_hangul(token.text)]
        japanese = [token for token in content if token.script_class in JP_CLASSES or _has_japanese(token.text)]
        if not hangul or not japanese:
            continue

        reading_tokens = [token for token in japanese if token.script_class in {"hiragana", "katakana"}]
        surface_tokens = [token for token in japanese if token not in reading_tokens]
        if not surface_tokens and reading_tokens:
            surface_tokens = [reading_tokens[0]]
            reading_tokens = reading_tokens[1:] or [surface_tokens[0]]
        if not reading_tokens and surface_tokens:
            reading_tokens = [surface_tokens[-1]]

        surface = text_of(surface_tokens, "").replace("□", "")
        reading = text_of(reading_tokens, "").replace("□", "")
        meaning = text_of(hangul, " ")
        bbox = union_bbox([token.bbox for token in content])
        confidence = min(token.confidence for token in content) if content else 0.5
        _, warnings = validator.validate_vocab(surface, reading)
        if len(content) < 3:
            warnings.append("Row has too few OCR tokens; verify manually.")
        items.append(
            {
                "id": new_id("vocab"),
                "type": "vocab_item",
                "surface": surface,
                "reading": reading,
                "meaning_ko": meaning,
                "field_evidence": {
                    "surface": token_evidence(surface_tokens, surface),
                    "reading": token_evidence(reading_tokens, reading),
                    "meaning_ko": token_evidence(hangul, meaning),
                },
                "evidence_tokens": [token.id for token in content],
                "bbox": bbox,
                "confidence": round(confidence, 3),
                "needs_review": bool(warnings),
                "warnings": warnings,
            }
        )
    return items


def extract_vocab_items_dual_ocr(
    japanese_tokens: list[OcrToken],
    korean_tokens: list[OcrToken],
    validator: DictionaryValidator | None = None,
    extraction_variant: str = DEFAULT_EXTRACTION_VARIANT,
) -> list[dict]:
    validator = validator or DictionaryValidator()
    components = extraction_variant_components(extraction_variant)
    candidate_components = _candidate_extraction_components(components)
    split_x = _column_split_x(japanese_tokens)
    items = _extract_vocab_items_from_layout(japanese_tokens, korean_tokens, split_x, validator, candidate_components)
    if items:
        return items

    rows = _japanese_vocab_rows(japanese_tokens)
    for row in rows:
        for column in ("left", "right"):
            side_tokens = [token for token in row if _token_column(token, split_x) == column]
            item = _dual_vocab_item(side_tokens, korean_tokens, split_x, column, validator)
            if item:
                items.append(item)
    return items


def extract_jp_ko_meaning_items(
    japanese_tokens: list[OcrToken],
    korean_tokens: list[OcrToken] | None = None,
    validator: DictionaryValidator | None = None,
) -> list[dict]:
    del validator
    combined = _unique_tokens([*japanese_tokens, *(korean_tokens or [])])
    items: list[dict] = []
    for line in group_tokens_by_line(combined):
        content = [token for token in line if token.script_class != "punctuation" and token.text not in {"□", "☐", "▢"}]
        if len(content) < 2:
            continue
        surface_candidates = [
            token
            for token in content
            if _has_japanese(token.text) and not _has_hangul(token.text) and _clean_surface(token.text)
        ]
        meaning_candidates = [token for token in content if _has_hangul(token.text) and _clean_korean_meaning(token.text)]
        if not surface_candidates or not meaning_candidates:
            continue
        surface_token = min(surface_candidates, key=lambda token: (token.bbox[0], -token.confidence))
        meaning_token = max(meaning_candidates, key=lambda token: (token.confidence, token.bbox[2] - token.bbox[0]))
        surface = _clean_surface(surface_token.text)
        meaning = _clean_korean_meaning(meaning_token.text)
        if not surface or not meaning:
            continue
        bbox = union_bbox([surface_token.bbox, meaning_token.bbox])
        confidence = min(surface_token.confidence, meaning_token.confidence)
        token_ids = [surface_token.id, meaning_token.id]
        items.append(
            {
                "id": new_id("vocab"),
                "type": "vocab_item",
                "vocab_type": "jp_ko_meaning",
                "surface": surface,
                "reading": "",
                "meaning_ko": meaning,
                "field_evidence": {
                    "surface": token_evidence([surface_token], surface),
                    "meaning_ko": token_evidence([meaning_token], meaning),
                },
                "evidence_tokens": token_ids,
                "bbox": bbox,
                "row_bbox": bbox,
                "surface_bbox": surface_token.bbox,
                "meaning_bbox": meaning_token.bbox,
                "confidence": round(confidence, 3),
                "needs_review": confidence < 0.75,
                "warnings": ["Meaning-only vocab row; no reading OCR field expected."],
                "study_writing": False,
                "study_reading": False,
                "study_meaning": True,
            }
        )
    return items


def _unique_tokens(tokens: list[OcrToken]) -> list[OcrToken]:
    unique: dict[str, OcrToken] = {}
    for token in tokens:
        unique.setdefault(token.id, token)
    return list(unique.values())


def _candidate_extraction_components(components: frozenset[str]) -> frozenset[str]:
    """Keep failed row-alignment experiments diagnostic until a benchmark gate admits them."""
    return frozenset(component for component in components if component not in ROW_ALIGNMENT_DIAGNOSTIC_COMPONENTS)


def vocab_alignment_diagnostics(
    japanese_tokens: list[OcrToken],
    korean_tokens: list[OcrToken],
    extraction_variant: str = DEFAULT_EXTRACTION_VARIANT,
) -> dict[str, Any]:
    components = extraction_variant_components(extraction_variant)
    diagnostic_components = components & ROW_ALIGNMENT_DIAGNOSTIC_COMPONENTS
    if not diagnostic_components:
        return {}

    split_x = _column_split_x(japanese_tokens)
    shadow_rows = _layout_vocab_rows(japanese_tokens, korean_tokens, split_x, components)
    warning_counts: dict[str, int] = {}
    rows_by_column: dict[str, int] = {}
    rows_by_section: dict[str, int] = {}
    meaning_token_ids: list[str] = []
    for row in shadow_rows:
        rows_by_column[row.column] = rows_by_column.get(row.column, 0) + 1
        rows_by_section[row.section or "unsectioned"] = rows_by_section.get(row.section or "unsectioned", 0) + 1
        for code in row.warning_codes:
            warning_counts[code] = warning_counts.get(code, 0) + 1
        if row.meaning_token:
            meaning_token_ids.append(row.meaning_token.id)

    clean_korean_token_ids = {token.id for token in korean_tokens if _clean_korean_meaning(token.text)}
    paired_korean_token_ids = set(meaning_token_ids) & clean_korean_token_ids
    complete_rows = [
        row
        for row in shadow_rows
        if row.surface and row.reading and row.meaning_ko and "SCRIPT_MISMATCH" not in row.warning_codes
    ]
    duplicate_korean_pairs = len(meaning_token_ids) - len(set(meaning_token_ids))
    confidence_values = [row.layout_confidence for row in shadow_rows]
    complete_ratio = len(complete_rows) / len(shadow_rows) if shadow_rows else None
    risk_level = _alignment_risk_level(
        shadow_row_count=len(shadow_rows),
        complete_ratio=complete_ratio,
        duplicate_korean_pairs=duplicate_korean_pairs,
        warning_counts=warning_counts,
    )
    return {
        "schema_version": 1,
        "candidate_replacement": "guarded_off",
        "components": sorted(diagnostic_components),
        "split_x": round(split_x, 3),
        "japanese_token_count": len(japanese_tokens),
        "korean_token_count": len(korean_tokens),
        "shadow_row_count": len(shadow_rows),
        "shadow_complete_row_count": len(complete_rows),
        "shadow_incomplete_row_count": max(0, len(shadow_rows) - len(complete_rows)),
        "shadow_complete_ratio": round(complete_ratio, 4) if complete_ratio is not None else None,
        "average_layout_confidence": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else None,
        "paired_korean_token_count": len(paired_korean_token_ids),
        "unpaired_korean_token_count": max(0, len(clean_korean_token_ids) - len(paired_korean_token_ids)),
        "duplicate_korean_pair_count": max(0, duplicate_korean_pairs),
        "warning_counts": dict(sorted(warning_counts.items())),
        "rows_by_column": dict(sorted(rows_by_column.items())),
        "rows_by_section": dict(sorted(rows_by_section.items())),
        "risk_level": risk_level,
    }


def _alignment_risk_level(
    *,
    shadow_row_count: int,
    complete_ratio: float | None,
    duplicate_korean_pairs: int,
    warning_counts: dict[str, int],
) -> str:
    if not shadow_row_count:
        return "not_applicable"
    if complete_ratio is not None and complete_ratio < 0.75:
        return "high"
    if duplicate_korean_pairs or warning_counts.get("SCRIPT_MISMATCH", 0):
        return "high"
    if complete_ratio is not None and complete_ratio < 0.9:
        return "medium"
    if warning_counts.get("WEAK_OCR_EVIDENCE", 0):
        return "medium"
    return "low"


def _extract_vocab_items_from_layout(
    japanese_tokens: list[OcrToken],
    korean_tokens: list[OcrToken],
    split_x: float,
    validator: DictionaryValidator,
    components: frozenset[str] = frozenset({DEFAULT_EXTRACTION_VARIANT}),
) -> list[dict]:
    layout_rows = _layout_vocab_rows(japanese_tokens, korean_tokens, split_x, components)
    items: list[dict] = []
    for layout_row in layout_rows:
        surface = layout_row.surface
        reading = layout_row.reading
        if "v5_token_split_v1" in components:
            surface, reading = _refine_vocab_surface_reading(surface, reading, validator)
        surface_normalization = "v5_ocr_dictionary_refinement" if surface != layout_row.surface else None
        reading_normalization = "v5_ocr_dictionary_refinement" if reading != layout_row.reading else None
        evidence = [token for token in (layout_row.surface_token, layout_row.reading_token, layout_row.meaning_token) if token]
        confidence = min([token.confidence for token in evidence], default=layout_row.layout_confidence)
        _, validator_warnings = validator.validate_vocab(surface, reading)
        warning_codes = _layout_warning_codes(
            surface_token=layout_row.surface_token,
            reading_token=layout_row.reading_token,
            meaning_token=layout_row.meaning_token,
            surface=surface,
            reading=reading,
            meaning=layout_row.meaning_ko,
        )
        if any("local dictionary" in warning for warning in validator_warnings):
            warning_codes.append("DICTIONARY_MISMATCH")
        warnings = _warnings_from_codes(warning_codes)
        items.append(
            {
                "id": new_id("vocab"),
                "type": "vocab_item",
                "surface": surface,
                "reading": reading,
                "meaning_ko": layout_row.meaning_ko,
                "field_evidence": {
                    "surface": _field_evidence(
                        layout_row.surface_token,
                        surface,
                        "surface",
                        bbox_override=layout_row.surface_bbox,
                        normalization_strategy=surface_normalization,
                    ),
                    "reading": _field_evidence(
                        layout_row.reading_token,
                        reading,
                        "reading",
                        bbox_override=layout_row.reading_bbox,
                        normalization_strategy=reading_normalization,
                    ),
                    "meaning_ko": _field_evidence(layout_row.meaning_token, layout_row.meaning_ko, "meaning_ko", bbox_override=layout_row.meaning_bbox),
                },
                "evidence_tokens": [token.id for token in evidence],
                "bbox": layout_row.row_bbox,
                "row_bbox": layout_row.row_bbox,
                "surface_bbox": layout_row.surface_bbox,
                "reading_bbox": layout_row.reading_bbox,
                "meaning_bbox": layout_row.meaning_bbox,
                "column": layout_row.column,
                "section": layout_row.section,
                "layout_confidence": layout_row.layout_confidence,
                "warning_codes": _unique(warning_codes),
                "confidence": round(confidence, 3),
                "needs_review": bool(warnings),
                "warnings": _unique(warnings),
            }
        )
    return items


def _layout_vocab_rows(
    japanese_tokens: list[OcrToken],
    korean_tokens: list[OcrToken],
    split_x: float,
    components: frozenset[str] = frozenset({DEFAULT_EXTRACTION_VARIANT}),
) -> list[VocabLayoutRow]:
    rows: list[VocabLayoutRow] = []
    seen_surface_tokens: set[str] = set()
    used_meaning_token_ids: set[str] = set()
    table_sections = _section_ranges(japanese_tokens)
    split_embedded = "v5_token_split_v1" in components
    consume_korean = bool({"v5_vocab_rows_v1", "ko_alignment_v1"} & components)
    row_tolerance_scale = 1.25 if "ko_alignment_v1" in components else (1.1 if "v5_vocab_rows_v1" in components else 1.0)
    for surface_token, surface, column, embedded_reading, surface_bbox, embedded_reading_bbox in _surface_candidates(
        japanese_tokens,
        split_x,
        split_embedded=split_embedded,
    ):
        if surface_token.id in seen_surface_tokens:
            continue
        seen_surface_tokens.add(surface_token.id)
        reading_token = surface_token if embedded_reading else _nearest_reading_token(
            japanese_tokens,
            surface_token,
            split_x,
            column,
            tolerance_scale=row_tolerance_scale,
            prefer_same_column=split_embedded,
            prefer_hiragana_reading=split_embedded,
            allow_noisy_trailing_reading=split_embedded,
        )
        reading = embedded_reading or (_clean_reading(reading_token.text, allow_noisy_trailing=split_embedded) if reading_token else "")
        reading_bbox = embedded_reading_bbox if embedded_reading else (reading_token.bbox if reading_token else None)
        row_y_values = [_cy(surface_token)]
        if reading_token:
            row_y_values.append(_cy(reading_token))
        row_y = sum(row_y_values) / len(row_y_values)
        meaning_token = _nearest_korean_meaning_token(
            korean_tokens,
            row_y,
            split_x,
            column,
            used_token_ids=used_meaning_token_ids if consume_korean else None,
            tolerance_scale=row_tolerance_scale,
            allow_closer_cross_column=split_embedded,
        )
        if meaning_token and consume_korean:
            used_meaning_token_ids.add(meaning_token.id)
        meaning = _clean_korean_meaning(meaning_token.text) if meaning_token else ""
        if not reading and not meaning:
            continue
        evidence = [token for token in (surface_token, reading_token, meaning_token) if token]
        row_bbox = _expanded_row_bbox(evidence, split_x, column)
        warning_codes = _layout_warning_codes(
            surface_token=surface_token,
            reading_token=reading_token,
            meaning_token=meaning_token,
            surface=surface,
            reading=reading,
            meaning=meaning,
        )
        rows.append(
            VocabLayoutRow(
                surface_token=surface_token,
                reading_token=reading_token,
                meaning_token=meaning_token,
                surface=surface,
                reading=reading,
                meaning_ko=meaning,
                surface_bbox=surface_bbox if surface_token else None,
                reading_bbox=reading_bbox,
                meaning_bbox=meaning_token.bbox if meaning_token else None,
                row_bbox=row_bbox,
                column=column,
                section=_section_for_y(table_sections, row_y),
                layout_confidence=_layout_confidence(evidence, warning_codes),
                warning_codes=warning_codes,
            )
        )
    rows.sort(key=lambda row: (row.row_bbox[1], row.row_bbox[0]))
    return rows


def _field_evidence(
    token: OcrToken | None,
    text: str,
    field: str,
    *,
    bbox_override: list[float] | None = None,
    normalization_strategy: str | None = None,
) -> dict[str, Any]:
    if token:
        evidence = token_evidence([token], text)
        if bbox_override and bbox_override != token.bbox:
            evidence["bbox"] = bbox_override
            evidence["derived_from_token_ids"] = [token.id]
            evidence["bbox_strategy"] = "split_merged_vocab_token"
        if normalization_strategy:
            evidence["raw_text"] = token.text
            evidence["normalization_strategy"] = normalization_strategy
            evidence.setdefault("derived_from_token_ids", [token.id])
        return evidence
    return static_evidence(text, f"missing_{field}")


def _layout_warning_codes(
    *,
    surface_token: OcrToken | None,
    reading_token: OcrToken | None,
    meaning_token: OcrToken | None,
    surface: str,
    reading: str,
    meaning: str,
) -> list[str]:
    codes: list[str] = []
    if not surface:
        codes.append("MISSING_SURFACE")
    if not reading:
        codes.append("MISSING_READING")
    if not meaning:
        codes.append("MISSING_KOREAN_MEANING")
    if any(token and token.confidence < 0.72 for token in (surface_token, reading_token, meaning_token)):
        codes.append("WEAK_OCR_EVIDENCE")
    if surface and _has_hangul(surface):
        codes.append("SCRIPT_MISMATCH")
    if reading and classify_script(reading) not in {"hiragana", "katakana", "mixed"}:
        codes.append("SCRIPT_MISMATCH")
    if meaning and not _has_hangul(meaning):
        codes.append("SCRIPT_MISMATCH")
    return _unique(codes)


def _layout_confidence(tokens: list[OcrToken], warning_codes: list[str]) -> float:
    if not tokens:
        return 0.0
    confidence = min(token.confidence for token in tokens)
    if any(code.startswith("MISSING_") for code in warning_codes):
        confidence = min(confidence, 0.49)
    elif warning_codes:
        confidence = min(confidence, 0.74)
    return round(confidence, 3)


def _warnings_from_codes(codes: list[str]) -> list[str]:
    return [WARNING_MESSAGES[code] for code in _unique(codes) if code in WARNING_MESSAGES]


def _expanded_row_bbox(tokens: list[OcrToken], split_x: float, column: str) -> list[float]:
    bbox = union_bbox([token.bbox for token in tokens])
    heights = [max(1.0, token.bbox[3] - token.bbox[1]) for token in tokens]
    padding_y = max(4.0, _median(heights) * 0.35)
    padding_x = max(6.0, _median(heights) * 0.6)
    if column == "left":
        right = min(max(bbox[2] + padding_x, split_x - padding_x), split_x + padding_x)
        return [max(0.0, bbox[0] - padding_x), max(0.0, bbox[1] - padding_y), right, bbox[3] + padding_y]
    left = max(min(bbox[0] - padding_x, split_x + padding_x), split_x - padding_x)
    return [left, max(0.0, bbox[1] - padding_y), bbox[2] + padding_x, bbox[3] + padding_y]


def _section_ranges(tokens: list[OcrToken]) -> list[tuple[float, str]]:
    sections: list[tuple[float, str]] = []
    for token in tokens:
        text = token.text.strip()
        if len(text) <= 2 and text and all(0x3040 <= ord(char) <= 0x30FF for char in text):
            sections.append((_cy(token), text))
    sections.sort()
    return sections


def _section_for_y(sections: list[tuple[float, str]], row_y: float) -> str:
    section = ""
    for section_y, name in sections:
        if section_y <= row_y:
            section = name
        else:
            break
    return section


def _japanese_vocab_rows(tokens: list[OcrToken]) -> list[list[OcrToken]]:
    rows: list[list[OcrToken]] = []
    for line in group_tokens_by_line(tokens, tolerance=24.0):
        content = [token for token in line if _has_japanese(token.text)]
        if len(content) < 2:
            continue
        rows.append(content)
    return rows


def _dual_vocab_item(
    side_tokens: list[OcrToken],
    korean_tokens: list[OcrToken],
    split_x: float,
    column: str,
    validator: DictionaryValidator,
) -> dict | None:
    ordered = sorted(side_tokens, key=lambda token: token.bbox[0])
    if len(ordered) < 2:
        return None

    surface_token = ordered[0]
    reading_token = _first_reading_token(ordered[1:])
    if not reading_token:
        return None

    surface = _clean_surface(surface_token.text)
    reading = _clean_reading(reading_token.text)
    if not surface or not reading:
        return None

    row_y = (_cy(surface_token) + _cy(reading_token)) / 2
    meaning_token = _nearest_korean_meaning_token(korean_tokens, row_y, split_x, column)
    if not meaning_token:
        return None

    meaning = _clean_korean_meaning(meaning_token.text)
    if not meaning:
        return None

    evidence = [surface_token, reading_token, meaning_token]
    bbox = union_bbox([token.bbox for token in evidence])
    confidence = min(token.confidence for token in evidence)
    _, warnings = validator.validate_vocab(surface, reading)
    if meaning_token.script_class not in {"hangul", "mixed", "number"}:
        warnings.append("Korean meaning came from a low-confidence or non-Hangul OCR token.")
    return {
        "id": new_id("vocab"),
        "type": "vocab_item",
        "surface": surface,
        "reading": reading,
        "meaning_ko": meaning,
        "field_evidence": {
            "surface": token_evidence([surface_token], surface),
            "reading": token_evidence([reading_token], reading),
            "meaning_ko": token_evidence([meaning_token], meaning),
        },
        "evidence_tokens": [token.id for token in evidence],
        "bbox": bbox,
        "confidence": round(confidence, 3),
        "needs_review": bool(warnings),
        "warnings": warnings,
    }


def _first_reading_token(tokens: list[OcrToken]) -> OcrToken | None:
    for token in tokens:
        if _clean_reading(token.text):
            return token
    return None


def _nearest_korean_meaning_token(
    tokens: list[OcrToken],
    row_y: float,
    split_x: float,
    column: str,
    *,
    used_token_ids: set[str] | None = None,
    tolerance_scale: float = 1.0,
    allow_closer_cross_column: bool = False,
) -> OcrToken | None:
    candidates = []
    cross_column_candidates = []
    tolerance = _row_match_tolerance(tokens) * tolerance_scale
    for token in tokens:
        if used_token_ids and token.id in used_token_ids:
            continue
        text = _clean_korean_meaning(token.text)
        if not text:
            continue
        vertical_delta = _cy(token) - row_y
        distance = abs(vertical_delta)
        if distance > tolerance:
            continue
        if _token_column(token, split_x) == column:
            candidates.append((distance, vertical_delta, token))
        else:
            cross_column_candidates.append((distance, vertical_delta, token))
    candidates.sort(key=lambda item: (item[0], -item[2].confidence))
    cross_column_candidates.sort(
        key=lambda item: (item[1] < -tolerance * 0.1, item[0], -item[2].confidence)
    )
    if allow_closer_cross_column and candidates and cross_column_candidates:
        same_column_distance = candidates[0][0]
        cross_column_distance = cross_column_candidates[0][0]
        if cross_column_distance + tolerance * 0.25 < same_column_distance:
            candidates = cross_column_candidates
    elif not candidates:
        candidates = cross_column_candidates
    if not candidates:
        return None
    return candidates[0][2]


def _column_split_x(tokens: list[OcrToken]) -> float:
    if not tokens:
        return 0.0
    surface_centers = sorted(
        (token.bbox[0] + token.bbox[2]) / 2
        for token in tokens
        if _surface_text_without_column_context(token.text)
    )
    centers = surface_centers or sorted((token.bbox[0] + token.bbox[2]) / 2 for token in tokens if _has_japanese(token.text))
    if len(centers) < 4:
        return _median([(token.bbox[0] + token.bbox[2]) / 2 for token in tokens])
    clustered_split = _two_cluster_split(centers)
    if clustered_split is not None:
        return clustered_split
    min_x = centers[0]
    max_x = centers[-1]
    gaps = [(right - left, left, right) for left, right in zip(centers, centers[1:])]
    median_gap = _median([gap for gap, _, _ in gaps]) or 1.0
    usable_gaps = [
        item
        for item in gaps
        if item[0] > median_gap * 2.75 and min_x + (max_x - min_x) * 0.25 <= (item[1] + item[2]) / 2 <= min_x + (max_x - min_x) * 0.75
    ]
    if usable_gaps:
        _, left, right = max(usable_gaps, key=lambda item: item[0])
        return (left + right) / 2
    return _median(centers)


def _two_cluster_split(centers: list[float]) -> float | None:
    ordered = sorted(centers)
    left_mean = ordered[len(ordered) // 4]
    right_mean = ordered[(len(ordered) * 3) // 4]
    if right_mean <= left_mean:
        return None
    left_cluster: list[float] = []
    right_cluster: list[float] = []
    for _ in range(8):
        left_cluster = []
        right_cluster = []
        for center in ordered:
            if abs(center - left_mean) <= abs(center - right_mean):
                left_cluster.append(center)
            else:
                right_cluster.append(center)
        if not left_cluster or not right_cluster:
            return None
        next_left = sum(left_cluster) / len(left_cluster)
        next_right = sum(right_cluster) / len(right_cluster)
        if abs(next_left - left_mean) < 0.5 and abs(next_right - right_mean) < 0.5:
            break
        left_mean, right_mean = next_left, next_right
    if len(left_cluster) < 3 or len(right_cluster) < 3:
        return None
    gap = min(right_cluster) - max(left_cluster)
    median_step = _median([right - left for left, right in zip(ordered, ordered[1:])]) or 1.0
    if gap < median_step * 1.5 and abs(right_mean - left_mean) < median_step * 6:
        return None
    return (left_mean + right_mean) / 2


def _surface_text_without_column_context(text: str) -> str:
    embedded_reading, surface = _split_combined_vocab_token(text)
    if embedded_reading and surface:
        return surface
    surface = _clean_surface(text)
    if not surface:
        return ""
    cleaned = SURFACE_LEADING_NOISE_RE.sub("", text.strip())
    if (
        KANA_PREFIX_RE.match(cleaned)
        and surface == _clean_surface(text)
        and not any(0x4E00 <= ord(char) <= 0x9FFF for char in cleaned)
    ):
        return ""
    return surface


def _token_column(token: OcrToken, split_x: float) -> str:
    return "left" if ((token.bbox[0] + token.bbox[2]) / 2) < split_x else "right"


def _cy(token: OcrToken) -> float:
    return (token.bbox[1] + token.bbox[3]) / 2


def _clean_surface(text: str) -> str:
    text = SURFACE_LEADING_NOISE_RE.sub("", text.strip())
    surface = "".join(ch for ch in text if _char_is_japanese(ch))
    if any(0x4E00 <= ord(ch) <= 0x9FFF for ch in surface) and surface and 0x30A0 <= ord(surface[-1]) <= 0x30FF:
        surface = surface[:-1]
    return surface


def _clean_reading(text: str, *, allow_noisy_trailing: bool = False) -> str:
    leading_noise = READING_LEADING_NOISE_RE if allow_noisy_trailing else SURFACE_LEADING_NOISE_RE
    text = leading_noise.sub("", text.strip())
    match = KANA_PREFIX_RE.match(text)
    if match:
        return match.group(0)
    if not allow_noisy_trailing:
        return ""
    trailing = re.search(r"[ぁ-ゖー]{2,}$", text)
    if trailing:
        return trailing.group(0)
    internal_runs = re.findall(r"[ぁ-ゖー]{4,}", text)
    return max(internal_runs, key=len) if internal_runs else ""


def _refine_vocab_surface_reading(
    surface: str,
    reading: str,
    validator: DictionaryValidator,
) -> tuple[str, str]:
    if not surface or not reading or not getattr(validator, "entries", None):
        return surface, reading
    if validator.validate_vocab(surface, reading)[0] == "valid":
        return surface, reading
    for surface_candidate in _surface_prefix_candidates(surface):
        reading_candidates = _reading_ocr_candidates(reading)
        reading_candidates.extend(_dictionary_completed_reading_candidates(surface_candidate, reading_candidates, validator))
        for reading_candidate in _unique(reading_candidates):
            if validator.validate_vocab(surface_candidate, reading_candidate)[0] == "valid":
                return surface_candidate, reading_candidate
    return surface, reading


def _surface_prefix_candidates(surface: str) -> list[str]:
    candidates = [surface]
    for end in range(len(surface) - 1, 0, -1):
        candidate = surface[:end].strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _reading_ocr_candidates(reading: str) -> list[str]:
    candidates = [reading]
    normalized = _normalize_yoon_ocr_reading(reading)
    if normalized and normalized not in candidates:
        candidates.append(normalized)
    for candidate in _reading_prefix_candidates(normalized):
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _dictionary_completed_reading_candidates(
    surface: str,
    readings: list[str],
    validator: DictionaryValidator,
) -> list[str]:
    entries = getattr(validator, "entries", set())
    if not entries:
        return []
    candidates: list[str] = []
    for entry_surface, entry_reading in entries:
        if entry_surface != surface:
            continue
        for reading in readings:
            missing_chars = len(entry_reading) - len(reading)
            if len(reading) >= 4 and 0 < missing_chars <= 1 and entry_reading.endswith(reading):
                candidates.append(entry_reading)
    return candidates


def _reading_prefix_candidates(reading: str) -> list[str]:
    candidates: list[str] = []
    for end in range(len(reading) - 1, 0, -1):
        candidate = reading[:end].strip()
        if candidate:
            candidates.append(candidate)
    return candidates


def _normalize_yoon_ocr_reading(reading: str) -> str:
    replacements = {
        "きや": "きゃ",
        "きゆ": "きゅ",
        "きよ": "きょ",
        "しや": "しゃ",
        "しゆ": "しゅ",
        "しよ": "しょ",
        "ちや": "ちゃ",
        "ちゆ": "ちゅ",
        "ちよ": "ちょ",
        "にや": "にゃ",
        "にゆ": "にゅ",
        "によ": "にょ",
        "ひや": "ひゃ",
        "ひゆ": "ひゅ",
        "ひよ": "ひょ",
        "みや": "みゃ",
        "みゆ": "みゅ",
        "みよ": "みょ",
        "りや": "りゃ",
        "りゆ": "りゅ",
        "りよ": "りょ",
        "ぎや": "ぎゃ",
        "ぎゆ": "ぎゅ",
        "ぎよ": "ぎょ",
        "じや": "じゃ",
        "じゆ": "じゅ",
        "じよ": "じょ",
        "びや": "びゃ",
        "びゆ": "びゅ",
        "びよ": "びょ",
        "ぴや": "ぴゃ",
        "ぴゆ": "ぴゅ",
        "ぴよ": "ぴょ",
    }
    normalized = reading
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def _clean_korean_meaning(text: str) -> str:
    text = KO_LEADING_NOISE_RE.sub("", text.strip())
    text = text.replace("。", ".").replace("，", ",")
    allowed = []
    for ch in text:
        if _char_is_hangul(ch) or ch.isdigit() or ch in {",", ".", " ", "(", ")", "（", "）"}:
            allowed.append(ch)
    return "".join(allowed).strip(" .,")


def _surface_candidates(
    tokens: list[OcrToken],
    split_x: float,
    *,
    split_embedded: bool = False,
) -> list[tuple[OcrToken, str, str, str, list[float] | None, list[float] | None]]:
    candidates: list[tuple[OcrToken, str, str, str, list[float] | None, list[float] | None]] = []
    for token in sorted(tokens, key=lambda item: (_cy(item), item.bbox[0])):
        embedded_reading, surface = _split_combined_vocab_token(
            token.text,
            protect_single_kana_prefix=split_embedded,
        )
        if not surface:
            surface = _clean_surface(token.text)
        if not surface or not _looks_like_surface_token(token, surface, split_x, strict_kana_noise=split_embedded):
            continue
        surface_bbox = token.bbox
        reading_bbox = token.bbox if embedded_reading else None
        if split_embedded and embedded_reading:
            reading_bbox, surface_bbox = _split_combined_vocab_bboxes(token, embedded_reading, surface)
        candidates.append((token, surface, _surface_column(token, split_x), embedded_reading, surface_bbox, reading_bbox))
    return candidates


def _split_combined_vocab_bboxes(token: OcrToken, reading: str, surface: str) -> tuple[list[float], list[float]]:
    x1, y1, x2, y2 = [float(value) for value in token.bbox]
    total_units = max(1, len(reading) + len(surface))
    split_x = x1 + (x2 - x1) * (len(reading) / total_units)
    split_x = max(x1, min(x2, split_x))
    return [x1, y1, split_x, y2], [split_x, y1, x2, y2]


def _split_combined_vocab_token(text: str, *, protect_single_kana_prefix: bool = False) -> tuple[str, str]:
    cleaned = SURFACE_LEADING_NOISE_RE.sub("", text.strip())
    reading_match = KANA_PREFIX_RE.match(cleaned)
    if not reading_match:
        return "", ""
    reading = reading_match.group(0)
    if protect_single_kana_prefix and len(reading) < 2:
        return "", ""
    suffix = cleaned[reading_match.end() :]
    kanji_indexes = [index for index, char in enumerate(suffix) if 0x4E00 <= ord(char) <= 0x9FFF]
    preferred_indexes = [
        index
        for index in kanji_indexes
        if suffix[index] not in EMBEDDED_READING_SURFACE_DELIMITERS
        and len("".join(ch for ch in suffix[index:] if _char_is_japanese(ch))) >= 2
    ]
    kanji_index = (preferred_indexes or kanji_indexes or [None])[0]
    if kanji_index is None:
        return "", ""
    surface = "".join(ch for ch in suffix[kanji_index:] if _char_is_japanese(ch))
    if len(surface) < 2:
        return "", ""
    if protect_single_kana_prefix and any(char in EMBEDDED_READING_SURFACE_DELIMITERS for char in suffix[:kanji_index]):
        return "", surface
    return reading, surface


def _looks_like_surface_token(
    token: OcrToken,
    surface: str,
    split_x: float,
    *,
    strict_kana_noise: bool = False,
) -> bool:
    if not surface:
        return False
    text = SURFACE_LEADING_NOISE_RE.sub("", token.text.strip())
    text_for_kana = text
    if strict_kana_noise:
        text_for_kana = _strip_leading_non_japanese(text_for_kana)
        text_for_kana = SURFACE_LEADING_NOISE_RE.sub("", text_for_kana)
    if (
        KANA_PREFIX_RE.match(text_for_kana)
        and surface == _clean_surface(text)
        and not any(0x4E00 <= ord(char) <= 0x9FFF for char in text_for_kana)
        and not (strict_kana_noise and _looks_like_katakana_surface(surface))
    ):
        return False
    if token.bbox[0] < split_x < token.bbox[2]:
        return True
    return _token_column(token, split_x) in {"left", "right"}


def _looks_like_katakana_surface(text: str) -> bool:
    kana_chars = [char for char in text if 0x3040 <= ord(char) <= 0x30FF]
    if len(kana_chars) < 2:
        return False
    return any(0x30A0 <= ord(char) <= 0x30FF for char in kana_chars) and not any(
        0x3040 <= ord(char) <= 0x309F for char in kana_chars
    )


def _strip_leading_non_japanese(text: str) -> str:
    for index, char in enumerate(text):
        if _char_is_japanese(char):
            return text[index:]
    return ""


def _surface_column(token: OcrToken, split_x: float) -> str:
    if token.bbox[0] < split_x < token.bbox[2] and "口" in token.text:
        return "right"
    return _token_column(token, split_x)


def _nearest_reading_token(
    tokens: list[OcrToken],
    surface_token: OcrToken,
    split_x: float,
    column: str,
    *,
    tolerance_scale: float = 1.0,
    prefer_same_column: bool = False,
    prefer_hiragana_reading: bool = False,
    allow_noisy_trailing_reading: bool = False,
) -> OcrToken | None:
    candidates = []
    cross_column_candidates = []
    tolerance = _row_match_tolerance(tokens) * tolerance_scale
    for token in tokens:
        if token is surface_token:
            continue
        distance_y = abs(_cy(token) - _cy(surface_token))
        if distance_y > tolerance:
            continue
        reading = _clean_reading(token.text, allow_noisy_trailing=allow_noisy_trailing_reading)
        if not reading or _looks_like_section_marker(token):
            continue
        edge_distance = _horizontal_edge_distance(surface_token.bbox, token.bbox)
        row_band = 0 if distance_y <= tolerance * 0.45 else 1
        token_column = _token_column(token, split_x)
        crosses_split = token.bbox[0] < split_x < token.bbox[2]
        column_penalty = 0 if token_column == column or not prefer_same_column else 1
        reading_penalty = _reading_script_penalty(reading) if prefer_hiragana_reading else 0
        crossing_penalty = 1 if prefer_same_column and crosses_split and token_column == column else 0
        candidate = (row_band, column_penalty, reading_penalty, crossing_penalty, edge_distance, distance_y, token)
        if token_column == column or crosses_split:
            candidates.append(candidate)
        else:
            cross_column_candidates.append(candidate)
    if not candidates:
        candidates = cross_column_candidates
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5], -item[6].confidence))
    return candidates[0][6]


def _reading_script_penalty(reading: str) -> int:
    kana = [char for char in reading if 0x3040 <= ord(char) <= 0x30FF]
    if not kana:
        return 3
    if all(0x3040 <= ord(char) <= 0x309F or char == "ー" for char in kana):
        return 0
    if any(0x3040 <= ord(char) <= 0x309F for char in kana):
        return 1
    return 2


def _horizontal_edge_distance(surface_bbox: list[float], reading_bbox: list[float]) -> float:
    if reading_bbox[0] >= surface_bbox[2]:
        return reading_bbox[0] - surface_bbox[2]
    if surface_bbox[0] >= reading_bbox[2]:
        return surface_bbox[0] - reading_bbox[2]
    return 0.0


def _looks_like_section_marker(token: OcrToken) -> bool:
    text = SURFACE_LEADING_NOISE_RE.sub("", token.text.strip())
    return len(text) == 1 and 0x3040 <= ord(text) <= 0x30FF and (token.bbox[2] - token.bbox[0]) < 48


def _char_is_hangul(ch: str) -> bool:
    return 0xAC00 <= ord(ch) <= 0xD7AF or 0x1100 <= ord(ch) <= 0x11FF or 0x3130 <= ord(ch) <= 0x318F


def _char_is_japanese(ch: str) -> bool:
    return 0x3040 <= ord(ch) <= 0x30FF or 0x4E00 <= ord(ch) <= 0x9FFF


def _kana_count(text: str) -> int:
    return sum(1 for ch in text if 0x3040 <= ord(ch) <= 0x30FF)


def _kanji_count(text: str) -> int:
    return sum(1 for ch in text if 0x4E00 <= ord(ch) <= 0x9FFF)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _row_match_tolerance(tokens: list[OcrToken]) -> float:
    heights = [max(1.0, token.bbox[3] - token.bbox[1]) for token in tokens]
    if not heights:
        return 36.0
    return max(24.0, min(58.0, _median(heights) * 1.65))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2)


def _has_hangul(text: str) -> bool:
    return any(0xAC00 <= ord(ch) <= 0xD7AF for ch in text)


def _has_japanese(text: str) -> bool:
    return any(0x3040 <= ord(ch) <= 0x30FF or 0x4E00 <= ord(ch) <= 0x9FFF for ch in text)
