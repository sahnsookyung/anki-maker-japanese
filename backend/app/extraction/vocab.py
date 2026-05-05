from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from app.core.ids import new_id
from app.core.script import classify_script
from app.extraction.field_evidence import static_evidence, token_evidence
from app.extraction.geometry import group_tokens_by_line, text_of, union_bbox
from app.models.schemas import OcrToken
from app.validation.dictionary import DictionaryValidator


JP_CLASSES = {"hiragana", "katakana", "kanji", "mixed"}
KANA_PREFIX_RE = re.compile(r"^[ぁ-ゖァ-ヺー]+")
SURFACE_LEADING_NOISE_RE = re.compile(r"^[□☐▢口日回]+")
KO_LEADING_NOISE_RE = re.compile(r"^[<ㄴhHzZVWC①②③④□☐▢口日回\s]+")

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
) -> list[dict]:
    validator = validator or DictionaryValidator()
    split_x = _column_split_x(japanese_tokens)
    items = _extract_vocab_items_from_layout(japanese_tokens, korean_tokens, split_x, validator)
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


def _extract_vocab_items_from_layout(
    japanese_tokens: list[OcrToken],
    korean_tokens: list[OcrToken],
    split_x: float,
    validator: DictionaryValidator,
) -> list[dict]:
    layout_rows = _layout_vocab_rows(japanese_tokens, korean_tokens, split_x)
    items: list[dict] = []
    for layout_row in layout_rows:
        evidence = [token for token in (layout_row.surface_token, layout_row.reading_token, layout_row.meaning_token) if token]
        confidence = min([token.confidence for token in evidence], default=layout_row.layout_confidence)
        _, validator_warnings = validator.validate_vocab(layout_row.surface, layout_row.reading)
        warning_codes = list(layout_row.warning_codes)
        if any("local dictionary" in warning for warning in validator_warnings):
            warning_codes.append("DICTIONARY_MISMATCH")
        warnings = _warnings_from_codes(warning_codes)
        items.append(
            {
                "id": new_id("vocab"),
                "type": "vocab_item",
                "surface": layout_row.surface,
                "reading": layout_row.reading,
                "meaning_ko": layout_row.meaning_ko,
                "field_evidence": {
                    "surface": _field_evidence(layout_row.surface_token, layout_row.surface, "surface"),
                    "reading": _field_evidence(layout_row.reading_token, layout_row.reading, "reading"),
                    "meaning_ko": _field_evidence(layout_row.meaning_token, layout_row.meaning_ko, "meaning_ko"),
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
) -> list[VocabLayoutRow]:
    rows: list[VocabLayoutRow] = []
    seen_surface_tokens: set[str] = set()
    table_sections = _section_ranges(japanese_tokens)
    for surface_token, surface, column, embedded_reading in _surface_candidates(japanese_tokens, split_x):
        if surface_token.id in seen_surface_tokens:
            continue
        seen_surface_tokens.add(surface_token.id)
        reading_token = surface_token if embedded_reading else _nearest_reading_token(japanese_tokens, surface_token, split_x, column)
        reading = embedded_reading or (_clean_reading(reading_token.text) if reading_token else "")
        row_y_values = [_cy(surface_token)]
        if reading_token:
            row_y_values.append(_cy(reading_token))
        row_y = sum(row_y_values) / len(row_y_values)
        meaning_token = _nearest_korean_meaning_token(korean_tokens, row_y, split_x, column)
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
                surface_bbox=surface_token.bbox if surface_token else None,
                reading_bbox=reading_token.bbox if reading_token else None,
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


def _field_evidence(token: OcrToken | None, text: str, field: str) -> dict[str, Any]:
    if token:
        return token_evidence([token], text)
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
) -> OcrToken | None:
    candidates = []
    cross_column_candidates = []
    tolerance = _row_match_tolerance(tokens)
    for token in tokens:
        text = _clean_korean_meaning(token.text)
        if not text:
            continue
        distance = abs(_cy(token) - row_y)
        if distance > tolerance:
            continue
        if _token_column(token, split_x) == column:
            candidates.append((distance, token))
        else:
            cross_column_candidates.append((distance, token))
    if not candidates:
        candidates = cross_column_candidates
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], -item[1].confidence))
    return candidates[0][1]


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
    if KANA_PREFIX_RE.match(cleaned) and surface == _clean_surface(text) and not any(0x4E00 <= ord(char) <= 0x9FFF for char in cleaned):
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


def _clean_reading(text: str) -> str:
    text = SURFACE_LEADING_NOISE_RE.sub("", text.strip())
    match = KANA_PREFIX_RE.match(text)
    return match.group(0) if match else ""


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
) -> list[tuple[OcrToken, str, str, str]]:
    candidates: list[tuple[OcrToken, str, str, str]] = []
    for token in sorted(tokens, key=lambda item: (_cy(item), item.bbox[0])):
        embedded_reading, surface = _split_combined_vocab_token(token.text)
        if not surface:
            surface = _clean_surface(token.text)
        if not surface or not _looks_like_surface_token(token, surface, split_x):
            continue
        candidates.append((token, surface, _surface_column(token, split_x), embedded_reading))
    return candidates


def _split_combined_vocab_token(text: str) -> tuple[str, str]:
    cleaned = SURFACE_LEADING_NOISE_RE.sub("", text.strip())
    reading_match = KANA_PREFIX_RE.match(cleaned)
    if not reading_match:
        return "", ""
    suffix = cleaned[reading_match.end() :]
    kanji_index = next((index for index, char in enumerate(suffix) if 0x4E00 <= ord(char) <= 0x9FFF), None)
    if kanji_index is None:
        return "", ""
    reading = reading_match.group(0)
    surface = "".join(ch for ch in suffix[kanji_index:] if _char_is_japanese(ch))
    if len(surface) < 2:
        return "", ""
    return reading, surface


def _looks_like_surface_token(token: OcrToken, surface: str, split_x: float) -> bool:
    if not surface:
        return False
    text = SURFACE_LEADING_NOISE_RE.sub("", token.text.strip())
    if KANA_PREFIX_RE.match(text) and surface == _clean_surface(text) and not any(0x4E00 <= ord(char) <= 0x9FFF for char in text):
        return False
    if token.bbox[0] < split_x < token.bbox[2]:
        return True
    return _token_column(token, split_x) in {"left", "right"}


def _surface_column(token: OcrToken, split_x: float) -> str:
    if token.bbox[0] < split_x < token.bbox[2] and "口" in token.text:
        return "right"
    return _token_column(token, split_x)


def _nearest_reading_token(
    tokens: list[OcrToken],
    surface_token: OcrToken,
    split_x: float,
    column: str,
) -> OcrToken | None:
    candidates = []
    cross_column_candidates = []
    tolerance = _row_match_tolerance(tokens)
    for token in tokens:
        if token is surface_token:
            continue
        distance_y = abs(_cy(token) - _cy(surface_token))
        if distance_y > tolerance:
            continue
        reading = _clean_reading(token.text)
        if not reading or _looks_like_section_marker(token):
            continue
        edge_distance = _horizontal_edge_distance(surface_token.bbox, token.bbox)
        row_band = 0 if distance_y <= tolerance * 0.45 else 1
        candidate = (row_band, edge_distance, distance_y, token)
        if _token_column(token, split_x) == column or token.bbox[0] < split_x < token.bbox[2]:
            candidates.append(candidate)
        else:
            cross_column_candidates.append(candidate)
    if not candidates:
        candidates = cross_column_candidates
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], -item[3].confidence))
    return candidates[0][3]


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
