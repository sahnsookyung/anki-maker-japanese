from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re

from app.core.config import KOREAN_GLOSSARY_PATH
from app.core.ids import new_id
from app.extraction.field_evidence import static_evidence, token_evidence
from app.extraction.geometry import group_tokens_by_line, text_of, union_bbox
from app.models.schemas import OcrToken
from app.validation.dictionary import DictionaryValidator


JP_CLASSES = {"hiragana", "katakana", "kanji", "mixed"}
KANA_PREFIX_RE = re.compile(r"^[ぁ-ゖァ-ヺー]+")
SURFACE_LEADING_NOISE_RE = re.compile(r"^[□☐▢口日回]+")
KO_LEADING_NOISE_RE = re.compile(r"^[<ㄴhHzZVWC①②③④□☐▢口日回\s]+")


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
    return _dedupe(items)


def extract_vocab_items_dual_ocr(
    japanese_tokens: list[OcrToken],
    korean_tokens: list[OcrToken],
    validator: DictionaryValidator | None = None,
) -> list[dict]:
    validator = validator or DictionaryValidator()
    split_x = _column_split_x(japanese_tokens)
    items = _extract_vocab_items_from_layout(japanese_tokens, korean_tokens, split_x, validator)
    if items:
        return _dedupe(items)

    rows = _japanese_vocab_rows(japanese_tokens)
    for row in rows:
        for column in ("left", "right"):
            side_tokens = [token for token in row if _token_column(token, split_x) == column]
            item = _dual_vocab_item(side_tokens, korean_tokens, split_x, column, validator)
            if item:
                items.append(item)
    return _dedupe(items)


def _extract_vocab_items_from_layout(
    japanese_tokens: list[OcrToken],
    korean_tokens: list[OcrToken],
    split_x: float,
    validator: DictionaryValidator,
) -> list[dict]:
    glossary = _load_korean_glossary()
    surface_tokens = _surface_candidates(japanese_tokens, split_x, glossary)
    items: list[dict] = []
    for surface_token, surface, column in surface_tokens:
        glossary_entry = glossary.get(surface)
        reading_token = _nearest_reading_token(japanese_tokens, surface_token, split_x, column, glossary_entry)
        reading = glossary_entry["reading"] if glossary_entry else ""
        if not reading and reading_token:
            reading = _clean_reading(reading_token.text)
        if not reading:
            continue

        row_y_values = [_cy(surface_token)]
        if reading_token:
            row_y_values.append(_cy(reading_token))
        row_y = sum(row_y_values) / len(row_y_values)
        meaning_token = _nearest_korean_meaning_token(korean_tokens, row_y, split_x, column)
        meaning = glossary_entry["meaning_ko"] if glossary_entry else ""
        if not meaning and meaning_token:
            meaning = _clean_korean_meaning(meaning_token.text)
        if not meaning:
            continue

        evidence = [surface_token]
        if reading_token:
            evidence.append(reading_token)
        if meaning_token:
            evidence.append(meaning_token)
        bbox = union_bbox([token.bbox for token in evidence])
        confidence = min(token.confidence for token in evidence)
        _, warnings = validator.validate_vocab(surface, reading)
        if glossary_entry and meaning_token:
            observed_meaning = _clean_korean_meaning(meaning_token.text)
            if observed_meaning and observed_meaning not in meaning:
                warnings.append(f"Korean OCR gloss '{observed_meaning}' was normalized with the local glossary.")
        elif glossary_entry and not meaning_token:
            warnings.append("Korean gloss was filled from the local glossary because OCR did not recover it.")
        items.append(
            {
                "id": new_id("vocab"),
                "type": "vocab_item",
                "surface": surface,
                "reading": reading,
                "meaning_ko": meaning,
                "field_evidence": {
                    "surface": token_evidence([surface_token], surface),
                    "reading": token_evidence([reading_token], reading) if reading_token else static_evidence(reading, "glossary"),
                    "meaning_ko": token_evidence([meaning_token], meaning) if meaning_token else static_evidence(meaning, "glossary"),
                },
                "evidence_tokens": [token.id for token in evidence],
                "bbox": bbox,
                "confidence": round(confidence, 3),
                "needs_review": bool(warnings),
                "warnings": warnings,
            }
        )
    items.extend(_supplement_vocab_items_from_glossary(japanese_tokens, korean_tokens, validator, items, glossary))
    return items


def _japanese_vocab_rows(tokens: list[OcrToken]) -> list[list[OcrToken]]:
    rows: list[list[OcrToken]] = []
    for line in group_tokens_by_line(tokens, tolerance=24.0):
        content = [token for token in line if _has_japanese(token.text)]
        if len(content) < 2:
            continue
        center_y = sum(_cy(token) for token in content) / len(content)
        if center_y < 360:
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
    for token in tokens:
        text = _clean_korean_meaning(token.text)
        if not text:
            continue
        if _token_column(token, split_x) != column:
            continue
        distance = abs(_cy(token) - row_y)
        if distance <= 36:
            candidates.append((distance, token))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], -item[1].confidence))
    return candidates[0][1]


def _column_split_x(tokens: list[OcrToken]) -> float:
    if not tokens:
        return 0.0
    max_x = max(token.bbox[2] for token in tokens)
    return max_x * 0.55


def _token_column(token: OcrToken, split_x: float) -> str:
    return "left" if ((token.bbox[0] + token.bbox[2]) / 2) < split_x else "right"


def _cy(token: OcrToken) -> float:
    return (token.bbox[1] + token.bbox[3]) / 2


def _clean_surface(text: str) -> str:
    text = SURFACE_LEADING_NOISE_RE.sub("", text.strip())
    return "".join(ch for ch in text if _char_is_japanese(ch))


def _clean_reading(text: str) -> str:
    match = KANA_PREFIX_RE.match(text.strip())
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
    glossary: dict[str, dict[str, str]],
) -> list[tuple[OcrToken, str, str]]:
    candidates: list[tuple[OcrToken, str, str]] = []
    surfaces = sorted(glossary, key=len, reverse=True)
    for token in sorted(tokens, key=lambda item: (_cy(item), item.bbox[0])):
        if _cy(token) < 390:
            continue
        matched_surface = _surface_from_glossary(token.text, surfaces)
        if matched_surface:
            surface = matched_surface
        else:
            surface = _clean_surface(token.text)
            if _kana_count(surface) > _kanji_count(surface):
                continue
        if not surface or not _looks_like_surface_token(token, surface, split_x):
            continue
        candidates.append((token, surface, _surface_column(token, split_x)))
    return candidates


def _surface_from_glossary(text: str, surfaces: list[str]) -> str:
    for surface in surfaces:
        if surface in text:
            return surface
    return ""


def _looks_like_surface_token(token: OcrToken, surface: str, split_x: float) -> bool:
    if not surface:
        return False
    text = token.text.strip()
    if KANA_PREFIX_RE.match(text) and surface == _clean_surface(text):
        return False
    if token.bbox[0] < split_x < token.bbox[2]:
        return True
    return _token_column(token, split_x) in {"left", "right"} and token.bbox[0] < split_x + 360


def _surface_column(token: OcrToken, split_x: float) -> str:
    if token.bbox[0] < split_x < token.bbox[2] and "口" in token.text:
        return "right"
    return _token_column(token, split_x)


def _nearest_reading_token(
    tokens: list[OcrToken],
    surface_token: OcrToken,
    split_x: float,
    column: str,
    glossary_entry: dict[str, str] | None,
) -> OcrToken | None:
    expected_reading = glossary_entry["reading"] if glossary_entry else ""
    candidates = []
    for token in tokens:
        if token is surface_token:
            continue
        if _token_column(token, split_x) != column and not (token.bbox[0] < split_x < token.bbox[2]):
            continue
        distance_y = abs(_cy(token) - _cy(surface_token))
        if distance_y > 40:
            continue
        reading = _clean_reading(token.text)
        has_expected = bool(expected_reading and expected_reading in token.text)
        if not reading and not has_expected:
            continue
        if token.bbox[0] < surface_token.bbox[0] and not has_expected:
            continue
        distance_x = abs(token.bbox[0] - surface_token.bbox[0])
        priority = 0 if has_expected else 1
        candidates.append((priority, distance_y, distance_x, token))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2], -item[3].confidence))
    return candidates[0][3]


def _supplement_vocab_items_from_glossary(
    japanese_tokens: list[OcrToken],
    korean_tokens: list[OcrToken],
    validator: DictionaryValidator,
    existing_items: list[dict],
    glossary: dict[str, dict[str, str]],
) -> list[dict]:
    if len(japanese_tokens) < 20 or len(existing_items) >= 30:
        return []
    existing = {(str(item.get("surface")), str(item.get("reading"))) for item in existing_items}
    supplements: list[dict] = []
    for entry in glossary.values():
        key = (entry["surface"], entry["reading"])
        if key in existing:
            continue
        evidence: list[OcrToken] = []
        reading_token = _token_containing(japanese_tokens, entry["reading"])
        meaning_token = _similar_meaning_token(korean_tokens, entry["meaning_ko"])
        if not reading_token and not _meaning_token_is_strong_match(meaning_token, entry["meaning_ko"]):
            meaning_token = None
        if reading_token:
            evidence.append(reading_token)
        if meaning_token:
            evidence.append(meaning_token)
        if not evidence:
            continue
        _, warnings = validator.validate_vocab(entry["surface"], entry["reading"])
        warnings.append("Entry was supplemented from the local glossary because OCR missed one or more fields.")
        bbox = union_bbox([token.bbox for token in evidence])
        confidence = min(token.confidence for token in evidence)
        supplements.append(
            {
                "id": new_id("vocab"),
                "type": "vocab_item",
                "surface": entry["surface"],
                "reading": entry["reading"],
                "meaning_ko": entry["meaning_ko"],
                "field_evidence": {
                    "surface": static_evidence(entry["surface"], "glossary"),
                    "reading": token_evidence([reading_token], entry["reading"]) if reading_token else static_evidence(entry["reading"], "glossary"),
                    "meaning_ko": (
                        token_evidence([meaning_token], entry["meaning_ko"])
                        if meaning_token
                        else static_evidence(entry["meaning_ko"], "glossary")
                    ),
                },
                "evidence_tokens": [token.id for token in evidence],
                "bbox": bbox,
                "confidence": round(confidence, 3),
                "needs_review": True,
                "warnings": warnings,
            }
        )
        existing.add(key)
    return supplements


def _token_containing(tokens: list[OcrToken], value: str) -> OcrToken | None:
    candidates = [token for token in tokens if value and value in token.text and _cy(token) >= 390]
    if not candidates:
        return None
    candidates.sort(key=lambda token: (-token.confidence, _cy(token)))
    return candidates[0]


def _similar_meaning_token(tokens: list[OcrToken], meaning: str) -> OcrToken | None:
    candidates = []
    for token in tokens:
        cleaned = _clean_korean_meaning(token.text)
        if not cleaned:
            continue
        score = _hangul_subsequence_score(cleaned, meaning)
        if score >= 0.6:
            candidates.append((score, token.confidence, token))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    return candidates[0][2]


def _hangul_subsequence_score(actual: str, expected: str) -> float:
    actual_hangul = [char for char in actual if _char_is_hangul(char)]
    expected_hangul = [char for char in expected if _char_is_hangul(char)]
    if not actual_hangul or not expected_hangul:
        return 0.0
    pos = 0
    matched = 0
    for char in expected_hangul:
        try:
            found = actual_hangul.index(char, pos)
        except ValueError:
            continue
        matched += 1
        pos = found + 1
    return matched / len(expected_hangul)


def _meaning_token_is_strong_match(token: OcrToken | None, meaning: str) -> bool:
    if not token:
        return False
    actual_hangul = [char for char in _clean_korean_meaning(token.text) if _char_is_hangul(char)]
    expected_hangul = [char for char in meaning if _char_is_hangul(char)]
    if len(expected_hangul) < 3 or not actual_hangul:
        return False
    return actual_hangul[0] == expected_hangul[0]


@lru_cache(maxsize=1)
def _load_korean_glossary() -> dict[str, dict[str, str]]:
    path = KOREAN_GLOSSARY_PATH
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    glossary: dict[str, dict[str, str]] = {}
    for item in data:
        surface = str(item.get("surface") or "")
        reading = str(item.get("reading") or "")
        meaning = str(item.get("meaning_ko") or "")
        if surface and reading and meaning:
            glossary[surface] = {"surface": surface, "reading": reading, "meaning_ko": meaning}
    return glossary


def _char_is_hangul(ch: str) -> bool:
    return 0xAC00 <= ord(ch) <= 0xD7AF or 0x1100 <= ord(ch) <= 0x11FF or 0x3130 <= ord(ch) <= 0x318F


def _char_is_japanese(ch: str) -> bool:
    return 0x3040 <= ord(ch) <= 0x30FF or 0x4E00 <= ord(ch) <= 0x9FFF


def _kana_count(text: str) -> int:
    return sum(1 for ch in text if 0x3040 <= ord(ch) <= 0x30FF)


def _kanji_count(text: str) -> int:
    return sum(1 for ch in text if 0x4E00 <= ord(ch) <= 0x9FFF)


def _dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for item in items:
        key = f"{item.get('surface')}|{item.get('reading')}|{item.get('meaning_ko')}"
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _has_hangul(text: str) -> bool:
    return any(0xAC00 <= ord(ch) <= 0xD7AF for ch in text)


def _has_japanese(text: str) -> bool:
    return any(0x3040 <= ord(ch) <= 0x30FF or 0x4E00 <= ord(ch) <= 0x9FFF for ch in text)
