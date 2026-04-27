from __future__ import annotations

from app.models.schemas import OcrToken


def union_bbox(boxes: list[list[float]]) -> list[float]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def group_tokens_by_line(tokens: list[OcrToken], tolerance: float = 18.0) -> list[list[OcrToken]]:
    ordered = sorted(tokens, key=lambda t: ((t.bbox[1] + t.bbox[3]) / 2, t.bbox[0]))
    lines: list[list[OcrToken]] = []
    centers: list[float] = []
    for token in ordered:
        cy = (token.bbox[1] + token.bbox[3]) / 2
        for idx, center in enumerate(centers):
            if abs(cy - center) <= tolerance:
                lines[idx].append(token)
                centers[idx] = (center * (len(lines[idx]) - 1) + cy) / len(lines[idx])
                break
        else:
            lines.append([token])
            centers.append(cy)
    for line in lines:
        line.sort(key=lambda t: t.bbox[0])
    return lines


def text_of(tokens: list[OcrToken], separator: str = " ") -> str:
    return separator.join(token.text for token in tokens if token.text).strip()
