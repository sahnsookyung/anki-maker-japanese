from __future__ import annotations

from pathlib import Path

from app.core.config import CROP_DIR


def crop_bbox(image_path: Path, bbox: list[float], crop_id: str, padding: int = 18) -> Path:
    from PIL import Image

    image = Image.open(image_path)
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    left = max(0, x1 - padding)
    top = max(0, y1 - padding)
    right = min(image.width, x2 + padding)
    bottom = min(image.height, y2 + padding)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    crop_path = CROP_DIR / f"{crop_id}.png"
    image.crop((left, top, right, bottom)).save(crop_path)
    return crop_path
