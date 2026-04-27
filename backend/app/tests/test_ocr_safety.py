from __future__ import annotations

from PIL import Image

from app.core.images import _resize_if_needed
from app.ocr.paddle_provider import PaddleOcrProvider


def test_resize_if_needed_downscales_large_images() -> None:
    image = Image.new("RGB", (4000, 2000), "white")
    warnings: list[str] = []

    resized = _resize_if_needed(image, 1800, warnings, "Preprocessed image")

    assert resized.size == (1800, 900)
    assert warnings == ["Preprocessed image was downscaled from 4000x2000 to 1800x900."]


def test_paddle_box_to_bbox_handles_rectangles_and_polygons() -> None:
    provider = PaddleOcrProvider.__new__(PaddleOcrProvider)

    assert provider._box_to_bbox([10, 20, 30, 40]) == [10.0, 20.0, 30.0, 40.0]
    assert provider._box_to_bbox([[10, 20], [30, 20], [30, 40], [10, 40]]) == [10.0, 20.0, 30.0, 40.0]
