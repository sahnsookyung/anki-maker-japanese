from __future__ import annotations

import pytest
from PIL import Image

from app.core.images import _resize_if_needed, preprocess_image
from app.ocr import paddle_provider
from app.ocr.paddle_provider import PaddleOcrProvider


def test_resize_if_needed_downscales_large_images() -> None:
    image = Image.new("RGB", (4000, 2000), "white")
    warnings: list[str] = []

    resized = _resize_if_needed(image, 1800, warnings, "Preprocessed image")

    assert resized.size == (1800, 900)
    assert warnings == ["Preprocessed image was downscaled from 4000x2000 to 1800x900."]


def test_preprocess_image_records_original_and_processed_dimensions(tmp_path) -> None:
    original = tmp_path / "original.png"
    processed = tmp_path / "processed.png"
    Image.new("RGB", (320, 180), "white").save(original)

    result = preprocess_image(original, processed)

    assert result.original_width == 320
    assert result.original_height == 180
    assert result.width == 320
    assert result.height == 180
    assert result.processed_path == processed
    assert processed.exists()


def test_paddle_box_to_bbox_handles_rectangles_and_polygons() -> None:
    provider = PaddleOcrProvider.__new__(PaddleOcrProvider)

    assert provider._box_to_bbox([10, 20, 30, 40]) == [10.0, 20.0, 30.0, 40.0]
    assert provider._box_to_bbox([[10, 20], [30, 20], [30, 40], [10, 40]]) == [10.0, 20.0, 30.0, 40.0]


def test_paddle_box_to_bbox_scales_back_to_processed_image_coordinates() -> None:
    provider = PaddleOcrProvider.__new__(PaddleOcrProvider)

    assert provider._box_to_bbox([100, 200, 300, 400], scale_x=1.125, scale_y=1.125) == [
        112.5,
        225.0,
        337.5,
        450.0,
    ]


def test_paddle_load_image_returns_manual_resize_scale(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paddle_provider, "PADDLE_OCR_MAX_SIDE_LEN", 1600)
    image_path = tmp_path / "page.png"
    Image.new("RGB", (1800, 900), "white").save(image_path)
    provider = PaddleOcrProvider.__new__(PaddleOcrProvider)

    image, scale_x, scale_y = provider._load_image(image_path)

    assert image.shape[:2] == (800, 1600)
    assert scale_x == pytest.approx(1.125)
    assert scale_y == pytest.approx(1.125)
