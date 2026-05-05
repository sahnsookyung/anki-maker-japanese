from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from app.core.config import PREPROCESS_MAX_SIDE_LEN


@dataclass
class PreprocessResult:
    processed_path: Path
    width: int | None
    height: int | None
    warnings: list[str]
    original_width: int | None = None
    original_height: int | None = None


def preprocess_image(original_path: Path, output_path: Path) -> PreprocessResult:
    warnings: list[str] = []
    try:
        from PIL import Image, ImageOps, ImageFilter
    except Exception:
        shutil.copyfile(original_path, output_path)
        return PreprocessResult(output_path, None, None, ["Pillow is not installed; copied original image without preprocessing."])

    image = Image.open(original_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_width, original_height = image.width, image.height
    image = _perspective_crop_if_available(image, warnings)
    image = _resize_if_needed(image, PREPROCESS_MAX_SIDE_LEN, warnings, "Preprocessed image")
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.SHARPEN)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return PreprocessResult(output_path, image.width, image.height, warnings, original_width, original_height)


def _perspective_crop_if_available(image: "Image.Image", warnings: list[str]) -> "Image.Image":
    try:
        import cv2
        import numpy as np
    except Exception:
        warnings.append("OpenCV is not installed; skipped page polygon crop/deskew.")
        return image

    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        warnings.append("Could not detect page contour; using full image.")
        return image

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = image.width * image.height
    for contour in contours[:8]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        area = cv2.contourArea(approx)
        if len(approx) == 4 and area > image_area * 0.18:
            pts = approx.reshape(4, 2).astype("float32")
            warped = _four_point_transform(arr, pts, cv2, np)
            return Image.fromarray(warped)

    warnings.append("Page contour confidence was low; using full image.")
    return image


def _four_point_transform(arr, pts, cv2, np):
    rect = _order_points(pts, np)
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = max(int(width_a), int(width_b), 1)
    max_height = max(int(height_a), int(height_b), 1)
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(arr, matrix, (max_width, max_height))


def _order_points(pts, np):
    rect = np.zeros((4, 2), dtype="float32")
    sums = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    diffs = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def _resize_if_needed(image: "Image.Image", max_side_len: int, warnings: list[str], label: str) -> "Image.Image":
    if max_side_len <= 0:
        return image

    longest_side = max(image.width, image.height)
    if longest_side <= max_side_len:
        return image

    scale = max_side_len / float(longest_side)
    new_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    resampling = getattr(getattr(image, "Resampling", None), "LANCZOS", None)
    if resampling is None:
        from PIL import Image as PilImage

        resampling = PilImage.LANCZOS
    warnings.append(f"{label} was downscaled from {image.width}x{image.height} to {new_size[0]}x{new_size[1]}.")
    return image.resize(new_size, resample=resampling)
