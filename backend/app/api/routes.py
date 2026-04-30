from __future__ import annotations

from pathlib import Path
import re
import shutil
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

from app.core.config import (
    CROP_DIR,
    EXPORT_DIR,
    OCR_COMPARE_PROVIDER,
    OCR_VL_PAGE_WORKER_MAX_RSS_MB,
    PROCESSED_DIR,
    UPLOAD_DIR,
)
from app.core.ids import new_id
from app.db import database
from app.export.anki_csv import write_csv
from app.export.tsv import write_tsv
from app.extraction.pipeline import process_page
from app.models.schemas import (
    CardUpdate,
    DocumentParseResult,
    ExportRequest,
    ExportResponse,
    FieldOcrPreviewRequest,
    FieldOcrPreviewResponse,
    OcrComparison,
    OcrRuntimeStatus,
    Page,
    PageUpdate,
)
from app.ocr.comparison import compare_ocr_tokens
from app.ocr.crop_worker import CropOcrError, crop_ocr_worker
from app.ocr.engines import PADDLEOCR_ENGINE, PADDLEOCR_VL_ENGINE, SUPPORTED_OCR_ENGINES, normalize_ocr_engine
from app.ocr.page_worker import run_document_parse_worker, run_page_process_worker
from app.ocr.runtime import ocr_runtime_job


router = APIRouter(prefix="/api")

PAGE_NOT_FOUND = "Page not found."
PAGE_IMAGE_NOT_FOUND = "Page image not found."
CARD_NOT_FOUND = "Card not found."
EXPORT_NOT_FOUND = "Export not found."
OCR_BUSY_DETAIL = "Another OCR job is already running."
RESPONSES = {
    400: {"description": "Unsupported request."},
    409: {"description": OCR_BUSY_DETAIL},
    404: {"description": "Requested resource was not found."},
    503: {"description": "Optional OCR/VLM service is unavailable."},
}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/pages")
def pages() -> list[Page]:
    return database.list_pages()


@router.post("/pages/upload", responses={400: RESPONSES[400]})
async def upload_page(file: Annotated[UploadFile, File(...)]) -> dict[str, str]:
    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Unsupported image type.")
    upload_name = Path(file.filename or "").name.strip()
    display_name = Path(upload_name).stem.strip()
    existing_page = database.get_page_by_upload_name(upload_name) if upload_name else None
    page_id = existing_page.id if existing_page else new_id("page")
    destination = UPLOAD_DIR / f"{page_id}{suffix}"
    page_display_name = (existing_page.display_name if existing_page else None) or display_name or page_id
    if existing_page:
        _delete_runtime_file(existing_page.original_image_path, UPLOAD_DIR)
        _delete_runtime_file(existing_page.processed_image_path, PROCESSED_DIR)
        _delete_page_crops(page_id)
        database.replace_tokens(page_id, [])
        database.replace_cards(page_id, [])
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    page = Page(
        id=page_id,
        original_image_path=str(destination),
        upload_name=upload_name or (existing_page.upload_name if existing_page else None),
        display_name=page_display_name,
        processed_image_path=None,
        page_type="uploaded",
        page_type_confidence=0.0,
        created_at=existing_page.created_at if existing_page else database.utc_now(),
    )
    database.upsert_page(page)
    return {"page_id": page_id, "status": "replaced" if existing_page else "uploaded"}


@router.patch("/pages/{page_id}", responses={404: RESPONSES[404]})
def update_page(page_id: str, patch: PageUpdate) -> Page:
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=PAGE_NOT_FOUND)
    display_name = patch.display_name.strip() if patch.display_name is not None else page.display_name
    if display_name == "":
        display_name = None
    updated = database.update_page_display_name(page_id, display_name)
    if not updated:
        raise HTTPException(status_code=404, detail=PAGE_NOT_FOUND)
    return updated


@router.delete("/pages/{page_id}", responses={404: RESPONSES[404]})
def delete_page(page_id: str) -> dict[str, str]:
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=PAGE_NOT_FOUND)
    database.delete_page(page_id)
    _delete_runtime_file(page.original_image_path, UPLOAD_DIR)
    _delete_runtime_file(page.processed_image_path, PROCESSED_DIR)
    _delete_page_crops(page_id)
    return {"page_id": page_id, "status": "deleted"}


@router.post("/pages/{page_id}/process", responses={400: RESPONSES[400], 404: RESPONSES[404], 409: RESPONSES[409], 503: RESPONSES[503]})
def process(
    page_id: str,
    engine: Annotated[
        str,
        Query(description=f"OCR engine to use for candidate generation: {', '.join(sorted(SUPPORTED_OCR_ENGINES))}."),
    ] = PADDLEOCR_ENGINE,
):
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=PAGE_NOT_FOUND)
    try:
        normalized_engine = normalize_ocr_engine(engine)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with ocr_runtime_job(blocking=False) as acquired:
        if not acquired:
            raise HTTPException(status_code=409, detail=OCR_BUSY_DETAIL)
        try:
            if normalized_engine == PADDLEOCR_VL_ENGINE:
                return run_page_process_worker(page.id, normalized_engine, max_rss_mb=OCR_VL_PAGE_WORKER_MAX_RSS_MB)
            return process_page(page, engine=normalized_engine)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/pages/dedupe")
def dedupe_pages() -> dict[str, object]:
    pages = database.list_pages()
    grouped: dict[str, list[Page]] = {}
    for page in pages:
        key = _page_cleanup_key(page)
        if not key:
            continue
        grouped.setdefault(key, []).append(page)
    removed: list[dict[str, str]] = []
    for group in grouped.values():
        if len(group) <= 1:
            continue
        keep = max(group, key=lambda item: item.created_at)
        for page in group:
            if page.id == keep.id:
                continue
            database.delete_page(page.id)
            _delete_runtime_file(page.original_image_path, UPLOAD_DIR)
            _delete_runtime_file(page.processed_image_path, PROCESSED_DIR)
            _delete_page_crops(page.id)
            removed.append({"page_id": page.id, "kept_page_id": keep.id})
    return {"removed_count": len(removed), "removed": removed}


@router.get("/pages/{page_id}/ocr", responses={404: RESPONSES[404]})
def page_ocr(page_id: str):
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=PAGE_NOT_FOUND)
    return {
        "page": page,
        "tokens": database.get_tokens(page_id),
    }


@router.get("/pages/{page_id}/ocr/compare", responses={404: RESPONSES[404], 409: RESPONSES[409]})
def compare_page_ocr(page_id: str, provider: Annotated[str, Query()] = OCR_COMPARE_PROVIDER) -> OcrComparison:
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=PAGE_NOT_FOUND)
    image_path = Path(page.processed_image_path or page.original_image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=PAGE_IMAGE_NOT_FOUND)
    primary_tokens = database.get_tokens(page_id)
    with ocr_runtime_job(blocking=False) as acquired:
        if not acquired:
            raise HTTPException(status_code=409, detail=OCR_BUSY_DETAIL)
        return compare_ocr_tokens(image_path, page_id, primary_tokens, provider)


@router.post(
    "/pages/{page_id}/document/parse",
    responses={404: RESPONSES[404], 409: RESPONSES[409], 503: RESPONSES[503]},
)
def parse_page_document(page_id: str) -> DocumentParseResult:
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=PAGE_NOT_FOUND)
    image_path = Path(page.processed_image_path or page.original_image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=PAGE_IMAGE_NOT_FOUND)
    with ocr_runtime_job(blocking=False) as acquired:
        if not acquired:
            raise HTTPException(status_code=409, detail=OCR_BUSY_DETAIL)
        try:
            return run_document_parse_worker(image_path, page_id, max_rss_mb=OCR_VL_PAGE_WORKER_MAX_RSS_MB)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=f"PaddleOCR-VL parse failed: {exc}") from exc


@router.get("/pages/{page_id}/cards", responses={404: RESPONSES[404]})
def page_cards(page_id: str):
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=PAGE_NOT_FOUND)
    return database.get_cards(page_id)


@router.patch("/cards/{card_id}", responses={404: RESPONSES[404]})
def update_card(card_id: str, patch: CardUpdate):
    card = database.get_card(card_id)
    if not card:
        raise HTTPException(status_code=404, detail=CARD_NOT_FOUND)
    data = card.model_dump()
    for key, value in patch.model_dump(exclude_unset=True).items():
        if value is not None:
            data[key] = value
    updated = type(card)(**data)
    database.upsert_card(updated)
    return updated


@router.post(
    "/cards/{card_id}/field-ocr/preview",
    responses={400: RESPONSES[400], 404: RESPONSES[404], 409: RESPONSES[409], 503: RESPONSES[503]},
)
def preview_card_field_ocr(card_id: str, request: FieldOcrPreviewRequest) -> FieldOcrPreviewResponse:
    card = database.get_card(card_id)
    if not card:
        raise HTTPException(status_code=404, detail=CARD_NOT_FOUND)
    page = database.get_page(card.page_id)
    if not page:
        raise HTTPException(status_code=404, detail=PAGE_NOT_FOUND)
    image_path = Path(page.processed_image_path or page.original_image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=PAGE_IMAGE_NOT_FOUND)
    page_width, page_height = _page_image_size(page, image_path)
    with ocr_runtime_job(blocking=False) as acquired:
        if not acquired:
            raise HTTPException(status_code=409, detail=OCR_BUSY_DETAIL)
        try:
            return crop_ocr_worker.preview(
                image_path=image_path,
                page_id=page.id,
                card_id=card.id,
                source=card.source,
                field=request.field,
                bbox=request.bbox,
                page_width=page_width,
                page_height=page_height,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except CropOcrError as exc:
            raise HTTPException(status_code=503, detail=f"Crop OCR preview failed: {exc}") from exc


@router.get("/ocr/runtime")
def ocr_runtime() -> OcrRuntimeStatus:
    crop_ocr_worker.offload_if_idle()
    return crop_ocr_worker.status()


@router.post("/cards/{card_id}/approve", responses={404: RESPONSES[404]})
def approve_card(card_id: str):
    card = database.get_card(card_id)
    if not card:
        raise HTTPException(status_code=404, detail=CARD_NOT_FOUND)
    card.status = "approved"
    database.upsert_card(card)
    return card


@router.post("/exports/tsv")
def export_tsv(request: ExportRequest) -> ExportResponse:
    cards = _exportable_cards(request)
    export_id = new_id("export")
    path = EXPORT_DIR / f"{export_id}.tsv"
    write_tsv(path, cards)
    return ExportResponse(
        export_id=export_id,
        path=str(path),
        card_count=len(cards),
        download_url=f"/api/exports/{export_id}.tsv",
    )


@router.post("/exports/csv")
def export_csv(request: ExportRequest) -> ExportResponse:
    cards = _exportable_cards(request)
    export_id = new_id("export")
    path = EXPORT_DIR / f"{export_id}.csv"
    write_csv(path, cards)
    return ExportResponse(
        export_id=export_id,
        path=str(path),
        card_count=len(cards),
        download_url=f"/api/exports/{export_id}.csv",
    )


@router.get("/exports/{filename}", responses={404: RESPONSES[404]})
def download_export(filename: str):
    if Path(filename).name != filename or Path(filename).suffix not in {".csv", ".tsv"}:
        raise HTTPException(status_code=404, detail=EXPORT_NOT_FOUND)
    path = EXPORT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=EXPORT_NOT_FOUND)
    media_type = "text/csv; charset=utf-8" if filename.endswith(".csv") else "text/tab-separated-values; charset=utf-8"
    return FileResponse(path, media_type=media_type, filename=filename)


def _exportable_cards(request: ExportRequest) -> list:
    page_ids = set(request.page_ids)
    cards = database.get_cards()
    if page_ids:
        cards = [card for card in cards if card.page_id in page_ids]
    if request.approved_only:
        cards = [card for card in cards if card.status == "approved"]
    cards = [card for card in cards if card.review_state != "red" or request.include_red]
    if not request.include_yellow:
        cards = [card for card in cards if card.review_state != "yellow"]
    return cards


def _delete_runtime_file(path_value: str | None, allowed_dir: Path) -> None:
    if not path_value:
        return
    path = Path(path_value)
    try:
        if path.is_file() and path.resolve().parent == allowed_dir.resolve():
            path.unlink()
    except OSError:
        return


def _delete_page_crops(page_id: str) -> None:
    for crop_path in CROP_DIR.glob(f"*{page_id}*"):
        try:
            if crop_path.is_file():
                crop_path.unlink()
        except OSError:
            continue


def _page_cleanup_key(page: Page) -> str | None:
    upload_name = (page.upload_name or "").strip()
    if upload_name:
        return f"upload:{upload_name.casefold()}"

    # Legacy rows created before upload_name tracking can still be deduped when their
    # stored original filename is a human-facing import name rather than an internal page id.
    original_name = Path(page.original_image_path).name.strip()
    if not original_name:
        return None
    original_stem = Path(original_name).stem.strip()
    if not original_stem or re.fullmatch(r"page_[0-9a-f]{12}", original_stem):
        return None
    display_name = (page.display_name or "").strip()
    if display_name and display_name != original_stem:
        return None
    return f"legacy:{original_name.casefold()}"


def _page_image_size(page: Page, image_path: Path) -> tuple[int, int]:
    if page.image_width and page.image_height:
        return page.image_width, page.image_height
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        return image.size
