from __future__ import annotations

from pathlib import Path
import shutil
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.core.config import CROP_DIR, EXPORT_DIR, OCR_COMPARE_PROVIDER, PROCESSED_DIR, UPLOAD_DIR
from app.core.ids import new_id
from app.db import database
from app.export.tsv import write_tsv
from app.extraction.pipeline import process_page
from app.models.schemas import CardUpdate, DocumentParseResult, ExportRequest, ExportResponse, OcrComparison, Page, PageUpdate
from app.ocr.comparison import compare_ocr_tokens
from app.vision.paddle_ocr_vl import get_paddle_ocr_vl_parser


router = APIRouter(prefix="/api")

PAGE_NOT_FOUND = "Page not found."
PAGE_IMAGE_NOT_FOUND = "Page image not found."
CARD_NOT_FOUND = "Card not found."
EXPORT_NOT_FOUND = "Export not found."
RESPONSES = {
    400: {"description": "Unsupported request."},
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
    page_id = new_id("page")
    destination = UPLOAD_DIR / f"{page_id}{suffix}"
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    page = Page(
        id=page_id,
        original_image_path=str(destination),
        display_name=Path(file.filename or page_id).stem or page_id,
        processed_image_path=None,
        page_type="uploaded",
        page_type_confidence=0.0,
        created_at=database.utc_now(),
    )
    database.upsert_page(page)
    return {"page_id": page_id, "status": "uploaded"}


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


@router.post("/pages/{page_id}/process", responses={404: RESPONSES[404]})
def process(page_id: str):
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=PAGE_NOT_FOUND)
    return process_page(page)


@router.get("/pages/{page_id}/ocr", responses={404: RESPONSES[404]})
def page_ocr(page_id: str):
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=PAGE_NOT_FOUND)
    return {
        "page": page,
        "tokens": database.get_tokens(page_id),
    }


@router.get("/pages/{page_id}/ocr/compare", responses={404: RESPONSES[404]})
def compare_page_ocr(page_id: str, provider: Annotated[str, Query()] = OCR_COMPARE_PROVIDER) -> OcrComparison:
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=PAGE_NOT_FOUND)
    image_path = Path(page.processed_image_path or page.original_image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=PAGE_IMAGE_NOT_FOUND)
    primary_tokens = database.get_tokens(page_id)
    return compare_ocr_tokens(image_path, page_id, primary_tokens, provider)


@router.post("/pages/{page_id}/document/parse", responses={404: RESPONSES[404], 503: RESPONSES[503]})
def parse_page_document(page_id: str) -> DocumentParseResult:
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=PAGE_NOT_FOUND)
    image_path = Path(page.processed_image_path or page.original_image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=PAGE_IMAGE_NOT_FOUND)
    try:
        return get_paddle_ocr_vl_parser().parse(image_path, page_id)
    except Exception as exc:
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
    page_ids = set(request.page_ids)
    cards = database.get_cards()
    if page_ids:
        cards = [card for card in cards if card.page_id in page_ids]
    if request.approved_only:
        cards = [card for card in cards if card.status == "approved"]
    cards = [card for card in cards if card.review_state != "red" or request.include_red]
    if not request.include_yellow:
        cards = [card for card in cards if card.review_state != "yellow"]

    export_id = new_id("export")
    path = EXPORT_DIR / f"{export_id}.tsv"
    write_tsv(path, cards)
    return ExportResponse(
        export_id=export_id,
        path=str(path),
        card_count=len(cards),
        download_url=f"/api/exports/{export_id}.tsv",
    )


@router.get("/exports/{filename}", responses={404: RESPONSES[404]})
def download_export(filename: str):
    if Path(filename).name != filename or not filename.endswith(".tsv"):
        raise HTTPException(status_code=404, detail=EXPORT_NOT_FOUND)
    path = EXPORT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=EXPORT_NOT_FOUND)
    return FileResponse(path, media_type="text/tab-separated-values; charset=utf-8", filename=filename)


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
