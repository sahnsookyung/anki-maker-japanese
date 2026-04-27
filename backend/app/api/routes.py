from __future__ import annotations

from pathlib import Path
import shutil

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.core.config import EXPORT_DIR, OCR_COMPARE_PROVIDER, UPLOAD_DIR
from app.core.ids import new_id
from app.db import database
from app.export.tsv import write_tsv
from app.extraction.pipeline import process_page
from app.models.schemas import CardUpdate, DocumentParseResult, ExportRequest, ExportResponse, OcrComparison, Page
from app.ocr.comparison import compare_ocr_tokens
from app.vision.paddle_ocr_vl import get_paddle_ocr_vl_parser


router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/pages")
def pages() -> list[Page]:
    return database.list_pages()


@router.post("/pages/upload")
async def upload_page(file: UploadFile = File(...)) -> dict[str, str]:
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
        processed_image_path=None,
        page_type="uploaded",
        page_type_confidence=0.0,
        created_at=database.utc_now(),
    )
    database.upsert_page(page)
    return {"page_id": page_id, "status": "uploaded"}


@router.post("/pages/{page_id}/process")
def process(page_id: str):
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found.")
    return process_page(page)


@router.get("/pages/{page_id}/ocr")
def page_ocr(page_id: str):
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found.")
    return {
        "page": page,
        "tokens": database.get_tokens(page_id),
    }


@router.get("/pages/{page_id}/ocr/compare", response_model=OcrComparison)
def compare_page_ocr(page_id: str, provider: str = Query(default=OCR_COMPARE_PROVIDER)):
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found.")
    image_path = Path(page.processed_image_path or page.original_image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Page image not found.")
    primary_tokens = database.get_tokens(page_id)
    return compare_ocr_tokens(image_path, page_id, primary_tokens, provider)


@router.post("/pages/{page_id}/document/parse", response_model=DocumentParseResult)
def parse_page_document(page_id: str) -> DocumentParseResult:
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found.")
    image_path = Path(page.processed_image_path or page.original_image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Page image not found.")
    try:
        return get_paddle_ocr_vl_parser().parse(image_path, page_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"PaddleOCR-VL parse failed: {exc}") from exc


@router.get("/pages/{page_id}/cards")
def page_cards(page_id: str):
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail="Page not found.")
    return database.get_cards(page_id)


@router.patch("/cards/{card_id}")
def update_card(card_id: str, patch: CardUpdate):
    card = database.get_card(card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found.")
    data = card.model_dump()
    for key, value in patch.model_dump(exclude_unset=True).items():
        if value is not None:
            data[key] = value
    updated = type(card)(**data)
    database.upsert_card(updated)
    return updated


@router.post("/cards/{card_id}/approve")
def approve_card(card_id: str):
    card = database.get_card(card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found.")
    card.status = "approved"
    database.upsert_card(card)
    return card


@router.post("/exports/tsv", response_model=ExportResponse)
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


@router.get("/exports/{filename}")
def download_export(filename: str):
    path = EXPORT_DIR / filename
    if not path.exists() or path.suffix != ".tsv":
        raise HTTPException(status_code=404, detail="Export not found.")
    return FileResponse(path, media_type="text/tab-separated-values; charset=utf-8", filename=filename)
