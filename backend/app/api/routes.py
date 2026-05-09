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
    OCR_PAGE_WORKER_MAX_RSS_MB,
    OCR_VL_PAGE_WORKER_MAX_RSS_MB,
    PROCESSED_DIR,
    UPLOAD_DIR,
)
from app.core import config as runtime_config
from app.core.images import preprocess_image
from app.core.ids import new_id
from app.db import database
from app.export.anki_csv import write_export_csvs
from app.extraction.pipeline import process_page
from app.models.schemas import (
    CardUpdate,
    DocumentParseResult,
    ExportRequest,
    ExportResponse,
    FieldOcrPreviewRequest,
    FieldOcrPreviewResponse,
    OcrComparison,
    OcrRun,
    OcrRuntimeStatus,
    Page,
    PageUpdate,
)
from app.ocr.comparison import compare_ocr_tokens
from app.ocr.crop_worker import CropOcrError, crop_ocr_worker
from app.ocr.engines import PADDLEOCR_ENGINE, PADDLEOCR_VL_ENGINE, SUPPORTED_OCR_ENGINES, normalize_ocr_engine
from app.ocr.page_worker import run_document_parse_worker, run_page_process_worker
from app.ocr.profiles import (
    BASELINE_MODEL_PROFILE,
    BENCHMARK_ONLY_EXTRACTION_VARIANTS,
    DEFAULT_EXTRACTION_VARIANT,
    DEFAULT_KOREAN_PROFILE,
    available_korean_profile_payload,
    available_variant_payload,
    available_profile_payload,
    normalize_korean_profile,
    normalize_extraction_variant,
    profile_env_overrides,
    resolve_korean_ocr_profile,
    resolve_ocr_model_profile,
)
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
        database.clear_page_runs(page_id)
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


@router.post(
    "/pages/{page_id}/process",
    responses={400: RESPONSES[400], 404: RESPONSES[404], 409: RESPONSES[409], 503: RESPONSES[503]},
)
def process(
    page_id: str,
    engine: Annotated[
        str,
        Query(description=f"OCR engine to use for candidate generation: {', '.join(sorted(SUPPORTED_OCR_ENGINES))}."),
    ] = PADDLEOCR_ENGINE,
    model_profile: Annotated[
        str,
        Query(description="Experimental OCR model profile. Default keeps the frozen production control."),
    ] = BASELINE_MODEL_PROFILE,
    korean_profile: Annotated[
        str,
        Query(description="Experimental Korean OCR sub-profile for the two-pass vocab pipeline."),
    ] = DEFAULT_KOREAN_PROFILE,
    extraction_variant: Annotated[
        str,
        Query(description="Experimental extraction variant. Default keeps the frozen current extractor."),
    ] = DEFAULT_EXTRACTION_VARIANT,
    allow_benchmark_variant: Annotated[
        bool,
        Query(description="Internal benchmark harness opt-in for benchmark-only extraction variants."),
    ] = False,
):
    page = database.get_page(page_id)
    if not page:
        raise HTTPException(status_code=404, detail=PAGE_NOT_FOUND)
    try:
        normalized_engine = normalize_ocr_engine(engine)
        profile = resolve_ocr_model_profile(model_profile)
        korean_ocr_profile = resolve_korean_ocr_profile(korean_profile)
        normalized_variant = normalize_extraction_variant(extraction_variant)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if normalized_variant in BENCHMARK_ONLY_EXTRACTION_VARIANTS and not allow_benchmark_variant:
        raise HTTPException(status_code=400, detail=f"Extraction variant {normalized_variant!r} is benchmark-only.")
    if not profile.creates_candidates and normalized_engine == PADDLEOCR_ENGINE:
        raise HTTPException(status_code=400, detail=f"Profile {profile.id!r} is diagnostic-only and cannot create Anki candidates.")
    with ocr_runtime_job(blocking=False) as acquired:
        if not acquired:
            raise HTTPException(status_code=409, detail=OCR_BUSY_DETAIL)
        try:
            use_worker = (
                normalized_engine == PADDLEOCR_VL_ENGINE
                or profile.id != BASELINE_MODEL_PROFILE
                or korean_ocr_profile.id != DEFAULT_KOREAN_PROFILE
                or normalized_variant != DEFAULT_EXTRACTION_VARIANT
                or not _profile_matches_active_runtime_config(profile.id, korean_ocr_profile.id)
            )
            if use_worker:
                return run_page_process_worker(
                    page.id,
                    normalized_engine,
                    max_rss_mb=OCR_VL_PAGE_WORKER_MAX_RSS_MB if normalized_engine == PADDLEOCR_VL_ENGINE else OCR_PAGE_WORKER_MAX_RSS_MB,
                    env_overrides=profile_env_overrides(profile.id, korean_ocr_profile.id),
                    model_profile=profile.id,
                    korean_profile=korean_ocr_profile.id,
                    extraction_variant=normalized_variant,
                )
            return process_page(
                page,
                engine=normalized_engine,
                model_profile=profile.id,
                korean_profile=korean_ocr_profile.id,
                extraction_variant=normalized_variant,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/ocr/profiles")
def ocr_profiles() -> dict[str, object]:
    return {
        "profiles": available_profile_payload(),
        "korean_profiles": available_korean_profile_payload(),
        "variants": available_variant_payload(),
        "default_profile": BASELINE_MODEL_PROFILE,
        "default_korean_profile": DEFAULT_KOREAN_PROFILE,
        "default_variant": DEFAULT_EXTRACTION_VARIANT,
    }


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
    tokens = database.get_tokens(page_id)
    page, tokens = _ensure_review_artifacts(page, tokens, database.get_cards(page_id))
    document_parse = None if _has_stale_evidence_warning(page) else database.get_active_document_parse(page_id)
    return {
        "page": page,
        "tokens": tokens,
        "document_parse": document_parse,
    }


@router.get("/pages/{page_id}/ocr/runs", responses={404: RESPONSES[404]})
def page_ocr_runs(page_id: str) -> list[OcrRun]:
    if not database.get_page(page_id):
        raise HTTPException(status_code=404, detail=PAGE_NOT_FOUND)
    return database.list_ocr_runs(page_id)


@router.post("/pages/{page_id}/ocr/runs/{run_id}/activate", responses={404: RESPONSES[404]})
def activate_page_ocr_run(page_id: str, run_id: str) -> OcrRun:
    run = database.activate_ocr_run(page_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="OCR run was not found or is not successful.")
    return run


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


@router.post(
    "/cards/{card_id}/field-ocr/apply",
    responses={400: RESPONSES[400], 404: RESPONSES[404], 409: RESPONSES[409], 503: RESPONSES[503]},
)
def apply_card_field_ocr(card_id: str, request: FieldOcrPreviewRequest):
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
            preview = crop_ocr_worker.preview(
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
    run = database.get_active_ocr_run(page.id)
    existing_tokens = database.get_tokens(page.id, run.id if run else None)
    database.replace_tokens(page.id, [*existing_tokens, *preview.tokens], run.id if run else None)
    updated = _card_with_applied_field_preview(card, preview)
    database.upsert_card(updated)
    return {"card": updated, "tokens": preview.tokens}


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


@router.post("/exports/csv")
def export_csv(request: ExportRequest) -> ExportResponse:
    cards = _exportable_cards(request)
    export_id = new_id("export")
    files, note_count, generated_card_count = write_export_csvs(EXPORT_DIR, export_id, cards)
    return ExportResponse(
        export_id=export_id,
        files=files,
        note_count=note_count,
        estimated_generated_card_count=generated_card_count,
    )


@router.get("/exports/{filename}", responses={404: RESPONSES[404]})
def download_export(filename: str):
    if Path(filename).name != filename or Path(filename).suffix != ".csv":
        raise HTTPException(status_code=404, detail=EXPORT_NOT_FOUND)
    path = EXPORT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=EXPORT_NOT_FOUND)
    return FileResponse(path, media_type="text/csv; charset=utf-8", filename=filename)


def _card_with_applied_field_preview(card, preview: FieldOcrPreviewResponse):
    source = dict(card.source)
    source.update(preview.suggested_source)
    evidence = source.get("field_evidence")
    field_evidence = dict(evidence) if isinstance(evidence, dict) else {}
    field_evidence[preview.field] = preview.field_evidence
    source["field_evidence"] = field_evidence
    token_ids = list(source.get("evidence_tokens") if isinstance(source.get("evidence_tokens"), list) else [])
    token_ids.extend(preview.field_evidence.get("token_ids", []) if isinstance(preview.field_evidence, dict) else [])
    source["evidence_tokens"] = list(dict.fromkeys(token_id for token_id in token_ids if isinstance(token_id, str)))
    bbox = _union_bboxes([card.source_bbox, preview.bbox])
    warnings = _merge_warnings(
        [warning for warning in card.warnings if not (preview.field.startswith("choice_") and "four choices" in warning.lower())],
        preview.warnings,
    )
    confidence = max(float(card.confidence or 0.0), float(preview.confidence or 0.0))
    return card.model_copy(update={"source": source, "source_bbox": bbox or card.source_bbox, "warnings": warnings, "confidence": confidence})


def _union_bboxes(values: list[object]) -> list[float] | None:
    bboxes = []
    for value in values:
        if isinstance(value, list) and len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            x1, y1, x2, y2 = [float(item) for item in value]
            bboxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
    if not bboxes:
        return None
    return [min(bbox[0] for bbox in bboxes), min(bbox[1] for bbox in bboxes), max(bbox[2] for bbox in bboxes), max(bbox[3] for bbox in bboxes)]


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


def _ensure_review_artifacts(page: Page, tokens: list, cards: list) -> tuple[Page, list]:
    has_persisted_evidence = bool(tokens) or any(card.source_bbox for card in cards) or _has_document_parse_evidence(page.id)
    processed_path = Path(page.processed_image_path) if page.processed_image_path else None
    original_path = Path(page.original_image_path)
    if processed_path and not processed_path.exists() and original_path.exists():
        regenerated, evidence_safe = _regenerate_processed_image(page, original_path, has_persisted_evidence)
        if regenerated:
            return regenerated, tokens if evidence_safe else []

    if has_persisted_evidence and _has_stale_evidence_warning(page):
        return page, []

    image_path = Path(page.processed_image_path or page.original_image_path)
    if (not page.image_width or not page.image_height) and image_path.exists():
        try:
            width, height = _read_image_size(image_path)
        except OSError:
            return page, tokens
        updated = page.model_copy(update={"image_width": width, "image_height": height})
        database.upsert_page(updated)
        return updated, tokens
    return page, tokens


def _regenerate_processed_image(page: Page, original_path: Path, has_persisted_evidence: bool) -> tuple[Page | None, bool]:
    target = PROCESSED_DIR / f"{page.id}.png"
    try:
        preprocess = preprocess_image(original_path, target)
    except OSError:
        return None, False
    evidence_safe = _regenerated_geometry_matches(page, preprocess.width, preprocess.height, has_persisted_evidence)
    warnings = _merge_warnings(
        page.warnings,
        [
            *preprocess.warnings,
            "Regenerated processed image cache from the original upload.",
            *([] if evidence_safe else ["Existing OCR evidence needs reprocessing before boxes can be shown safely."]),
        ],
    )
    updated = page.model_copy(
        update={
            "processed_image_path": str(target),
            "image_width": preprocess.width if evidence_safe else None,
            "image_height": preprocess.height if evidence_safe else None,
            "warnings": warnings,
        }
    )
    database.upsert_page(updated)
    return updated, evidence_safe


def _regenerated_geometry_matches(
    page: Page,
    width: int | None,
    height: int | None,
    has_persisted_evidence: bool,
) -> bool:
    if not has_persisted_evidence:
        return True
    return bool(page.image_width and page.image_height and page.image_width == width and page.image_height == height)


def _read_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        return image.size


def _merge_warnings(existing: list[str], additions: list[str]) -> list[str]:
    merged = list(existing)
    for warning in additions:
        if warning and warning not in merged:
            merged.append(warning)
    return merged


def _has_stale_evidence_warning(page: Page) -> bool:
    return "Existing OCR evidence needs reprocessing before boxes can be shown safely." in page.warnings


def _has_document_parse_evidence(page_id: str) -> bool:
    document_parse = database.get_active_document_parse(page_id)
    return bool(document_parse and any(block.bbox for block in document_parse.blocks))


def _profile_matches_active_runtime_config(profile_id: str, korean_profile_id: str | None = None) -> bool:
    runtime_values = {
        "PADDLE_OCR_USE_LANGUAGE_PROFILE": str(runtime_config.PADDLE_OCR_USE_LANGUAGE_PROFILE).lower(),
        "PADDLE_OCR_LANG": runtime_config.PADDLE_OCR_LANG,
        "PADDLE_OCR_TEXT_DETECTION_MODEL_NAME": runtime_config.PADDLE_OCR_TEXT_DETECTION_MODEL_NAME,
        "PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME": runtime_config.PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME,
        "PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME": runtime_config.PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME,
        "PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME": runtime_config.PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME,
        "PADDLE_OCR_KOREAN_USE_LANGUAGE_PROFILE": str(runtime_config.PADDLE_OCR_KOREAN_USE_LANGUAGE_PROFILE).lower(),
        "PADDLE_OCR_KOREAN_LANG": runtime_config.PADDLE_OCR_KOREAN_LANG,
    }
    for key, expected in profile_env_overrides(profile_id, normalize_korean_profile(korean_profile_id)).items():
        actual = runtime_values.get(key)
        normalized_expected = (
            str(expected).lower()
            if key in {"PADDLE_OCR_USE_LANGUAGE_PROFILE", "PADDLE_OCR_KOREAN_USE_LANGUAGE_PROFILE"}
            else expected
        )
        if actual != normalized_expected:
            return False
    return True


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
