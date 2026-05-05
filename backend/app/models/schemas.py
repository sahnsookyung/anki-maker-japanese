from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Literal


BBox = list[float]


class OcrToken(BaseModel):
    id: str
    page_id: str
    text: str
    bbox: BBox
    confidence: float = 0.0
    script_class: str = "unknown"
    source: str = "unknown"


class Page(BaseModel):
    id: str
    original_image_path: str
    upload_name: str | None = None
    display_name: str | None = None
    processed_image_path: str | None = None
    active_ocr_run_id: str | None = None
    active_ocr_engine: str | None = None
    active_ocr_completed_at: str | None = None
    active_ocr_duration_ms: int | None = None
    page_type: str = "unprocessed"
    page_type_confidence: float = 0.0
    image_width: int | None = None
    image_height: int | None = None
    warnings: list[str] = Field(default_factory=list)
    created_at: str
    card_count: int = 0
    approved_card_count: int = 0
    red_card_count: int = 0


class CardCandidate(BaseModel):
    id: str
    page_id: str
    run_id: str | None = None
    source_type: str
    source_id: str
    source: dict[str, Any] = Field(default_factory=dict)
    note_type: str
    front: str
    back: str
    tags: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    status: Literal["pending_review", "approved", "skipped"] = "pending_review"
    review_state: Literal["green", "yellow", "red"] = "yellow"
    source_bbox: BBox | None = None
    warnings: list[str] = Field(default_factory=list)


class ProcessResult(BaseModel):
    page: Page
    tokens: list[OcrToken]
    cards: list[CardCandidate]
    script_summary: dict[str, int]
    answer_map: dict[int, int] = Field(default_factory=dict)
    ocr_run: "OcrRun | None" = None
    document_parse: "DocumentParseResult | None" = None


class OcrRun(BaseModel):
    id: str
    page_id: str
    engine: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"] = "queued"
    image_sha256: str | None = None
    processed_image_path: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    provider_config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    started_at: str
    completed_at: str | None = None
    duration_ms: int | None = None


class OcrComparison(BaseModel):
    primary_provider: str
    compare_provider: str
    primary_token_count: int
    compare_token_count: int
    agreement: float
    missing_from_primary: list[str] = Field(default_factory=list)
    missing_from_comparison: list[str] = Field(default_factory=list)
    compare_tokens: list[OcrToken] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DocumentParseBlock(BaseModel):
    id: str | None = None
    label: str
    content: str = ""
    bbox: BBox | None = None
    order: int | None = None
    confidence: float | None = None


class DocumentParseResult(BaseModel):
    page_id: str
    provider: str
    source_image_path: str
    backend: str
    block_count: int
    blocks: list[DocumentParseBlock] = Field(default_factory=list)
    markdown_text: str = ""
    warnings: list[str] = Field(default_factory=list)


class FieldOcrPreviewRequest(BaseModel):
    field: str
    bbox: BBox


class FieldOcrPreviewResponse(BaseModel):
    card_id: str
    page_id: str
    field: str
    bbox: BBox
    provider: str
    text: str
    confidence: float = 0.0
    tokens: list[OcrToken] = Field(default_factory=list)
    suggested_source: dict[str, Any] = Field(default_factory=dict)
    field_evidence: dict[str, Any] = Field(default_factory=dict)
    worker: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class OcrRuntimeStatus(BaseModel):
    state: str
    pid: int | None = None
    loaded_provider: str | None = None
    idle_deadline: str | None = None
    current_rss_mb: float | None = None
    jobs_handled: int = 0
    last_error: str | None = None


class CardUpdate(BaseModel):
    front: str | None = None
    back: str | None = None
    tags: list[str] | None = None
    confidence: float | None = None
    status: Literal["pending_review", "approved", "skipped"] | None = None
    review_state: Literal["green", "yellow", "red"] | None = None
    source: dict[str, Any] | None = None
    source_bbox: BBox | None = None
    warnings: list[str] | None = None


class PageUpdate(BaseModel):
    display_name: str | None = None


class ExportRequest(BaseModel):
    page_ids: list[str] = Field(default_factory=list)
    include_yellow: bool = True
    include_red: bool = False
    approved_only: bool = True


class ExportFile(BaseModel):
    kind: Literal["vocab", "mcq"]
    filename: str
    path: str
    download_url: str
    row_count: int


class ExportResponse(BaseModel):
    export_id: str
    files: list[ExportFile] = Field(default_factory=list)
    note_count: int = 0
    estimated_generated_card_count: int = 0
