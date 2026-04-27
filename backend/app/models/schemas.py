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
    display_name: str | None = None
    processed_image_path: str | None = None
    page_type: str = "unprocessed"
    page_type_confidence: float = 0.0
    image_width: int | None = None
    image_height: int | None = None
    warnings: list[str] = Field(default_factory=list)
    created_at: str


class CardCandidate(BaseModel):
    id: str
    page_id: str
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
    label: str
    content: str = ""
    bbox: BBox | None = None
    order: int | None = None


class DocumentParseResult(BaseModel):
    page_id: str
    provider: str
    source_image_path: str
    backend: str
    block_count: int
    blocks: list[DocumentParseBlock] = Field(default_factory=list)
    markdown_text: str = ""
    warnings: list[str] = Field(default_factory=list)


class CardUpdate(BaseModel):
    front: str | None = None
    back: str | None = None
    tags: list[str] | None = None
    status: Literal["pending_review", "approved", "skipped"] | None = None
    review_state: Literal["green", "yellow", "red"] | None = None
    source: dict[str, Any] | None = None


class PageUpdate(BaseModel):
    display_name: str | None = None


class ExportRequest(BaseModel):
    page_ids: list[str] = Field(default_factory=list)
    include_yellow: bool = True
    include_red: bool = False
    approved_only: bool = True


class ExportResponse(BaseModel):
    export_id: str
    path: str
    card_count: int
    download_url: str
