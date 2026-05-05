from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.extraction.geometry import group_tokens_by_line, union_bbox
from app.models.schemas import CardCandidate, DocumentParseResult, OcrToken


DOCUMENT_GRAPH_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DocumentGraph:
    page_id: str
    source: str
    text_nodes: list[dict[str, Any]]
    line_nodes: list[dict[str, Any]]
    region_nodes: list[dict[str, Any]]
    table_cells: list[dict[str, Any]]
    selection_marks: list[dict[str, Any]]
    field_hypotheses: list[dict[str, Any]]
    row_hypotheses: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    transform: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return {
            "schema_version": DOCUMENT_GRAPH_SCHEMA_VERSION,
            "page_id": self.page_id,
            "source": self.source,
            "text_nodes": self.text_nodes,
            "line_nodes": self.line_nodes,
            "region_nodes": self.region_nodes,
            "table_cells": self.table_cells,
            "selection_marks": self.selection_marks,
            "field_hypotheses": self.field_hypotheses,
            "row_hypotheses": self.row_hypotheses,
            "relationships": self.relationships,
            "transform": self.transform,
            "metrics": graph_metrics(self),
        }


def graph_from_tokens(
    page_id: str,
    tokens: list[OcrToken],
    *,
    source: str = "paddleocr",
    transform: dict[str, Any] | None = None,
) -> DocumentGraph:
    text_nodes = [
        {
            "id": token.id,
            "text": token.text,
            "bbox": token.bbox,
            "confidence": token.confidence,
            "script_class": token.script_class,
            "source": token.source,
        }
        for token in tokens
    ]
    line_nodes = []
    region_nodes = [
        {
            "id": "region_page",
            "type": "page",
            "bbox": _union_valid_bboxes([token.bbox for token in tokens]),
            "source": source,
        }
    ]
    table_cells = []
    selection_marks = []
    relationships = []
    for index, line in enumerate(group_tokens_by_line(tokens)):
        token_ids = [token.id for token in line]
        bbox = union_bbox([token.bbox for token in line]) if line else None
        line_id = f"line_{index + 1:04d}"
        region_id = f"region_{line_id}"
        line_nodes.append(
            {
                "id": line_id,
                "text": "".join(token.text for token in sorted(line, key=lambda item: item.bbox[0])),
                "bbox": bbox,
                "token_ids": token_ids,
                "order": index,
                "confidence": min((token.confidence for token in line), default=0.0),
            }
        )
        region_nodes.append({"id": region_id, "type": "line", "bbox": bbox, "line_id": line_id, "source": source})
        relationships.append({"from": "region_page", "to": region_id, "type": "contains"})
        relationships.append({"from": region_id, "to": line_id, "type": "contains"})
        relationships.extend({"from": line_id, "to": token_id, "type": "contains"} for token_id in token_ids)
        for cell_index, token in enumerate(sorted(line, key=lambda item: item.bbox[0])):
            cell_id = f"cell_{index + 1:04d}_{cell_index + 1:02d}"
            table_cells.append(
                {
                    "id": cell_id,
                    "line_id": line_id,
                    "token_ids": [token.id],
                    "text": token.text,
                    "bbox": token.bbox,
                    "confidence": token.confidence,
                    "source": token.source,
                }
            )
            relationships.append({"from": line_id, "to": cell_id, "type": "has_cell"})
            relationships.append({"from": cell_id, "to": token.id, "type": "contains"})
            if token.text.strip() in {"□", "☐", "▢", "☑", "✓"}:
                mark_id = f"mark_{token.id}"
                selection_marks.append(
                    {
                        "id": mark_id,
                        "token_id": token.id,
                        "bbox": token.bbox,
                        "state": "checked" if token.text.strip() in {"☑", "✓"} else "unchecked",
                        "confidence": token.confidence,
                        "source": token.source,
                    }
                )
                relationships.append({"from": mark_id, "to": token.id, "type": "detected_from"})
    return DocumentGraph(
        page_id=page_id,
        source=source,
        text_nodes=text_nodes,
        line_nodes=line_nodes,
        region_nodes=region_nodes,
        table_cells=table_cells,
        selection_marks=selection_marks,
        field_hypotheses=[],
        row_hypotheses=[],
        relationships=relationships,
        transform=transform or {"schema_version": DOCUMENT_GRAPH_SCHEMA_VERSION, "coordinate_space": "processed_image"},
    )


def graph_from_document_parse(
    result: DocumentParseResult,
    *,
    transform: dict[str, Any] | None = None,
) -> DocumentGraph:
    text_nodes = [
        {
            "id": block.id or f"block_{index + 1:04d}",
            "text": block.content,
            "bbox": block.bbox,
            "confidence": block.confidence,
            "script_class": "mixed",
            "source": result.provider,
        }
        for index, block in enumerate(result.blocks)
    ]
    line_nodes = [
        {
            "id": node["id"],
            "text": node["text"],
            "bbox": node["bbox"],
            "token_ids": [node["id"]],
            "order": index,
            "confidence": node["confidence"],
        }
        for index, node in enumerate(text_nodes)
    ]
    return DocumentGraph(
        page_id=result.page_id,
        source=result.provider,
        text_nodes=text_nodes,
        line_nodes=line_nodes,
        region_nodes=[
            {
                "id": f"region_{node['id']}",
                "type": "document_block",
                "bbox": node.get("bbox"),
                "block_id": node["id"],
                "source": result.provider,
            }
            for node in text_nodes
        ],
        table_cells=[],
        selection_marks=[],
        field_hypotheses=[],
        row_hypotheses=[],
        relationships=[{"from": node["id"], "to": node["id"], "type": "block_text"} for node in text_nodes],
        transform=transform or {"schema_version": DOCUMENT_GRAPH_SCHEMA_VERSION, "coordinate_space": "processed_image"},
    )


def graph_with_card_hypotheses(graph: DocumentGraph, cards: list[CardCandidate]) -> DocumentGraph:
    field_hypotheses: list[dict[str, Any]] = []
    row_hypotheses: list[dict[str, Any]] = []
    relationships = list(graph.relationships)
    seen_rows: set[tuple[str, str]] = set()
    for card in cards:
        source_key = (card.source_type, card.source_id)
        field_ids: list[str] = []
        field_evidence = card.source.get("field_evidence")
        if isinstance(field_evidence, dict):
            for field, evidence in sorted(field_evidence.items()):
                if not isinstance(evidence, dict):
                    continue
                field_id = f"field_{card.source_id}_{field}"
                field_ids.append(field_id)
                token_ids = [token_id for token_id in evidence.get("token_ids", []) if isinstance(token_id, str)]
                block_ids = [block_id for block_id in evidence.get("block_ids", []) if isinstance(block_id, str)]
                field_hypotheses.append(
                    {
                        "id": field_id,
                        "card_id": card.id,
                        "source_type": card.source_type,
                        "source_id": card.source_id,
                        "field": field,
                        "text": evidence.get("text") or card.source.get(field) or "",
                        "bbox": evidence.get("bbox"),
                        "token_ids": token_ids,
                        "block_ids": block_ids,
                        "provenance": evidence.get("provenance"),
                        "confidence": evidence.get("confidence", card.confidence),
                    }
                )
                relationships.extend({"from": field_id, "to": token_id, "type": "supported_by_token"} for token_id in token_ids)
                relationships.extend({"from": field_id, "to": block_id, "type": "supported_by_block"} for block_id in block_ids)
        if source_key not in seen_rows:
            seen_rows.add(source_key)
            row_id = f"row_{card.source_type}_{card.source_id}"
            row_hypotheses.append(
                {
                    "id": row_id,
                    "source_type": card.source_type,
                    "source_id": card.source_id,
                    "card_ids": [item.id for item in cards if item.source_type == card.source_type and item.source_id == card.source_id],
                    "field_ids": field_ids,
                    "bbox": card.source_bbox or card.source.get("bbox"),
                    "confidence": card.confidence,
                    "review_state": card.review_state,
                    "warning_count": len(card.warnings),
                }
            )
            relationships.extend({"from": row_id, "to": field_id, "type": "has_field"} for field_id in field_ids)
    return DocumentGraph(
        page_id=graph.page_id,
        source=graph.source,
        text_nodes=graph.text_nodes,
        line_nodes=graph.line_nodes,
        region_nodes=graph.region_nodes,
        table_cells=graph.table_cells,
        selection_marks=graph.selection_marks,
        field_hypotheses=field_hypotheses,
        row_hypotheses=row_hypotheses,
        relationships=relationships,
        transform=graph.transform,
    )


def graph_metrics(graph: DocumentGraph) -> dict[str, Any]:
    boxed_nodes = [node for node in graph.text_nodes if _valid_bbox(node.get("bbox"))]
    low_confidence = [
        node
        for node in graph.text_nodes
        if isinstance(node.get("confidence"), (int, float)) and float(node.get("confidence") or 0.0) < 0.75
    ]
    boxed_fields = [field for field in graph.field_hypotheses if _valid_bbox(field.get("bbox"))]
    boxed_rows = [row for row in graph.row_hypotheses if _valid_bbox(row.get("bbox"))]
    return {
        "text_node_count": len(graph.text_nodes),
        "line_node_count": len(graph.line_nodes),
        "region_node_count": len(graph.region_nodes),
        "table_cell_count": len(graph.table_cells),
        "selection_mark_count": len(graph.selection_marks),
        "field_hypothesis_count": len(graph.field_hypotheses),
        "row_hypothesis_count": len(graph.row_hypotheses),
        "boxed_text_node_count": len(boxed_nodes),
        "bbox_coverage": round(len(boxed_nodes) / len(graph.text_nodes), 4) if graph.text_nodes else 0.0,
        "low_confidence_node_count": len(low_confidence),
        "field_bbox_coverage": round(len(boxed_fields) / len(graph.field_hypotheses), 4) if graph.field_hypotheses else 0.0,
        "row_bbox_coverage": round(len(boxed_rows) / len(graph.row_hypotheses), 4) if graph.row_hypotheses else 0.0,
        "evidence_alignment_score": _evidence_alignment_score(graph, boxed_fields, boxed_rows),
    }


def _valid_bbox(value: object) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(isinstance(item, (int, float)) for item in value)


def _union_valid_bboxes(values: list[list[float]]) -> list[float] | None:
    valid = [value for value in values if _valid_bbox(value)]
    return union_bbox(valid) if valid else None


def _evidence_alignment_score(
    graph: DocumentGraph,
    boxed_fields: list[dict[str, Any]],
    boxed_rows: list[dict[str, Any]],
) -> float:
    expected = len(graph.field_hypotheses) + len(graph.row_hypotheses)
    if not expected:
        return 0.0
    return round((len(boxed_fields) + len(boxed_rows)) / expected, 4)
