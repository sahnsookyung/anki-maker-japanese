from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path
from typing import NamedTuple


EXPECTED_COLUMNS = ["notetype", "front", "back", "source_page", "source_bbox", "confidence", "tags"]


class ImportRow(NamedTuple):
    notetype: str
    front: str
    back: str
    source_page: str
    source_bbox: str
    confidence: str
    tags: str


def parse_anki_csv(path: Path) -> list[ImportRow]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [line for line in lines if line.startswith("#")]
    body = [line for line in lines if not line.startswith("#")]
    _validate_headers(headers)
    rows = list(csv.reader(body))
    parsed: list[ImportRow] = []
    for index, row in enumerate(rows, start=1):
        if len(row) != len(EXPECTED_COLUMNS):
            raise ValueError(f"Data row {index} has {len(row)} columns; expected {len(EXPECTED_COLUMNS)}.")
        parsed.append(ImportRow(*row))
    return parsed


def verify_with_anki(path: Path) -> int:
    rows = parse_anki_csv(path)
    try:
        from anki.collection import Collection
        from anki.import_export_pb2 import ImportCsvRequest
    except ImportError as exc:
        raise RuntimeError("Install the optional Anki verifier dependency with `uv sync --group dev --extra anki-import`.") from exc

    with tempfile.TemporaryDirectory() as temp_dir:
        collection_path = Path(temp_dir) / "collection.anki2"
        collection = Collection(str(collection_path))
        try:
            for row in rows:
                _ensure_model(collection, row.notetype)
            metadata = collection.get_csv_metadata(str(path), None)
            response = collection.import_csv(ImportCsvRequest(path=str(path), metadata=metadata))
            collection.save()
            _verify_imported_notes(collection, rows)
            imported = len(response.log.new)
            if imported != len(rows):
                raise RuntimeError(f"Anki imported {imported} rows; expected {len(rows)}.")
            return len(rows)
        finally:
            collection.close()


def _validate_headers(headers: list[str]) -> None:
    required = {
        "#separator:Comma",
        "#html:true",
        f"#columns:{','.join(EXPECTED_COLUMNS)}",
        "#notetype column:1",
        "#tags column:7",
    }
    missing = sorted(required.difference(headers))
    if missing:
        raise ValueError(f"Missing Anki CSV header(s): {', '.join(missing)}")


def _ensure_model(collection, name: str) -> int:
    existing = collection.models.by_name(name)
    if existing:
        return existing["id"]
    model = collection.models.new(name)
    for field_name in ["Front", "Back", "Source Page", "Source BBox", "Confidence"]:
        collection.models.add_field(model, collection.models.new_field(field_name))
    template = collection.models.new_template("Card 1")
    template["qfmt"] = "{{Front}}"
    template["afmt"] = "{{FrontSide}}<hr id=answer>{{Back}}"
    collection.models.add_template(model, template)
    collection.models.add(model)
    return model["id"]


def _verify_imported_notes(collection, rows: list[ImportRow]) -> None:
    imported = collection.db.all("select flds, tags, mid from notes")
    if len(imported) != len(rows):
        raise RuntimeError(f"Temporary Anki collection contains {len(imported)} notes; expected {len(rows)}.")
    expected_by_front = {row.front: row for row in rows}
    for fields, tags, model_id in imported:
        front, back, source_page, source_bbox, confidence = fields.split("\x1f")
        expected = expected_by_front.get(front)
        if expected is None:
            raise RuntimeError(f"Imported unexpected Anki note front: {front!r}.")
        model = collection.models.get(model_id)
        if model["name"] != expected.notetype:
            raise RuntimeError(f"Imported note type {model['name']!r}; expected {expected.notetype!r}.")
        if (back, source_page, source_bbox, confidence) != (
            expected.back,
            expected.source_page,
            expected.source_bbox,
            expected.confidence,
        ):
            raise RuntimeError(f"Imported fields for {front!r} did not match the CSV row.")
        imported_tags = set(tags.split())
        expected_tags = set(expected.tags.split())
        if imported_tags != expected_tags:
            raise RuntimeError(f"Imported tags {sorted(imported_tags)}; expected {sorted(expected_tags)}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an Anki CSV export can be loaded into a temporary Anki collection.")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    imported = verify_with_anki(args.csv_path)
    print(f"Verified {imported} rows in a temporary Anki collection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
