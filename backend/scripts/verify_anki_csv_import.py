from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path
from typing import NamedTuple, Union


MCQ_COLUMNS = ["notetype", "front", "back", "source_page", "source_bbox", "confidence", "tags"]
VOCAB_COLUMNS = [
    "VocabKey",
    "Surface",
    "Reading",
    "MeaningKo",
    "StudyWriting",
    "StudyReading",
    "StudyMeaning",
    "SourcePage",
    "SourceBBox",
    "Confidence",
    "Warnings",
    "tags",
]


class McqImportRow(NamedTuple):
    notetype: str
    front: str
    back: str
    source_page: str
    source_bbox: str
    confidence: str
    tags: str


class VocabImportRow(NamedTuple):
    vocab_key: str
    surface: str
    reading: str
    meaning_ko: str
    study_writing: str
    study_reading: str
    study_meaning: str
    source_page: str
    source_bbox: str
    confidence: str
    warnings: str
    tags: str


ImportRow = Union[McqImportRow, VocabImportRow]


def parse_anki_csv(path: Path) -> list[ImportRow]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = [line for line in lines if line.startswith("#")]
    body = [line for line in lines if not line.startswith("#")]
    if "#notetype:jp_vocab_entry" in headers:
        _validate_headers(headers, VOCAB_COLUMNS, tags_column=12, fixed_notetype="#notetype:jp_vocab_entry")
        return _parse_rows(body, VOCAB_COLUMNS, VocabImportRow)
    _validate_headers(headers, MCQ_COLUMNS, tags_column=7, notetype_column=True)
    return _parse_rows(body, MCQ_COLUMNS, McqImportRow)


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
                _ensure_model(collection, _row_notetype(row))
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


def _parse_rows(body: list[str], expected_columns: list[str], row_type):
    rows = list(csv.reader(body))
    parsed = []
    for index, row in enumerate(rows, start=1):
        if len(row) != len(expected_columns):
            raise ValueError(f"Data row {index} has {len(row)} columns; expected {len(expected_columns)}.")
        parsed.append(row_type(*row))
    return parsed


def _validate_headers(
    headers: list[str],
    columns: list[str],
    *,
    tags_column: int,
    fixed_notetype: str | None = None,
    notetype_column: bool = False,
) -> None:
    required = {
        "#separator:Comma",
        "#html:true",
        f"#columns:{','.join(columns)}",
        f"#tags column:{tags_column}",
    }
    if fixed_notetype:
        required.add(fixed_notetype)
    if notetype_column:
        required.add("#notetype column:1")
    missing = sorted(required.difference(headers))
    if missing:
        raise ValueError(f"Missing Anki CSV header(s): {', '.join(missing)}")


def _ensure_model(collection, name: str) -> int:
    existing = collection.models.by_name(name)
    if existing:
        return existing["id"]
    if name == "jp_vocab_entry":
        return _ensure_vocab_model(collection)
    return _ensure_front_back_model(collection, name)


def _ensure_front_back_model(collection, name: str) -> int:
    model = collection.models.new(name)
    for field_name in ["Front", "Back", "Source Page", "Source BBox", "Confidence"]:
        collection.models.add_field(model, collection.models.new_field(field_name))
    template = collection.models.new_template("Card 1")
    template["qfmt"] = "{{Front}}"
    template["afmt"] = "{{FrontSide}}<hr id=answer>{{Back}}"
    collection.models.add_template(model, template)
    collection.models.add(model)
    return model["id"]


def _ensure_vocab_model(collection) -> int:
    model = collection.models.new("jp_vocab_entry")
    for field_name in VOCAB_COLUMNS[:-1]:
        collection.models.add_field(model, collection.models.new_field(field_name))
    templates = [
        (
            "Kana to Kanji",
            "{{#StudyWriting}}{{Reading}}{{/StudyWriting}}",
            "{{FrontSide}}<hr id=answer>{{Surface}}<br><details><summary>Meaning</summary>{{MeaningKo}}</details>",
        ),
        (
            "Kanji to Kana",
            "{{#StudyReading}}{{Surface}}{{/StudyReading}}",
            "{{FrontSide}}<hr id=answer>{{Reading}}<br><details><summary>Meaning</summary>{{MeaningKo}}</details>",
        ),
        (
            "Meaning to Japanese",
            "{{#StudyMeaning}}{{MeaningKo}}{{/StudyMeaning}}",
            "{{FrontSide}}<hr id=answer>{{Surface}}<br>{{Reading}}",
        ),
    ]
    for name, question, answer in templates:
        template = collection.models.new_template(name)
        template["qfmt"] = question
        template["afmt"] = answer
        collection.models.add_template(model, template)
    collection.models.add(model)
    return model["id"]


def _verify_imported_notes(collection, rows: list[ImportRow]) -> None:
    imported = collection.db.all("select flds, tags, mid from notes")
    if len(imported) != len(rows):
        raise RuntimeError(f"Temporary Anki collection contains {len(imported)} notes; expected {len(rows)}.")
    expected_by_first_field = {_first_field(row): row for row in rows}
    for fields, tags, model_id in imported:
        imported_fields = fields.split("\x1f")
        expected = expected_by_first_field.get(imported_fields[0])
        if expected is None:
            raise RuntimeError(f"Imported unexpected Anki note key/front: {imported_fields[0]!r}.")
        model = collection.models.get(model_id)
        if model["name"] != _row_notetype(expected):
            raise RuntimeError(f"Imported note type {model['name']!r}; expected {_row_notetype(expected)!r}.")
        _verify_row_fields(expected, imported_fields)
        imported_tags = set(tags.split())
        expected_tags = set(expected.tags.split())
        if imported_tags != expected_tags:
            raise RuntimeError(f"Imported tags {sorted(imported_tags)}; expected {sorted(expected_tags)}.")


def _verify_row_fields(expected: ImportRow, imported_fields: list[str]) -> None:
    if isinstance(expected, VocabImportRow):
        expected_fields = list(expected[:-1])
    else:
        expected_fields = [expected.front, expected.back, expected.source_page, expected.source_bbox, expected.confidence]
    if imported_fields != expected_fields:
        raise RuntimeError(f"Imported fields for {_first_field(expected)!r} did not match the CSV row.")


def _row_notetype(row: ImportRow) -> str:
    return "jp_vocab_entry" if isinstance(row, VocabImportRow) else row.notetype


def _first_field(row: ImportRow) -> str:
    return row.vocab_key if isinstance(row, VocabImportRow) else row.front


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an Anki CSV export can be loaded into a temporary Anki collection.")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    imported = verify_with_anki(args.csv_path)
    print(f"Verified {imported} rows in a temporary Anki collection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
