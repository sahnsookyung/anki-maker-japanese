from __future__ import annotations

from pathlib import Path
import json
import re


KANA_RE = re.compile(r"^[ぁ-ゖァ-ヺー]+$")


class DictionaryValidator:
    def __init__(self, dictionary_path: Path | None = None) -> None:
        self.entries: set[tuple[str, str]] = set()
        if dictionary_path and dictionary_path.exists():
            data = json.loads(dictionary_path.read_text(encoding="utf-8"))
            for item in data:
                surface = item.get("surface")
                reading = item.get("reading")
                if surface and reading:
                    self.entries.add((surface, reading))

    def validate_vocab(self, surface: str, reading: str) -> tuple[str, list[str]]:
        warnings: list[str] = []
        if not surface:
            warnings.append("Missing written form.")
        if not reading:
            warnings.append("Missing reading.")
        if reading and not KANA_RE.match(reading):
            warnings.append("Reading is not kana-only.")
        if self.entries and surface and reading and (surface, reading) not in self.entries:
            warnings.append("Surface-reading pair was not found in the local dictionary.")
        if warnings:
            return "review", warnings
        return "valid", []
