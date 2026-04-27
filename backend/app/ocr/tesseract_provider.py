from __future__ import annotations

from pathlib import Path
import csv
import shutil
import subprocess

from app.models.schemas import OcrToken
from app.ocr.providers import make_token


class TesseractOcrProvider:
    name = "tesseract"

    def __init__(self, lang: str = "jpn+kor+eng") -> None:
        if not shutil.which("tesseract"):
            raise RuntimeError("tesseract binary is not available")
        self.lang = self._available_lang(lang)

    def recognize(self, image_path: Path, page_id: str) -> list[OcrToken]:
        cmd = ["tesseract", str(image_path), "stdout", "-l", self.lang, "--psm", "6", "tsv"]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "tesseract failed")
        rows = csv.DictReader(proc.stdout.splitlines(), delimiter="\t")
        tokens: list[OcrToken] = []
        for row in rows:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            try:
                conf = max(float(row.get("conf") or 0.0), 0.0) / 100.0
                x = float(row.get("left") or 0.0)
                y = float(row.get("top") or 0.0)
                w = float(row.get("width") or 0.0)
                h = float(row.get("height") or 0.0)
            except ValueError:
                continue
            tokens.append(make_token(page_id, text, [x, y, x + w, y + h], conf, self.name))
        return tokens

    def _available_lang(self, requested: str) -> str:
        proc = subprocess.run(["tesseract", "--list-langs"], text=True, capture_output=True, check=False)
        available = set(proc.stdout.splitlines()[1:])
        requested_parts = requested.split("+")
        usable = [part for part in requested_parts if part in available]
        if usable:
            return "+".join(usable)
        if "eng" in available:
            return "eng"
        return requested_parts[-1]
