from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from app.ocr import service


def main() -> int:
    current_provider = ""
    for line in sys.stdin:
        try:
            request = json.loads(line)
            provider = str(request["provider"])
            if provider != current_provider:
                service.release_ocr_provider_cache()
                current_provider = provider
            tokens, warnings = service.recognize_with_provider(
                Path(str(request["image_path"])),
                str(request["page_id"]),
                provider,
            )
            _write(
                {
                    "ok": True,
                    "provider": provider,
                    "tokens": [token.model_dump() for token in tokens],
                    "warnings": warnings,
                }
            )
        except Exception as exc:
            _write({"ok": False, "error": str(exc), "tokens": [], "warnings": [str(exc)]})
    return 0


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
