from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from typing import Iterator


_OCR_RUNTIME_LOCK = Lock()


@contextmanager
def ocr_runtime_job(blocking: bool = False) -> Iterator[bool]:
    acquired = _OCR_RUNTIME_LOCK.acquire(blocking=blocking)
    try:
        yield acquired
    finally:
        if acquired:
            _OCR_RUNTIME_LOCK.release()
