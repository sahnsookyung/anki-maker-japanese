from __future__ import annotations

from pathlib import Path
import os

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")
load_dotenv(BACKEND_DIR / ".env")

UPLOAD_DIR = BACKEND_DIR / "uploads"
PROCESSED_DIR = BACKEND_DIR / "processed"
CROP_DIR = BACKEND_DIR / "crops"
EXPORT_DIR = BACKEND_DIR / "exports"
GOOGLE_CREDENTIALS_DIR = BACKEND_DIR / "credentials"
DB_PATH = BACKEND_DIR / os.getenv("ANKI_MAKER_DB", "app.db")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "auto")
OCR_COMPARE_PROVIDER = os.getenv("OCR_COMPARE_PROVIDER", "google_vision")
PREPROCESS_MAX_SIDE_LEN = int(os.getenv("PREPROCESS_MAX_SIDE_LEN", "1800"))
PADDLE_OCR_MAX_SIDE_LEN = int(os.getenv("PADDLE_OCR_MAX_SIDE_LEN", "1600"))
PADDLE_OCR_USE_DOC_ORIENTATION_CLASSIFY = os.getenv(
    "PADDLE_OCR_USE_DOC_ORIENTATION_CLASSIFY", "false"
).lower() in {"1", "true", "yes", "on"}
PADDLE_OCR_USE_DOC_UNWARPING = os.getenv("PADDLE_OCR_USE_DOC_UNWARPING", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PADDLE_OCR_USE_TEXTLINE_ORIENTATION = os.getenv(
    "PADDLE_OCR_USE_TEXTLINE_ORIENTATION", "false"
).lower() in {"1", "true", "yes", "on"}
PADDLE_OCR_TEXT_DETECTION_MODEL_NAME = os.getenv("PADDLE_OCR_TEXT_DETECTION_MODEL_NAME", "PP-OCRv3_mobile_det")
PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME = os.getenv(
    "PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME", "japan_PP-OCRv3_mobile_rec"
)
PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME = os.getenv(
    "PADDLE_OCR_KOREAN_TEXT_DETECTION_MODEL_NAME", "PP-OCRv5_mobile_det"
)
PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME = os.getenv(
    "PADDLE_OCR_KOREAN_TEXT_RECOGNITION_MODEL_NAME", "korean_PP-OCRv5_mobile_rec"
)
VOCAB_DUAL_OCR_ENABLED = os.getenv("VOCAB_DUAL_OCR_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
PADDLE_OCR_VL_BACKEND = os.getenv("PADDLE_OCR_VL_BACKEND", "native")
PADDLE_OCR_VL_SERVER_URL = os.getenv("PADDLE_OCR_VL_SERVER_URL", "")
PADDLE_OCR_VL_API_MODEL_NAME = os.getenv("PADDLE_OCR_VL_API_MODEL_NAME", "")
PADDLE_OCR_VL_API_KEY = os.getenv("PADDLE_OCR_VL_API_KEY", "")
PADDLE_OCR_VL_MAX_PIXELS = int(os.getenv("PADDLE_OCR_VL_MAX_PIXELS", "1000000"))
PADDLE_OCR_VL_MAX_NEW_TOKENS = int(os.getenv("PADDLE_OCR_VL_MAX_NEW_TOKENS", "1024"))
PADDLE_OCR_VL_USE_LAYOUT_DETECTION = os.getenv(
    "PADDLE_OCR_VL_USE_LAYOUT_DETECTION", "false"
).lower() in {"1", "true", "yes", "on"}
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
VLM_PROVIDER = os.getenv("VLM_PROVIDER", "ollama")
LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080")
LLAMA_CPP_MODEL = os.getenv("LLAMA_CPP_MODEL", "Qwen3-VL-8B-Instruct")
DICTIONARY_PATH = Path(os.getenv("DICTIONARY_PATH", BACKEND_DIR / "data" / "dictionaries" / "jmdict_min.json"))
KOREAN_GLOSSARY_PATH = Path(
    os.getenv("KOREAN_GLOSSARY_PATH", BACKEND_DIR / "data" / "dictionaries" / "jlpt_basic_ko.json")
)
VLM_CLEANUP_ENABLED = os.getenv("VLM_CLEANUP_ENABLED", "false").lower() in {"1", "true", "yes", "on"}


def ensure_dirs() -> None:
    for path in (UPLOAD_DIR, PROCESSED_DIR, CROP_DIR, EXPORT_DIR, GOOGLE_CREDENTIALS_DIR):
        path.mkdir(parents=True, exist_ok=True)
