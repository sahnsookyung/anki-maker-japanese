from __future__ import annotations

from pathlib import Path
import os

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")
load_dotenv(BACKEND_DIR / ".env")


def _normalize_google_credentials_env() -> None:
    credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials:
        return
    credentials_path = Path(credentials).expanduser()
    if credentials_path.is_absolute():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
        return
    for base_dir in (ROOT_DIR, BACKEND_DIR):
        candidate = (base_dir / credentials_path).resolve()
        if candidate.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate)
            return
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str((ROOT_DIR / credentials_path).resolve())


def _configured_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    if path.is_absolute():
        return path
    for base_dir in (ROOT_DIR, BACKEND_DIR):
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
    return (ROOT_DIR / path).resolve()


_normalize_google_credentials_env()

UPLOAD_DIR = BACKEND_DIR / "uploads"
PROCESSED_DIR = BACKEND_DIR / os.getenv("ANKI_MAKER_PROCESSED_DIR", "processed")
CROP_DIR = BACKEND_DIR / "crops"
EXPORT_DIR = BACKEND_DIR / "exports"
GOOGLE_CREDENTIALS_DIR = BACKEND_DIR / "credentials"
OCR_CACHE_DIR = BACKEND_DIR / "ocr_cache"
USAGE_DIR = BACKEND_DIR / "usage"
DB_PATH = BACKEND_DIR / os.getenv("ANKI_MAKER_DB", "app.db")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "paddle")
OCR_PROVIDER_CACHE_ENABLED = os.getenv("OCR_PROVIDER_CACHE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
OCR_COMPARE_PROVIDER = os.getenv("OCR_COMPARE_PROVIDER", "google_vision")
GOOGLE_VISION_ALLOW_CLOUD = os.getenv("GOOGLE_VISION_ALLOW_CLOUD", "false").lower() in {"1", "true", "yes", "on"}
GOOGLE_VISION_MONTHLY_CAP = int(os.getenv("GOOGLE_VISION_MONTHLY_CAP", "1000"))
GOOGLE_VISION_CACHE_ENABLED = os.getenv("GOOGLE_VISION_CACHE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
GOOGLE_VISION_API_ENDPOINT = os.getenv("GOOGLE_VISION_API_ENDPOINT", "").strip()
OCR_CROP_WORKER_IDLE_SECONDS = float(os.getenv("OCR_CROP_WORKER_IDLE_SECONDS", "120"))
OCR_CROP_WORKER_MAX_RSS_MB = float(os.getenv("OCR_CROP_WORKER_MAX_RSS_MB", "6144"))
OCR_CROP_JOB_TIMEOUT_SECONDS = float(os.getenv("OCR_CROP_JOB_TIMEOUT_SECONDS", "45"))
OCR_PAGE_WORKER_MAX_RSS_MB = float(os.getenv("OCR_PAGE_WORKER_MAX_RSS_MB", "6144"))
OCR_VL_PAGE_WORKER_MAX_RSS_MB = float(os.getenv("OCR_VL_PAGE_WORKER_MAX_RSS_MB", "14336"))
OCR_PAGE_JOB_TIMEOUT_SECONDS = float(os.getenv("OCR_PAGE_JOB_TIMEOUT_SECONDS", "300"))
OCR_CROP_MIN_SIDE = int(os.getenv("OCR_CROP_MIN_SIDE", "8"))
OCR_CROP_MAX_SIDE = int(os.getenv("OCR_CROP_MAX_SIDE", "1800"))
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
PADDLE_OCR_VL_BACKEND = os.getenv("PADDLE_OCR_VL_BACKEND", "")
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
DICTIONARY_PATH = _configured_path("DICTIONARY_PATH", BACKEND_DIR / "data" / "dictionaries" / "jlpt_basic_ko.json")
KOREAN_GLOSSARY_PATH = _configured_path("KOREAN_GLOSSARY_PATH", BACKEND_DIR / "data" / "dictionaries" / "jlpt_basic_ko.json")
VLM_CLEANUP_ENABLED = os.getenv("VLM_CLEANUP_ENABLED", "false").lower() in {"1", "true", "yes", "on"}


def ensure_dirs() -> None:
    for path in (UPLOAD_DIR, PROCESSED_DIR, CROP_DIR, EXPORT_DIR, GOOGLE_CREDENTIALS_DIR, OCR_CACHE_DIR, USAGE_DIR):
        path.mkdir(parents=True, exist_ok=True)
