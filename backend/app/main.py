from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import CROP_DIR, EXPORT_DIR, PROCESSED_DIR, UPLOAD_DIR, ensure_dirs
from app.db.database import init_db


ensure_dirs()
init_db()

app = FastAPI(title="Japanese Study Image to Anki")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):[0-9]+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.mount("/files/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/files/processed", StaticFiles(directory=PROCESSED_DIR), name="processed")
app.mount("/files/crops", StaticFiles(directory=CROP_DIR), name="crops")
app.mount("/files/exports", StaticFiles(directory=EXPORT_DIR), name="exports")
