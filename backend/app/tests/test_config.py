from __future__ import annotations

import os

from app.core import config


def test_google_credentials_relative_path_is_normalized_from_repo_root(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    backend_dir = repo_root / "backend"
    credentials_path = backend_dir / "credentials" / "service-account.json"
    credentials_path.parent.mkdir(parents=True)
    credentials_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(config, "ROOT_DIR", repo_root)
    monkeypatch.setattr(config, "BACKEND_DIR", backend_dir)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "backend/credentials/service-account.json")

    config._normalize_google_credentials_env()

    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(credentials_path.resolve())
