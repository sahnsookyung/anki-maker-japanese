from __future__ import annotations

import os

from app.core import config
from app.extraction import pipeline
from app.ocr.profiles import normalize_extraction_variant, resolve_ocr_model_profile


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


def test_ocr_model_profile_manifest_records_runtime_and_model_config() -> None:
    profile = resolve_ocr_model_profile("jp_v5_mobile")

    manifest = profile.manifest(
        engine="paddleocr",
        extraction_variant="table_graph_v1",
        cache_hit=False,
        cache_key="abc123",
        preprocessing_config={"processed_width": 100, "processed_height": 200},
    )

    assert profile.id == "jp_v5_mobile_general"
    assert manifest["schema_version"] == 1
    assert manifest["profile_id"] == "jp_v5_mobile_general"
    assert manifest["budget"] == "safe_local"
    assert manifest["env"]["PADDLE_OCR_TEXT_RECOGNITION_MODEL_NAME"] == "PP-OCRv5_mobile_rec"
    assert manifest["model_config"]["japanese_detection_model"] == "PP-OCRv5_mobile_det"
    assert manifest["language_config"]["use_language_profile"] is False
    assert manifest["preprocessing_config"]["processed_width"] == 100
    assert "device" in manifest["runtime"]
    assert manifest["cache"]["hit"] is False
    assert manifest["cache"]["key"] == "abc123"
    assert "model_cache_fingerprints" in manifest["cache"]["key_components"]
    assert "extraction_variant" not in manifest["cache"]["key_components"]
    assert "python" in manifest["runtime"]


def test_extraction_variant_normalization_rejects_unknown_values() -> None:
    assert normalize_extraction_variant("baseline") == "baseline_current"

    try:
        normalize_extraction_variant("secret_variant")
    except ValueError as exc:
        assert "Unsupported extraction variant" in str(exc)
    else:
        raise AssertionError("Expected unsupported extraction variant to fail.")


def test_preprocessing_hash_and_ocr_cache_key_include_config(monkeypatch) -> None:
    transform = {
        "schema_version": 1,
        "coordinate_space": "processed_image",
        "original_to_processed": {"pipeline": ["resize", "sharpen"]},
    }
    monkeypatch.setattr(pipeline, "PREPROCESS_MAX_SIDE_LEN", 1800)
    first_config = pipeline._preprocessing_config(100, 200, [], transform)
    first_hash = pipeline._preprocessing_hash(first_config)
    monkeypatch.setattr(pipeline, "PREPROCESS_MAX_SIDE_LEN", 2400)
    second_config = pipeline._preprocessing_config(100, 200, [], transform)
    second_hash = pipeline._preprocessing_hash(second_config)

    profile = resolve_ocr_model_profile("jp_v3_mobile_current")
    first_manifest = profile.manifest(
        engine="paddleocr",
        extraction_variant="baseline_current",
        preprocessing_config=first_config,
    )
    second_manifest = profile.manifest(
        engine="paddleocr",
        extraction_variant="baseline_current",
        preprocessing_config=second_config,
    )
    first_key = pipeline._ocr_cache_key(
        image_sha="same-image",
        preprocessing_hash=first_hash,
        profile_manifest=first_manifest,
        engine="paddleocr",
        extraction_variant="baseline_current",
    )
    second_key = pipeline._ocr_cache_key(
        image_sha="same-image",
        preprocessing_hash=second_hash,
        profile_manifest=second_manifest,
        engine="paddleocr",
        extraction_variant="baseline_current",
    )

    assert first_config["preprocess_max_side_len"] == 1800
    assert second_config["preprocess_max_side_len"] == 2400
    assert first_hash != second_hash
    assert first_key != second_key


def test_preprocessing_cache_fingerprint_ignores_runtime_paths(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "PREPROCESS_MAX_SIDE_LEN", 1800)
    first_config = pipeline._preprocessing_config(
        100,
        200,
        [],
        {
            "coordinate_space": "processed_image",
            "original_image_path": "/uploads/page-a.jpg",
            "processed_image_path": "/processed/page-a.png",
            "original_to_processed": {"pipeline": ["resize"]},
        },
    )
    second_config = pipeline._preprocessing_config(
        100,
        200,
        [],
        {
            "coordinate_space": "processed_image",
            "original_image_path": "/uploads/page-b.jpg",
            "processed_image_path": "/processed/page-b.png",
            "original_to_processed": {"pipeline": ["resize"]},
        },
    )
    profile = resolve_ocr_model_profile("jp_v3_mobile_current")
    first_manifest = profile.manifest(
        engine="paddleocr",
        extraction_variant="baseline_current",
        preprocessing_config=first_config,
    )
    second_manifest = profile.manifest(
        engine="paddleocr",
        extraction_variant="baseline_current",
        preprocessing_config=second_config,
    )

    assert pipeline._preprocessing_hash(first_config) == pipeline._preprocessing_hash(second_config)
    assert pipeline._ocr_cache_key(
        image_sha="same-image",
        preprocessing_hash=pipeline._preprocessing_hash(first_config),
        profile_manifest=first_manifest,
        engine="paddleocr",
        extraction_variant="baseline_current",
    ) == pipeline._ocr_cache_key(
        image_sha="same-image",
        preprocessing_hash=pipeline._preprocessing_hash(second_config),
        profile_manifest=second_manifest,
        engine="paddleocr",
        extraction_variant="baseline_current",
    )


def test_ocr_cache_key_ignores_extraction_variant_for_reusable_payloads(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "PREPROCESS_MAX_SIDE_LEN", 1800)
    preprocessing_config = pipeline._preprocessing_config(
        100,
        200,
        [],
        {
            "schema_version": 1,
            "coordinate_space": "processed_image",
            "original_to_processed": {"pipeline": ["resize"]},
        },
    )
    preprocessing_hash = pipeline._preprocessing_hash(preprocessing_config)
    profile = resolve_ocr_model_profile("jp_v3_mobile_current")
    line_manifest = profile.manifest(
        engine="paddleocr",
        extraction_variant="line_graph_v1",
        preprocessing_config=preprocessing_config,
    )
    table_manifest = profile.manifest(
        engine="paddleocr",
        extraction_variant="table_graph_v1",
        preprocessing_config=preprocessing_config,
    )

    assert line_manifest["extraction_variant"] == "line_graph_v1"
    assert table_manifest["extraction_variant"] == "table_graph_v1"
    assert "extraction_variant" not in line_manifest["cache"]["key_components"]
    assert pipeline._ocr_cache_key(
        image_sha="same-image",
        preprocessing_hash=preprocessing_hash,
        profile_manifest=line_manifest,
        engine="paddleocr",
        extraction_variant="line_graph_v1",
    ) == pipeline._ocr_cache_key(
        image_sha="same-image",
        preprocessing_hash=preprocessing_hash,
        profile_manifest=table_manifest,
        engine="paddleocr",
        extraction_variant="table_graph_v1",
    )
