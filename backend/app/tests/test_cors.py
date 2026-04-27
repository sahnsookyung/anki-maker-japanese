from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.parametrize("origin", ["http://127.0.0.1:3000", "http://localhost:3000", "http://127.0.0.1:3001"])
@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/api/pages", "GET"),
        ("/api/pages/upload", "POST"),
    ],
)
def test_dev_origins_can_preflight_api_routes(origin: str, path: str, method: str) -> None:
    client = TestClient(app)

    response = client.options(
        path,
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert method in response.headers["access-control-allow-methods"]


@pytest.mark.parametrize("origin", ["http://127.0.0.1:3000", "http://localhost:3000"])
def test_dev_origins_receive_cors_headers_on_page_list(origin: str) -> None:
    client = TestClient(app)

    response = client.get("/api/pages", headers={"Origin": origin})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
