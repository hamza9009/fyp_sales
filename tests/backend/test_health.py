"""
Phase 1 — Backend health endpoint tests.
Run with: pytest tests/backend/test_health.py -v
Requires a running PostgreSQL instance (or mock the DB check).
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)


def test_health_check_ok():
    """Health endpoint returns 200 when DB is reachable."""
    with patch("app.routers.health.check_db_connection", return_value=True):
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"
    assert "version" in data


def test_health_check_db_down():
    """Health endpoint returns 503 when DB is unreachable."""
    with patch("app.routers.health.check_db_connection", return_value=False):
        response = client.get("/api/v1/health")
    assert response.status_code == 503
    assert "Database connection failed" in response.json()["detail"]


def test_root_endpoint():
    """Root endpoint returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()
