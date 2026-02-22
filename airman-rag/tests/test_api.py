"""
tests/test_api.py — Basic API tests for AIRMAN RAG system
Run: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock OpenAI before importing app
import unittest.mock as mock

with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
    from app.main import app

client = TestClient(app)


def test_health():
    """Health endpoint returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_ask_without_ingestion():
    """Ask before ingestion returns 400."""
    response = client.post("/ask", json={"question": "What is the DALR?"})
    assert response.status_code == 400


def test_health_shows_no_index():
    """Health shows no index before ingestion."""
    response = client.get("/health")
    data = response.json()
    assert data["index_loaded"] == False


def test_ask_request_schema():
    """Ask endpoint accepts correct schema."""
    # This will 400 (no index) but validates schema is accepted
    response = client.post("/ask", json={
        "question": "test question",
        "debug": True,
        "top_k": 5
    })
    # 400 is expected (no index), but not 422 (schema error)
    assert response.status_code in [400, 200]


def test_ask_bad_schema():
    """Ask endpoint rejects bad schema."""
    response = client.post("/ask", json={"wrong_field": "value"})
    assert response.status_code == 422
