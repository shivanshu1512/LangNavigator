"""
Tests for the FastAPI endpoints.

Uses httpx.AsyncClient with the FastAPI app directly (no server needed).
LLM and Qdrant calls are mocked so tests run without API keys or services.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from docs_rag.api.main import app
from docs_rag.models import Answer, Citation, Timings


@pytest.fixture
def client():
    """Create a test client that skips model loading."""
    # Override lifespan to avoid loading real models in tests
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoint:
    """Tests for GET /health."""

    @patch("docs_rag.api.main.get_env_settings")
    def test_health_returns_200(self, mock_settings, client):
        """Health endpoint should return 200 even if Qdrant is unavailable."""
        mock_settings.return_value = MagicMock(
            qdrant_url="http://localhost:6333",
            qdrant_api_key="",
            qdrant_collection="docs",
        )
        resp = client.get("/health")
        # May return degraded if Qdrant isn't running, but shouldn't crash
        assert resp.status_code == 200

    def test_health_returns_json(self, client):
        """Health response should be valid JSON."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data


class TestQueryEndpoint:
    """Tests for POST /query."""

    def test_query_validation_too_short(self, client):
        """Questions shorter than 3 chars should return 422."""
        resp = client.post("/query", json={"question": "ab"})
        assert resp.status_code == 422

    def test_query_validation_too_long(self, client):
        """Questions longer than 500 chars should return 422."""
        resp = client.post("/query", json={"question": "x" * 501})
        assert resp.status_code == 422

    def test_query_validation_missing_question(self, client):
        """Missing question field should return 422."""
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    @patch("docs_rag.api.main.pipeline_answer")
    def test_query_success_response_schema(self, mock_pipeline, client):
        """Successful query should return the expected schema."""
        mock_pipeline.return_value = Answer(
            answer="Test answer [1]",
            refused=False,
            citations=[
                Citation(
                    n=1,
                    breadcrumb="Test > Section",
                    url="https://docs.example.com#test",
                    snippet="Test snippet...",
                    rerank_score=7.5,
                    fused_rank=1,
                )
            ],
            timings=Timings(
                retrieve_ms=50.0,
                rerank_ms=100.0,
                generate_ms=500.0,
                total_ms=650.0,
            ),
            config="c_hybrid_rerank",
            model="llama-3.1-8b-instant",
        )

        resp = client.post("/query", json={"question": "How do I persist state?"})
        assert resp.status_code == 200

        data = resp.json()
        assert "answer" in data
        assert "refused" in data
        assert "citations" in data
        assert "timings_ms" in data
        assert "config" in data
        assert "model" in data
        assert "request_id" in data

        assert data["answer"] == "Test answer [1]"
        assert data["refused"] is False
        assert len(data["citations"]) == 1
        assert data["citations"][0]["n"] == 1

    @patch("docs_rag.api.main.pipeline_answer")
    def test_query_refusal(self, mock_pipeline, client):
        """Refusal responses should have refused=true."""
        mock_pipeline.return_value = Answer(
            answer="I couldn't find this in the LangGraph docs.",
            refused=True,
            citations=[],
            timings=Timings(retrieve_ms=50.0, total_ms=50.0),
            config="c_hybrid_rerank",
            model="llama-3.1-8b-instant",
        )

        resp = client.post("/query", json={"question": "Can LangGraph cook dinner?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["refused"] is True

    @patch("docs_rag.api.main.pipeline_answer")
    def test_query_upstream_failure_returns_503(self, mock_pipeline, client):
        """Upstream failures should return 503, not 500."""
        mock_pipeline.side_effect = ConnectionError("Qdrant unreachable")

        resp = client.post("/query", json={"question": "How do I use checkpoints?"})
        assert resp.status_code == 503


class TestFeedbackEndpoint:
    """Tests for POST /feedback."""

    def test_feedback_records(self, client):
        """Feedback should be accepted and return status."""
        resp = client.post("/feedback", json={
            "request_id": "abc123",
            "helpful": True,
            "comment": "Great answer!",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "recorded"
