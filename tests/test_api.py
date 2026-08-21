"""Unit and integration tests for Mogra Movie Recommender Agent REST API."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_healthcheck() -> None:
    """Verify GET /health returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"

def test_swagger_ui_docs() -> None:
    """Verify interactive Swagger UI (/docs) and OpenAPI JSON endpoints are served."""
    response_docs = client.get("/docs")
    assert response_docs.status_code == 200
    assert "swagger-ui" in response_docs.text.lower()

    response_openapi = client.get("/openapi.json")
    assert response_openapi.status_code == 200
    spec = response_openapi.json()
    assert "paths" in spec
    assert "/api/v1/recommendations" in spec["paths"]

def test_intent_parse_endpoint() -> None:
    """Verify POST /api/v1/intent/parse parses queries into QueryIntentPayload."""
    payload = {"prompt": "dystopian cyberpunk thriller with mind games"}
    response = client.post("/api/v1/intent/parse", json=payload)
    assert response.status_code == 200
    intent = response.json()
    assert "hard_filters" in intent
    assert "seed_anchors" in intent

def test_recommendations_endpoint() -> None:
    """Verify POST /api/v1/recommendations returns candidates, synthesis, and telemetry."""
    payload = {
        "prompt": "movies like Blade Runner",
        "top_k": 2,
        "min_similarity_threshold": 0.25
    }
    response = client.post("/api/v1/recommendations", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "record_id" in data
    assert "user_prompt" in data
    assert data["user_prompt"] == "movies like Blade Runner"
    assert "recommended_movies" in data
    assert len(data["recommended_movies"]) > 0
    assert "synthesis_explanation" in data
    assert "telemetry_metrics" in data

def test_feedback_endpoints() -> None:
    """Verify submitting feedback via API and checking summary stats."""
    # 1. First run recommendation query to obtain valid record_id
    rec_res = client.post("/api/v1/recommendations", json={"prompt": "space horror like Alien", "top_k": 1})
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    record_id = rec_data["record_id"]

    # 2. Submit rating feedback
    feedback_payload = {
        "record_id": record_id,
        "overall_rating": "APPROVED",
        "card_ratings": {"348": "APPROVED"},
        "developer_notes": "Perfect match"
    }
    fb_res = client.post("/api/v1/feedback", json=feedback_payload)
    assert fb_res.status_code == 200
    assert fb_res.json()["status"] == "SUCCESS"

    # 3. Retrieve summary
    summary_res = client.get("/api/v1/feedback/summary")
    assert summary_res.status_code == 200
    summary_data = summary_res.json()
    assert summary_data["total_records"] >= 1
    assert summary_data["approved_count"] >= 1

def test_system_stats_endpoint() -> None:
    """Verify GET /api/v1/metadata/stats returns node counts and status."""
    response = client.get("/api/v1/metadata/stats")
    assert response.status_code == 200
    stats = response.json()

    assert stats["status"] == "HEALTHY"
    assert stats["total_movies"] > 0
    assert stats["total_genres"] > 0
    assert "feedback_stats" in stats
