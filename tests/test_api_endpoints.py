"""Tests for FastAPI endpoints using TestClient."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from aipa_engine.main import create_app, app
from aipa_engine.config import get_settings


# ──────────────────────────────────────────────
# Client fixture
# ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    """TestClient wrapping the FastAPI app."""
    # Clear settings cache to avoid stale state from other tests
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c


# ──────────────────────────────────────────────
# Health check tests
# ──────────────────────────────────────────────


class TestHealthCheck:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_body(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


# ──────────────────────────────────────────────
# POST /api/v1/personas/generate
# ──────────────────────────────────────────────


class TestPersonasGenerate:
    def test_generate_default(self, client):
        """Generate personas with minimal config."""
        response = client.post(
            "/api/v1/personas/generate",
            json={"panel_count": 3},
        )
        assert response.status_code == 200
        personas = response.json()
        assert len(personas) == 3
        for p in personas:
            assert "id" in p
            assert "name" in p
            assert "attributes" in p
            assert "age_group" in p["attributes"]
            assert "gender" in p["attributes"]

    def test_generate_with_age_filter(self, client):
        response = client.post(
            "/api/v1/personas/generate",
            json={
                "panel_count": 5,
                "age_groups": ["20대", "30대"],
            },
        )
        assert response.status_code == 200
        personas = response.json()
        assert len(personas) == 5
        for p in personas:
            assert p["attributes"]["age_group"] in ["20대", "30대"]

    def test_generate_invalid_age_group_returns_400(self, client):
        response = client.post(
            "/api/v1/personas/generate",
            json={
                "panel_count": 3,
                "age_groups": ["invalid_age"],
            },
        )
        assert response.status_code == 400

    def test_generate_with_occupations(self, client):
        response = client.post(
            "/api/v1/personas/generate",
            json={
                "panel_count": 5,
                "occupations": ["개발자", "디자이너"],
            },
        )
        assert response.status_code == 200
        personas = response.json()
        for p in personas:
            assert p["attributes"]["occupation"] in ["개발자", "디자이너"]

    def test_panel_count_validation_too_high(self, client):
        """panel_count > 200 should fail validation."""
        response = client.post(
            "/api/v1/personas/generate",
            json={"panel_count": 999},
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_panel_count_validation_zero(self, client):
        """panel_count < 1 should fail validation."""
        response = client.post(
            "/api/v1/personas/generate",
            json={"panel_count": 0},
        )
        assert response.status_code == 422


class TestPersonaTemplates:
    def test_get_templates(self, client):
        response = client.get("/api/v1/personas/templates")
        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
        assert len(data["templates"]) > 0
        for template in data["templates"]:
            assert "id" in template
            assert "name" in template
            assert "config" in template


# ──────────────────────────────────────────────
# POST /api/v1/simulations/
# ──────────────────────────────────────────────


class TestSimulations:
    def test_create_simulation_returns_session(self, client):
        """Creating a simulation should return a session with PENDING status."""
        response = client.post(
            "/api/v1/simulations/",
            json={
                "panel_count": 3,
                "questions": [
                    {
                        "text": "테스트 질문",
                        "choices": ["A", "B", "C"],
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["status"] == "pending"

    def test_create_simulation_no_questions_fails(self, client):
        """At least one question is required."""
        response = client.post(
            "/api/v1/simulations/",
            json={
                "panel_count": 3,
                "questions": [],
            },
        )
        assert response.status_code == 422

    def test_get_nonexistent_session_returns_404(self, client):
        response = client.get("/api/v1/simulations/nonexistent-id")
        assert response.status_code == 404

    def test_get_result_nonexistent_returns_404(self, client):
        response = client.get("/api/v1/simulations/nonexistent-id/result")
        assert response.status_code == 404

    def test_create_and_get_session(self, client):
        """Create a session and verify it can be retrieved."""
        create_resp = client.post(
            "/api/v1/simulations/",
            json={
                "panel_count": 2,
                "questions": [
                    {"text": "Q1", "choices": ["Yes", "No"]},
                ],
            },
        )
        session_id = create_resp.json()["id"]

        get_resp = client.get(f"/api/v1/simulations/{session_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == session_id

    def test_get_result_before_completion_returns_400(self, client):
        """Fetching result of a non-COMPLETED simulation should return 400.

        Because the background task may complete very quickly in mock mode,
        we inject a PENDING session directly into the in-memory store.
        """
        from aipa_engine.api.simulations import sessions
        from aipa_engine.models.simulation import SimulationSession, SimulationStatus
        from aipa_engine.models.persona import PersonaConfig

        session_id = "test-pending-session"
        sessions[session_id] = SimulationSession(
            id=session_id,
            config=PersonaConfig(panel_count=1),
            questions=[],
            status=SimulationStatus.PENDING,
        )

        result_resp = client.get(f"/api/v1/simulations/{session_id}/result")
        assert result_resp.status_code == 400
        assert "not completed" in result_resp.json()["detail"].lower()

        # Clean up
        del sessions[session_id]


# ──────────────────────────────────────────────
# GET /api/v1/statistics/population/age
# ──────────────────────────────────────────────


class TestStatistics:
    def test_age_distribution_returns_200(self, client):
        """Age distribution endpoint should always return data (fallback if KOSIS fails)."""
        response = client.get("/api/v1/statistics/population/age")
        assert response.status_code == 200
        data = response.json()
        assert "distribution" in data
        assert "source" in data
        # Should have age group keys
        dist = data["distribution"]
        assert len(dist) > 0

    def test_gender_distribution_returns_200(self, client):
        response = client.get("/api/v1/statistics/population/gender")
        assert response.status_code == 200
        data = response.json()
        assert "distribution" in data
        dist = data["distribution"]
        assert "male" in dist or "female" in dist

    def test_occupation_distribution_returns_200(self, client):
        response = client.get("/api/v1/statistics/occupation")
        assert response.status_code == 200
        data = response.json()
        assert "distribution" in data
        assert len(data["distribution"]) > 0

    def test_age_distribution_values_sum_near_one(self, client):
        """Distribution values should approximately sum to 1.0."""
        response = client.get("/api/v1/statistics/population/age")
        data = response.json()
        total = sum(data["distribution"].values())
        assert pytest.approx(total, abs=0.05) == 1.0
