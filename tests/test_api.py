"""Tests for the FastAPI application (src/api/main.py).

All tests use the FastAPI TestClient and mock the database engine so that
no live Postgres connection is needed.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src import config
from src.api.main import app

TEST_API_KEY = "test-key"
AUTH = {"X-API-Key": TEST_API_KEY}


# ── helpers ───────────────────────────────────────────────────────────
def _mock_engine():
    """Return a MagicMock that behaves like a SQLAlchemy engine context."""
    return MagicMock()


@pytest.fixture(autouse=True)
def _patch_engine():
    """Inject a mock engine into the module-level ``engine`` variable."""
    import src.api.main as api_mod

    api_mod.engine = _mock_engine()
    yield


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch):
    """Give the guarded routes a key to compare against.

    ``require_api_key`` reads ``config.API_KEY`` through the module, so patching
    the attribute is enough — no need to re-import src.api.main.
    """
    monkeypatch.setattr(config, "API_KEY", TEST_API_KEY)
    yield


@pytest.fixture()
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_ok(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ── /runs/latest ──────────────────────────────────────────────────────
class TestLatestRun:
    def test_latest_run_found(self, client: TestClient):
        fake_row = {
            "run_id": "20260211_135056",
            "model_name": "logreg",
            "created_at": "2026-02-11T13:50:56",
            "threshold": 0.5,
        }
        import src.api.main as api_mod

        mock_conn = MagicMock()
        mock_conn.execute.return_value.mappings.return_value.first.return_value = fake_row
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_conn)
        mock_context.__exit__ = MagicMock(return_value=False)
        api_mod.engine.begin = MagicMock(return_value=mock_context)

        resp = client.get("/runs/latest", headers=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == "20260211_135056"
        assert body["threshold"] == 0.5

    def test_latest_run_not_found(self, client: TestClient):
        import src.api.main as api_mod

        mock_conn = MagicMock()
        mock_conn.execute.return_value.mappings.return_value.first.return_value = None
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_conn)
        mock_context.__exit__ = MagicMock(return_value=False)
        api_mod.engine.begin = MagicMock(return_value=mock_context)

        resp = client.get("/runs/latest", headers=AUTH)
        assert resp.status_code == 404


# ── /predict_by_id ────────────────────────────────────────────────────
class TestPredictById:
    def test_predict_found(self, client: TestClient):
        fake_row = {
            "customer_id": "7590-VHVEG",
            "run_id": "20260211_135056",
            "scored_at": "2026-02-11T14:00:00",
            "proba_churn": 0.82,
            "pred_label": 1,
            "threshold": 0.5,
        }
        import src.api.main as api_mod

        mock_conn = MagicMock()
        mock_conn.execute.return_value.mappings.return_value.first.return_value = fake_row
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_conn)
        mock_context.__exit__ = MagicMock(return_value=False)
        api_mod.engine.begin = MagicMock(return_value=mock_context)

        resp = client.post("/predict_by_id", json={"customer_id": "7590-VHVEG"}, headers=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert body["customer_id"] == "7590-VHVEG"
        assert 0 <= body["proba_churn"] <= 1

    def test_predict_not_found(self, client: TestClient):
        import src.api.main as api_mod

        mock_conn = MagicMock()
        mock_conn.execute.return_value.mappings.return_value.first.return_value = None
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_conn)
        mock_context.__exit__ = MagicMock(return_value=False)
        api_mod.engine.begin = MagicMock(return_value=mock_context)

        resp = client.post("/predict_by_id", json={"customer_id": "UNKNOWN"}, headers=AUTH)
        assert resp.status_code == 404

    def test_predict_missing_body(self, client: TestClient):
        resp = client.post("/predict_by_id", json={}, headers=AUTH)
        assert resp.status_code == 422  # Pydantic validation error


# ── /top_at_risk ──────────────────────────────────────────────────────
class TestTopAtRisk:
    def test_top_at_risk_returns_list(self, client: TestClient):
        fake_rows = [
            {"customer_id": f"C-{i}", "proba_churn": 0.9 - i * 0.01, "pred_label": 1}
            for i in range(5)
        ]
        import src.api.main as api_mod

        mock_conn = MagicMock()
        mock_conn.execute.return_value.mappings.return_value.all.return_value = fake_rows
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_conn)
        mock_context.__exit__ = MagicMock(return_value=False)
        api_mod.engine.begin = MagicMock(return_value=mock_context)

        resp = client.get("/top_at_risk", params={"n": 5}, headers=AUTH)
        assert resp.status_code == 200
        body = resp.json()
        assert body["n"] == 5
        assert len(body["rows"]) == 5

    def test_top_at_risk_caps_n(self, client: TestClient):
        """n is capped at TOP_AT_RISK_MAX (500) server-side."""
        import src.api.main as api_mod

        mock_conn = MagicMock()
        mock_conn.execute.return_value.mappings.return_value.all.return_value = []
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_conn)
        mock_context.__exit__ = MagicMock(return_value=False)
        api_mod.engine.begin = MagicMock(return_value=mock_context)

        resp = client.get("/top_at_risk", params={"n": 9999}, headers=AUTH)
        assert resp.status_code == 200


# ── API-key guard ─────────────────────────────────────────────────────
class TestApiKeyGuard:
    """The data endpoints are guarded; /health deliberately is not."""

    def test_health_needs_no_key(self, client: TestClient):
        assert client.get("/health").status_code == 200

    def test_missing_key_is_rejected(self, client: TestClient):
        resp = client.get("/runs/latest")
        assert resp.status_code == 401

    def test_wrong_key_is_rejected(self, client: TestClient):
        resp = client.get("/runs/latest", headers={"X-API-Key": "nope"})
        assert resp.status_code == 401

    def test_unconfigured_server_key_fails_closed(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        """No API_KEY in the environment must not mean "no auth"."""
        monkeypatch.setattr(config, "API_KEY", "")
        resp = client.get("/runs/latest", headers=AUTH)
        assert resp.status_code == 503

    def test_guard_applies_to_all_data_routes(self, client: TestClient):
        assert client.get("/runs/latest").status_code == 401
        assert client.post("/predict_by_id", json={"customer_id": "x"}).status_code == 401
        assert client.get("/top_at_risk").status_code == 401

    def test_non_ascii_key_is_rejected_not_crashed(self, client: TestClient):
        """A non-ASCII header must be a 401, not a 500.

        Sent as raw bytes because httpx refuses non-ASCII header *strings* — a real
        HTTP client has no such scruples. Starlette latin-1-decodes it, and
        compare_digest would raise TypeError on the resulting str, hence the
        byte-wise comparison in require_api_key.
        """
        resp = client.get("/runs/latest", headers={"X-API-Key": "schlüssel".encode()})
        assert resp.status_code == 401
