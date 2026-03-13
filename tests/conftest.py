"""Shared fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ── Dummy data that mirrors the raw_churn schema (after ingest rename) ──
@pytest.fixture()
def sample_raw_df() -> pd.DataFrame:
    """Small DataFrame that looks like the ingested raw_churn table."""
    rng = np.random.RandomState(42)
    n = 50
    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "senior_citizen": rng.choice([0, 1], n),
            "partner": rng.choice([True, False], n),
            "dependents": rng.choice([True, False], n),
            "tenure": rng.randint(0, 72, n),
            "phone_service": rng.choice([True, False], n),
            "multiple_lines": rng.choice(["Yes", "No", "No phone service"], n),
            "internet_service": rng.choice(["DSL", "Fiber optic", "No"], n),
            "online_security": rng.choice(["Yes", "No", "No internet service"], n),
            "online_backup": rng.choice(["Yes", "No", "No internet service"], n),
            "device_protection": rng.choice(["Yes", "No", "No internet service"], n),
            "tech_support": rng.choice(["Yes", "No", "No internet service"], n),
            "streaming_tv": rng.choice(["Yes", "No", "No internet service"], n),
            "streaming_movies": rng.choice(["Yes", "No", "No internet service"], n),
            "contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "paperless_billing": rng.choice([True, False], n),
            "payment_method": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "monthly_charges": rng.uniform(18, 120, n).round(2),
            "total_charges": rng.uniform(18, 8000, n).round(2),
            "churn": rng.choice([0, 1], n, p=[0.73, 0.27]),
        }
    )


@pytest.fixture()
def feature_df(sample_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix (X) ready for model training / inference."""
    return sample_raw_df.drop(columns=["customer_id", "churn"])


@pytest.fixture()
def target_series(sample_raw_df: pd.DataFrame) -> np.ndarray:
    return sample_raw_df["churn"].values


@pytest.fixture()
def api_client():
    """TestClient that does NOT need a live database.

    We patch the engine away at the lifespan level and override
    specific routes in individual test functions.
    """
    from src.api.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
