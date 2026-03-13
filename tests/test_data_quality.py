"""Data quality & data science best-practice tests.

These tests validate:
  - Raw CSV integrity (schema, no duplicates, value ranges)
  - Target distribution (class imbalance sanity)
  - Feature expectations (no constant columns, no extreme nulls)
  - Saved model artifact sanity
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import ARTIFACTS_DIR, EXPECTED_RAW_COLS, RAW_CSV_PATH, RENAME_MAP


# ── Raw CSV tests ─────────────────────────────────────────────────────
class TestRawCSV:
    @pytest.fixture(autouse=True)
    def _load_csv(self):
        self.df = pd.read_csv(RAW_CSV_PATH)

    def test_csv_not_empty(self):
        assert len(self.df) > 0

    def test_expected_row_count(self):
        """Telco churn dataset has 7043 rows."""
        assert len(self.df) == 7043

    def test_no_duplicate_customer_ids(self):
        assert self.df["customerID"].is_unique

    def test_expected_columns_present(self):
        renamed = self.df.rename(columns=RENAME_MAP)
        for col in EXPECTED_RAW_COLS:
            assert col in renamed.columns, f"Missing column: {col}"

    def test_target_is_binary(self):
        assert set(self.df["Churn"].dropna().unique()) == {"Yes", "No"}

    def test_monthly_charges_positive(self):
        charges = pd.to_numeric(self.df["MonthlyCharges"], errors="coerce")
        assert (charges.dropna() >= 0).all()

    def test_tenure_non_negative(self):
        tenure = pd.to_numeric(self.df["tenure"], errors="coerce")
        assert (tenure.dropna() >= 0).all()

    def test_senior_citizen_binary(self):
        assert set(self.df["SeniorCitizen"].dropna().unique()).issubset({0, 1})


# ── Target distribution ──────────────────────────────────────────────
class TestTargetDistribution:
    def test_churn_rate_within_expected_range(self):
        """Churn rate should be roughly 26-27 % for this dataset."""
        df = pd.read_csv(RAW_CSV_PATH)
        churn_rate = (df["Churn"] == "Yes").mean()
        assert 0.15 < churn_rate < 0.40, f"Unexpected churn rate: {churn_rate:.2%}"

    def test_no_class_has_zero_samples(self):
        df = pd.read_csv(RAW_CSV_PATH)
        counts = df["Churn"].value_counts()
        assert counts.min() > 0


# ── Feature quality (on renamed DF) ──────────────────────────────────
class TestFeatureQuality:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.df = pd.read_csv(RAW_CSV_PATH).rename(columns=RENAME_MAP)[
            EXPECTED_RAW_COLS
        ]

    def test_no_constant_columns(self):
        for col in self.df.columns:
            n_unique = self.df[col].nunique(dropna=True)
            assert n_unique > 1, f"Column '{col}' is constant"

    def test_null_fraction_below_threshold(self, threshold: float = 0.30):
        """No column should have > 30 % nulls."""
        for col in self.df.columns:
            null_frac = self.df[col].isna().mean()
            assert null_frac <= threshold, f"Column '{col}' has {null_frac:.1%} nulls"

    def test_no_fully_null_rows(self):
        fully_null = self.df.isna().all(axis=1).sum()
        assert fully_null == 0


# ── Saved artifact tests ─────────────────────────────────────────────
class TestArtifact:
    @pytest.fixture(autouse=True)
    def _find_latest(self):
        runs = sorted(ARTIFACTS_DIR.glob("baseline_v1_*"))
        if not runs:
            pytest.skip("No artifact runs found")
        self.run_dir = runs[-1]

    def test_model_file_exists(self):
        assert (self.run_dir / "model.joblib").exists()

    def test_metrics_file_exists(self):
        assert (self.run_dir / "metrics.json").exists()

    def test_metrics_json_valid(self):
        metrics = json.loads((self.run_dir / "metrics.json").read_text())
        assert "val" in metrics
        assert "test" in metrics

    def test_roc_auc_above_baseline(self):
        """Model ROC-AUC should beat a random classifier (0.5)."""
        metrics = json.loads((self.run_dir / "metrics.json").read_text())
        assert metrics["test"]["roc_auc"] > 0.5

    def test_threshold_between_0_and_1(self):
        metrics = json.loads((self.run_dir / "metrics.json").read_text())
        t = metrics["val"]["threshold"]
        assert 0 < t < 1

    def test_model_loads_and_predicts(self):
        import joblib

        model = joblib.load(self.run_dir / "model.joblib")
        # Build a tiny fake row with the right feature names
        df = pd.read_csv(RAW_CSV_PATH).rename(columns=RENAME_MAP)[EXPECTED_RAW_COLS]
        X = df.drop(columns=["customer_id", "churn"]).head(5)
        proba = model.predict_proba(X)[:, 1]
        assert proba.shape == (5,)
        assert np.all((proba >= 0) & (proba <= 1))
