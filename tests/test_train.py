"""Tests for the training pipeline logic (src/train_baseline_v1.py).

These tests exercise the pure-Python helpers without touching the database
or MLflow remote server.
"""

import numpy as np

from src.train_baseline_v1 import (
    build_base_pipeline,
    metrics_at_threshold,
    pick_threshold_by_cost,
)


class TestBuildBasePipeline:
    def test_pipeline_has_expected_steps(self, feature_df):
        pipe = build_base_pipeline(feature_df)
        step_names = [name for name, _ in pipe.steps]
        assert "preprocess" in step_names
        assert "model" in step_names

    def test_pipeline_fit_predict(self, feature_df, target_series):
        pipe = build_base_pipeline(feature_df)
        pipe.fit(feature_df, target_series)
        proba = pipe.predict_proba(feature_df)[:, 1]
        assert proba.shape == (len(feature_df),)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_pipeline_predictions_are_binary(self, feature_df, target_series):
        pipe = build_base_pipeline(feature_df)
        pipe.fit(feature_df, target_series)
        preds = pipe.predict(feature_df)
        assert set(preds).issubset({0, 1})


class TestPickThresholdByCost:
    def test_returns_float_threshold(self):
        rng = np.random.RandomState(0)
        y = rng.choice([0, 1], 200, p=[0.7, 0.3])
        proba = rng.uniform(0, 1, 200)
        t, cost = pick_threshold_by_cost(y, proba, c_fn=1, c_fp=1)
        assert isinstance(t, float)
        assert 0 < t < 1
        assert isinstance(cost, float)

    def test_higher_c_fn_lowers_threshold(self):
        """When missed churners are more expensive, the threshold should drop."""
        rng = np.random.RandomState(1)
        y = rng.choice([0, 1], 500, p=[0.7, 0.3])
        proba = rng.uniform(0, 1, 500)
        t_equal, _ = pick_threshold_by_cost(y, proba, c_fn=1, c_fp=1)
        t_high_fn, _ = pick_threshold_by_cost(y, proba, c_fn=10, c_fp=1)
        assert t_high_fn <= t_equal


class TestMetricsAtThreshold:
    def test_keys_present(self):
        y = np.array([0, 1, 1, 0, 1])
        proba = np.array([0.1, 0.8, 0.6, 0.3, 0.9])
        m = metrics_at_threshold(y, proba, t=0.5)
        assert "threshold" in m
        assert "confusion_matrix" in m
        assert "classification_report" in m

    def test_confusion_matrix_shape(self):
        y = np.array([0, 1, 1, 0, 1])
        proba = np.array([0.1, 0.8, 0.6, 0.3, 0.9])
        m = metrics_at_threshold(y, proba, t=0.5)
        cm = m["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2

    def test_all_positive_at_zero_threshold(self):
        y = np.array([0, 1, 0, 1])
        proba = np.array([0.2, 0.8, 0.3, 0.9])
        m = metrics_at_threshold(y, proba, t=0.0)
        cm = m["confusion_matrix"]
        # All predicted positive → FN should be 0
        assert cm[1][0] == 0  # fn = 0
