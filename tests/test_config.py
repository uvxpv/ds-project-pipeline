"""Tests for src/config.py — make sure constants are sane."""

from pathlib import Path

from src.config import (
    ARTIFACTS_DIR,
    C_FN,
    C_FP,
    DB_URL,
    EXPECTED_RAW_COLS,
    PROJECT_ROOT,
    RANDOM_STATE,
    RAW_CSV_PATH,
    RENAME_MAP,
    TARGET_COL,
    TOP_AT_RISK_MAX,
)


class TestPaths:
    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_artifacts_dir_is_under_root(self):
        assert ARTIFACTS_DIR.parent == PROJECT_ROOT

    def test_raw_csv_exists(self):
        assert RAW_CSV_PATH.exists()


class TestConstants:
    def test_random_state_is_int(self):
        assert isinstance(RANDOM_STATE, int)

    def test_costs_positive(self):
        assert C_FN > 0
        assert C_FP > 0

    def test_top_at_risk_max_positive(self):
        assert TOP_AT_RISK_MAX > 0

    def test_target_col_is_string(self):
        assert isinstance(TARGET_COL, str)

    def test_db_url_has_scheme(self):
        assert DB_URL.startswith("postgresql")

    def test_expected_cols_include_target(self):
        assert TARGET_COL in EXPECTED_RAW_COLS

    def test_rename_map_covers_expected_cols(self):
        mapped_values = set(RENAME_MAP.values())
        for col in EXPECTED_RAW_COLS:
            assert col in mapped_values, f"'{col}' not in RENAME_MAP values"
