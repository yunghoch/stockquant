"""Tests for QVM Screener."""

import pandas as pd
import pytest
from lasps.models.qvm_screener import QVMScreener


@pytest.fixture
def sample_stocks():
    return pd.DataFrame({
        "code": [f"{i:06d}" for i in range(100)],
        "market_cap": [i * 1e10 for i in range(1, 101)],
        "per": [10 + i * 0.5 for i in range(100)],
        "pbr": [0.5 + i * 0.05 for i in range(100)],
        "roe": [5 + i * 0.3 for i in range(100)],
        "debt_ratio": [50 + i for i in range(100)],
        "volume_avg_20": [1e6 + i * 1e4 for i in range(100)],
    })


def test_screen_returns_50(sample_stocks):
    screener = QVMScreener()
    result = screener.screen(sample_stocks, top_n=50)
    assert len(result) == 50
    assert "qvm_score" in result.columns
