"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_yields():
    """Sample yield data for testing."""
    return np.array([2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5])


@pytest.fixture
def sample_maturities():
    """Sample maturities corresponding to yields."""
    return np.array([0.25, 0.5, 1, 2, 5, 10, 30])


@pytest.fixture
def sample_dates():
    """Sample date range for testing."""
    return pd.date_range("2020-01-31", "2020-12-31", freq='M')


@pytest.fixture
def sample_fx_data(sample_dates):
    """Sample FX data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'WPUUSD': 1.0 + np.random.randn(len(sample_dates)) * 0.02
    }, index=sample_dates)
