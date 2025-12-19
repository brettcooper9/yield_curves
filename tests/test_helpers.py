"""
Unit tests for helper functions.

Run with: pytest tests/test_helpers.py
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

import sys
sys.path.insert(0, '../src')

from yield_curves.helpers import (
    monthly_bond_return,
    round_month_to_maturity,
    generate_month_end_series,
)


class TestMonthlyBondReturn:
    """Tests for monthly_bond_return function."""

    def test_basic_return_calculation(self):
        """Test basic bond return calculation."""
        yields = np.array([2.5, 2.6, 2.4])
        result = monthly_bond_return(yields, 1)
        assert isinstance(result, float)
        # Result should be reasonable (within -10% to +10% for monthly return)
        assert -0.1 < result < 0.1

    def test_invalid_index_low(self):
        """Test that index < 1 raises ValueError."""
        yields = np.array([2.5, 2.6, 2.4])
        with pytest.raises(ValueError, match="out of valid range"):
            monthly_bond_return(yields, 0)

    def test_invalid_index_high(self):
        """Test that index >= len(yields) raises ValueError."""
        yields = np.array([2.5, 2.6, 2.4])
        with pytest.raises(ValueError, match="out of valid range"):
            monthly_bond_return(yields, 3)

    def test_zero_yield_raises_error(self):
        """Test that zero current yield raises ValueError."""
        yields = np.array([2.5, 0.0])
        with pytest.raises(ValueError, match="must be positive"):
            monthly_bond_return(yields, 1)

    def test_negative_yield_raises_error(self):
        """Test that negative current yield raises ValueError."""
        yields = np.array([2.5, -1.0])
        with pytest.raises(ValueError, match="must be positive"):
            monthly_bond_return(yields, 1)


class TestRoundMonthToMaturity:
    """Tests for round_month_to_maturity function."""

    def test_exact_years(self):
        """Test maturity calculation for exact years."""
        result = round_month_to_maturity("2020-01-15", "2025-01-15")
        assert result == 5.0

    def test_half_year(self):
        """Test maturity calculation for half year."""
        result = round_month_to_maturity("2020-01-15", "2020-07-15")
        assert result == 0.5

    def test_timestamp_input(self):
        """Test that pd.Timestamp inputs work."""
        result = round_month_to_maturity(
            pd.Timestamp("2020-01-15"),
            pd.Timestamp("2021-01-15")
        )
        assert result == 1.0

    def test_past_maturity_returns_zero(self):
        """Test that past maturity returns 0."""
        result = round_month_to_maturity("2025-01-15", "2020-01-15")
        assert result == 0.0

    def test_custom_digits(self):
        """Test custom number of digits for rounding."""
        result = round_month_to_maturity(
            "2020-01-01", "2020-03-15",
            num_digits=2
        )
        # Should be around 0.17 to 0.25 years
        assert 0 <= result <= 1


class TestGenerateMonthEndSeries:
    """Tests for generate_month_end_series function."""

    def test_basic_generation(self):
        """Test basic month-end series generation."""
        df = generate_month_end_series(
            "2020-01-15",
            "2020-06-30",
            columns=["yield", "return"]
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) == 2
        assert "yield" in df.columns
        assert "return" in df.columns
        # Should have month-end dates
        assert all(df.index.is_month_end)

    def test_default_column(self):
        """Test with default column name."""
        df = generate_month_end_series("2020-01-01", "2020-03-31")
        assert len(df.columns) == 1
        assert df.columns[0] == "Value"

    def test_all_values_nan(self):
        """Test that all values are initially NaN."""
        df = generate_month_end_series(
            "2020-01-01", "2020-03-31",
            columns=["test"]
        )
        assert df["test"].isna().all()

    def test_invalid_date_order(self):
        """Test that origination > maturity raises error."""
        with pytest.raises(ValueError, match="before maturity"):
            generate_month_end_series("2025-01-01", "2020-01-01")

    def test_future_origination(self):
        """Test that future origination date raises error."""
        future_date = pd.Timestamp.now() + pd.Timedelta(days=365)
        with pytest.raises(ValueError, match="before today"):
            generate_month_end_series(
                future_date.strftime("%Y-%m-%d"),
                "2030-01-01"
            )

    def test_caps_at_today(self):
        """Test that series stops at today if maturity is in future."""
        future_date = pd.Timestamp.now() + pd.Timedelta(days=365)
        df = generate_month_end_series(
            "2020-01-01",
            future_date.strftime("%Y-%m-%d")
        )
        # Last date should be <= today
        assert df.index[-1] <= pd.Timestamp.now()


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_bond_return_time_series(self):
        """Test calculating bond returns for a time series."""
        # Create mock yield data
        dates = pd.date_range("2020-01-31", "2020-06-30", freq='M')
        yields = np.array([2.5, 2.6, 2.4, 2.7, 2.5, 2.6])

        # Calculate returns for each month
        returns = []
        for i in range(1, len(yields)):
            ret = monthly_bond_return(yields, i)
            returns.append(ret)

        assert len(returns) == len(yields) - 1
        assert all(isinstance(r, float) for r in returns)

    def test_maturity_decreases_over_time(self):
        """Test that time to maturity decreases as time passes."""
        maturity_date = "2025-12-31"
        dates = ["2020-01-15", "2021-01-15", "2022-01-15"]

        maturities = [
            round_month_to_maturity(date, maturity_date)
            for date in dates
        ]

        # Each maturity should be less than the previous
        for i in range(1, len(maturities)):
            assert maturities[i] < maturities[i-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
