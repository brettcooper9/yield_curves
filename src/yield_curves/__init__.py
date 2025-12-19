"""
Yield Curve Analysis Package

This package provides tools for analyzing bond yields, currency swaps,
and yield curve fitting using Nelson-Siegel and Svensson models.
"""

__version__ = "0.1.0"

from .helpers import (
    monthly_bond_return,
    round_month_to_maturity,
    generate_month_end_series,
)

__all__ = [
    "monthly_bond_return",
    "round_month_to_maturity",
    "generate_month_end_series",
]
