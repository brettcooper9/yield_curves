"""
Helper functions for yield curve calculations.

This module contains utility functions for bond return calculations,
time to maturity computations, and date series generation.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


def monthly_bond_return(yields: np.ndarray, i: int) -> float:
    """
    Calculate monthly bond return using Henckel's formula.

    This function computes the monthly return on a bond based on yield changes
    between consecutive months, accounting for both coupon payments and
    price appreciation/depreciation.

    Parameters
    ----------
    yields : np.ndarray
        Array of monthly bond yields (in percentage terms)
    i : int
        Current index in the yields array (must be >= 1)

    Returns
    -------
    float
        Monthly bond return

    Raises
    ------
    ValueError
        If i < 1 or i >= len(yields), or if calculation fails

    Examples
    --------
    >>> yields = np.array([2.5, 2.6, 2.4])
    >>> monthly_bond_return(yields, 1)
    0.0123  # example output

    Notes
    -----
    The formula assumes a 10-year bond with semi-annual compounding.
    Formula: prior_yield/1200 + (prior_yield/curr_yield) *
             (1 - (1 + curr_yield/200)^(-2*(10 - 1/12))) +
             (1 + curr_yield/200)^(-2*(10 - 1/12)) - 1
    """
    if i < 1 or i >= len(yields):
        raise ValueError(
            f"Index {i} out of valid range. Must be between 1 and {len(yields)-1}"
        )

    prior_mo = yields[i - 1]
    curr_mo = yields[i]

    if curr_mo <= 0:
        raise ValueError(f"Current yield must be positive, got {curr_mo}")

    try:
        # Henckel's formula for monthly bond return
        result = (
            prior_mo / 1200
            + (prior_mo / curr_mo) * (1 - (1 + curr_mo / 200) ** (-2 * (10 - 1/12)))
            + (1 + curr_mo / 200) ** (-2 * (10 - 1/12))
            - 1
        )
        return result
    except Exception as e:
        raise ValueError(f"Unable to calculate bond return: {str(e)}")


def round_month_to_maturity(
    current_date: pd.Timestamp | str,
    maturity_date: pd.Timestamp | str,
    num_digits: int = 4
) -> float:
    """
    Calculate time to maturity in years based on whole months.

    This function computes the time remaining until a specified maturity date,
    expressed in years, based on the number of months between the current date
    and the maturity date.

    Parameters
    ----------
    current_date : pd.Timestamp or str
        The current date or valuation date
    maturity_date : pd.Timestamp or str
        The maturity date of the instrument
    num_digits : int, default=4
        Number of decimal places for rounding the result

    Returns
    -------
    float
        Time to maturity in fractional years, rounded to num_digits

    Examples
    --------
    >>> round_month_to_maturity("2025-01-15", "2030-01-15")
    5.0

    >>> round_month_to_maturity("2025-01-15", "2026-07-15")
    1.5
    """
    # Convert to pandas Timestamp
    current_date = pd.Timestamp(current_date)
    maturity_date = pd.Timestamp(maturity_date)

    # Return 0 if already past maturity
    if current_date > maturity_date:
        return 0.0

    # Calculate month difference
    months_diff = (
        (maturity_date.year - current_date.year) * 12
        + (maturity_date.month - current_date.month)
    )

    # Convert to years and round
    years_to_maturity = round(months_diff / 12, num_digits)

    return years_to_maturity


def generate_month_end_series(
    origination_date: pd.Timestamp | str,
    maturity_date: pd.Timestamp | str,
    columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Generate a DataFrame with month-end dates from origination to maturity.

    Creates a time series DataFrame indexed by month-end dates, spanning from
    the origination date to the earlier of maturity date or today. This is
    useful for setting up structures to hold bond or swap calculations.

    Parameters
    ----------
    origination_date : pd.Timestamp or str
        Start date for the series
    maturity_date : pd.Timestamp or str
        End date for the series
    columns : list of str, optional
        Column names for the DataFrame. If None, defaults to ["Value"]

    Returns
    -------
    pd.DataFrame
        DataFrame with month-end dates as index and specified columns

    Raises
    ------
    ValueError
        If origination_date > maturity_date or origination_date > today

    Examples
    --------
    >>> df = generate_month_end_series(
    ...     "2020-01-15",
    ...     "2020-06-30",
    ...     columns=["yield", "return"]
    ... )
    >>> df.index
    DatetimeIndex(['2020-01-31', '2020-02-29', '2020-03-31',
                   '2020-04-30', '2020-05-31', '2020-06-30'])
    """
    # Convert to pandas Timestamp
    orig_date = pd.Timestamp(origination_date)
    mat_date = pd.Timestamp(maturity_date)
    today = pd.Timestamp.now().normalize()

    # Validate dates
    if orig_date > mat_date:
        raise ValueError("Origination date must be before maturity date")
    if orig_date > today:
        raise ValueError("Origination date must be before today")

    # End date is earlier of maturity or today
    final_date = min(mat_date, today)

    # Generate month-end dates
    # Start from the month containing orig_date
    month_ends = pd.date_range(
        start=orig_date,
        end=final_date,
        freq='M'  # Month end frequency
    )

    # Set default columns if not provided
    if columns is None:
        columns = ["Value"]

    # Create DataFrame with NaN values
    df = pd.DataFrame(
        index=month_ends,
        columns=columns,
        dtype=float
    )

    return df
