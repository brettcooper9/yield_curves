"""
Data validation schemas using Pandera.

This module defines validation rules for bond data, FX data,
and yield curves to ensure data quality and catch errors early.
"""

import pandera as pa
from pandera import Column, DataFrameSchema, Check


# Schema for FX exchange rate data
fx_data_schema = DataFrameSchema(
    {
        "WPUUSD": Column(
            float,
            checks=[
                Check.greater_than(0, error="Exchange rate must be positive"),
                Check.less_than(10, error="Exchange rate seems unrealistic"),
            ],
            nullable=True,
        ),
    },
    index=pa.Index(pa.DateTime, name="Date"),
    strict=False,  # Allow additional currency columns
    coerce=True,
)


# Schema for bond yield data
bond_yield_schema = DataFrameSchema(
    {
        "country": Column(
            str,
            checks=[
                Check.str_length(2, 3),
                Check(
                    lambda s: s.str.isupper().all(),
                    error="Country codes must be uppercase"
                ),
            ]
        ),
        "maturity": Column(
            float,
            checks=[
                Check.greater_than(0, error="Maturity must be positive"),
                Check.less_than(100, error="Maturity exceeds reasonable range"),
            ]
        ),
        "yield": Column(
            float,
            checks=[
                Check.greater_than(-5, error="Yield seems too negative"),
                Check.less_than(50, error="Yield seems unrealistically high"),
            ],
            nullable=True,
        ),
        "date": Column(pa.DateTime),
    },
    coerce=True,
)


# Schema for weight data (WPU constituents)
weight_schema = DataFrameSchema(
    strict=False,
    coerce=True,
    checks=[
        # Check that each row sums to approximately 100%
        Check(
            lambda df: df.select_dtypes(include='number').sum(axis=1).between(99, 101).all(),
            error="Weights should sum to approximately 100% per date",
            ignore_na=True,
        )
    ],
)


# Schema for fitted yield curve results
fitted_curve_schema = DataFrameSchema(
    {
        "maturity": Column(
            float,
            checks=[
                Check.greater_than(0),
                Check.less_than(100),
            ]
        ),
        "yield": Column(
            float,
            checks=[
                Check.greater_than(-5),
                Check.less_than(50),
            ]
        ),
    },
    coerce=True,
)


# Schema for swap analysis results
swap_results_schema = DataFrameSchema(
    {
        "time_to_maturity": Column(float, nullable=True),
        "long_bond_yield": Column(float, nullable=True),
        "short_bond_yield": Column(float, nullable=True),
        "lb_return": Column(
            float,
            checks=[
                Check.greater_than(-0.5, error="Bond return < -50% seems extreme"),
                Check.less_than(0.5, error="Bond return > 50% seems extreme"),
            ],
            nullable=True,
        ),
        "sb_return": Column(
            float,
            checks=[
                Check.greater_than(-0.5),
                Check.less_than(0.5),
            ],
            nullable=True,
        ),
        "long_ccy_return": Column(
            float,
            checks=[
                Check.greater_than(-0.3, error="FX return < -30% seems extreme"),
                Check.less_than(0.3, error="FX return > 30% seems extreme"),
            ],
            nullable=True,
        ),
        "short_ccy_return": Column(float, nullable=True),
        "lb_cumulative": Column(
            float,
            checks=[Check.greater_than(0, error="Cumulative return index must be positive")],
            nullable=True,
        ),
        "sb_cumulative": Column(
            float,
            checks=[Check.greater_than(0)],
            nullable=True,
        ),
        "swap_cumulative": Column(
            float,
            checks=[Check.greater_than(0)],
            nullable=True,
        ),
    },
    index=pa.Index(pa.DateTime),
    coerce=True,
)


def validate_fx_data(df: pa.typing.DataFrame) -> pa.typing.DataFrame:
    """
    Validate FX exchange rate data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Date index and currency columns

    Returns
    -------
    pd.DataFrame
        Validated DataFrame

    Raises
    ------
    pandera.errors.SchemaError
        If validation fails
    """
    return fx_data_schema.validate(df)


def validate_bond_yields(df: pa.typing.DataFrame) -> pa.typing.DataFrame:
    """
    Validate bond yield data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with country, maturity, yield, and date columns

    Returns
    -------
    pd.DataFrame
        Validated DataFrame

    Raises
    ------
    pandera.errors.SchemaError
        If validation fails
    """
    return bond_yield_schema.validate(df)


def validate_weights(df: pa.typing.DataFrame) -> pa.typing.DataFrame:
    """
    Validate weight data for composite curves.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date index and country weight columns

    Returns
    -------
    pd.DataFrame
        Validated DataFrame

    Raises
    ------
    pandera.errors.SchemaError
        If validation fails
    """
    return weight_schema.validate(df)


def validate_swap_results(df: pa.typing.DataFrame) -> pa.typing.DataFrame:
    """
    Validate swap analysis results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with swap calculation results

    Returns
    -------
    pd.DataFrame
        Validated DataFrame

    Raises
    ------
    pandera.errors.SchemaError
        If validation fails
    """
    return swap_results_schema.validate(df)
