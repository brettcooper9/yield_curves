"""
Yield curve fitting using Nelson-Siegel and Svensson models.

This module provides functions for fitting yield curves to bond data
using parametric models (Nelson-Siegel and Svensson).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from nelson_siegel_svensson import NelsonSiegelCurve, SvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_ns_ols, calibrate_nss_ols


def fit_nelson_siegel(
    yields: np.ndarray,
    maturities: np.ndarray
) -> Tuple[NelsonSiegelCurve, np.ndarray]:
    """
    Fit Nelson-Siegel curve to yield data.

    The Nelson-Siegel model represents the yield curve with 4 parameters:
    - beta0: long-term level
    - beta1: short-term component
    - beta2: medium-term component
    - tau: decay factor

    Parameters
    ----------
    yields : np.ndarray
        Array of observed yields (in decimal form, e.g., 0.025 for 2.5%)
    maturities : np.ndarray
        Array of corresponding maturities in years

    Returns
    -------
    curve : NelsonSiegelCurve
        Fitted Nelson-Siegel curve object
    fitted_yields : np.ndarray
        Model-fitted yields at the given maturities

    Examples
    --------
    >>> maturities = np.array([0.25, 0.5, 1, 2, 5, 10, 30])
    >>> yields = np.array([0.015, 0.018, 0.020, 0.022, 0.025, 0.028, 0.030])
    >>> curve, fitted = fit_nelson_siegel(yields, maturities)
    """
    # Remove any NaN values
    valid_idx = ~np.isnan(yields)
    clean_yields = yields[valid_idx]
    clean_maturities = maturities[valid_idx]

    # Calibrate the Nelson-Siegel model
    curve, status = calibrate_ns_ols(clean_maturities, clean_yields)

    # Get fitted values for all original maturities
    fitted_yields = np.array([curve(t) for t in maturities])

    return curve, fitted_yields


def fit_svensson(
    yields: np.ndarray,
    maturities: np.ndarray
) -> Tuple[SvenssonCurve, np.ndarray]:
    """
    Fit Svensson curve to yield data.

    The Svensson model extends Nelson-Siegel with 6 parameters for
    better flexibility in fitting complex yield curve shapes.

    Parameters
    ----------
    yields : np.ndarray
        Array of observed yields (in decimal form)
    maturities : np.ndarray
        Array of corresponding maturities in years

    Returns
    -------
    curve : SvenssonCurve
        Fitted Svensson curve object
    fitted_yields : np.ndarray
        Model-fitted yields at the given maturities

    Examples
    --------
    >>> maturities = np.array([0.25, 0.5, 1, 2, 5, 10, 30])
    >>> yields = np.array([0.015, 0.018, 0.020, 0.022, 0.025, 0.028, 0.030])
    >>> curve, fitted = fit_svensson(yields, maturities)
    """
    # Remove any NaN values
    valid_idx = ~np.isnan(yields)
    clean_yields = yields[valid_idx]
    clean_maturities = maturities[valid_idx]

    # Calibrate the Svensson model
    curve, status = calibrate_nss_ols(clean_maturities, clean_yields)

    # Get fitted values for all original maturities
    fitted_yields = np.array([curve(t) for t in maturities])

    return curve, fitted_yields


def fit_country_curves(
    bond_data: pd.DataFrame,
    country: str,
    date: pd.Timestamp,
    maturities_to_fit: np.ndarray,
    model: str = 'svensson'
) -> Dict[str, np.ndarray]:
    """
    Fit yield curve for a specific country and date.

    Parameters
    ----------
    bond_data : pd.DataFrame
        DataFrame containing bond yields with columns for country, maturity, date
    country : str
        ISO country code (e.g., 'US', 'AU', 'GB')
    date : pd.Timestamp
        Date for which to fit the curve
    maturities_to_fit : np.ndarray
        Array of maturities at which to evaluate the fitted curve
    model : str, default='svensson'
        Model to use: 'nelson_siegel' or 'svensson'

    Returns
    -------
    dict
        Dictionary containing:
        - 'fitted_yields': yields at specified maturities
        - 'curve': fitted curve object
        - 'observed_yields': original observed yields
        - 'observed_maturities': original maturities

    Examples
    --------
    >>> result = fit_country_curves(
    ...     bond_data, 'US', pd.Timestamp('2025-01-31'),
    ...     np.array([1, 2, 5, 10]), model='svensson'
    ... )
    """
    # Filter data for specific country and date
    country_data = bond_data[
        (bond_data['country'] == country) &
        (bond_data['date'] == date)
    ]

    if len(country_data) == 0:
        raise ValueError(f"No data found for {country} on {date}")

    # Extract yields and maturities
    observed_maturities = country_data['maturity'].values
    observed_yields = country_data['yield'].values / 100  # Convert to decimal

    # Fit the appropriate model
    if model.lower() == 'nelson_siegel':
        curve, _ = fit_nelson_siegel(observed_yields, observed_maturities)
    elif model.lower() == 'svensson':
        curve, _ = fit_svensson(observed_yields, observed_maturities)
    else:
        raise ValueError(f"Unknown model: {model}. Use 'nelson_siegel' or 'svensson'")

    # Get fitted yields at requested maturities
    fitted_yields = np.array([curve(t) for t in maturities_to_fit])

    return {
        'fitted_yields': fitted_yields * 100,  # Convert back to percentage
        'curve': curve,
        'observed_yields': observed_yields * 100,
        'observed_maturities': observed_maturities
    }


def create_weighted_curve(
    country_curves: Dict[str, np.ndarray],
    weights: Dict[str, float],
    maturities: np.ndarray
) -> pd.DataFrame:
    """
    Create a weighted composite yield curve from multiple countries.

    This is useful for creating synthetic curves like the WPU (World Public Unit)
    which combines yields from multiple countries weighted by economic factors.

    Parameters
    ----------
    country_curves : dict
        Dictionary mapping country codes to arrays of fitted yields
    weights : dict
        Dictionary mapping country codes to weight values (should sum to 1)
    maturities : np.ndarray
        Array of maturities corresponding to the yield values

    Returns
    -------
    pd.DataFrame
        DataFrame with maturities and weighted composite yields

    Examples
    --------
    >>> curves = {'US': np.array([2.5, 3.0, 3.5]), 'GB': np.array([2.0, 2.5, 3.0])}
    >>> weights = {'US': 0.6, 'GB': 0.4}
    >>> maturities = np.array([1, 5, 10])
    >>> wpu_curve = create_weighted_curve(curves, weights, maturities)
    """
    # Verify weights sum to approximately 1
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0, atol=0.01):
        raise ValueError(f"Weights sum to {total_weight}, should be close to 1.0")

    # Calculate weighted average yields
    weighted_yields = np.zeros(len(maturities))

    for country, country_yields in country_curves.items():
        if country not in weights:
            continue
        weight = weights[country]
        weighted_yields += weight * country_yields

    # Create DataFrame
    result = pd.DataFrame({
        'maturity': maturities,
        'yield': weighted_yields
    })

    return result
