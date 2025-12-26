"""
Yield curve fitting using Nelson-Siegel and Svensson models.

This module provides functions for fitting yield curves to bond data
using parametric models (Nelson-Siegel and Svensson).

Note: This uses scipy for optimization instead of nelson-siegel-svensson package
which has compatibility issues. The implementation follows the same mathematical
formulas as the R YieldCurve package.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Callable
from scipy.optimize import least_squares


class NelsonSiegelCurve:
    """Nelson-Siegel yield curve model."""

    def __init__(self, beta0: float, beta1: float, beta2: float, tau: float):
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau

    def __call__(self, t: float) -> float:
        """Evaluate yield at maturity t."""
        if t <= 0:
            return self.beta0 + self.beta1
        tau_t = t / self.tau
        exp_term = np.exp(-tau_t)
        return (self.beta0 +
                self.beta1 * (1 - exp_term) / tau_t +
                self.beta2 * ((1 - exp_term) / tau_t - exp_term))

    def __repr__(self):
        return f"NelsonSiegelCurve(β0={self.beta0:.4f}, β1={self.beta1:.4f}, β2={self.beta2:.4f}, τ={self.tau:.4f})"


class SvenssonCurve:
    """Svensson yield curve model (extension of Nelson-Siegel)."""

    def __init__(self, beta0: float, beta1: float, beta2: float, beta3: float,
                 tau1: float, tau2: float):
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.tau1 = tau1
        self.tau2 = tau2

    def __call__(self, t: float) -> float:
        """Evaluate yield at maturity t."""
        if t <= 0:
            return self.beta0 + self.beta1
        tau1_t = t / self.tau1
        tau2_t = t / self.tau2
        exp1 = np.exp(-tau1_t)
        exp2 = np.exp(-tau2_t)
        return (self.beta0 +
                self.beta1 * (1 - exp1) / tau1_t +
                self.beta2 * ((1 - exp1) / tau1_t - exp1) +
                self.beta3 * ((1 - exp2) / tau2_t - exp2))

    def __repr__(self):
        return (f"SvenssonCurve(β0={self.beta0:.4f}, β1={self.beta1:.4f}, "
                f"β2={self.beta2:.4f}, β3={self.beta3:.4f}, "
                f"τ1={self.tau1:.4f}, τ2={self.tau2:.4f})")


def fit_nelson_siegel(
    yields: np.ndarray,
    maturities: np.ndarray
) -> Tuple[NelsonSiegelCurve, np.ndarray]:
    """
    Fit Nelson-Siegel curve to yield data using OLS.

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
    """
    # Remove any NaN values
    valid_idx = ~np.isnan(yields)
    clean_yields = yields[valid_idx]
    clean_maturities = maturities[valid_idx]

    # Objective function to minimize
    def objective(params):
        beta0, beta1, beta2, tau = params
        curve = NelsonSiegelCurve(beta0, beta1, beta2, tau)
        predicted = np.array([curve(t) for t in clean_maturities])
        return predicted - clean_yields

    # Initial guess
    x0 = [clean_yields.mean(), -0.02, 0.02, 2.0]

    # Bounds: tau must be positive
    bounds = ([-np.inf, -np.inf, -np.inf, 0.01],
              [np.inf, np.inf, np.inf, 100])

    # Fit
    result = least_squares(objective, x0, bounds=bounds)

    # Create curve object
    curve = NelsonSiegelCurve(*result.x)

    # Get fitted values for all original maturities
    fitted_yields = np.array([curve(t) for t in maturities])

    return curve, fitted_yields


def fit_svensson(
    yields: np.ndarray,
    maturities: np.ndarray
) -> Tuple[SvenssonCurve, np.ndarray]:
    """
    Fit Svensson curve to yield data using OLS.

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
    """
    # Remove any NaN values
    valid_idx = ~np.isnan(yields)
    clean_yields = yields[valid_idx]
    clean_maturities = maturities[valid_idx]

    # Objective function
    def objective(params):
        beta0, beta1, beta2, beta3, tau1, tau2 = params
        curve = SvenssonCurve(beta0, beta1, beta2, beta3, tau1, tau2)
        predicted = np.array([curve(t) for t in clean_maturities])
        return predicted - clean_yields

    # Initial guess
    x0 = [clean_yields.mean(), -0.02, 0.02, 0.01, 2.0, 5.0]

    # Bounds: both taus must be positive
    bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, 0.01, 0.01],
              [np.inf, np.inf, np.inf, np.inf, 100, 100])

    # Fit
    result = least_squares(objective, x0, bounds=bounds)

    # Create curve object
    curve = SvenssonCurve(*result.x)

    # Get fitted values
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
