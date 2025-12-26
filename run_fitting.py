"""
Run yield curve fitting without Jupyter
Extracts and runs the core fitting logic from 02_yield_curve_fitting.ipynb
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from pathlib import Path
from yield_curves.yield_curve_fitting import (
    fit_nelson_siegel, fit_svensson, fit_cubic_spline, fit_bspline, calculate_rmse
)

print("Loading raw bond data...")

# Load the data (wide format from Excel/CSV export)
DATA_DIR = Path('data/raw')
bond_data = pd.read_csv(DATA_DIR / 'bond_dat.csv', header=None)

print(f"Bond data shape: {bond_data.shape}")

# Extract metadata from specific rows (matching R code structure)
# Row 1 (index 0): Maturities
# Row 3 (index 2): Countries
# Row 8+ (index 7+): Dates and yields

maturities = pd.to_numeric(bond_data.iloc[0, 1:], errors='coerce').values
countries = bond_data.iloc[2, 1:].values
dates = pd.to_datetime(bond_data.iloc[7:, 0])

# Extract yield matrix (rows 8+, columns 2+)
yields_matrix = bond_data.iloc[7:, 1:].apply(pd.to_numeric, errors='coerce').values

print(f"Number of dates: {len(dates)}")
print(f"Number of series: {len(countries)}")
print(f"Unique countries: {sorted(set(countries))}")

# Define fitting period
# Default: Start from when RUB was dropped from WPU basket (2024-08-30)
# Can be changed to include more historical data
START_DATE = '2024-08-30'  # When Russia dropped from WPU basket
fitting_dates = dates[dates >= START_DATE]

# Get unique values for output
unique_maturities = sorted(set(maturities[~np.isnan(maturities)]))
unique_countries = sorted(set(countries))

print(f"\nFitting period: {fitting_dates.min()} to {fitting_dates.max()}")
print(f"Number of dates: {len(fitting_dates)}")
print(f"Countries: {unique_countries}")
print(f"Number of maturities: {len(unique_maturities)}")

# Initialize arrays to store fitted yields for all 4 models
# Shape: (model, country, date, maturity)
model_names = ['Nelson-Siegel', 'Svensson', 'Cubic Spline', 'B-Spline']
n_models = len(model_names)
all_fitted_yields = np.zeros((n_models, len(unique_countries), len(fitting_dates), len(unique_maturities)))
all_rmse = np.zeros((n_models, len(unique_countries), len(fitting_dates)))

print(f"\nFitting {n_models} models for each country-date...")
print(f"Models: {', '.join(model_names)}")

# Fit curves for each country and date
errors = []

for date_idx, date in enumerate(fitting_dates):
    # Get the row index in original data
    row_idx = dates[dates == date].index[0] - dates.index[0]

    for country_idx, country in enumerate(unique_countries):
        # Get data for this country and date
        country_mask = countries == country
        obs_maturities = maturities[country_mask]
        obs_yields = yields_matrix[row_idx, country_mask] / 100

        # Remove NaN values
        valid = ~np.isnan(obs_yields)
        if valid.sum() < 3:  # Need at least 3 points
            continue

        clean_mats = obs_maturities[valid]
        clean_yields = obs_yields[valid]

        # Get all maturities' observed yields for RMSE calculation
        all_obs_yields = obs_yields.copy()

        # Determine the range of observed maturities for spline clipping
        min_observed_mat = clean_mats.min()
        max_observed_mat = clean_mats.max()

        # Fit all 4 models
        for model_idx, model_name in enumerate(model_names):
            try:
                if model_name == 'Nelson-Siegel':
                    curve, fitted = fit_nelson_siegel(clean_yields, clean_mats)
                elif model_name == 'Svensson':
                    curve, fitted = fit_svensson(clean_yields, clean_mats)
                elif model_name == 'Cubic Spline':
                    curve, fitted = fit_cubic_spline(clean_yields, clean_mats)
                elif model_name == 'B-Spline':
                    curve, fitted = fit_bspline(clean_yields, clean_mats)

                # Get fitted values at standard maturities
                # Allow all models to extrapolate for smooth WPU curves
                fitted_at_maturities = np.zeros(len(unique_maturities))
                for mat_idx, maturity in enumerate(unique_maturities):
                    fitted_at_maturities[mat_idx] = curve(maturity)

                # Store fitted yields (convert to percentage)
                all_fitted_yields[model_idx, country_idx, date_idx, :] = fitted_at_maturities * 100

                # Calculate RMSE against observed yields
                # Create array of fitted yields at observed maturities for RMSE
                fitted_at_obs = np.zeros(len(obs_maturities))
                for i, mat in enumerate(obs_maturities):
                    fitted_at_obs[i] = curve(mat)

                rmse = calculate_rmse(fitted_at_obs, all_obs_yields)
                all_rmse[model_idx, country_idx, date_idx] = rmse

            except Exception as e:
                errors.append((date, country, model_name, str(e)))
                # Fill with NaN on error
                all_fitted_yields[model_idx, country_idx, date_idx, :] = np.nan
                all_rmse[model_idx, country_idx, date_idx] = np.nan

    if (date_idx + 1) % 10 == 0:
        print(f"  Processed {date_idx + 1}/{len(fitting_dates)} dates")

print(f"\nFitting complete!")
print(f"Errors encountered: {len(errors)}")

# Find best model for each country-date based on RMSE
best_model_idx = np.nanargmin(all_rmse, axis=0)  # Shape: (country, date)
print(f"\nBest model selection (by RMSE):")
for country_idx, country in enumerate(unique_countries):
    model_counts = np.bincount(best_model_idx[country_idx, :][~np.isnan(all_rmse[:, country_idx, :]).all(axis=0)], minlength=n_models)
    print(f"  {country}: ", end="")
    for m_idx, count in enumerate(model_counts):
        if count > 0:
            print(f"{model_names[m_idx]}={count} ", end="")
    print()

# Calculate WPU weighted composite curve using best model for each country
print("\nCalculating WPU weighted composite curve...")

# Load WPU weights
wpu_weights = pd.read_excel(DATA_DIR / 'wpu_weights.xlsx')
wpu_weights['Date'] = pd.to_datetime(wpu_weights['Column1'])
wpu_weights = wpu_weights.set_index('Date')

print(f"Loaded WPU weights from {wpu_weights.index.min().date()} to {wpu_weights.index.max().date()}")

# Map country codes to weight columns
country_map = {
    'AU': 'AUD', 'BR': 'BRL', 'CA': 'CAD', 'CH': 'CHF',
    'CN': 'CNY', 'EU': 'EUR', 'GB': 'GBP', 'IN': 'INR',
    'JP': 'JPY', 'MX': 'MXN', 'US': 'USD'
}

# Initialize WPU yields array for each model
wpu_fitted_yields = np.zeros((n_models, len(fitting_dates), len(unique_maturities)))

# Calculate weighted average for each date and model
for model_idx in range(n_models):
    for date_idx, date in enumerate(fitting_dates):
        # Get weights for this date (use most recent available)
        weight_date_list = wpu_weights.index[wpu_weights.index <= date]
        if len(weight_date_list) == 0:
            continue

        weight_date = weight_date_list[-1]
        weights_row = wpu_weights.loc[weight_date]

        # Calculate weighted average across countries for each maturity
        for mat_idx in range(len(unique_maturities)):
            weighted_sum = 0
            total_weight = 0

            for country_idx, country in enumerate(unique_countries):
                if country in country_map:
                    weight_col = country_map[country]
                    weight = weights_row[weight_col]

                    # Get yield for this country/date/maturity from this model
                    country_yield = all_fitted_yields[model_idx, country_idx, date_idx, mat_idx]

                    if not np.isnan(country_yield) and weight > 0:
                        weighted_sum += country_yield * weight
                        total_weight += weight

            # Store weighted average
            if total_weight > 0:
                wpu_fitted_yields[model_idx, date_idx, mat_idx] = weighted_sum / total_weight

print(f"WPU curve calculated for {len(fitting_dates)} dates across {n_models} models")

# Add WPU to countries list
all_countries = list(unique_countries) + ['WPU']

# Expand arrays to include WPU
# Shape: (model, country+WPU, date, maturity)
all_fitted_yields_with_wpu = np.concatenate([
    all_fitted_yields,
    wpu_fitted_yields[:, np.newaxis, :, :]
], axis=1)

# WPU has no RMSE (it's a composite)
wpu_rmse = np.full((n_models, 1, len(fitting_dates)), np.nan)
all_rmse_with_wpu = np.concatenate([all_rmse, wpu_rmse], axis=1)

# Expand best_model_idx to include WPU (use Nelson-Siegel as default)
best_model_idx_with_wpu = np.concatenate([
    best_model_idx,
    np.zeros((1, len(fitting_dates)), dtype=int)  # WPU defaults to first model
], axis=0)

print(f"\nTotal yield curves: {len(all_countries)} (includes WPU)")

# Save results
output_path = Path('data/processed/fitted_yield_curves.npz')
output_path.parent.mkdir(parents=True, exist_ok=True)

np.savez(
    output_path,
    # All model yields: shape (model, country, date, maturity)
    all_model_yields=all_fitted_yields_with_wpu,
    # RMSE for each model: shape (model, country, date)
    all_model_rmse=all_rmse_with_wpu,
    # Best model index for each country-date: shape (country, date)
    best_model_idx=best_model_idx_with_wpu,
    # Model names
    model_names=np.array(model_names),
    # Metadata
    countries=np.array(all_countries),
    dates=fitting_dates.values,
    maturities=np.array(unique_maturities)
)

print(f"\nSaved fitted yield curves to {output_path}")
print(f"  All models shape: {all_fitted_yields_with_wpu.shape}")
print(f"  Models: {len(model_names)}")
print(f"  Countries: {len(all_countries)}")
print(f"  Dates: {len(fitting_dates)}")
print(f"  Maturities: {len(unique_maturities)}")
print("\nDone! You can now run: streamlit run yield_curve_app.py")
