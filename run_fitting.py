"""
Run yield curve fitting without Jupyter
Extracts and runs the core fitting logic from 02_yield_curve_fitting.ipynb
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from pathlib import Path
from yield_curves.yield_curve_fitting import fit_nelson_siegel, fit_svensson

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

# Initialize array to store fitted yields
ns_fitted_yields = np.zeros((len(unique_countries), len(fitting_dates), len(unique_maturities)))

print("\nFitting Nelson-Siegel curves for each country-date...")

# Fit curves for each country and date
errors = []

for date_idx, date in enumerate(fitting_dates):
    # Get the row index in original data
    row_idx = dates[dates == date].index[0] - dates.index[0]

    for country_idx, country in enumerate(unique_countries):
        try:
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

            # Fit Nelson-Siegel curve
            ns_curve, _ = fit_nelson_siegel(clean_yields, clean_mats)

            # Get fitted values at standard maturities
            for mat_idx, maturity in enumerate(unique_maturities):
                ns_fitted_yields[country_idx, date_idx, mat_idx] = ns_curve(maturity) * 100

        except Exception as e:
            errors.append((date, country, str(e)))

    if (date_idx + 1) % 10 == 0:
        print(f"  Processed {date_idx + 1}/{len(fitting_dates)} dates")

print(f"\nFitting complete!")
print(f"Errors encountered: {len(errors)}")

# Calculate WPU weighted composite curve
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

# Initialize WPU yields array
wpu_fitted_yields = np.zeros((len(fitting_dates), len(unique_maturities)))

# Calculate weighted average for each date
for date_idx, date in enumerate(fitting_dates):
    # Get weights for this date (use most recent available)
    weight_date_list = wpu_weights.index[wpu_weights.index <= date]
    if len(weight_date_list) == 0:
        print(f"Warning: No weights available for {date}, skipping")
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

                # Get yield for this country/date/maturity
                country_yield = ns_fitted_yields[country_idx, date_idx, mat_idx]

                if not np.isnan(country_yield) and weight > 0:
                    weighted_sum += country_yield * weight
                    total_weight += weight

        # Store weighted average
        if total_weight > 0:
            wpu_fitted_yields[date_idx, mat_idx] = weighted_sum / total_weight

print(f"WPU curve calculated for {len(fitting_dates)} dates")

# Combine country yields and WPU yields
all_countries = list(unique_countries) + ['WPU']
all_ns_yields = np.concatenate([
    ns_fitted_yields,
    wpu_fitted_yields.reshape(1, len(fitting_dates), len(unique_maturities))
], axis=0)

print(f"\nTotal yield curves: {len(all_countries)} (includes WPU)")

# Save results
output_path = Path('data/processed/fitted_yield_curves.npz')
output_path.parent.mkdir(parents=True, exist_ok=True)

np.savez(
    output_path,
    ns_yields=all_ns_yields,
    countries=np.array(all_countries),
    dates=fitting_dates.values,
    maturities=unique_maturities
)

print(f"\nSaved fitted yield curves to {output_path}")
print(f"  Shape: {all_ns_yields.shape}")
print(f"  Countries: {len(all_countries)}")
print(f"  Dates: {len(fitting_dates)}")
print(f"  Maturities: {len(unique_maturities)}")
print("\nDone! You can now run: streamlit run yield_curve_app.py")
