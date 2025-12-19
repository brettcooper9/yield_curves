# Yield Curve Analysis

A Python package for analyzing bond yields, fitting yield curves, and evaluating currency swap strategies.

**Author:** Brett Cooper
**Version:** 0.1.0
**Status:** Active Development

---

## Overview

This project analyzes government bond yields across multiple countries, fits parametric yield curve models (Nelson-Siegel and Svensson), and evaluates currency swap trading strategies. The code was converted from R to Python with improvements in reproducibility, testing, and documentation following [Openscapes](https://www.openscapes.org/) principles.

### Key Features

- **Yield Curve Fitting:** Nelson-Siegel and Svensson models for smooth yield curves
- **Multi-Country Analysis:** Process bond data for multiple countries simultaneously
- **Currency Swap Analysis:** Evaluate bond returns including FX effects
- **Data Validation:** Automatic data quality checks using Pandera
- **Reproducible Workflows:** Conda environment and Jupyter notebooks
- **Well-Tested:** Unit tests with pytest

---

## Project Structure

```
yield_curves/
├── data/
│   ├── raw/              # Original data files (CSV, Excel)
│   └── processed/        # Cleaned and processed data
├── src/
│   └── yield_curves/     # Python package
│       ├── __init__.py
│       ├── helpers.py              # Utility functions
│       ├── yield_curve_fitting.py  # Curve fitting models
│       ├── swap_analysis.py        # Swap calculations
│       └── validation.py           # Data validation schemas
├── notebooks/
│   ├── 01_bond_swap_analysis.ipynb
│   └── 02_yield_curve_fitting.ipynb
├── tests/
│   ├── test_helpers.py
│   └── conftest.py
├── docs/                 # Additional documentation
├── environment.yml       # Conda environment specification
├── README.md            # This file
└── .gitignore
```

---

## Getting Started

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   cd /path/to/yield_curves
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment:**
   ```bash
   conda activate yield_curves
   ```

4. **Verify installation:**
   ```bash
   python -c "import yield_curves; print('Success!')"
   ```

### Quick Start

1. **Run the notebooks:**
   ```bash
   jupyter lab
   ```

   Open and run:
   - `notebooks/02_yield_curve_fitting.ipynb` - Fit yield curves to bond data
   - `notebooks/01_bond_swap_analysis.ipynb` - Analyze swap strategies

2. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

---

## Data

### Required Data Files

Place these files in `data/raw/`:

- `bond_dat.csv` - Bond yield data for multiple countries and maturities
- `wpu_exchange_rates.csv` - WPU/USD exchange rate time series
- `wpu_weights.xlsx` - Country weights for WPU basket
- `yields_fit_all_mat.Rdata` - (Optional) Pre-fitted yield curves from R

### Data Format

**Bond Data** (`bond_dat.csv`):
- Row 1: Maturities (in years)
- Row 3: Country codes (e.g., US, GB, AU)
- Row 8+: Dates and yield values

**FX Data** (`wpu_exchange_rates.csv`):
```
Date,WPUUSD
2020-01-31,1.0234
2020-02-29,1.0189
...
```

**Weights** (`wpu_weights.xlsx`):
- Date column + country weight columns (should sum to ~100%)

---

## Usage Examples

### Fit a Yield Curve

```python
from yield_curves.yield_curve_fitting import fit_svensson
import numpy as np

# Sample data
maturities = np.array([0.25, 0.5, 1, 2, 5, 10, 30])
yields = np.array([0.015, 0.018, 0.020, 0.022, 0.025, 0.028, 0.030])

# Fit Svensson model
curve, fitted_yields = fit_svensson(yields, maturities)

# Get yield at any maturity
yield_at_7y = curve(7.0)
```

### Calculate Bond Returns

```python
from yield_curves.helpers import monthly_bond_return
import numpy as np

# Monthly yields (in percentage)
yields = np.array([2.5, 2.6, 2.4, 2.7])

# Calculate returns
returns = [monthly_bond_return(yields, i) for i in range(1, len(yields))]
```

### Run Swap Analysis

```python
from yield_curves.swap_analysis import SwapAnalysis

# Initialize
swap = SwapAnalysis(
    origination_date='2020-01-31',
    maturity_date='2025-12-31'
)

# Set data (yields_df, fx_returns)
swap.set_yields(wpu_yields, us_yields)
swap.set_currency_returns(fx_returns)
swap.calculate_returns()

# Get performance metrics
summary = swap.get_performance_summary()
print(summary)
```

---

## Methodology

### Yield Curve Models

**Nelson-Siegel (4 parameters):**
- β₀: Long-term level
- β₁: Short-term component
- β₂: Medium-term component
- τ: Decay factor

**Svensson (6 parameters):**
- Extends Nelson-Siegel for better fit to complex shapes

### Bond Return Calculation

Uses Henckel's formula accounting for:
- Coupon payments (semi-annual)
- Price appreciation/depreciation from yield changes
- 10-year maturity assumption

### Currency Swap Strategy

**Long Position:** Foreign bond + FX exposure
**Short Position:** USD bond (no FX)
**Swap Return:** (1 + bond return) × (1 + FX return) - (1 + USD bond return)

---

## Openscapes Principles

This project follows [Openscapes](https://www.openscapes.org/) best practices:

✅ **Reproducible:** Conda environment ensures consistent dependencies
✅ **Shareable:** Clear documentation and modular code
✅ **Collaborative:** Version controlled with Git
✅ **Data-Driven:** Transparent data processing pipeline
✅ **Well-Documented:** Docstrings, notebooks, and README
✅ **Tested:** Unit tests for core functionality

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/yield_curves tests/

# Run specific test file
pytest tests/test_helpers.py -v
```

### Code Style

Format code with Black:
```bash
black src/yield_curves/
```

Lint with flake8:
```bash
flake8 src/yield_curves/
```

### Adding New Features

1. Write functions in `src/yield_curves/`
2. Add docstrings with examples
3. Create tests in `tests/`
4. Update notebooks to demonstrate usage
5. Run tests before committing

---

## Migration from R

This project was converted from R. Key changes:

| R Package | Python Equivalent |
|-----------|------------------|
| `xts` | `pandas` with DatetimeIndex |
| `YieldCurve` | `nelson-siegel-svensson` |
| `tidyverse` | `pandas` + `numpy` |
| `PerformanceAnalytics` | Custom functions |

### File Mapping

- `helper_fn.R` → `src/yield_curves/helpers.py`
- `fitting_yield_curves.R` → `src/yield_curves/yield_curve_fitting.py`
- `main.R` → `notebooks/01_bond_swap_analysis.ipynb`

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

## License

[Specify your license here]

---

## Contact

**Brett Cooper**
[Your email/contact info]

---

## References

- Nelson, C. R., & Siegel, A. F. (1987). Parsimonious modeling of yield curves. *Journal of Business*, 473-489.
- Svensson, L. E. (1994). Estimating and interpreting forward interest rates: Sweden 1992-1994. *NBER Working Paper*.
- Openscapes: https://www.openscapes.org/

---

## Changelog

### Version 0.1.0 (2025-01-15)
- Initial Python conversion from R
- Added Conda environment
- Created Jupyter notebooks
- Implemented data validation
- Added unit tests
- Structured following Openscapes principles
