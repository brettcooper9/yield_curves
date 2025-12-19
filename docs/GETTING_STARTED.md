# Getting Started with Yield Curves

Welcome! This guide will help you get up and running with the yield curves analysis project.

## Prerequisites

Before you begin, ensure you have:

1. **Anaconda or Miniconda** installed
   - Download from: https://www.anaconda.com/download or https://docs.conda.io/en/latest/miniconda.html

2. **Git** installed (should already be available)
   - Check with: `git --version`

## Step-by-Step Setup

### 1. Set Up the Environment

From the project root directory:

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate yield_curves
```

### 2. Verify Installation

```bash
# Test Python imports
python -c "import numpy, pandas, scipy; print('Core libraries OK')"

# Run the test suite
pytest tests/ -v
```

### 3. Prepare Your Data

Ensure these files are in `data/raw/`:
- `bond_dat.csv` - Bond yield data
- `wpu_exchange_rates.csv` - FX data
- `wpu_weights.xlsx` - Country weights

### 4. Start Jupyter Lab

```bash
jupyter lab
```

This will open Jupyter Lab in your browser.

## Your First Analysis

### Option 1: Run the Notebooks

1. Open `notebooks/02_yield_curve_fitting.ipynb`
2. Run all cells (Cell → Run All)
3. This will fit yield curves and save results to `data/processed/`

4. Open `notebooks/01_bond_swap_analysis.ipynb`
5. Run all cells to analyze swap strategies

### Option 2: Use Python Scripts

Create a new script `my_analysis.py`:

```python
from yield_curves.helpers import round_month_to_maturity
import pandas as pd

# Calculate time to maturity
ttm = round_month_to_maturity('2025-01-15', '2030-01-15')
print(f"Time to maturity: {ttm} years")

# Load FX data
fx_data = pd.read_csv('data/raw/wpu_exchange_rates.csv',
                      parse_dates=['Date'], index_col='Date')
print(fx_data.head())
```

Run it:
```bash
python my_analysis.py
```

## Understanding the Workflow

The typical workflow is:

1. **Load Data** → Bond yields, FX rates, weights
2. **Fit Curves** → Nelson-Siegel or Svensson models
3. **Analyze Swaps** → Calculate returns with FX effects
4. **Evaluate Performance** → Summary statistics and plots

## Common Tasks

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_helpers.py -v

# With coverage report
pytest --cov=src/yield_curves tests/
```

### Code Formatting

```bash
# Format Python code
black src/yield_curves/

# Check style
flake8 src/yield_curves/
```

### Updating Dependencies

If you need to add a package:

1. Edit `environment.yml`
2. Update the environment:
   ```bash
   conda env update -f environment.yml --prune
   ```

## Troubleshooting

### Environment Issues

**Problem:** `conda env create` fails

**Solution:**
```bash
# Remove existing environment
conda env remove -n yield_curves

# Try again
conda env create -f environment.yml
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'yield_curves'`

**Solution:**
```bash
# Make sure you're in the project root
cd /path/to/yield_curves

# Install in development mode
pip install -e .
```

### Data Loading Issues

**Problem:** `FileNotFoundError` when loading data

**Solution:**
- Check that data files are in `data/raw/`
- Verify file names match exactly (case-sensitive)
- Use absolute paths if needed

## Next Steps

Once you're set up:

1. **Explore the notebooks** - They contain detailed examples
2. **Read the docstrings** - Each function has usage examples
3. **Check the tests** - They show how functions work
4. **Modify for your needs** - Add new analyses or data sources

## Getting Help

- Check the main [README.md](../README.md) for detailed documentation
- Look at test files in `tests/` for examples
- Review notebook outputs for expected results

## Openscapes Resources

Learn more about open science practices:
- Openscapes: https://www.openscapes.org/
- Reproducible research guide: https://the-turing-way.netlify.app/

Happy analyzing! 🎉
