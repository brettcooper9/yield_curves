# R vs Python Comparison Guide

This guide helps you verify that the Python conversion produces the same results as the original R code.

## Quick Comparison

### Step 1: Run R Test Script

In RStudio or R console:

```r
# Make sure you're in the project directory
setwd("c:/Users/BrettCooper/MPG001 Dropbox/Brett Cooper/Data/brett/yield_curves")

# Run the comparison script
source("compare_outputs.R")
```

This will print test outputs for:
- Bond return calculations
- Time to maturity calculations
- Date series generation

### Step 2: Run Python Test Notebook

```bash
# Activate the conda environment
conda activate yield_curves

# Start Jupyter
jupyter lab

# Open: notebooks/03_r_vs_python_comparison.ipynb
# Run all cells
```

### Step 3: Compare Results

The outputs should match to ~10 decimal places. Any differences beyond that are due to floating-point precision differences between R and Python.

## Full Analysis Comparison

### Run Original R Analysis

```r
# Run the full analysis
source("main.R")

# This creates:
# - Performance charts
# - Annual returns table
# - xts_res object with all calculations
```

Save the results:
```r
# Save key outputs
write.csv(as.data.frame(xts_res), "r_outputs_main.csv", row.names=TRUE)
print(table.AnnualizedReturns(xts_res[,c("Swap", "WPU_Bond", "US_10Y", "WPU")]))
```

### Run Python Analysis

```bash
conda activate yield_curves
jupyter lab
```

Open `notebooks/01_bond_swap_analysis.ipynb` and run all cells.

Compare:
1. Monthly returns (should match closely)
2. Cumulative returns (should match closely)
3. Annualized statistics (should match closely)

## What Should Match Exactly

- ✅ Date sequences
- ✅ Time to maturity calculations
- ✅ Month-end dates

## What Should Match Closely (within 1e-6)

- ✅ Bond return calculations
- ✅ FX returns
- ✅ Cumulative returns
- ✅ Performance statistics

## What Might Differ Slightly

- ⚠️ Yield curve fitting (different optimization algorithms)
- ⚠️ Svensson parameters (multiple local optima possible)
- ⚠️ Plots/visualizations (different plotting libraries)

## Troubleshooting

### R Script Errors

**Problem:** `Error in source("helper_fn.R")`

**Solution:** Make sure you're in the correct directory with `getwd()` and `setwd()`

### Python Import Errors

**Problem:** `ModuleNotFoundError: No module named 'yield_curves'`

**Solution:**
```bash
cd /path/to/yield_curves
pip install -e .
```

### Data File Not Found

Both R and Python need access to the same data files. Make sure:
- `wpu_exchange_rates.csv` is in the project root for R
- `wpu_exchange_rates.csv` is copied to `data/raw/` for Python

## Side-by-Side Workflow

You can keep both versions:

### Current State
```
yield_curves/
├── *.R              # Original R files (not in git)
├── src/             # New Python package (in git)
├── notebooks/       # New Jupyter notebooks (in git)
├── data/raw/        # Shared data (not in git, copied)
└── ...
```

### Recommendation

If you want to keep using R while transitioning:
1. Keep the R files for now
2. Use Python for new analyses
3. Compare outputs occasionally
4. Once confident, you can archive the R files

Or run them truly side-by-side:
1. R for production (what you know works)
2. Python for development (new features, better testing)
3. Gradually migrate workflows as you validate
