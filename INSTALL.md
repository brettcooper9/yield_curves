# Installation Instructions

Quick guide to get this project running on your machine.

## Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git (optional, but recommended)

## Setup Steps

### 1. Extract the ZIP file

Unzip the package to your desired location, e.g.:
```
C:/Users/YourName/projects/yield_curves/
```

### 2. Create the Conda Environment

Open Anaconda Prompt (or your terminal) and navigate to the project directory:

```bash
cd path/to/yield_curves
conda env create -f environment.yml
```

This will install all required Python packages.

### 3. Activate the Environment

```bash
conda activate yield_curves
```

### 4. Verify Installation

Run the test script:
```bash
python test_comparison.py
```

You should see output with bond return calculations.

### 5. Run the Analysis

Option A - Interactive Notebooks (Recommended):
```bash
jupyter lab
```
Then open `notebooks/02_yield_curve_fitting.ipynb`

Option B - Python Scripts:
```bash
python -c "from yield_curves.helpers import round_month_to_maturity; print(round_month_to_maturity('2020-01-01', '2025-01-01'))"
```

## Running Tests

```bash
pytest tests/ -v
```

## Troubleshooting

**Problem:** `conda: command not found`
- Make sure Anaconda/Miniconda is installed
- Restart your terminal after installation

**Problem:** `ModuleNotFoundError: No module named 'yield_curves'`
```bash
pip install -e .
```

**Problem:** Tests fail
- Make sure you activated the environment: `conda activate yield_curves`
- Check that all data files are in `data/raw/`

## Project Structure

```
yield_curves/
├── data/raw/           # Input data files
├── data/processed/     # Output location
├── src/yield_curves/   # Python package
├── notebooks/          # Jupyter notebooks
├── tests/              # Unit tests
├── docs/              # Documentation
└── environment.yml     # Dependencies
```

## Next Steps

See [GETTING_STARTED.md](docs/GETTING_STARTED.md) for a detailed tutorial.
