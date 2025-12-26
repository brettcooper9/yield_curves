# Running the Yield Curve App

The yield curve visualization app lets you explore government bond yield curves across countries and time interactively.

## Prerequisites

1. **Conda environment activated:**
   ```bash
   conda activate yield_curves
   ```

2. **Data generated:** You need to run the yield curve fitting notebook first to generate the data file.

## Step 1: Generate the Data

If you haven't already, run the fitting notebook:

```bash
jupyter lab
```

Then:
1. Open `notebooks/02_yield_curve_fitting.ipynb`
2. Run all cells (this will take a few minutes)
3. This creates `data/processed/fitted_yield_curves.npz`
4. Close Jupyter when done

## Step 2: Install Streamlit (if needed)

If you get an error about streamlit not being installed:

```bash
conda install -c conda-forge streamlit
```

Or update your environment:

```bash
conda env update -f environment.yml --prune
```

## Step 3: Run the App

From the project root directory:

```bash
streamlit run yield_curve_app.py
```

The app will automatically open in your web browser at `http://localhost:8501`

## Using the App

### Main Features

1. **Date Slider** (left sidebar)
   - Drag to select different dates
   - Shows yield curves for that specific month

2. **Country Highlighting**
   - Select which countries to emphasize
   - Highlighted countries show in color with thicker lines
   - Non-highlighted countries appear as thin gray lines

3. **Auto-play Animation** (checkbox in sidebar)
   - Enable to automatically cycle through dates
   - Watch how yield curves evolve over time

4. **Interactive Chart**
   - Hover over lines to see exact values
   - Zoom and pan using plotly controls
   - Double-click to reset view

### Example Use Cases

**Compare US vs WPU yields:**
1. Select "US" and "WPU" in the highlight section
2. Use the slider to navigate through time
3. Observe the yield differentials

**Analyze yield curve evolution:**
1. Highlight a single country
2. Enable auto-play animation
3. Watch how the curve shape changes (inversion, steepening, etc.)

**View all countries at once:**
1. Highlight all countries (or none)
2. Compare relative positions across maturities

## Troubleshooting

### "Data file not found"

**Problem:** The app can't find `data/processed/fitted_yield_curves.npz`

**Solution:** Run the fitting notebook first:
```bash
jupyter lab
# Open notebooks/02_yield_curve_fitting.ipynb
# Run all cells
```

### "ModuleNotFoundError: No module named 'streamlit'"

**Solution:**
```bash
conda install -c conda-forge streamlit
```

### App runs but shows error

**Check:**
1. You're in the project root directory
2. The conda environment is activated
3. The data file exists: `ls data/processed/fitted_yield_curves.npz`

### Animation is slow

**Normal behavior:** The animation pauses 1.5 seconds between frames to match the R Shiny app timing. This is intentional for better viewing.

## Stopping the App

Press `Ctrl+C` in the terminal where the app is running.

## Comparison with R Shiny App

This Python/Streamlit app provides the same functionality as `yield_curve_app.R`:

| Feature | R (Shiny) | Python (Streamlit) |
|---------|-----------|-------------------|
| Time slider | ✅ | ✅ |
| Country highlighting | ✅ | ✅ |
| Interactive plots | ✅ (plotly) | ✅ (plotly) |
| Animation | ✅ | ✅ |
| Hover tooltips | ✅ | ✅ |
| Summary table | ❌ | ✅ (bonus) |

## Next Steps

- Experiment with different country combinations
- Watch the animation to see major economic events
- Export charts using the plotly toolbar (camera icon)
- Compare with the R version if you have it running
