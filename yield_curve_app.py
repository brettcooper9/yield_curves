"""
Interactive Yield Curve Visualization App

A Streamlit app for visualizing yield curves across countries and time.
Similar to the R Shiny app but in Python.

Run with: streamlit run yield_curve_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, 'src')

# Page config
st.set_page_config(
    page_title="Yield Curves by Country",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Yield Curves by Country")

# Configuration
# Start date: when Russia (RUB) was dropped from WPU basket
APP_START_DATE = pd.Timestamp('2024-08-30')

# Load data
@st.cache_data
def load_yield_data():
    """Load fitted yield curve data with multiple models and observed data markers."""
    data_path = Path('data/processed/fitted_yield_curves.npz')

    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        st.info("Please run: python run_fitting.py")
        st.stop()

    # Load the numpy archive
    data = np.load(data_path, allow_pickle=True)

    # Extract arrays
    # Check if this is old format (ns_yields) or new format (all_model_yields)
    if 'all_model_yields' in data:
        # New multi-model format
        all_model_yields = data['all_model_yields']  # shape: (model, country, date, maturity)
        all_model_rmse = data['all_model_rmse']  # shape: (model, country, date)
        best_model_idx = data['best_model_idx']  # shape: (country, date)
        model_names = data['model_names']
        countries = data['countries']
        dates = pd.to_datetime(data['dates'])
        maturities = data['maturities']
    else:
        # Old single-model format - convert to new format
        st.warning("⚠️ Using old data format. Please run: python run_fitting.py to get multi-model support")
        ns_yields = data['ns_yields']  # shape: (country, date, maturity)
        countries = data['countries']
        dates = pd.to_datetime(data['dates'])
        maturities = data['maturities']

        # Convert to multi-model format with only Nelson-Siegel
        model_names = np.array(['Nelson-Siegel'])
        all_model_yields = ns_yields[np.newaxis, :, :, :]  # Add model dimension
        all_model_rmse = np.full((1, len(countries), len(dates)), np.nan)
        best_model_idx = np.zeros((len(countries), len(dates)), dtype=int)

    # Filter to dates >= APP_START_DATE
    date_mask = dates >= APP_START_DATE
    dates = dates[date_mask]
    all_model_yields = all_model_yields[:, :, date_mask, :]
    all_model_rmse = all_model_rmse[:, :, date_mask]
    best_model_idx = best_model_idx[:, date_mask]

    # Load raw data to identify observed points and get raw yields
    DATA_DIR = Path('data/raw')
    bond_data = pd.read_csv(DATA_DIR / 'bond_dat.csv', header=None)

    raw_maturities = pd.to_numeric(bond_data.iloc[0, 1:], errors='coerce').values
    raw_countries = bond_data.iloc[2, 1:].values
    raw_dates = pd.to_datetime(bond_data.iloc[7:, 0])
    raw_yields_matrix = bond_data.iloc[7:, 1:].apply(pd.to_numeric, errors='coerce').values

    # Return all data needed for model selection
    return {
        'all_model_yields': all_model_yields,
        'all_model_rmse': all_model_rmse,
        'best_model_idx': best_model_idx,
        'model_names': model_names,
        'countries': countries,
        'dates': dates,
        'maturities': maturities,
        'raw_maturities': raw_maturities,
        'raw_countries': raw_countries,
        'raw_dates': raw_dates,
        'raw_yields_matrix': raw_yields_matrix
    }

# Try to load data
try:
    data_dict = load_yield_data()
    all_model_yields = data_dict['all_model_yields']
    all_model_rmse = data_dict['all_model_rmse']
    best_model_idx = data_dict['best_model_idx']
    model_names = data_dict['model_names']
    countries = data_dict['countries']
    dates = data_dict['dates']
    maturities = data_dict['maturities']
    raw_maturities = data_dict['raw_maturities']
    raw_countries = data_dict['raw_countries']
    raw_dates = data_dict['raw_dates']
    raw_yields_matrix = data_dict['raw_yields_matrix']
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("""
    To generate the data file:
    1. Open notebooks/02_yield_curve_fitting.ipynb
    2. Run all cells
    3. This will create data/processed/fitted_yield_curves.npz
    4. Then restart this app
    """)
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")

# Date slider with start and end dates displayed
selected_date_idx = st.sidebar.slider(
    f"Select Date ({dates.min().strftime('%b %Y')} - {dates.max().strftime('%b %Y')}):",
    min_value=0,
    max_value=len(dates) - 1,
    value=0,
    format=""
)
selected_date = dates[selected_date_idx]

# Display selected date
st.sidebar.markdown(f"**Selected:** {selected_date.strftime('%B %Y')}")

# Country selection for highlighting
st.sidebar.subheader("Highlight Countries")

# Check if WPU is in the data
if 'WPU' not in countries:
    st.sidebar.warning("⚠️ WPU curve not found in data. Make sure to run all cells in the fitting notebook.")

# Default: select WPU and top 4 currencies by weight (US, EU, GB, JP)
default_countries = ['WPU', 'US', 'EU', 'GB', 'JP'] if 'WPU' in countries else ['US', 'EU', 'GB', 'JP']
highlighted_countries = st.sidebar.multiselect(
    "Select countries to highlight:",
    options=sorted(countries),
    default=default_countries
)

# Maturity range selector (below country selection)
st.sidebar.subheader("Maturity Range")
max_maturity = st.sidebar.selectbox(
    "Maximum maturity to display:",
    options=[1, 3, 5, 10, 30, 50],
    index=3  # Default to 10 years
)

# Model selection
st.sidebar.subheader("Model Selection")
st.sidebar.info(
    "📊 By default, the best-fitting model (lowest RMSE) is automatically selected for each currency.\n\n"
    f"**Available models:** {', '.join(model_names)}"
)

# Checkbox to enable manual model selection
manual_model_selection = st.sidebar.checkbox(
    "Manually select models per currency",
    value=False,
    help="Override automatic model selection and choose models for each currency"
)

# Create dict to store selected model for each country
selected_models = {}

if manual_model_selection:
    st.sidebar.markdown("**Select model for each currency:**")
    # Show dropdown for each non-WPU country
    for country in sorted([c for c in countries if c != 'WPU']):
        # Get best model for this country at selected date
        c_idx = np.where(countries == country)[0][0]
        best_idx = best_model_idx[c_idx, selected_date_idx]
        default_model = model_names[best_idx]

        selected_model = st.sidebar.selectbox(
            f"{country}:",
            options=list(model_names),
            index=int(best_idx),
            key=f"model_{country}",
            help=f"Best model (lowest RMSE): {default_model}"
        )
        selected_models[country] = np.where(model_names == selected_model)[0][0]

    # WPU uses weighted average of selected models
    st.sidebar.caption("💡 WPU uses weighted average of each currency's selected model")
else:
    # Use best model for each country
    for country in countries:
        if country != 'WPU':
            c_idx = np.where(countries == country)[0][0]
            selected_models[country] = best_model_idx[c_idx, selected_date_idx]

# Build dataframe with selected models
def build_dataframe_with_model_selection(selected_date_idx, selected_models, max_maturity):
    """Build dataframe using selected models for each country."""
    rows = []

    for c_idx, country in enumerate(countries):
        if country == 'WPU':
            # Calculate WPU as weighted average of selected country models
            # Load WPU weights
            try:
                DATA_DIR = Path('data/raw')
                wpu_weights = pd.read_excel(DATA_DIR / 'wpu_weights.xlsx')
                wpu_weights['Date'] = pd.to_datetime(wpu_weights['Column1'])
                wpu_weights = wpu_weights.set_index('Date')

                selected_date = dates[selected_date_idx]
                weight_date_list = wpu_weights.index[wpu_weights.index <= selected_date]

                if len(weight_date_list) > 0:
                    weight_date = weight_date_list[-1]
                    weights_row = wpu_weights.loc[weight_date]

                    country_map = {
                        'AU': 'AUD', 'BR': 'BRL', 'CA': 'CAD', 'CH': 'CHF',
                        'CN': 'CNY', 'EU': 'EUR', 'GB': 'GBP', 'IN': 'INR',
                        'JP': 'JPY', 'MX': 'MXN', 'US': 'USD'
                    }

                    # Calculate WPU for each maturity
                    for m_idx, maturity in enumerate(maturities):
                        if maturity > max_maturity:
                            continue

                        weighted_sum = 0
                        total_weight = 0

                        for other_country in countries:
                            if other_country == 'WPU' or other_country not in country_map:
                                continue

                            weight = weights_row[country_map[other_country]]
                            other_c_idx = np.where(countries == other_country)[0][0]
                            model_idx = selected_models[other_country]

                            country_yield = all_model_yields[model_idx, other_c_idx, selected_date_idx, m_idx]

                            if not np.isnan(country_yield) and weight > 0:
                                weighted_sum += country_yield * weight
                                total_weight += weight

                        wpu_yield = weighted_sum / total_weight if total_weight > 0 else np.nan

                        rows.append({
                            'country': 'WPU',
                            'date': dates[selected_date_idx],
                            'maturity': maturity,
                            'yield': wpu_yield,
                            'raw_yield': np.nan,
                            'is_observed': False,
                            'model': 'Composite'
                        })
            except:
                # Fallback: WPU unavailable
                pass
        else:
            # Regular country - use selected model
            model_idx = selected_models[country]

            # Get raw data for this country and date
            raw_date_mask = raw_dates == dates[selected_date_idx]
            if raw_date_mask.sum() > 0:
                raw_date_idx = np.where(raw_dates == dates[selected_date_idx])[0][0] - raw_dates.index[0]
                country_mask = raw_countries == country
                obs_maturities = raw_maturities[country_mask]
                obs_yields = raw_yields_matrix[raw_date_idx, country_mask]

                raw_yield_dict = {}
                for mat, yld in zip(obs_maturities, obs_yields):
                    if not np.isnan(yld):
                        raw_yield_dict[mat] = yld
            else:
                raw_yield_dict = {}

            for m_idx, maturity in enumerate(maturities):
                if maturity > max_maturity:
                    continue

                is_observed = maturity in raw_yield_dict
                fitted_yield = all_model_yields[model_idx, c_idx, selected_date_idx, m_idx]

                rows.append({
                    'country': country,
                    'date': dates[selected_date_idx],
                    'maturity': maturity,
                    'yield': fitted_yield,
                    'raw_yield': raw_yield_dict.get(maturity, np.nan),
                    'is_observed': is_observed,
                    'model': model_names[model_idx]
                })

    return pd.DataFrame(rows)

# Build dataframe with selected models
df_filtered = build_dataframe_with_model_selection(selected_date_idx, selected_models, max_maturity)
df_filtered['is_highlighted'] = df_filtered['country'].isin(highlighted_countries)

# Define consistent color mapping for countries
# Professional color palette for financial data
plotly_colors = [
    '#2E5090',  # Deep Blue - Australia
    '#E85D75',  # Coral Red - Brazil
    '#4A90A4',  # Teal - Canada
    '#D64545',  # Red - Switzerland
    '#E8B55D',  # Gold - China
    '#4169A1',  # Royal Blue - Eurozone
    '#8B4789',  # Purple - UK
    '#E87D3E',  # Orange - India
    '#5C9D73',  # Green - Japan
    '#C9596B',  # Rose - Mexico
    '#2F5F8F',  # Navy - United States
    '#6B7B8C'   # Gray - WPU
]

# Map country codes to full names and colors
country_to_name = {
    'AU': 'Australia', 'BR': 'Brazil', 'CA': 'Canada', 'CH': 'Switzerland',
    'CN': 'China', 'EU': 'Eurozone', 'GB': 'UK', 'IN': 'India',
    'JP': 'Japan', 'MX': 'Mexico', 'US': 'United States', 'WPU': 'WPU'
}

sorted_countries = sorted(df_filtered['country'].unique())
country_colors = {country: plotly_colors[i % len(plotly_colors)]
                  for i, country in enumerate(sorted_countries)}

# Create the plot
fig = go.Figure()

# Plot each country
for country in sorted_countries:
    df_country = df_filtered[df_filtered['country'] == country].sort_values('maturity')
    is_highlighted = country in highlighted_countries

    # Style based on highlighting
    if is_highlighted:
        line_width = 3
        opacity = 1.0
        showlegend = True
        line_color = country_colors[country]  # Use consistent color
    else:
        line_width = 2  # Bolder gray lines
        opacity = 0.6  # More visible
        showlegend = False
        line_color = '#888888'  # Darker gray

    # Add line trace (for all points)
    fig.add_trace(go.Scatter(
        x=df_country['maturity'],
        y=df_country['yield'],
        mode='lines',
        name=country,
        line=dict(
            width=line_width,
            color=line_color
        ),
        opacity=opacity,
        showlegend=showlegend,
        hoverinfo='skip'
    ))

    # Add markers for observed points (filled circles) - use raw yields
    df_observed = df_country[df_country['is_observed'] == True]
    if not df_observed.empty:
        fig.add_trace(go.Scatter(
            x=df_observed['maturity'],
            y=df_observed['raw_yield'],  # Use raw observed yield, not fitted
            mode='markers',
            name=f"{country} (observed)",
            marker=dict(
                size=8 if is_highlighted else 4,
                color=line_color,
                line=dict(width=0)
            ),
            opacity=opacity,
            showlegend=False,
            hovertemplate=(
                f"<b>{country}</b><br>"
                "Maturity: %{x:.1f} yr<br>"
                "Yield: %{y:.2f}%<br>"
                "<i>Observed Market Data</i><br>"
                "<extra></extra>"
            )
        ))

    # Add markers for interpolated points (hollow circles)
    df_interpolated = df_country[df_country['is_observed'] == False]
    if not df_interpolated.empty:
        # Get model name for this country
        model_name = df_interpolated['model'].iloc[0] if len(df_interpolated) > 0 else 'Unknown'

        fig.add_trace(go.Scatter(
            x=df_interpolated['maturity'],
            y=df_interpolated['yield'],
            mode='markers',
            name=f"{country} (fitted)",
            marker=dict(
                size=8 if is_highlighted else 4,
                color='white',
                line=dict(width=2 if is_highlighted else 1, color=line_color)
            ),
            opacity=opacity,
            showlegend=False,
            hovertemplate=(
                f"<b>{country}</b><br>"
                "Maturity: %{x:.1f} yr<br>"
                "Yield: %{y:.2f}%<br>"
                f"<i>Fitted ({model_name})</i><br>"
                "<extra></extra>"
            )
        ))

# Update layout
fig.update_layout(
    title=dict(
        text=f"Yield Curves - {selected_date.strftime('%B %Y')}",
        font=dict(size=24)
    ),
    xaxis=dict(
        title="Maturity (Years)",
        gridcolor='lightgray',
        showgrid=True
    ),
    yaxis=dict(
        title="Yield (%)",
        gridcolor='lightgray',
        showgrid=True
    ),
    hovermode='closest',
    height=600,
    plot_bgcolor='white',
    legend=dict(
        orientation='h',
        yanchor='top',
        y=-0.15,
        xanchor='center',
        x=0.5
    ),
    annotations=[
        dict(
            text="● Observed&nbsp;&nbsp;○ Fitted",
            xref="paper",
            yref="paper",
            x=1.0,
            y=-0.15,
            xanchor='left',
            yanchor='top',
            showarrow=False,
            font=dict(size=10, color='#666666')
        )
    ]
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# WPU Composition Chart (if WPU is in data)
if 'WPU' in countries:
    st.subheader("WPU Composition")

    # Load WPU weights
    import pandas as pd
    from pathlib import Path

    try:
        DATA_DIR = Path('data/raw')
        wpu_weights = pd.read_excel(DATA_DIR / 'wpu_weights.xlsx')
        wpu_weights['Date'] = pd.to_datetime(wpu_weights['Column1'])
        wpu_weights = wpu_weights.set_index('Date')

        # Get weights for selected date (use most recent available)
        weight_date_list = wpu_weights.index[wpu_weights.index <= selected_date]
        if len(weight_date_list) > 0:
            weight_date = weight_date_list[-1]
            weights_row = wpu_weights.loc[weight_date]

            # Currency to country code mapping (for color lookup)
            currency_to_code = {
                'AUD': 'AU', 'BRL': 'BR', 'CAD': 'CA', 'CHF': 'CH',
                'CNY': 'CN', 'EUR': 'EU', 'GBP': 'GB', 'INR': 'IN',
                'JPY': 'JP', 'MXN': 'MX', 'USD': 'US'
            }

            # Extract weights
            weight_data = []
            for currency, country_code in currency_to_code.items():
                if currency in weights_row.index:
                    weight = weights_row[currency]
                    if not pd.isna(weight) and weight > 0:
                        country_name = country_to_name.get(country_code, country_code)
                        country_color = country_colors.get(country_code, '#cccccc')
                        weight_data.append({
                            'Country': country_name,
                            'CountryCode': country_code,
                            'Weight': weight,
                            'Color': country_color
                        })

            # Sort by weight descending for stacking order (largest first)
            weight_df = pd.DataFrame(weight_data).sort_values('Weight', ascending=False)

            # Create stacked horizontal bar chart
            fig_weights = go.Figure()

            for idx, row in weight_df.iterrows():
                # Show country code (e.g., "US") for weights > 5%, otherwise just percentage
                if row['Weight'] > 5:
                    text_label = f"{row['CountryCode']}<br>{row['Weight']:.1f}%"
                else:
                    text_label = f"{row['Weight']:.1f}%"

                fig_weights.add_trace(go.Bar(
                    x=[row['Weight']],
                    y=['WPU'],
                    orientation='h',
                    name=row['Country'],
                    marker=dict(color=row['Color']),  # Use matching color from yield curve chart
                    text=text_label,
                    textposition='inside',
                    textfont=dict(color='white', size=11),
                    hovertemplate=f"<b>{row['Country']}</b><br>Weight: {row['Weight']:.2f}%<extra></extra>",
                    showlegend=False  # Remove from legend
                ))

            fig_weights.update_layout(
                title=dict(
                    text=f"WPU Weight Composition - {selected_date.strftime('%B %Y')}<br><sub>Weight date: {weight_date.strftime('%Y-%m-%d')}</sub>",
                    font=dict(size=18)
                ),
                xaxis=dict(
                    title="Weight (%)",
                    gridcolor='lightgray',
                    showgrid=True,
                    range=[0, 105]
                ),
                yaxis=dict(
                    title="",
                    showticklabels=False
                ),
                barmode='stack',
                height=150,
                plot_bgcolor='white',
                margin=dict(l=50, r=50, t=80, b=30),
                showlegend=False  # No legend needed
            )

            st.plotly_chart(fig_weights, use_container_width=True)

            # Show total
            total_weight = weight_df['Weight'].sum()
            st.caption(f"Total weight: {total_weight:.2f}% (excludes Russia which is not in yield curve data)")

    except Exception as e:
        st.warning(f"Could not load WPU weights: {e}")

# Statistics table
st.subheader("Current Yield Levels")
if manual_model_selection:
    st.caption("Values shown with * are observed market data. Others are fitted using the selected model for each currency.")
else:
    st.caption("Values shown with * are observed market data. Others are fitted using the best RMSE model for each currency (auto-selected).")

# Create summary table with observed markers
if not df_filtered[df_filtered['is_highlighted']].empty:
    # Pivot for fitted yields
    fitted_df = df_filtered[df_filtered['is_highlighted']].pivot(
        index='country',
        columns='maturity',
        values='yield'
    ).round(2)

    # Pivot for raw yields
    raw_df = df_filtered[df_filtered['is_highlighted']].pivot(
        index='country',
        columns='maturity',
        values='raw_yield'
    ).round(2)

    # Pivot for observed flag
    observed_df = df_filtered[df_filtered['is_highlighted']].pivot(
        index='country',
        columns='maturity',
        values='is_observed'
    )

    # Format with asterisks for observed values, using raw yields when observed
    formatted_df = fitted_df.copy()
    for country in formatted_df.index:
        for maturity in formatted_df.columns:
            if observed_df.loc[country, maturity]:
                # Use raw yield for observed data
                formatted_df.loc[country, maturity] = f"{raw_df.loc[country, maturity]:.2f}*"
            else:
                # Use fitted yield for interpolated data
                formatted_df.loc[country, maturity] = f"{fitted_df.loc[country, maturity]:.2f}"

    st.dataframe(formatted_df, use_container_width=True)
else:
    st.info("Select countries to highlight to see yield levels")

# Additional info
with st.expander("ℹ️ About this app"):
    st.markdown("""
    This app visualizes government bond yield curves across multiple countries over time.

    **Features:**
    - **Time slider**: Navigate through historical yield curves
    - **Maturity range selector**: Choose max maturity (1, 3, 5, 10, 30, or 50 years)
    - **Country highlighting**: Select countries to emphasize
    - **Interactive tooltips**: Hover over curves for detailed values
    - **Observed vs Fitted**: Filled markers show observed data, hollow markers show fitted values
    - **Multi-model fitting**: 4 models available (Nelson-Siegel, Svensson, Cubic Spline, B-Spline)
    - **Auto model selection**: Best-fitting model (lowest RMSE) automatically chosen for each currency
    - **Manual model override**: Optional per-currency model selection
    - **WPU composition**: View weighted currency basket breakdown

    **Data:**
    - Yield curves fitted using 4 different models
    - Models: Nelson-Siegel, Svensson, Cubic Spline, B-Spline
    - Best model auto-selected based on RMSE (Root Mean Square Error)
    - Data source: Multi-country government bond yields
    - Maturities: 3 months to 50 years
    - Start date: August 2024 (when Russia dropped from WPU basket)

    **Model Selection:**
    - **Default (Auto)**: Best RMSE model is automatically selected for each currency-date
    - **Manual Override**: Check "Manually select models per currency" to choose specific models
    - **WPU Calculation**: Uses weighted average of each currency's selected model
    - Hover over fitted points to see which model was used

    **Chart Legend:**
    - **Filled circles**: Observed market yields
    - **Hollow circles**: Fitted yields (model-interpolated)
    - Hover over any point to see if it's observed or fitted, and which model was used

    **Usage:**
    1. Use the date slider to select a time period
    2. Choose maximum maturity to display (default: 10 years)
    3. Select countries to highlight in the sidebar
    4. (Optional) Enable manual model selection to override defaults
    5. Hover over points to see exact values, data source, and model used
    6. View WPU composition chart below the main plot
    7. Check the table - values with * are observed data
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption(f"Data from {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
st.sidebar.caption(f"Total dates: {len(dates)}")
st.sidebar.caption(f"Countries: {len(countries)}")
