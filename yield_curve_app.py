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

# Load data
@st.cache_data
def load_yield_data():
    """Load fitted yield curve data."""
    data_path = Path('data/processed/fitted_yield_curves.npz')

    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        st.info("Please run the yield curve fitting notebook first: notebooks/02_yield_curve_fitting.ipynb")
        st.stop()

    # Load the numpy archive
    data = np.load(data_path, allow_pickle=True)

    # Extract arrays
    ns_yields = data['ns_yields']  # shape: (countries, dates, maturities)
    countries = data['countries']
    dates = pd.to_datetime(data['dates'])
    maturities = data['maturities']

    # Convert to long-form DataFrame
    rows = []
    for c_idx, country in enumerate(countries):
        for d_idx, date in enumerate(dates):
            for m_idx, maturity in enumerate(maturities):
                rows.append({
                    'country': country,
                    'date': date,
                    'maturity': maturity,
                    'yield': ns_yields[c_idx, d_idx, m_idx]
                })

    df = pd.DataFrame(rows)
    return df, countries, dates, maturities

# Try to load data
try:
    yield_df, countries, dates, maturities = load_yield_data()
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

# Date slider
selected_date_idx = st.sidebar.slider(
    "Select Date:",
    min_value=0,
    max_value=len(dates) - 1,
    value=0,
    format=""
)
selected_date = dates[selected_date_idx]

# Display selected date
st.sidebar.markdown(f"**Selected:** {selected_date.strftime('%B %Y')}")

# Animation checkbox
animate = st.sidebar.checkbox("Auto-play animation", value=False)

# Country selection for highlighting
st.sidebar.subheader("Highlight Countries")

# Check if WPU is in the data
if 'WPU' not in countries:
    st.sidebar.warning("⚠️ WPU curve not found in data. Make sure to run all cells in the fitting notebook.")

default_countries = ['WPU', 'US'] if 'WPU' in countries and 'US' in countries else countries[:2].tolist()
highlighted_countries = st.sidebar.multiselect(
    "Select countries to highlight:",
    options=sorted(countries),
    default=default_countries
)

# Filter data for selected date
df_filtered = yield_df[yield_df['date'] == selected_date].copy()
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
        mode = 'lines+markers'
        showlegend = True
        line_color = country_colors[country]  # Use consistent color
    else:
        line_width = 2  # Bolder gray lines
        opacity = 0.6  # More visible
        mode = 'lines'
        showlegend = False
        line_color = '#888888'  # Darker gray

    fig.add_trace(go.Scatter(
        x=df_country['maturity'],
        y=df_country['yield'],
        mode=mode,
        name=country,
        line=dict(
            width=line_width,
            color=line_color
        ),
        opacity=opacity,
        marker=dict(size=6 if is_highlighted else 3),
        hovertemplate=(
            f"<b>{country}</b><br>"
            "Maturity: %{x:.1f} yr<br>"
            "Yield: %{y:.2f}%<br>"
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
    )
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

# Animation logic
if animate:
    import time
    placeholder = st.empty()

    for i in range(selected_date_idx, len(dates)):
        # Update via rerun (Streamlit's way of handling animation)
        time.sleep(1.5)
        st.rerun()

# Statistics table
st.subheader("Current Yield Levels")

# Create summary table
summary_df = df_filtered[df_filtered['is_highlighted']].pivot(
    index='country',
    columns='maturity',
    values='yield'
).round(2)

if not summary_df.empty:
    st.dataframe(summary_df, use_container_width=True)
else:
    st.info("Select countries to highlight to see yield levels")

# Additional info
with st.expander("ℹ️ About this app"):
    st.markdown("""
    This app visualizes government bond yield curves across multiple countries over time.

    **Features:**
    - **Time slider**: Navigate through historical yield curves
    - **Country highlighting**: Select countries to emphasize
    - **Interactive tooltips**: Hover over curves for detailed values
    - **Animation**: Auto-play to see yield curve evolution

    **Data:**
    - Yield curves fitted using Nelson-Siegel model
    - Data source: Multi-country government bond yields
    - Maturities: 3 months to 30 years

    **Usage:**
    1. Use the slider to select a date
    2. Check countries to highlight in the sidebar
    3. Enable animation to see changes over time
    4. Hover over lines to see exact values
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption(f"Data from {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
st.sidebar.caption(f"Total dates: {len(dates)}")
st.sidebar.caption(f"Countries: {len(countries)}")
