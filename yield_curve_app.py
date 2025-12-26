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
    """Load fitted yield curve data and observed data markers."""
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

    # Filter to dates >= APP_START_DATE
    date_mask = dates >= APP_START_DATE
    dates = dates[date_mask]
    ns_yields = ns_yields[:, date_mask, :]

    # Load raw data to identify observed points and get raw yields
    DATA_DIR = Path('data/raw')
    bond_data = pd.read_csv(DATA_DIR / 'bond_dat.csv', header=None)

    raw_maturities = pd.to_numeric(bond_data.iloc[0, 1:], errors='coerce').values
    raw_countries = bond_data.iloc[2, 1:].values
    raw_dates = pd.to_datetime(bond_data.iloc[7:, 0])
    raw_yields_matrix = bond_data.iloc[7:, 1:].apply(pd.to_numeric, errors='coerce').values

    # Convert to long-form DataFrame with observed flag and raw yields
    rows = []
    for c_idx, country in enumerate(countries):
        if country == 'WPU':  # WPU is always interpolated (weighted composite)
            for d_idx, date in enumerate(dates):
                for m_idx, maturity in enumerate(maturities):
                    rows.append({
                        'country': country,
                        'date': date,
                        'maturity': maturity,
                        'yield': ns_yields[c_idx, d_idx, m_idx],
                        'raw_yield': np.nan,  # WPU has no raw yield
                        'is_observed': False
                    })
        else:
            for d_idx, date in enumerate(dates):
                # Find corresponding row in raw data
                raw_date_mask = raw_dates == date
                if raw_date_mask.sum() == 0:
                    continue
                raw_date_idx = np.where(raw_dates == date)[0][0] - raw_dates.index[0]

                # Get country mask in raw data
                country_mask = raw_countries == country
                obs_maturities = raw_maturities[country_mask]
                obs_yields = raw_yields_matrix[raw_date_idx, country_mask]

                # Create dictionary of maturity -> raw yield
                raw_yield_dict = {}
                for mat, yld in zip(obs_maturities, obs_yields):
                    if not np.isnan(yld):
                        raw_yield_dict[mat] = yld

                for m_idx, maturity in enumerate(maturities):
                    is_observed = maturity in raw_yield_dict
                    rows.append({
                        'country': country,
                        'date': date,
                        'maturity': maturity,
                        'yield': ns_yields[c_idx, d_idx, m_idx],  # Fitted yield
                        'raw_yield': raw_yield_dict.get(maturity, np.nan),  # Raw observed yield
                        'is_observed': is_observed
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

# Filter data for selected date and maturity range
df_filtered = yield_df[
    (yield_df['date'] == selected_date) &
    (yield_df['maturity'] <= max_maturity)
].copy()
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
                "<i>Fitted (Nelson-Siegel)</i><br>"
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
st.caption("Values shown with * are observed market data. Others are fitted using Nelson-Siegel model.")

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
    - **WPU composition**: View weighted currency basket breakdown

    **Data:**
    - Yield curves fitted using Nelson-Siegel model
    - Data source: Multi-country government bond yields
    - Maturities: 3 months to 50 years
    - Start date: August 2024 (when Russia dropped from WPU basket)

    **Chart Legend:**
    - **Filled circles**: Observed market yields
    - **Hollow circles**: Fitted yields (Nelson-Siegel interpolation)
    - Hover over any point to see if it's observed or fitted

    **Usage:**
    1. Use the date slider to select a time period
    2. Choose maximum maturity to display (default: 10 years)
    3. Select countries to highlight in the sidebar
    4. Hover over points to see exact values and data source
    5. View WPU composition chart below the main plot
    6. Check the table - values with * are observed data
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption(f"Data from {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
st.sidebar.caption(f"Total dates: {len(dates)}")
st.sidebar.caption(f"Countries: {len(countries)}")
