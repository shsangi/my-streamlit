# app.py - Main Streamlit Application
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Global City Population Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, beautiful UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Gradient background for headers */
    .gradient-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card styling for metrics */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Filter section styling */
    .filter-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid #e9ecef;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Header with gradient background
st.markdown("""
<div class="gradient-header">
    <h1 style="margin:0; font-size:2.5rem;">🌍 Global City Population Dashboard</h1>
    <p style="margin:0.5rem 0 0 0; opacity:0.9;">Interactive visualization of population data across countries and cities</p>
</div>
""", unsafe_allow_html=True)

# Google Sheets CSV URL
GSHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQY4rE12Yqty9vWRQjteO0Zs9nvCFBuzfI30iqZW8wdkjcVc8aqmsTNcc_QGHYgTdiofjSjopQ25_ZK/pub?gid=1256584885&single=true&output=csv"

# Function to load data from Google Sheets
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_google_sheets_data():
    """Load data from published Google Sheets CSV"""
    try:
        df = pd.read_csv(GSHEET_CSV_URL)
        return df
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {str(e)}")
        return None

# Function to create sample data as fallback
def create_sample_data():
    """Create sample data if Google Sheets loading fails"""
    np.random.seed(42)
    countries = ['United States', 'China', 'India', 'Brazil', 'Indonesia', 'Pakistan', 'Nigeria', 'Bangladesh', 'Russia', 'Mexico']
    cities_per_country = {
        'United States': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
        'China': ['Shanghai', 'Beijing', 'Guangzhou', 'Shenzhen', 'Chengdu'],
        'India': ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Chennai'],
        'Brazil': ['São Paulo', 'Rio de Janeiro', 'Brasília', 'Salvador', 'Fortaleza'],
        'Indonesia': ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Semarang'],
        'Pakistan': ['Karachi', 'Lahore', 'Faisalabad', 'Rawalpindi', 'Multan'],
        'Nigeria': ['Lagos', 'Kano', 'Ibadan', 'Abuja', 'Port Harcourt'],
        'Bangladesh': ['Dhaka', 'Chittagong', 'Khulna', 'Rajshahi', 'Sylhet'],
        'Russia': ['Moscow', 'Saint Petersburg', 'Novosibirsk', 'Yekaterinburg', 'Kazan'],
        'Mexico': ['Mexico City', 'Guadalajara', 'Monterrey', 'Puebla', 'Tijuana']
    }
    
    data = []
    years = range(1990, 2024)
    
    for country in countries:
        cities = cities_per_country.get(country, ['Unknown'])
        for city in cities:
            base_pop = np.random.randint(500000, 15000000)
            for year in years:
                growth = np.random.normal(0.02, 0.01)  # 2% average growth with some variation
                population = int(base_pop * (1 + growth) ** (year - 1990))
                # Add some random lat/lng for each city
                lat = np.random.uniform(-40, 60)
                lng = np.random.uniform(-120, 150)
                data.append({
                    'Country or Area': country,
                    'Year': year,
                    'City': city,
                    'Value': population,
                    'lat': lat,
                    'lng': lng
                })
    
    return pd.DataFrame(data)

# Sidebar for data loading
with st.sidebar:
    st.markdown("### 📁 Data Source")
    
    # Option to choose data source
    data_source = st.radio(
        "Choose data source",
        ["🌐 Google Sheets (Live Data)", "📤 Upload CSV", "🎲 Sample Data"],
        index=0,
        help="Select where to load the population data from"
    )
    
    df = None
    load_button = st.button("🔄 Load Data", type="primary", use_container_width=True)
    
    if load_button:
        with st.spinner("Loading data..."):
            if data_source == "🌐 Google Sheets (Live Data)":
                df = load_google_sheets_data()
                if df is not None:
                    st.success(f"✅ Successfully loaded {len(df):,} records from Google Sheets")
                else:
                    st.warning("Falling back to sample data...")
                    df = create_sample_data()
                    st.success(f"✅ Loaded {len(df):,} sample records")
            
            elif data_source == "📤 Upload CSV":
                uploaded_file = st.file_uploader(
                    "Upload your population data (CSV format)",
                    type=['csv'],
                    help="Upload a CSV file with columns: Country or Area, Year, City, Value, lat, lng"
                )
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.success(f"✅ Successfully loaded {len(df):,} records")
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
            
            else:  # Sample Data
                df = create_sample_data()
                st.success(f"✅ Loaded {len(df):,} sample records")
            
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
    
    # Display data info if loaded
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("### 📊 Data Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Cities", f"{df['City'].nunique():,}")
        
        # Year range
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            year_min = int(df['Year'].min())
            year_max = int(df['Year'].max())
            st.metric("Year Range", f"{year_min} - {year_max}")
        
        # Data preview
        with st.expander("🔍 Data Preview"):
            st.dataframe(df.head(10), use_container_width=True)

# Main content area - only show if data is loaded
if st.session_state.data_loaded and st.session_state.df is not None:
    df = st.session_state.df.copy()
    
    # Data preprocessing
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
    df = df.dropna(subset=['Value', 'Year', 'lat', 'lng'])
    
    # Get unique values for filters
    countries = sorted(df['Country or Area'].unique())
    cities = sorted(df['City'].unique())
    years = sorted(df['Year'].unique(), reverse=True)
    
    # Filter section
    with st.container():
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        
        # Create three columns for filters
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            selected_countries = st.multiselect(
                "🌍 Select Countries",
                options=countries,
                default=countries[:3] if len(countries) > 3 else countries,
                help="Choose one or more countries to analyze"
            )
        
        with col2:
            selected_cities = st.multiselect(
                "🏙️ Select Cities",
                options=cities,
                help="Choose specific cities (leave empty for all)"
            )
        
        with col3:
            if years:
                year_range = st.slider(
                    "📅 Year Range",
                    min_value=int(min(years)),
                    max_value=int(max(years)),
                    value=(int(min(years)), int(max(years))),
                    help="Select time period"
                )
            else:
                year_range = (1990, 2023)
        
        with col4:
            min_population = st.number_input(
                "👥 Min Population",
                min_value=0,
                value=100000,
                step=100000,
                help="Filter cities by minimum population"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data based on selections
    filtered_df = df[
        (df['Country or Area'].isin(selected_countries)) &
        (df['Year'].between(year_range[0], year_range[1])) &
        (df['Value'] >= min_population)
    ]
    
    if selected_cities:
        filtered_df = filtered_df[filtered_df['City'].isin(selected_cities)]
    
    if filtered_df.empty:
        st.warning("⚠️ No data matches the selected filters. Please adjust your filters.")
    else:
        # Get latest data for maps
        latest_data = filtered_df.sort_values('Year').groupby(
            ['Country or Area', 'City', 'lat', 'lng']
        ).last().reset_index()
        
        # Metrics row
        st.markdown("### 📈 Key Statistics")
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
        
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color:#667eea; margin:0;">🏙️</h3>
                <h2 style="margin:0.5rem 0;">{filtered_df['City'].nunique():,}</h2>
                <p style="color:#666; margin:0;">Cities</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color:#667eea; margin:0;">🌍</h3>
                <h2 style="margin:0.5rem 0;">{filtered_df['Country or Area'].nunique():,}</h2>
                <p style="color:#666; margin:0;">Countries</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            total_pop = filtered_df.groupby('Year')['Value'].sum().max()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color:#667eea; margin:0;">👥</h3>
                <h2 style="margin:0.5rem 0;">{total_pop:,.0f}</h2>
                <p style="color:#666; margin:0;">Max Total Population</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col4:
            avg_pop = filtered_df['Value'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color:#667eea; margin:0;">📊</h3>
                <h2 style="margin:0.5rem 0;">{avg_pop:,.0f}</h2>
                <p style="color:#666; margin:0;">Avg City Population</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col5:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color:#667eea; margin:0;">📅</h3>
                <h2 style="margin:0.5rem 0;">{year_range[0]}-{year_range[1]}</h2>
                <p style="color:#666; margin:0;">Years Range</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Interactive Maps", "📊 Data Explorer", "📈 Trends", "ℹ️ About"])
        
        with tab1:
            st.markdown("### 🗺️ Population Distribution Map")
            
            # Map type selector
            map_type = st.radio(
                "Map View",
                ["📍 Current Population", "⏰ Timeline Animation"],
                horizontal=True,
                help="Choose between static map or animated timeline"
            )
            
            if map_type == "📍 Current Population":
                # Static map with latest data
                if not latest_data.empty:
                    fig_map = px.scatter_mapbox(
                        latest_data,
                        lat='lat',
                        lon='lng',
                        size='Value',
                        color='Value',
                        hover_name='City',
                        hover_data={
                            'Country or Area': True,
                            'Year': True,
                            'Value': ':,.0f',
                            'lat': False,
                            'lng': False
                        },
                        color_continuous_scale='Viridis',
                        size_max=50,
                        zoom=1,
                        title=f'<b>City Populations ({year_range[1]})</b>'
                    )
                    
                    fig_map.update_layout(
                        mapbox_style='carto-positron',
                        height=600,
                        title_font_size=16,
                        margin={"r":0, "t":40, "l":0, "b":0},
                        coloraxis_colorbar=dict(
                            title="Population",
                            tickformat=',.0f',
                            thickness=15,
                            len=0.5
                        )
                    )
                    
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.warning("No data available for the selected filters")
            
            else:
                # Animated timeline map
                if not filtered_df.empty:
                    # Get top cities for animation
                    top_cities = filtered_df.groupby('City')['Value'].max().nlargest(50).index
                    anim_data = filtered_df[filtered_df['City'].isin(top_cities)].copy()
                    
                    if not anim_data.empty:
                        fig_anim = px.scatter_mapbox(
                            anim_data,
                            lat='lat',
                            lon='lng',
                            size='Value',
                            color='Value',
                            animation_frame='Year',
                            animation_group='City',
                            hover_name='City',
                            hover_data={
                                'Country or Area': True,
                                'Value': ':,.0f'
                            },
                            color_continuous_scale='Plasma',
                            size_max=50,
                            zoom=1,
                            title='<b>Population Changes Over Time</b>'
                        )
                        
                        fig_anim.update_layout(
                            mapbox_style='carto-positron',
                            height=650,
                            title_font_size=16,
                            margin={"r":0, "t":40, "l":0, "b":0},
                            coloraxis_colorbar=dict(
                                title="Population",
                                tickformat=',.0f',
                                thickness=15,
                                len=0.5
                            ),
                            updatemenus=[dict(
                                type="buttons",
                                buttons=[dict(
                                    label="▶️ Play",
                                    method="animate",
                                    args=[None, {"frame": {"duration": 500, "redraw": True}}]
                                )]
                            )]
                        )
                        
                        st.plotly_chart(fig_anim, use_container_width=True)
                    else:
                        st.warning("Insufficient data for animation")
                else:
                    st.warning("No data available for the selected filters")
        
        with tab2:
            st.markdown("### 📊 Data Explorer")
            
            # Data view options
            view_type = st.radio(
                "View Type",
                ["📋 Table View", "📊 Chart View"],
                horizontal=True
            )
            
            if view_type == "📋 Table View":
                # Interactive data table
                table_data = filtered_df.copy()
                table_data['Population'] = table_data['Value'].apply(lambda x: f"{x:,.0f}")
                table_data['Coordinates'] = table_data['lat'].round(4).astype(str) + ', ' + table_data['lng'].round(4).astype(str)
                
                display_cols = ['Country or Area', 'City', 'Year', 'Population', 'Coordinates']
                
                # Add search/filter
                search = st.text_input("🔍 Search in table", placeholder="Type to filter...")
                
                if search:
                    mask = table_data[display_cols].astype(str).apply(
                        lambda x: x.str.contains(search, case=False)
                    ).any(axis=1)
                    table_data = table_data[mask]
                
                st.dataframe(
                    table_data[display_cols],
                    use_container_width=True,
                    height=500,
                    column_config={
                        "Population": st.column_config.TextColumn("Population", width="medium"),
                        "Coordinates": st.column_config.TextColumn("Coordinates", width="medium")
                    }
                )
                
                # Download button
                csv = table_data[display_cols].to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"population_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            else:
                # Chart view with multiple chart types
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot"]
                )
                
                if chart_type == "Bar Chart":
                    # Top cities bar chart
                    top_n = st.slider("Number of cities to show", 5, 30, 15)
                    chart_data = filtered_df.sort_values('Year').groupby(
                        ['Country or Area', 'City']
                    ).last().reset_index().nlargest(top_n, 'Value')
                    
                    fig = px.bar(
                        chart_data,
                        x='Value',
                        y='City',
                        color='Country or Area',
                        orientation='h',
                        title=f'<b>Top {top_n} Cities by Population</b>',
                        labels={'Value': 'Population', 'City': ''},
                        hover_data={'Year': True}
                    )
                    
                    fig.update_layout(
                        height=600,
                        xaxis_tickformat=',.0f',
                        showlegend=True,
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Line Chart":
                    # Time series for selected cities
                    if selected_cities:
                        line_data = filtered_df[filtered_df['City'].isin(selected_cities)]
                    else:
                        top_cities = filtered_df.groupby('City')['Value'].max().nlargest(5).index
                        line_data = filtered_df[filtered_df['City'].isin(top_cities)]
                    
                    if not line_data.empty:
                        fig = px.line(
                            line_data,
                            x='Year',
                            y='Value',
                            color='City',
                            line_group='City',
                            title='<b>Population Trends Over Time</b>',
                            labels={'Value': 'Population', 'Year': ''},
                            hover_data={'Country or Area': True}
                        )
                        
                        fig.update_traces(line_width=3)
                        fig.update_layout(
                            height=500,
                            yaxis_tickformat=',.0f',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No data available for line chart")
                
                elif chart_type == "Scatter Plot":
                    fig = px.scatter(
                        filtered_df,
                        x='Year',
                        y='Value',
                        color='Country or Area',
                        hover_name='City',
                        size='Value',
                        title='<b>Population Distribution by Year</b>',
                        labels={'Value': 'Population', 'Year': ''}
                    )
                    
                    fig.update_layout(
                        height=500,
                        yaxis_tickformat=',.0f'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:  # Box Plot
                    fig = px.box(
                        filtered_df,
                        x='Country or Area',
                        y='Value',
                        color='Country or Area',
                        title='<b>Population Distribution by Country</b>',
                        labels={'Value': 'Population', 'Country or Area': ''},
                        points='all'
                    )
                    
                    fig.update_layout(
                        height=500,
                        yaxis_tickformat=',.0f',
                        xaxis_tickangle=-45,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 📈 Trends Analysis")
            
            # Two columns for trend charts
            trend_col1, trend_col2 = st.columns(2)
            
            with trend_col1:
                # Population growth rate
                st.subheader("📊 Population Growth Rate")
                
                growth_data = []
                for (country, city), group in filtered_df.groupby(['Country or Area', 'City']):
                    if len(group) >= 2:
                        group = group.sort_values('Year')
                        first_val = group['Value'].iloc[0]
                        last_val = group['Value'].iloc[-1]
                        first_year = group['Year'].iloc[0]
                        last_year = group['Year'].iloc[-1]
                        
                        if first_val > 0 and (last_year - first_year) >= 5:
                            total_growth = ((last_val - first_val) / first_val) * 100
                            annual_growth = total_growth / (last_year - first_year)
                            
                            growth_data.append({
                                'City': city,
                                'Country': country,
                                'Annual Growth %': annual_growth,
                                'Total Growth %': total_growth,
                                'Period': f"{int(first_year)}-{int(last_year)}"
                            })
                
                if growth_data:
                    growth_df = pd.DataFrame(growth_data)
                    top_growth = growth_df.nlargest(10, 'Annual Growth %')
                    
                    fig_growth = px.bar(
                        top_growth,
                        x='Annual Growth %',
                        y='City',
                        color='Annual Growth %',
                        orientation='h',
                        title='<b>Fastest Growing Cities</b>',
                        labels={'Annual Growth %': 'Annual Growth Rate (%)', 'City': ''},
                        hover_data=['Country', 'Period'],
                        color_continuous_scale='Greens',
                        text=top_growth['Annual Growth %'].round(1).astype(str) + '%'
                    )
                    
                    fig_growth.update_traces(textposition='outside')
                    fig_growth.update_layout(height=500)
                    st.plotly_chart(fig_growth, use_container_width=True)
                else:
                    st.info("Insufficient data for growth rate calculation")
            
            with trend_col2:
                # Year-over-year comparison
                st.subheader("📅 Year-over-Year Comparison")
                
                if len(years) >= 2:
                    compare_years = st.multiselect(
                        "Select years to compare",
                        options=years[:10],
                        default=years[:2] if len(years) >= 2 else years[:1],
                        max_selections=3
                    )
                    
                    if len(compare_years) >= 2:
                        comp_data = filtered_df[filtered_df['Year'].isin(compare_years)]
                        comp_pivot = comp_data.pivot_table(
                            index='City',
                            columns='Year',
                            values='Value',
                            aggfunc='mean'
                        ).reset_index()
                        
                        # Calculate change
                        year1, year2 = compare_years[0], compare_years[1]
                        comp_pivot['Change'] = comp_pivot[year2] - comp_pivot[year1]
                        comp_pivot['Change %'] = (comp_pivot[year2] / comp_pivot[year1] - 1) * 100
                        
                        top_changes = comp_pivot.nlargest(10, 'Change %')
                        
                        if not top_changes.empty:
                            fig_comp = px.bar(
                                top_changes,
                                x='Change %',
                                y='City',
                                color='Change %',
                                orientation='h',
                                title=f'<b>Population Change ({int(year1)} → {int(year2)})</b>',
                                labels={'Change %': 'Change (%)', 'City': ''},
                                color_continuous_scale='RdBu',
                                text=top_changes['Change %'].round(1).astype(str) + '%'
                            )
                            
                            fig_comp.update_traces(textposition='outside')
                            fig_comp.update_layout(height=500)
                            st.plotly_chart(fig_comp, use_container_width=True)
                        else:
                            st.info("No significant changes found")
                    else:
                        st.info("Select at least 2 years to compare")
                else:
                    st.info("Not enough years for comparison")
        
        with tab4:
            st.markdown("### ℹ️ About This Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h3 style="color:#667eea;">🎯 Features</h3>
                    <ul style="list-style-type: none; padding: 0;">
                        <li style="margin: 1rem 0;">✓ Interactive maps with zoom and hover</li>
                        <li style="margin: 1rem 0;">✓ Real-time data filtering</li>
                        <li style="margin: 1rem 0;">✓ Multiple chart types</li>
                        <li style="margin: 1rem 0;">✓ Population trend analysis</li>
                        <li style="margin: 1rem 0;">✓ Data export functionality</li>
                        <li style="margin: 1rem 0;">✓ Responsive design</li>
                        <li style="margin: 1rem 0;">✓ Live data from Google Sheets</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h3 style="color:#667eea;">📊 Data Description</h3>
                    <p><strong>Total Records:</strong> {len(df):,}</p>
                    <p><strong>Countries:</strong> {df['Country or Area'].nunique():,}</p>
                    <p><strong>Cities:</strong> {df['City'].nunique():,}</p>
                    <p><strong>Year Range:</strong> {int(df['Year'].min())} - {int(df['Year'].max())}</p>
                    <p><strong>Average Population:</strong> {df['Value'].mean():,.0f}</p>
                    <p><strong>Data Source:</strong> Google Sheets (Live)</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="margin-top: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white;">
                <h3 style="margin-top: 0;">🚀 How to Use</h3>
                <ol style="margin-bottom: 0;">
                    <li>Data automatically loads from Google Sheets (click "Load Data" in sidebar if needed)</li>
                    <li>Filter data by country, city, year range, and minimum population</li>
                    <li>Explore different visualizations in the tabs above</li>
                    <li>Download filtered data for further analysis</li>
                    <li>Hover over charts for detailed information</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

else:
    # Welcome screen when no data is loaded
    st.markdown("""
    <div style="text-align: center; padding: 4rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">👋 Welcome to the Global City Population Dashboard</h1>
        <p style="font-size: 1.2rem; margin-bottom: 2rem; opacity: 0.9;">Click "Load Data" in the sidebar to get started with live data from Google Sheets</p>
        <div style="background: rgba(255,255,255,0.2); padding: 2rem; border-radius: 15px; max-width: 600px; margin: 0 auto;">
            <h3>📋 Data Format:</h3>
            <p style="font-family: monospace; margin: 1rem 0;">
                Country or Area, Year, City, Value, lat, lng
            </p>
            <p>The dashboard loads data directly from a published Google Sheets CSV.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
