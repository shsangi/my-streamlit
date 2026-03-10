# app.py - Simple Population Dashboard
import streamlit as st
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(page_title="Population Data", layout="wide")

# Google Sheets CSV URL
GSHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQY4rE12Yqty9vWRQjteO0Zs9nvCFBuzfI30iqZW8wdkjcVc8aqmsTNcc_QGHYgTdiofjSjopQ25_ZK/pub?gid=1256584885&single=true&output=csv"

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(GSHEET_CSV_URL)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_data()

if df is not None:
    # Clean data
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Value', 'Year'])
    
    # Get unique values for filters
    countries = sorted(df['Country or Area'].unique())
    cities = sorted(df['City'].unique())
    years = sorted(df['Year'].unique(), reverse=True)
    
    # Filter data based on selections (will be updated after dropdowns)
    filtered_df = df.copy()
    display_df = df.copy()
    
    # Create placeholders for dynamic content
    header_placeholder = st.empty()
    metrics_placeholder = st.empty()
    filters_placeholder = st.empty()
    map_table_placeholder = st.empty()
    
    with filters_placeholder.container():
        # Three filters in one row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_country = st.selectbox("Country", ["All"] + countries, key="country")
        
        with col2:
            # Filter cities based on selected country
            if selected_country != "All":
                city_options = sorted(df[df['Country or Area'] == selected_country]['City'].unique())
            else:
                city_options = cities
            selected_city = st.selectbox("City", ["All"] + city_options, key="city")
        
        with col3:
            selected_year = st.selectbox("Year", ["All"] + years, key="year")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_country != "All":
        filtered_df = filtered_df[filtered_df['Country or Area'] == selected_country]
    
    if selected_city != "All":
        filtered_df = filtered_df[filtered_df['City'] == selected_city]
    
    # Further filter by year for display if specific year selected
    if selected_year != "All":
        display_df = filtered_df[filtered_df['Year'] == selected_year]
        year_text = f"for {int(selected_year)}"
    else:
        display_df = filtered_df
        year_text = "for All Years"
    
    # Score cards
    if not filtered_df.empty:
        with header_placeholder.container():
            # Show population for selected year or latest if "All"
            if selected_year != "All":
                pop_value = display_df['Value'].sum()
            else:
                # Get latest year population
                latest_year = filtered_df['Year'].max()
                pop_value = filtered_df[filtered_df['Year'] == latest_year]['Value'].sum()
            
            # Title with population
            st.title(f"Population Data: {pop_value:,.0f}")
        
        with metrics_placeholder.container():
            col1, col2 = st.columns(2)
            
            with col1:
                if selected_year != "All":
                    label = f"Population ({int(selected_year)})"
                else:
                    latest_year = filtered_df['Year'].max()
                    label = f"Current Population ({int(latest_year)})"
                st.metric(label, f"{pop_value:,.0f}")
            
            with col2:
                city_count = filtered_df['City'].nunique()
                st.metric("Number of Cities", city_count)
        
        with map_table_placeholder.container():
            # Two columns for map and table
            map_col, table_col = st.columns(2)
            
            with map_col:
                st.subheader(f"Map {year_text}")
                if 'lat' in display_df.columns and 'lng' in display_df.columns:
                    if not display_df.empty:
                        fig = px.scatter_mapbox(
                            display_df,
                            lat='lat',
                            lon='lng',
                            size='Value',
                            hover_name='City',
                            hover_data={'Country or Area': True, 'Year': True, 'Value': ':,.0f'},
                            zoom=1,
                            height=500
                        )
                        fig.update_layout(mapbox_style='carto-positron', margin={"r":0,"t":0,"l":0,"b":0})
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No data for selected year")
                else:
                    st.info("Location data not available")
            
            with table_col:
                st.subheader(f"Table {year_text}")
                if not display_df.empty:
                    table_data = display_df[['Country or Area', 'City', 'Year', 'Value']].copy()
                    table_data['Value'] = table_data['Value'].apply(lambda x: f"{x:,.0f}")
                    table_data.columns = ['Country', 'City', 'Year', 'Population']
                    st.dataframe(table_data, use_container_width=True, height=500)
                else:
                    st.info("No data for selected year")
    
    else:
        st.warning("No data for selected filters")

else:
    st.error("Failed to load data")
