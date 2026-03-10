# app.py - Simple Streamlit Application
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration - simple and clean
st.set_page_config(
    page_title="Population Data",
    layout="wide"
)

# Simple CSS - just basic spacing
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 0rem;}
    .stMetric {background-color: #f0f2f6; padding: 1rem; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# Google Sheets CSV URL - FIXED DATA SOURCE
GSHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQY4rE12Yqty9vWRQjteO0Zs9nvCFBuzfI30iqZW8wdkjcVc8aqmsTNcc_QGHYgTdiofjSjopQ25_ZK/pub?gid=1256584885&single=true&output=csv"

# Simple title
st.title("🌍 Population Data")

# Load data directly - no sidebar, no options
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(GSHEET_CSV_URL)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load the data
with st.spinner("Loading data..."):
    df = load_data()

if df is not None:
    # Basic data cleaning
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
    df = df.dropna(subset=['Value', 'Year', 'lat', 'lng'])
    
    # Get unique values for filters
    countries = sorted(df['Country or Area'].unique())
    cities = sorted(df['City'].unique())
    years = sorted(df['Year'].unique())
    
    # Simple filters in one row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_country = st.selectbox("Country", ["All"] + countries)
    
    with col2:
        selected_city = st.selectbox("City", ["All"] + cities)
    
    with col3:
        year_range = st.slider(
            "Year",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=(int(min(years)), int(max(years)))
        )
    
    with col4:
        min_pop = st.number_input("Min Population", min_value=0, value=0, step=100000)
    
    # Filter data
    filtered_df = df.copy()
    
    if selected_country != "All":
        filtered_df = filtered_df[filtered_df['Country or Area'] == selected_country]
    
    if selected_city != "All":
        filtered_df = filtered_df[filtered_df['City'] == selected_city]
    
    filtered_df = filtered_df[
        (filtered_df['Year'] >= year_range[0]) &
        (filtered_df['Year'] <= year_range[1]) &
        (filtered_df['Value'] >= min_pop)
    ]
    
    # Simple metrics in one row
    if not filtered_df.empty:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cities", filtered_df['City'].nunique())
        m2.metric("Countries", filtered_df['Country or Area'].nunique())
        m3.metric("Total Population", f"{filtered_df['Value'].sum():,.0f}")
        m4.metric("Avg Population", f"{filtered_df['Value'].mean():,.0f}")
        
        # Two columns for map and table
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.subheader("📍 Map")
            
            # Get latest data for map
            map_data = filtered_df.sort_values('Year').groupby(
                ['Country or Area', 'City', 'lat', 'lng']
            ).last().reset_index()
            
            if not map_data.empty:
                fig = px.scatter_mapbox(
                    map_data,
                    lat='lat',
                    lon='lng',
                    size='Value',
                    color='Value',
                    hover_name='City',
                    hover_data={'Country or Area': True, 'Year': True, 'Value': ':,.0f'},
                    color_continuous_scale='Viridis',
                    size_max=30,
                    zoom=1
                )
                
                fig.update_layout(
                    mapbox_style='carto-positron',
                    height=500,
                    margin={"r":0, "t":0, "l":0, "b":0}
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.subheader("📋 Data")
            
            # Simple table
            display_df = filtered_df[['Country or Area', 'City', 'Year', 'Value']].copy()
            display_df['Value'] = display_df['Value'].apply(lambda x: f"{x:,.0f}")
            display_df.columns = ['Country', 'City', 'Year', 'Population']
            
            st.dataframe(display_df, use_container_width=True, height=500)
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="population_data.csv",
                mime="text/csv"
            )
        
        # Simple line chart at bottom
        st.subheader("📈 Trends")
        
        if selected_city == "All":
            # Show top 5 cities if none selected
            top_cities = filtered_df.groupby('City')['Value'].max().nlargest(5).index
            trend_data = filtered_df[filtered_df['City'].isin(top_cities)]
        else:
            trend_data = filtered_df
        
        if not trend_data.empty:
            fig_line = px.line(
                trend_data,
                x='Year',
                y='Value',
                color='City',
                labels={'Value': 'Population'}
            )
            
            fig_line.update_layout(height=400)
            st.plotly_chart(fig_line, use_container_width=True)
    
    else:
        st.warning("No data matches the selected filters")

else:
    st.error("Failed to load data. Please check the Google Sheets URL.")
