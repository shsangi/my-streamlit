import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Population Data", layout="wide")

GSHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQY4rE12Yqty9vWRQjteO0Zs9nvCFBuzfI30iqZW8wdkjcVc8aqmsTNcc_QGHYgTdiofjSjopQ25_ZK/pub?gid=1256584885&single=true&output=csv"

@st.cache_data
def load_data():
    df = pd.read_csv(GSHEET_CSV_URL)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    return df.dropna(subset=['Value', 'Year'])

df = load_data()

if df is not None:
    countries = sorted(df['Country or Area'].unique())
    cities = sorted(df['City'].unique())
    years = sorted(df['Year'].unique(), reverse=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_country = st.selectbox("Country", ["All"] + countries, index=0)
    with col2:
        if selected_country != "All":
            city_options = sorted(df[df['Country or Area'] == selected_country]['City'].unique())
        else:
            city_options = cities
        selected_city = st.selectbox("City", ["All"] + city_options)
    with col3:
        selected_year = st.selectbox("Year", ["All"] + years)
    
    filtered_df = df.copy()
    if selected_country != "All":
        filtered_df = filtered_df[filtered_df['Country or Area'] == selected_country]
    if selected_city != "All":
        filtered_df = filtered_df[filtered_df['City'] == selected_city]
    
    if selected_year != "All":
        display_df = filtered_df[filtered_df['Year'] == selected_year]
        year_text = f"for {int(selected_year)}"
        pop_value = display_df['Value'].sum()
        city_count = display_df['City'].nunique()
    else:
        display_df = filtered_df
        year_text = "for All Years"
        pop_value = filtered_df['Value'].sum()
        city_count = filtered_df['City'].nunique()
    
    st.title(f"World Population: {pop_value:,.0f}, Total Cities: {city_count:,}")
    
    if not display_df.empty:
        map_col, table_col = st.columns(2)
        with map_col:
            st.subheader(f"Map {year_text}")
            fig = px.scatter_mapbox(display_df, lat='lat', lon='lng', size='Value', 
                                   hover_name='City', hover_data={'Country or Area': True, 'Year': True, 'Value': ':,.0f'},
                                   zoom=1, height=500)
            fig.update_layout(mapbox_style='carto-positron', margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
        with table_col:
            st.subheader(f"Table {year_text}")
            table_data = display_df[['Country or Area', 'City', 'Year', 'Value']].copy()
            table_data['Value'] = table_data['Value'].apply(lambda x: f"{x:,.0f}")
            table_data.columns = ['Country', 'City', 'Year', 'Population']
            st.dataframe(table_data, use_container_width=True, height=500)
