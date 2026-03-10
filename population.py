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
    page_title="🌍 Global City Population Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM MODERN THEME & CSS
# =============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container with animated gradient background */
    .main {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        min-height: 100vh;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glass morphism effect for containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Hero header with glass morphism */
    .hero-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 30px;
        padding: 2.5rem;
        margin: 1rem 0 2rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .hero-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .hero-header p {
        font-size: 1.2rem;
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Modern metric cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-card-modern {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card-modern:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
        border-color: rgba(255,255,255,0.5);
    }
    
    .metric-card-modern::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .metric-card-modern:hover::before {
        left: 100%;
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Filter panel with glass morphism */
    .filter-panel {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 0.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.9);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        border-color: rgba(255, 255, 255, 0.5);
    }
    
    /* Selectbox styling */
    .stSelectbox, .stMultiSelect {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Chart container */
    .chart-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    /* Loading animation */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner::after {
        content: '';
        width: 50px;
        height: 50px;
        border: 5px solid rgba(255,255,255,0.2);
        border-top-color: #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Tooltip styling */
    .tooltip-icon {
        display: inline-block;
        width: 20px;
        height: 20px;
        background: rgba(255,255,255,0.2);
        border-radius: 50%;
        text-align: center;
        line-height: 20px;
        font-size: 12px;
        color: white;
        cursor: help;
        margin-left: 5px;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255,255,255,0.6);
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# =============================================================================
# GOOGLE SHEETS CSV URL
# =============================================================================
GSHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQY4rE12Yqty9vWRQjteO0Zs9nvCFBuzfI30iqZW8wdkjcVc8aqmsTNcc_QGHYgTdiofjSjopQ25_ZK/pub?gid=1256584885&single=true&output=csv"

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
@st.cache_data(ttl=3600)
def load_google_sheets_data():
    """Load data from published Google Sheets CSV"""
    try:
        df = pd.read_csv(GSHEET_CSV_URL)
        return df
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {str(e)}")
        return None

def create_sample_data():
    """Create beautiful sample data if Google Sheets loading fails"""
    np.random.seed(42)
    
    # Realistic city data
    world_cities = {
        'North America': {
            'United States': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
            'Canada': ['Toronto', 'Montreal', 'Vancouver', 'Calgary', 'Edmonton', 'Ottawa', 'Quebec City', 'Winnipeg', 'Hamilton', 'Halifax'],
            'Mexico': ['Mexico City', 'Guadalajara', 'Monterrey', 'Puebla', 'Tijuana', 'Leon', 'Ciudad Juárez', 'Zapopan', 'Monterrey', 'Nezahualcoyotl']
        },
        'Europe': {
            'United Kingdom': ['London', 'Birmingham', 'Manchester', 'Glasgow', 'Liverpool', 'Edinburgh', 'Leeds', 'Bristol', 'Sheffield', 'Newcastle'],
            'Germany': ['Berlin', 'Hamburg', 'Munich', 'Cologne', 'Frankfurt', 'Stuttgart', 'Düsseldorf', 'Dortmund', 'Essen', 'Leipzig'],
            'France': ['Paris', 'Marseille', 'Lyon', 'Toulouse', 'Nice', 'Nantes', 'Strasbourg', 'Montpellier', 'Bordeaux', 'Lille'],
            'Italy': ['Rome', 'Milan', 'Naples', 'Turin', 'Palermo', 'Genoa', 'Bologna', 'Florence', 'Bari', 'Catania'],
            'Spain': ['Madrid', 'Barcelona', 'Valencia', 'Seville', 'Zaragoza', 'Malaga', 'Murcia', 'Palma', 'Las Palmas', 'Bilbao']
        },
        'Asia': {
            'China': ['Shanghai', 'Beijing', 'Guangzhou', 'Shenzhen', 'Chengdu', 'Tianjin', 'Wuhan', 'Dongguan', 'Chongqing', 'Nanjing'],
            'India': ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad', 'Ahmedabad', 'Pune', 'Surat', 'Jaipur'],
            'Japan': ['Tokyo', 'Yokohama', 'Osaka', 'Nagoya', 'Sapporo', 'Fukuoka', 'Kobe', 'Kyoto', 'Kawasaki', 'Saitama'],
            'South Korea': ['Seoul', 'Busan', 'Incheon', 'Daegu', 'Daejeon', 'Gwangju', 'Suwon', 'Ulsan', 'Changwon', 'Seongnam'],
            'Indonesia': ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Semarang', 'Makassar', 'Palembang', 'Depok', 'Tangerang', 'Bekasi']
        },
        'South America': {
            'Brazil': ['São Paulo', 'Rio de Janeiro', 'Brasília', 'Salvador', 'Fortaleza', 'Belo Horizonte', 'Manaus', 'Curitiba', 'Recife', 'Porto Alegre'],
            'Argentina': ['Buenos Aires', 'Córdoba', 'Rosario', 'Mendoza', 'La Plata', 'San Miguel de Tucumán', 'Mar del Plata', 'Salta', 'Santa Fe', 'San Juan'],
            'Colombia': ['Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'Cartagena', 'Cúcuta', 'Soledad', 'Ibagué', 'Bucaramanga', 'Villavicencio'],
            'Peru': ['Lima', 'Arequipa', 'Callao', 'Trujillo', 'Chiclayo', 'Piura', 'Iquitos', 'Cusco', 'Huancayo', 'Tacna']
        },
        'Africa': {
            'Nigeria': ['Lagos', 'Kano', 'Ibadan', 'Abuja', 'Port Harcourt', 'Benin City', 'Maiduguri', 'Zaria', 'Aba', 'Jos'],
            'Egypt': ['Cairo', 'Alexandria', 'Giza', 'Shubra El Kheima', 'Port Said', 'Suez', 'Luxor', 'Aswan', 'Ismailia', 'Tanta'],
            'South Africa': ['Johannesburg', 'Cape Town', 'Durban', 'Pretoria', 'Port Elizabeth', 'Bloemfontein', 'Pietermaritzburg', 'Welkom', 'East London', 'Kimberley'],
            'Kenya': ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Thika', 'Malindi', 'Kitale', 'Garissa', 'Kakamega']
        },
        'Oceania': {
            'Australia': ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Gold Coast', 'Canberra', 'Newcastle', 'Wollongong', 'Hobart'],
            'New Zealand': ['Auckland', 'Wellington', 'Christchurch', 'Hamilton', 'Tauranga', 'Dunedin', 'Palmerston North', 'Napier', 'Hastings', 'Nelson']
        }
    }
    
    data = []
    years = range(1990, 2024)
    
    # Realistic lat/lng for major cities
    city_coords = {
        'New York': (40.7128, -74.0060), 'Los Angeles': (34.0522, -118.2437), 'Chicago': (41.8781, -87.6298),
        'London': (51.5074, -0.1278), 'Paris': (48.8566, 2.3522), 'Tokyo': (35.6762, 139.6503),
        'Shanghai': (31.2304, 121.4737), 'Beijing': (39.9042, 116.4074), 'Mumbai': (19.0760, 72.8777),
        'Sydney': (-33.8688, 151.2093), 'Rio de Janeiro': (-22.9068, -43.1729), 'Cairo': (30.0444, 31.2357),
        'Moscow': (55.7558, 37.6173), 'Istanbul': (41.0082, 28.9784), 'Dubai': (25.2048, 55.2708),
        'Singapore': (1.3521, 103.8198), 'Hong Kong': (22.3193, 114.1694), 'Bangkok': (13.7563, 100.5018)
    }
    
    for continent, countries in world_cities.items():
        for country, cities in countries.items():
            for city in cities:
                # Get coordinates or generate random ones
                if city in city_coords:
                    lat, lng = city_coords[city]
                else:
                    lat = np.random.uniform(-40, 60)
                    lng = np.random.uniform(-120, 150)
                
                # Generate realistic population with growth trend
                base_pop = np.random.randint(500000, 15000000)
                
                for year in years:
                    # Add some realistic growth patterns
                    if year < 2000:
                        growth = np.random.normal(0.015, 0.005)  # Slower growth pre-2000
                    elif year < 2010:
                        growth = np.random.normal(0.02, 0.007)   # Moderate growth
                    else:
                        growth = np.random.normal(0.025, 0.01)   # Faster growth recent years
                    
                    # Add some random variation for realism
                    random_factor = np.random.normal(1, 0.02)
                    population = int(base_pop * (1 + growth) ** (year - 1990) * random_factor)
                    
                    data.append({
                        'Country or Area': country,
                        'Year': year,
                        'City': city,
                        'Value': population,
                        'lat': lat,
                        'lng': lng
                    })
    
    return pd.DataFrame(data)

# =============================================================================
# SIDEBAR - MODERN DESIGN
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h1 style="font-family: 'Space Grotesk', sans-serif; font-size: 2rem; background: linear-gradient(135deg, #fff 0%, #e0e0e0 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🌍 Data Explorer</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Data source selection with icons
    data_source = st.radio(
        "📡 Data Source",
        ["🌐 Google Sheets (Live)", "📤 Upload CSV", "🎲 Sample Data"],
        index=0,
        help="Choose where to load the population data from"
    )
    
    # Load button with animation
    load_button = st.button("🚀 LOAD DATA", use_container_width=True)
    
    if load_button:
        with st.spinner('✨ Loading beautiful data...'):
            if data_source == "🌐 Google Sheets (Live)":
                df = load_google_sheets_data()
                if df is not None:
                    st.success("✅ Data loaded successfully from Google Sheets!")
                else:
                    st.warning("⚠️ Using sample data as fallback")
                    df = create_sample_data()
                    st.success("✅ Sample data loaded!")
            
            elif data_source == "📤 Upload CSV":
                uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.success("✅ File uploaded successfully!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        df = None
                else:
                    df = None
                    st.info("📁 Please upload a CSV file")
            
            else:  # Sample Data
                df = create_sample_data()
                st.success("✅ Sample data loaded!")
            
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.balloons()
    
    # Display data info if loaded
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        
        st.markdown("---")
        st.markdown("### 📊 Data Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Cities", f"{df['City'].nunique():,}")
        
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            year_min = int(df['Year'].min())
            year_max = int(df['Year'].max())
            st.metric("Year Range", f"{year_min} - {year_max}")
        
        # Data quality indicator
        completeness = (df['Value'].notna().sum() / len(df)) * 100
        st.progress(completeness / 100, text=f"Data Quality: {completeness:.1f}%")
        
        with st.expander("🔍 Data Preview", expanded=False):
            st.dataframe(df.head(5), use_container_width=True)

# =============================================================================
# MAIN CONTENT - HERO HEADER
# =============================================================================
if st.session_state.data_loaded and st.session_state.df is not None:
    df = st.session_state.df.copy()
    
    # Hero header
    st.markdown("""
    <div class="hero-header">
        <h1>🌍 Global City Population Dashboard</h1>
        <p>Explore population dynamics across continents • Interactive visualization • Real-time analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # =========================================================================
    # FILTER SECTION - MODERN GLASS DESIGN
    # =========================================================================
    with st.container():
        st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
        
        st.markdown("### 🔍 Smart Filters")
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            selected_countries = st.multiselect(
                "🌍 Countries",
                options=countries,
                default=countries[:3] if len(countries) > 3 else countries,
                help="Select one or more countries to analyze"
            )
        
        with col2:
            selected_cities = st.multiselect(
                "🏙️ Cities",
                options=cities,
                help="Choose specific cities (leave empty for all)"
            )
        
        with col3:
            if years:
                year_range = st.slider(
                    "📅 Time Period",
                    min_value=int(min(years)),
                    max_value=int(max(years)),
                    value=(int(min(years)), int(max(years))),
                    help="Drag to select year range"
                )
            else:
                year_range = (1990, 2023)
        
        with col4:
            min_population = st.number_input(
                "👥 Min Population",
                min_value=0,
                value=100000,
                step=100000,
                help="Filter by minimum population"
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
        st.warning("⚠️ No data matches your filters. Try adjusting the criteria.")
    else:
        # Get latest data for maps
        latest_data = filtered_df.sort_values('Year').groupby(
            ['Country or Area', 'City', 'lat', 'lng']
        ).last().reset_index()
        
        # =========================================================================
        # METRICS SECTION - MODERN CARDS
        # =========================================================================
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card-modern">
                <div class="metric-icon">🏙️</div>
                <div class="metric-value">{filtered_df['City'].nunique():,}</div>
                <div class="metric-label">Cities Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card-modern">
                <div class="metric-icon">🌍</div>
                <div class="metric-value">{filtered_df['Country or Area'].nunique():,}</div>
                <div class="metric-label">Countries</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_pop = filtered_df['Value'].sum()
            st.markdown(f"""
            <div class="metric-card-modern">
                <div class="metric-icon">👥</div>
                <div class="metric-value">{total_pop:,.0f}</div>
                <div class="metric-label">Total Population</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_pop = filtered_df['Value'].mean()
            st.markdown(f"""
            <div class="metric-card-modern">
                <div class="metric-icon">📊</div>
                <div class="metric-value">{avg_pop:,.0f}</div>
                <div class="metric-label">Average City Pop</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            growth_rate = ((filtered_df['Value'].max() - filtered_df['Value'].min()) / filtered_df['Value'].min()) * 100
            st.markdown(f"""
            <div class="metric-card-modern">
                <div class="metric-icon">📈</div>
                <div class="metric-value">{growth_rate:.1f}%</div>
                <div class="metric-label">Growth Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # =========================================================================
        # TABS FOR VISUALIZATIONS
        # =========================================================================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ Interactive Maps", 
            "📊 Data Explorer", 
            "📈 Trends Analysis",
            "🏆 Rankings",
            "ℹ️ Insights"
        ])
        
        with tab1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Map type selector with modern design
            map_type = st.radio(
                "Select Visualization",
                ["📍 Current Population", "⏰ Timeline Animation", "🌡️ Heat Map"],
                horizontal=True,
                help="Choose the type of map visualization"
            )
            
            if map_type == "📍 Current Population":
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
                            'Value': ':,.0f'
                        },
                        color_continuous_scale='Viridis',
                        size_max=50,
                        zoom=1,
                        title='Global Population Distribution'
                    )
                    
                    fig_map.update_layout(
                        mapbox_style='carto-positron',
                        height=600,
                        margin={"r":0, "t":30, "l":0, "b":0},
                        coloraxis_colorbar=dict(
                            title="Population",
                            tickformat=',.0f',
                            thickness=15,
                            len=0.5
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    
                    st.plotly_chart(fig_map, use_container_width=True)
                    
                    # Stats below map
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        largest_city = latest_data.loc[latest_data['Value'].idxmax()]
                        st.info(f"🏆 Largest: {largest_city['City']} ({largest_city['Value']:,.0f})")
                    with col2:
                        st.info(f"📍 Total Cities: {len(latest_data):,}")
                    with col3:
                        st.info(f"📊 Avg Population: {latest_data['Value'].mean():,.0f}")
            
            elif map_type == "⏰ Timeline Animation":
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
                        title='Population Evolution Over Time'
                    )
                    
                    fig_anim.update_layout(
                        mapbox_style='carto-positron',
                        height=650,
                        margin={"r":0, "t":30, "l":0, "b":0},
                        coloraxis_colorbar=dict(
                            title="Population",
                            tickformat=',.0f',
                            thickness=15,
                            len=0.5
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
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
            
            else:  # Heat Map
                # Create a density heatmap
                fig_heat = px.density_mapbox(
                    latest_data,
                    lat='lat',
                    lon='lng',
                    z='Value',
                    radius=30,
                    hover_name='City',
                    hover_data={'Value': ':,.0f'},
                    color_continuous_scale='Hot',
                    zoom=1,
                    title='Population Density Heatmap'
                )
                
                fig_heat.update_layout(
                    mapbox_style='carto-positron',
                    height=600,
                    margin={"r":0, "t":30, "l":0, "b":0},
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig_heat, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            view_type = st.radio(
                "View Mode",
                ["📋 Interactive Table", "📊 Charts Gallery", "📈 Comparison View"],
                horizontal=True
            )
            
            if view_type == "📋 Interactive Table":
                # Search and filter
                search = st.text_input("🔍 Search cities or countries", placeholder="Type to filter...")
                
                table_data = filtered_df.copy()
                table_data['Population'] = table_data['Value'].apply(lambda x: f"{x:,.0f}")
                table_data['Coordinates'] = table_data['lat'].round(4).astype(str) + ', ' + table_data['lng'].round(4).astype(str)
                
                display_cols = ['Country or Area', 'City', 'Year', 'Population', 'Coordinates']
                
                if search:
                    mask = table_data[display_cols].astype(str).apply(
                        lambda x: x.str.contains(search, case=False)
                    ).any(axis=1)
                    table_data = table_data[mask]
                
                # Style the dataframe
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
                    label="📥 Download Data",
                    data=csv,
                    file_name=f"population_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            elif view_type == "📊 Charts Gallery":
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Top cities bar chart
                    top_n = st.slider("Number of cities", 5, 30, 15, key='top_n')
                    chart_data = filtered_df.sort_values('Year').groupby(
                        ['Country or Area', 'City']
                    ).last().reset_index().nlargest(top_n, 'Value')
                    
                    fig_bar = px.bar(
                        chart_data,
                        x='Value',
                        y='City',
                        color='Country or Area',
                        orientation='h',
                        title=f'Top {top_n} Cities',
                        labels={'Value': 'Population', 'City': ''},
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    
                    fig_bar.update_layout(
                        height=400,
                        xaxis_tickformat=',.0f',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with chart_col2:
                    # Distribution by country
                    country_totals = filtered_df.groupby('Country or Area')['Value'].sum().reset_index()
                    
                    fig_pie = px.pie(
                        country_totals,
                        values='Value',
                        names='Country or Area',
                        title='Population by Country',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    
                    fig_pie.update_layout(
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Time series
                if selected_cities:
                    line_data = filtered_df[filtered_df['City'].isin(selected_cities)]
                else:
                    top_cities = filtered_df.groupby('City')['Value'].max().nlargest(5).index
                    line_data = filtered_df[filtered_df['City'].isin(top_cities)]
                
                if not line_data.empty:
                    fig_line = px.line(
                        line_data,
                        x='Year',
                        y='Value',
                        color='City',
                        title='Population Trends',
                        labels={'Value': 'Population'},
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    
                    fig_line.update_traces(line_width=3)
                    fig_line.update_layout(
                        height=400,
                        yaxis_tickformat=',.0f',
                        hovermode='x unified',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    
                    st.plotly_chart(fig_line, use_container_width=True)
            
            else:  # Comparison View
                # Year comparison
                if len(years) >= 2:
                    compare_years = st.multiselect(
                        "Select years to compare",
                        options=years[:10],
                        default=years[:2] if len(years) >= 2 else years[:1],
                        max_selections=3
                    )
                    
                    if len(compare_years) >= 2:
                        comp_data = filtered_df[filtered_df['Year'].isin(compare_years)]
                        
                        # Create comparison chart
                        fig_comp = go.Figure()
                        
                        for year in compare_years:
                            year_data = comp_data[comp_data['Year'] == year]
                            year_totals = year_data.groupby('Country or Area')['Value'].sum().reset_index()
                            
                            fig_comp.add_trace(go.Bar(
                                name=str(int(year)),
                                x=year_totals['Country or Area'],
                                y=year_totals['Value'],
                                text=year_totals['Value'].apply(lambda x: f'{x:,.0f}'),
                                textposition='outside'
                            ))
                        
                        fig_comp.update_layout(
                            title='Population Comparison by Year',
                            barmode='group',
                            height=500,
                            yaxis_tickformat=',.0f',
                            xaxis_tickangle=-45,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        
                        st.plotly_chart(fig_comp, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            trend_col1, trend_col2 = st.columns(2)
            
            with trend_col1:
                st.subheader("🚀 Fastest Growing Cities")
                
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
                                'Growth Rate': annual_growth,
                                'Total Growth': total_growth,
                                'Period': f"{int(first_year)}-{int(last_year)}"
                            })
                
                if growth_data:
                    growth_df = pd.DataFrame(growth_data)
                    top_growth = growth_df.nlargest(10, 'Growth Rate')
                    
                    fig_growth = px.bar(
                        top_growth,
                        x='Growth Rate',
                        y='City',
                        color='Growth Rate',
                        orientation='h',
                        labels={'Growth Rate': 'Annual Growth (%)'},
                        color_continuous_scale='Greens',
                        text=top_growth['Growth Rate'].round(1).astype(str) + '%'
                    )
                    
                    fig_growth.update_traces(textposition='outside')
                    fig_growth.update_layout(
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    
                    st.plotly_chart(fig_growth, use_container_width=True)
            
            with trend_col2:
                st.subheader("📊 Population Distribution")
                
                # Box plot by country
                fig_box = px.box(
                    filtered_df,
                    x='Country or Area',
                    y='Value',
                    color='Country or Area',
                    points='outliers',
                    title='Population Distribution'
                )
                
                fig_box.update_layout(
                    height=500,
                    yaxis_tickformat=',.0f',
                    xaxis_tickangle=-45,
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig_box, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            rank_col1, rank_col2 = st.columns(2)
            
            with rank_col1:
                st.subheader("🏆 Top 10 Cities")
                top_cities = filtered_df.sort_values('Year').groupby(
                    ['City', 'Country or Area']
                ).last().reset_index().nlargest(10, 'Value')
                
                fig_top = px.bar(
                    top_cities,
                    x='Value',
                    y='City',
                    color='Country or Area',
                    orientation='h',
                    title='Largest Cities',
                    labels={'Value': 'Population'},
                    text=top_cities['Value'].apply(lambda x: f'{x:,.0f}')
                )
                
                fig_top.update_traces(textposition='outside')
                fig_top.update_layout(
                    height=500,
                    xaxis_tickformat=',.0f',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig_top, use_container_width=True)
            
            with rank_col2:
                st.subheader("🌍 Top 10 Countries")
                country_totals = filtered_df.groupby('Country or Area')['Value'].sum().reset_index()
                top_countries = country_totals.nlargest(10, 'Value')
                
                fig_country = px.pie(
                    top_countries,
                    values='Value',
                    names='Country or Area',
                    title='Population Share',
                    hole=0.4
                )
                
                fig_country.update_layout(
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig_country, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab5:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Key Insights")
                
                # Calculate insights
                total_cities = filtered_df['City'].nunique()
                total_countries = filtered_df['Country or Area'].nunique()
                avg_population = filtered_df['Value'].mean()
                median_population = filtered_df['Value'].median()
                largest_city = filtered_df.loc[filtered_df['Value'].idxmax()]
                smallest_city = filtered_df.loc[filtered_df['Value'].idxmin()]
                
                insights = [
                    f"• **Total Cities:** {total_cities:,}",
                    f"• **Total Countries:** {total_countries:,}",
                    f"• **Average Population:** {avg_population:,.0f}",
                    f"• **Median Population:** {median_population:,.0f}",
                    f"• **Largest City:** {largest_city['City']} ({largest_city['Value']:,.0f})",
                    f"• **Smallest City:** {smallest_city['City']} ({smallest_city['Value']:,.0f})",
                    f"• **Year Range:** {year_range[0]} - {year_range[1]}",
                    f"• **Total Population:** {filtered_df['Value'].sum():,.0f}"
                ]
                
                for insight in insights:
                    st.markdown(insight)
            
            with col2:
                st.subheader("📊 Data Quality")
                
                # Data quality metrics
                completeness = {
                    'Country': filtered_df['Country or Area'].notna().mean() * 100,
                    'City': filtered_df['City'].notna().mean() * 100,
                    'Year': filtered_df['Year'].notna().mean() * 100,
                    'Population': filtered_df['Value'].notna().mean() * 100,
                    'Coordinates': (filtered_df['lat'].notna() & filtered_df['lng'].notna()).mean() * 100
                }
                
                fig_quality = go.Figure(data=[
                    go.Bar(
                        x=list(completeness.keys()),
                        y=list(completeness.values()),
                        marker_color=['#667eea', '#764ba2', '#23a6d5', '#23d5ab', '#ee7752'],
                        text=[f"{v:.1f}%" for v in completeness.values()],
                        textposition='outside'
                    )
                ])
                
                fig_quality.update_layout(
                    title='Data Completeness',
                    yaxis=dict(range=[0, 100]),
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig_quality, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div class="footer">
            <p>✨ Built with Streamlit & Plotly • Data sourced from Google Sheets • Updated in real-time</p>
            <p style="font-size: 0.8rem;">© 2024 Global City Population Dashboard • All visualizations are interactive</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Welcome screen with modern design
    st.markdown("""
    <div style="text-align: center; padding: 4rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%); backdrop-filter: blur(10px); border-radius: 30px; margin: 2rem 0;">
        <h1 style="font-size: 4rem; margin-bottom: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">👋 Welcome!</h1>
        <p style="font-size: 1.5rem; color: white; margin-bottom: 2rem;">Click the button in the sidebar to load beautiful population data</p>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; max-width: 900px; margin: 3rem auto;">
            <div style="background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 20px;">
                <div style="font-size: 3rem;">🗺️</div>
                <h3 style="color: white;">Interactive Maps</h3>
                <p style="color: rgba(255,255,255,0.7);">Explore population distribution on a global map</p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 20px;">
                <div style="font-size: 3rem;">📊</div>
                <h3 style="color: white;">Rich Analytics</h3>
                <p style="color: rgba(255,255,255,0.7);">Multiple chart types and trend analysis</p>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 20px;">
                <div style="font-size: 3rem;">⚡</div>
                <h3 style="color: white;">Real-time Data</h3>
                <p style="color: rgba(255,255,255,0.7);">Live updates from Google Sheets</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
