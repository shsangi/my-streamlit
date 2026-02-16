import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Project Pulse • Live Dashboard",
    page_icon="📊",
    initial_sidebar_state="collapsed"
)

# --- Constants ---
REFRESH_INTERVAL = 5  # seconds
PAKISTAN_TZ = pytz.timezone('Asia/Karachi')
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQFttuVQlH84hCC-brrcJFa6eyrMeyc25Aqm_dLgfpuEBr0WCdc4OTKKZVK2Y6IfOoPdQFbYmSdrSYP/pub?output=xlsx"

# --- Initialize Session State ---
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 'PROJECT_MASTER'

# --- Auto-refresh ---
count = st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="auto_refresh")

# --- Data Loading Function ---
@st.cache_data(ttl=REFRESH_INTERVAL)
def load_data():
    """Load data from Google Sheets"""
    try:
        xl = pd.ExcelFile(DATA_URL)
        sheets = {
            'PROJECT_MASTER': pd.read_excel(xl, 'PROJECT_MASTER'),
            'DAILY_WORK_LOG': pd.read_excel(xl, 'DAILY_WORK_LOG'),
            'EMPLOYEE_COST': pd.read_excel(xl, 'EMPLOYEE_COST'),
            'RESOURCE_LINKS': pd.read_excel(xl, 'RESOURCE_LINKS'),
            'TASK_PLAN': pd.read_excel(xl, 'TASK PLAN + RESPONSIBILITY')
        }
        
        # Clean date columns
        for df in sheets.values():
            for col in df.select_dtypes(include=['object']):
                if 'date' in col.lower() or 'time' in col.lower():
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return sheets, datetime.now(PAKISTAN_TZ)
    except Exception as e:
        st.error(f"⚠️ Data load failed: {str(e)}")
        return {name: pd.DataFrame() for name in ['PROJECT_MASTER', 'DAILY_WORK_LOG', 'EMPLOYEE_COST', 'RESOURCE_LINKS', 'TASK_PLAN']}, None

# --- Load Data ---
data_sheets, last_update = load_data()

# --- Custom CSS for Modern Look ---
st.markdown("""
<style>
    /* Modern gradient header */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Status bar */
    .status-bar {
        background: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    
    .live-badge {
        background: #ff4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .timestamp {
        color: #666;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .pk-badge {
        background: #2c3e50;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Modern tabs */
    .tab-container {
        background: white;
        border-radius: 15px;
        padding: 0.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    
    .tab-button {
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        background: transparent;
        font-weight: 500;
        color: #666;
        transition: all 0.2s;
        cursor: pointer;
        width: 100%;
        text-align: left;
        margin: 2px 0;
    }
    
    .tab-button:hover {
        background: #f8f9fa;
        color: #667eea;
    }
    
    .tab-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header-container">
    <div style="display: flex; align-items: center; gap: 1rem;">
        <h1 style="margin: 0; font-size: 2rem;">📊 Project Pulse</h1>
        <span style="background: rgba(255,255,255,0.2); padding: 0.25rem 1rem; border-radius: 50px; font-size: 0.9rem;">
            Live Dashboard
        </span>
    </div>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Real-time project analytics & tracking</p>
</div>
""", unsafe_allow_html=True)

# --- Status Bar (Single, Clean) ---
if last_update:
    timestamp_str = last_update.strftime("%a, %d %b, %Y, %I:%M:%S %p")
    st.markdown(f"""
    <div class="status-bar">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span class="live-badge">LIVE</span>
            <span class="timestamp">
                <span>🔄 Last updated: {timestamp_str}</span>
                <span class="pk-badge">🇵🇰 PKT</span>
            </span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; color: #666;">
            <span>⏱️ Auto-refresh: {REFRESH_INTERVAL}s</span>
            <span style="color: #22c55e;">●</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Main Layout ---
col1, col2 = st.columns([1, 4])

# --- Left Column: Navigation ---
with col1:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.markdown("##### 📋 Navigation")
    
    tabs = {
        'PROJECT_MASTER': ('📁', 'Projects'),
        'DAILY_WORK_LOG': ('📝', 'Work Log'),
        'EMPLOYEE_COST': ('💰', 'Costs'),
        'RESOURCE_LINKS': ('🔗', 'Resources'),
        'TASK_PLAN': ('✅', 'Tasks')
    }
    
    for tab_key, (icon, label) in tabs.items():
        active_class = "active" if st.session_state.selected_tab == tab_key else ""
        if st.button(f"{icon} {label}", key=f"nav_{tab_key}", use_container_width=True):
            st.session_state.selected_tab = tab_key
        st.markdown(f'<style>div[data-testid="stButton"]:has(button[key="nav_{tab_key}"]) button {{background: {"linear-gradient(135deg, #667eea 0%, #764ba2 100%)" if st.session_state.selected_tab == tab_key else "transparent"}; color: {"white" if st.session_state.selected_tab == tab_key else "#666"};}}</style>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Right Column: Content ---
with col2:
    current_tab = st.session_state.selected_tab
    st.markdown(f"### {tabs[current_tab][1]} Dashboard")
    
    df = data_sheets[current_tab]
    
    if df.empty:
        st.warning("No data available")
    else:
        # Tab-specific content
        if current_tab == 'PROJECT_MASTER':
            # Filters in expander
            with st.expander("🔍 Filters", expanded=False):
                cols = st.columns(3)
                with cols[0]:
                    companies = ['All'] + list(df['Company name'].unique()) if 'Company name' in df.columns else ['All']
                    company = st.selectbox('Company', companies, key='filter_company')
                with cols[1]:
                    statuses = ['All'] + list(df['Status'].unique()) if 'Status' in df.columns else ['All']
                    status = st.selectbox('Status', statuses, key='filter_status')
                with cols[2]:
                    quarters = ['All'] + list(df['Quarter'].unique()) if 'Quarter' in df.columns else ['All']
                    quarter = st.selectbox('Quarter', quarters, key='filter_quarter')
            
            # Filter data
            filtered = df.copy()
            if company != 'All': filtered = filtered[filtered['Company name'] == company]
            if status != 'All': filtered = filtered[filtered['Status'] == status]
            if quarter != 'All': filtered = filtered[filtered['Quarter'] == quarter]
            
            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Projects", len(filtered))
            with m2:
                if 'Budget' in filtered.columns:
                    st.metric("Total Budget", f"${filtered['Budget'].sum():,.0f}")
            with m3:
                if 'Account Manager' in filtered.columns:
                    st.metric("Account Managers", filtered['Account Manager'].nunique())
            with m4:
                if 'Status' in filtered.columns:
                    st.metric("Active", len(filtered[filtered['Status'] == 'Active']))
            
            # Charts
            if 'Company name' in filtered.columns and 'Budget' in filtered.columns:
                fig = px.bar(
                    filtered.groupby('Company name')['Budget'].sum().reset_index(),
                    x='Company name', y='Budget',
                    title="Budget by Company",
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(plot_bgcolor='white', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            with st.expander("📋 View Details", expanded=True):
                st.dataframe(filtered, use_container_width=True, hide_index=True)
        
        elif current_tab == 'TASK_PLAN':
            # Similar concise layout for other tabs
            with st.expander("🔍 Filters", expanded=False):
                cols = st.columns(2)
                with cols[0]:
                    priorities = ['All'] + list(df['Priority'].unique()) if 'Priority' in df.columns else ['All']
                    priority = st.selectbox('Priority', priorities)
                with cols[1]:
                    statuses = ['All'] + list(df['Status'].unique()) if 'Status' in df.columns else ['All']
                    status = st.selectbox('Status', statuses)
            
            filtered = df.copy()
            if priority != 'All': filtered = filtered[filtered['Priority'] == priority]
            if status != 'All': filtered = filtered[filtered['Status'] == status]
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Tasks", len(filtered))
            with m2:
                if 'Status' in filtered.columns:
                    st.metric("Pending", len(filtered[filtered['Status'] != 'Done']))
            with m3:
                if 'Priority' in filtered.columns:
                    st.metric("High Priority", len(filtered[filtered['Priority'] == 'High']))
            
            if 'Owner (Team / Client)' in filtered.columns:
                fig = px.pie(
                    filtered['Owner (Team / Client)'].value_counts().reset_index(),
                    values='count', names='Owner (Team / Client)',
                    title="Tasks by Owner",
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📋 View Details", expanded=True):
                st.dataframe(filtered, use_container_width=True, hide_index=True)
        
        # Add similar concise layouts for other tabs...

# --- Minimal Footer ---
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #999; font-size: 0.8rem;'>"
    f"Project Pulse • Live Data • Updated every {REFRESH_INTERVAL}s</div>",
    unsafe_allow_html=True
)
