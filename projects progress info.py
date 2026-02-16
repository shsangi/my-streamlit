import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pytz
import numpy as np

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Project Pulse", page_icon="📊", initial_sidebar_state="collapsed")

# --- Constants ---
REFRESH_INTERVAL = 5
PAKISTAN_TZ = pytz.timezone('Asia/Karachi')
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQFttuVQlH84hCC-brrcJFa6eyrMeyc25Aqm_dLgfpuEBr0WCdc4OTKKZVK2Y6IfOoPdQFbYmSdrSYP/pub?output=xlsx"

# --- Initialize Session State ---
for key in ['selected_tab', 'last_update', 'data_sheets', 'show_original', 'filters']:
    if key not in st.session_state:
        if key == 'selected_tab': st.session_state[key] = 'PROJECT_MASTER'
        elif key == 'filters': st.session_state[key] = {}
        else: st.session_state[key] = None if key != 'show_original' else False

# --- Data Loading Function ---
@st.cache_data(ttl=REFRESH_INTERVAL)
def load_data():
    try:
        xl = pd.ExcelFile(DATA_URL)
        sheets = {
            'PROJECT_MASTER': pd.read_excel(xl, 'PROJECT_MASTER'),
            'DAILY_WORK_LOG': pd.read_excel(xl, 'DAILY_WORK_LOG'),
            'EMPLOYEE_COST': pd.read_excel(xl, 'EMPLOYEE_COST'),
            'RESOURCE_LINKS': pd.read_excel(xl, 'RESOURCE_LINKS'),
            'TASK_PLAN': pd.read_excel(xl, 'TASK PLAN + RESPONSIBILITY')
        }
        return sheets, datetime.now(PAKISTAN_TZ)
    except Exception as e:
        st.error(f"⚠️ Data load failed: {str(e)}")
        return {name: pd.DataFrame() for name in ['PROJECT_MASTER', 'DAILY_WORK_LOG', 'EMPLOYEE_COST', 'RESOURCE_LINKS', 'TASK_PLAN']}, None

# --- Load Data ---
data_sheets, last_update = load_data()
if last_update:
    st.session_state.last_update = last_update
    st.session_state.data_sheets = data_sheets

# --- Custom CSS with Modern Design ---
st.markdown("""
<style>
    /* Modern Color Palette */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f52e0;
        --secondary: #8b5cf6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark: #1f2937;
        --light: #f9fafb;
        --gray: #6b7280;
    }
    
    /* Header Styles */
    .header-container {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 1.5rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04);
    }
    
    .header-title {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        margin: 0.5rem 0 0 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .live-badge {
        background: var(--danger);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
        animation: pulse 2s infinite;
        letter-spacing: 0.5px;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    .timestamp {
        color: rgba(255,255,255,0.9);
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .pk-badge {
        background: rgba(255,255,255,0.2);
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        backdrop-filter: blur(5px);
    }
    
    /* Scorecards */
    .header-scorecards {
        display: flex;
        gap: 1rem;
        margin-top: 0.5rem;
    }
    
    .header-scorecard {
        background: rgba(255,255,255,0.15);
        padding: 0.75rem 1.25rem;
        border-radius: 16px;
        text-align: center;
        min-width: 100px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.2s;
    }
    
    .header-scorecard:hover {
        transform: translateY(-2px);
        background: rgba(255,255,255,0.2);
    }
    
    .header-scorecard-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        line-height: 1.2;
    }
    
    .header-scorecard-label {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.9);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .header-scorecard-sub {
        font-size: 0.65rem;
        color: rgba(255,255,255,0.8);
    }
    
    /* Secondary Scorecards */
    .secondary-scorecards {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    
    .scorecard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .scorecard {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #e5e7eb;
        transition: all 0.2s;
    }
    
    .scorecard:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        border-color: var(--primary);
    }
    
    .scorecard-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--dark);
        line-height: 1.2;
    }
    
    .scorecard-label {
        font-size: 0.85rem;
        color: var(--gray);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }
    
    /* Navigation */
    .tab-container {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    
    .nav-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--dark);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 500;
        transition: all 0.2s;
        border: 1px solid #e5e7eb;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    
    .reset-btn {
        background: linear-gradient(135deg, var(--danger) 0%, #dc2626 100%);
        color: white;
    }
    
    /* Table Header */
    .table-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding: 0.5rem 1rem;
        background: #f9fafb;
        border-radius: 12px;
    }
    
    .table-name {
        font-weight: 600;
        color: var(--dark);
        font-size: 1rem;
    }
    
    .table-shape {
        background: white;
        padding: 0.25rem 1rem;
        border-radius: 30px;
        font-size: 0.8rem;
        color: var(--gray);
        border: 1px solid #e5e7eb;
    }
    
    .no-filters-msg {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 12px;
        color: var(--gray);
        font-size: 0.9rem;
        text-align: center;
        border: 1px dashed #d1d5db;
    }
    
    /* Filter Section */
    .filter-container {
        background: #f9fafb;
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid #e5e7eb;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu, footer, .stDeployButton { visibility: hidden; }
    
    /* Custom Chart Container */
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Modern Chart Functions ---
def create_gradient_bar_chart(data, x, y, title, color_scale='viridis'):
    """Create a beautiful gradient bar chart"""
    fig = px.bar(data, x=x, y=y, title=title, 
                 color=y, color_continuous_scale=color_scale,
                 text=y)
    
    fig.update_traces(
        texttemplate='%{text:.0f}', 
        textposition='outside',
        marker=dict(line=dict(width=1, color='white'))
    )
    
    fig.update_layout(
        height=350,
        margin=dict(t=50, l=50, r=20, b=50),
        title=dict(font=dict(size=16, color='#1f2937'), x=0.5),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#e5e7eb', showgrid=True, gridwidth=1),
        yaxis=dict(gridcolor='#e5e7eb', showgrid=True, gridwidth=1),
        hoverlabel=dict(bgcolor='white', font_size=12, font_family='Arial'),
        coloraxis_showscale=False
    )
    
    return fig

def create_donut_chart(data, names, values, title, colors=None):
    """Create a modern donut chart"""
    if colors is None:
        colors = ['#6366f1', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444']
    
    fig = go.Figure(data=[go.Pie(
        labels=data[names],
        values=data[values],
        hole=.6,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=12, color='#1f2937'),
        hoverinfo='label+value+percent',
        hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Percent: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        height=350,
        title=dict(text=title, font=dict(size=16, color='#1f2937'), x=0.5),
        margin=dict(t=50, l=50, r=50, b=50),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(text=f'Total<br>{sum(data[values]):,.0f}', 
                         x=0.5, y=0.5, font_size=14, showarrow=False)]
    )
    
    return fig

def create_treemap_chart(data, path, values, title):
    """Create an interactive treemap"""
    fig = px.treemap(data, path=path, values=values, title=title,
                     color=values, color_continuous_scale='viridis')
    
    fig.update_layout(
        height=400,
        title=dict(font=dict(size=16, color='#1f2937'), x=0.5),
        margin=dict(t=50, l=0, r=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.update_traces(
        textinfo="label+value+percent root",
        hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Percent of total: %{percentRoot:.1%}<extra></extra>'
    )
    
    return fig

def create_stacked_bar_chart(data, x, y, color, title):
    """Create a stacked bar chart"""
    fig = px.bar(data, x=x, y=y, color=color, title=title,
                 barmode='stack', color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.update_layout(
        height=350,
        title=dict(font=dict(size=16, color='#1f2937'), x=0.5),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#e5e7eb', showgrid=True),
        yaxis=dict(gridcolor='#e5e7eb', showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_gauge_chart(value, title, max_value, color_scheme=None):
    """Create a modern gauge chart"""
    if color_scheme is None:
        color_scheme = ["#ef4444", "#f59e0b", "#10b981"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14, 'color': '#1f2937'}},
        number={'font': {'size': 24, 'color': '#1f2937'}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': 'darkgray'},
            'bar': {'color': '#6366f1', 'thickness': 0.3},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': 'gray',
            'steps': [
                {'range': [0, max_value * 0.3], 'color': color_scheme[0]},
                {'range': [max_value * 0.3, max_value * 0.7], 'color': color_scheme[1]},
                {'range': [max_value * 0.7, max_value], 'color': color_scheme[2]}
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(t=50, l=30, r=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_time_series_chart(data, x, y, title):
    """Create a time series chart with area fill"""
    fig = px.area(data, x=x, y=y, title=title,
                  line_shape='spline', color_discrete_sequence=['#6366f1'])
    
    fig.update_traces(
        fill='tozeroy',
        fillcolor='rgba(99, 102, 241, 0.2)',
        line=dict(width=3)
    )
    
    fig.update_layout(
        height=350,
        title=dict(font=dict(size=16, color='#1f2937'), x=0.5),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#e5e7eb', showgrid=True),
        yaxis=dict(gridcolor='#e5e7eb', showgrid=True),
        hovermode='x unified'
    )
    
    return fig

# --- Header ---
if st.session_state.data_sheets and not st.session_state.show_original:
    df_p = st.session_state.data_sheets['PROJECT_MASTER']
    df_w = st.session_state.data_sheets['DAILY_WORK_LOG']
    df_c = st.session_state.data_sheets['EMPLOYEE_COST']
    df_t = st.session_state.data_sheets['TASK_PLAN']
    df_r = st.session_state.data_sheets['RESOURCE_LINKS']
    
    metrics = {
        'total_projects': len(df_p) if not df_p.empty else 0,
        'active_projects': len(df_p[df_p['Status'] == 'Active']) if not df_p.empty and 'Status' in df_p.columns else 0,
        'total_hours': df_w['Hours Worked'].sum() if not df_w.empty else 0,
        'total_salary': df_c['Monthly Salary'].sum() if not df_c.empty and 'Monthly Salary' in df_c.columns else 0,
        'total_tasks': len(df_t) if not df_t.empty else 0,
        'pending_tasks': len(df_t[df_t['Status'] != 'Done']) if not df_t.empty and 'Status' in df_t.columns else 0,
        'total_resources': len(df_r) if not df_r.empty else 0
    }
    
    timestamp = st.session_state.last_update.strftime("%a, %d %b, %Y, %I:%M:%S %p") if st.session_state.last_update else ""
    
    st.markdown(f"""
    <div class="header-container">
        <div>
            <h1 class="header-title">📊 Project Pulse</h1>
            <p class="header-subtitle">
                <span class="live-badge">LIVE</span>
                <span class="timestamp">🔄 {timestamp} <span class="pk-badge">PKT</span></span>
            </p>
        </div>
        <div class="header-scorecards">
            <div class="header-scorecard">
                <div class="header-scorecard-value">{metrics['total_projects']}</div>
                <div class="header-scorecard-label">Projects</div>
                <div class="header-scorecard-sub">{metrics['active_projects']} Active</div>
            </div>
            <div class="header-scorecard">
                <div class="header-scorecard-value">{metrics['total_hours']:.0f}</div>
                <div class="header-scorecard-label">Hours</div>
                <div class="header-scorecard-sub">Work Log</div>
            </div>
            <div class="header-scorecard">
                <div class="header-scorecard-value">${metrics['total_salary']:,.0f}</div>
                <div class="header-scorecard-label">Salary</div>
                <div class="header-scorecard-sub">Monthly</div>
            </div>
            <div class="header-scorecard">
                <div class="header-scorecard-value">{metrics['total_tasks']}</div>
                <div class="header-scorecard-label">Tasks</div>
                <div class="header-scorecard-sub">{metrics['pending_tasks']} Pending</div>
            </div>
            <div class="header-scorecard">
                <div class="header-scorecard-value">{metrics['total_resources']}</div>
                <div class="header-scorecard-label">Resources</div>
                <div class="header-scorecard-sub">Links</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="header-container">
        <div>
            <h1 class="header-title">📊 Project Pulse</h1>
            <p class="header-subtitle">Original Sheets View</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Main Layout ---
if st.session_state.show_original:
    st.markdown("### 📋 Original Sheets Data")
    if st.button("← Back to Dashboard", type="primary"):
        st.session_state.show_original = False
        st.rerun()
    
    tabs = st.tabs(list(st.session_state.data_sheets.keys()))
    for i, (sheet_name, df) in enumerate(st.session_state.data_sheets.items()):
        with tabs[i]:
            st.markdown(f"""
            <div class="table-header">
                <span class="table-name">📄 {sheet_name.replace('_', ' ').title()}</span>
                <span class="table-shape">{df.shape[0]} rows × {df.shape[1]} columns</span>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(f"📥 Download {sheet_name}", csv, f"{sheet_name}.csv", "text/csv")
else:
    left_col, right_col = st.columns([1, 4])
    
    with left_col:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        st.markdown('<div class="nav-header">📋 Navigation</div>', unsafe_allow_html=True)
        
        tabs = {'PROJECT_MASTER': '📁 Projects', 'DAILY_WORK_LOG': '📝 Work Log', 'EMPLOYEE_COST': '💰 Costs', 
                'RESOURCE_LINKS': '🔗 Resources', 'TASK_PLAN': '✅ Tasks'}
        
        for key, label in tabs.items():
            if st.button(label, key=f"nav_{key}", use_container_width=True, 
                        type="secondary" if st.session_state.selected_tab != key else "primary"):
                st.session_state.selected_tab = key
                st.session_state.filters = {}
                st.rerun()
        
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("📋 View Original Sheets", use_container_width=True):
            st.session_state.show_original = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with right_col:
        current = st.session_state.selected_tab
        st.markdown(f"### {tabs[current]}")
        
        if st.session_state.data_sheets and not st.session_state.data_sheets[current].empty:
            df = st.session_state.data_sheets[current].copy()
            
            # Drop NaN values from filter columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('Unknown')
            
            # Filters Section
            if current != 'RESOURCE_LINKS':
                with st.expander("🔍 Filters", expanded=True):
                    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
                    cols = st.columns([4, 1])
                    with cols[0]:
                        if current == 'PROJECT_MASTER':
                            filter_cols = st.columns(3)
                            with filter_cols[0]:
                                companies = ['All'] + sorted([x for x in df['Company name'].unique() if x != 'Unknown']) if 'Company name' in df.columns else ['All']
                                st.session_state.filters['company'] = st.selectbox('Company', companies, index=0, key='f_comp')
                            with filter_cols[1]:
                                statuses = ['All'] + sorted([x for x in df['Status'].unique() if x != 'Unknown']) if 'Status' in df.columns else ['All']
                                st.session_state.filters['status'] = st.selectbox('Status', statuses, index=0, key='f_stat')
                            with filter_cols[2]:
                                quarters = ['All'] + sorted([x for x in df['Quarter'].unique() if x != 'Unknown']) if 'Quarter' in df.columns else ['All']
                                st.session_state.filters['quarter'] = st.selectbox('Quarter', quarters, index=0, key='f_quart')
                        
                        elif current == 'TASK_PLAN':
                            filter_cols = st.columns(3)
                            with filter_cols[0]:
                                priorities = ['All'] + sorted([x for x in df['Priority'].unique() if x != 'Unknown']) if 'Priority' in df.columns else ['All']
                                st.session_state.filters['priority'] = st.selectbox('Priority', priorities, index=0, key='f_pri')
                            with filter_cols[1]:
                                statuses = ['All'] + sorted([x for x in df['Status'].unique() if x != 'Unknown']) if 'Status' in df.columns else ['All']
                                st.session_state.filters['status'] = st.selectbox('Status', statuses, index=0, key='f_stat')
                            with filter_cols[2]:
                                owners = ['All'] + sorted([x for x in df['Owner (Team / Client)'].unique() if x != 'Unknown']) if 'Owner (Team / Client)' in df.columns else ['All']
                                st.session_state.filters['owner'] = st.selectbox('Owner', owners, index=0, key='f_own')
                        
                        elif current == 'DAILY_WORK_LOG':
                            employees = ['All'] + sorted([x for x in df['Employee Name'].unique() if x != 'Unknown']) if 'Employee Name' in df.columns else ['All']
                            st.session_state.filters['employee'] = st.selectbox('Employee', employees, index=0, key='f_emp')
                        
                        elif current == 'EMPLOYEE_COST':
                            roles = ['All'] + sorted([x for x in df['Role'].unique() if x != 'Unknown']) if 'Role' in df.columns else ['All']
                            st.session_state.filters['role'] = st.selectbox('Role', roles, index=0, key='f_role')
                    
                    with cols[1]:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🔄 Reset All Filters", use_container_width=True, type="primary"):
                            st.session_state.filters = {}
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="no-filters-msg">🔗 Resource Links View - No filters available</div>', unsafe_allow_html=True)
            
            # Apply filters
            filtered = df.copy()
            if current != 'RESOURCE_LINKS':
                if current == 'PROJECT_MASTER':
                    if st.session_state.filters.get('company', 'All') != 'All': filtered = filtered[filtered['Company name'] == st.session_state.filters['company']]
                    if st.session_state.filters.get('status', 'All') != 'All': filtered = filtered[filtered['Status'] == st.session_state.filters['status']]
                    if st.session_state.filters.get('quarter', 'All') != 'All': filtered = filtered[filtered['Quarter'] == st.session_state.filters['quarter']]
                elif current == 'TASK_PLAN':
                    if st.session_state.filters.get('priority', 'All') != 'All': filtered = filtered[filtered['Priority'] == st.session_state.filters['priority']]
                    if st.session_state.filters.get('status', 'All') != 'All': filtered = filtered[filtered['Status'] == st.session_state.filters['status']]
                    if st.session_state.filters.get('owner', 'All') != 'All': filtered = filtered[filtered['Owner (Team / Client)'] == st.session_state.filters['owner']]
                elif current == 'DAILY_WORK_LOG' and st.session_state.filters.get('employee', 'All') != 'All':
                    filtered = filtered[filtered['Employee Name'] == st.session_state.filters['employee']]
                elif current == 'EMPLOYEE_COST' and st.session_state.filters.get('role', 'All') != 'All':
                    filtered = filtered[filtered['Role'] == st.session_state.filters['role']]
            
            # Secondary Scorecards with Modern Design
            if not filtered.empty:
                st.markdown('<div class="secondary-scorecards">', unsafe_allow_html=True)
                st.markdown("##### 📊 Filtered View Metrics")
                st.markdown('<div class="scorecard-grid">', unsafe_allow_html=True)
                
                if current == 'PROJECT_MASTER':
                    cols = st.columns(4)
                    with cols[0]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered)}</div><div class='scorecard-label'>Projects</div></div>", unsafe_allow_html=True)
                    with cols[1]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>${filtered['Budget'].sum():,.0f}</div><div class='scorecard-label'>Total Budget</div></div>" if 'Budget' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Budget</div></div>", unsafe_allow_html=True)
                    with cols[2]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered[filtered['Status']=='Active'])}</div><div class='scorecard-label'>Active</div></div>" if 'Status' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Active</div></div>", unsafe_allow_html=True)
                    with cols[3]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Company name'].nunique()}</div><div class='scorecard-label'>Companies</div></div>" if 'Company name' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Companies</div></div>", unsafe_allow_html=True)
                
                elif current == 'TASK_PLAN':
                    cols = st.columns(4)
                    with cols[0]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered)}</div><div class='scorecard-label'>Tasks</div></div>", unsafe_allow_html=True)
                    with cols[1]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered[filtered['Status']=='Done'])}</div><div class='scorecard-label'>Completed</div></div>" if 'Status' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Completed</div></div>", unsafe_allow_html=True)
                    with cols[2]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered[filtered['Priority']=='High'])}</div><div class='scorecard-label'>High Priority</div></div>" if 'Priority' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>High Priority</div></div>", unsafe_allow_html=True)
                    with cols[3]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Owner (Team / Client)'].nunique()}</div><div class='scorecard-label'>Owners</div></div>" if 'Owner (Team / Client)' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Owners</div></div>", unsafe_allow_html=True)
                
                elif current == 'DAILY_WORK_LOG':
                    cols = st.columns(4)
                    with cols[0]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered)}</div><div class='scorecard-label'>Entries</div></div>", unsafe_allow_html=True)
                    with cols[1]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Hours Worked'].sum():.1f}</div><div class='scorecard-label'>Total Hours</div></div>" if 'Hours Worked' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Hours</div></div>", unsafe_allow_html=True)
                    with cols[2]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Hours Worked'].mean():.1f}</div><div class='scorecard-label'>Avg Hours</div></div>" if 'Hours Worked' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Avg Hours</div></div>", unsafe_allow_html=True)
                    with cols[3]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Employee Name'].nunique()}</div><div class='scorecard-label'>Employees</div></div>" if 'Employee Name' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Employees</div></div>", unsafe_allow_html=True)
                
                elif current == 'EMPLOYEE_COST':
                    cols = st.columns(4)
                    with cols[0]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered)}</div><div class='scorecard-label'>Employees</div></div>", unsafe_allow_html=True)
                    with cols[1]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>${filtered['Monthly Salary'].sum():,.0f}</div><div class='scorecard-label'>Total Salary</div></div>" if 'Monthly Salary' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>$0</div><div class='scorecard-label'>Salary</div></div>", unsafe_allow_html=True)
                    with cols[2]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>${filtered['Monthly Salary'].mean():,.0f}</div><div class='scorecard-label'>Avg Salary</div></div>" if 'Monthly Salary' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>$0</div><div class='scorecard-label'>Avg Salary</div></div>", unsafe_allow_html=True)
                    with cols[3]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Role'].nunique()}</div><div class='scorecard-label'>Roles</div></div>" if 'Role' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Roles</div></div>", unsafe_allow_html=True)
                
                elif current == 'RESOURCE_LINKS':
                    cols = st.columns(3)
                    with cols[0]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered)}</div><div class='scorecard-label'>Resources</div></div>", unsafe_allow_html=True)
                    with cols[1]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Type'].nunique() if 'Type' in filtered.columns else 0}</div><div class='scorecard-label'>Types</div></div>", unsafe_allow_html=True)
                    with cols[2]: 
                        st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Category'].nunique() if 'Category' in filtered.columns else 0}</div><div class='scorecard-label'>Categories</div></div>", unsafe_allow_html=True)
                
                st.markdown('</div></div>', unsafe_allow_html=True)
                
                # Modern Charts Section
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                
                if current == 'PROJECT_MASTER':
                    if 'Company name' in filtered.columns and 'Budget' in filtered.columns:
                        # Main Chart - Gradient Bar
                        chart_data = filtered.groupby('Company name')['Budget'].sum().reset_index()
                        fig = create_gradient_bar_chart(chart_data, 'Company name', 'Budget', "Budget Distribution by Company", 'viridis')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Secondary Chart - Treemap for detailed view
                        if 'Status' in filtered.columns and 'Quarter' in filtered.columns:
                            fig2 = create_treemap_chart(filtered, ['Company name', 'Status', 'Quarter'], 'Budget', "Budget Treemap by Company, Status, and Quarter")
                            st.plotly_chart(fig2, use_container_width=True)
                    
                elif current == 'TASK_PLAN':
                    if 'Owner (Team / Client)' in filtered.columns:
                        # Main Chart - Donut
                        chart_data = filtered['Owner (Team / Client)'].value_counts().reset_index()
                        chart_data.columns = ['Owner (Team / Client)', 'count']
                        fig = create_donut_chart(chart_data, 'Owner (Team / Client)', 'count', "Task Distribution by Owner")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Secondary Chart - Stacked Bar by Priority and Status
                        if 'Priority' in filtered.columns and 'Status' in filtered.columns:
                            chart_data2 = filtered.groupby(['Priority', 'Status']).size().reset_index(name='count')
                            fig2 = create_stacked_bar_chart(chart_data2, 'Priority', 'count', 'Status', "Tasks by Priority and Status")
                            st.plotly_chart(fig2, use_container_width=True)
                    
                elif current == 'DAILY_WORK_LOG':
                    if 'Employee Name' in filtered.columns and 'Hours Worked' in filtered.columns:
                        # Main Chart - Gradient Bar
                        chart_data = filtered.groupby('Employee Name')['Hours Worked'].sum().reset_index()
                        fig = create_gradient_bar_chart(chart_data, 'Employee Name', 'Hours Worked', "Total Hours by Employee", 'plasma')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Time Series if Date column exists
                        if 'Date' in filtered.columns:
                            try:
                                filtered['Date'] = pd.to_datetime(filtered['Date'])
                                time_data = filtered.groupby('Date')['Hours Worked'].sum().reset_index()
                                fig2 = create_time_series_chart(time_data, 'Date', 'Hours Worked', "Daily Work Hours Trend")
                                st.plotly_chart(fig2, use_container_width=True)
                            except:
                                pass
                    
                elif current == 'EMPLOYEE_COST':
                    if 'Role' in filtered.columns and 'Monthly Salary' in filtered.columns:
                        # Main Chart - Gradient Bar
                        chart_data = filtered.groupby('Role')['Monthly Salary'].sum().reset_index()
                        fig = create_gradient_bar_chart(chart_data, 'Role', 'Monthly Salary', "Salary Distribution by Role", 'magma')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Gauge Chart for total salary vs target
                        total_salary = filtered['Monthly Salary'].sum()
                        target_salary = total_salary * 1.2  # Example target (20% higher)
                        fig2 = create_gauge_chart(total_salary, "Total Salary Progress", target_salary)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Data Table
                with st.expander("📋 Details", expanded=True):
                    st.markdown(f"""
                    <div class="table-header">
                        <span class="table-name">📄 {current.replace('_', ' ').title()}</span>
                        <span class="table-shape">{filtered.shape[0]} rows × {filtered.shape[1]} columns</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Format numbers for better display
                    display_df = filtered.copy()
                    for col in display_df.select_dtypes(include=['float64', 'int64']).columns:
                        if 'salary' in col.lower() or 'budget' in col.lower() or 'cost' in col.lower():
                            display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else x)
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("📭 No data available")

# --- Auto-refresh and Footer ---
st.markdown(f'<meta http-equiv="refresh" content="{REFRESH_INTERVAL}">', unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 1rem;'>
    <span style='color: #999; font-size: 0.8rem;'>Project Pulse • Auto-refreshes every {REFRESH_INTERVAL}s</span>
    <br>
    <span style='color: #ccc; font-size: 0.7rem;'>Built with Streamlit & Plotly • Modern Dashboard Design</span>
</div>
""", unsafe_allow_html=True)
