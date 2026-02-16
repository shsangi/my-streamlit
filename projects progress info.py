import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pytz

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Project Pulse • Live Analytics", page_icon="🎯", initial_sidebar_state="collapsed")

# --- Constants ---
REFRESH_INTERVAL = 5
PAKISTAN_TZ = pytz.timezone('Asia/Karachi')
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQFttuVQlH84hCC-brrcJFa6eyrMeyc25Aqm_dLgfpuEBr0WCdc4OTKKZVK2Y6IfOoPdQFbYmSdrSYP/pub?output=xlsx"
SHEETS = ['PROJECT_MASTER', 'DAILY_WORK_LOG', 'EMPLOYEE_COST', 'RESOURCE_LINKS', 'TASK_PLAN']

# --- Session State Init ---
for key in ['selected_tab', 'last_update', 'data_sheets', 'show_original']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'selected_tab' else 'PROJECT_MASTER'

# --- Data Loading ---
@st.cache_data(ttl=REFRESH_INTERVAL)
def load_data():
    try:
        xl = pd.ExcelFile(DATA_URL)
        sheets = {name: pd.read_excel(xl, name) for name in SHEETS}
        for df in sheets.values():
            for col in df.select_dtypes(include=['object']):
                if any(k in col.lower() for k in ['date', 'time']):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        return sheets, datetime.now(PAKISTAN_TZ)
    except:
        return {name: pd.DataFrame() for name in SHEETS}, None

data_sheets, last_update = load_data()
if last_update:
    st.session_state.last_update = last_update
    st.session_state.data_sheets = data_sheets

# --- Modern CSS ---
st.markdown("""
<style>
    /* Global */
    .stApp { background: #f8fafc; }
    .block-container { padding: 1.5rem 2rem; }
    
    /* Header */
    .header { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 1.5rem 2rem; border-radius: 20px; margin-bottom: 1.5rem; color: white; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1); }
    
    /* Status */
    .status-bar { background: white; padding: 0.75rem 1.5rem; border-radius: 50px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .live-badge { background: #ef4444; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
    
    /* KPI Cards */
    .kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
    .kpi-card { background: white; padding: 1.25rem; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; transition: all 0.2s; }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #1e293b; line-height: 1.2; }
    .kpi-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    
    /* Navigation */
    .nav-card { background: white; border-radius: 16px; padding: 1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .stButton > button { width: 100%; text-align: left; border-radius: 12px; margin: 0.25rem 0; border: 1px solid #e2e8f0; background: white; transition: all 0.2s; }
    .stButton > button:hover { transform: translateX(5px); border-color: #3b82f6; background: #f8fafc; }
    
    /* Charts */
    .chart-container { background: white; border-radius: 16px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header">
    <div style="display: flex; align-items: center; gap: 1rem;">
        <h1 style="margin: 0; font-size: 2rem;">🎯 Project Pulse</h1>
        <span style="background: rgba(255,255,255,0.2); padding: 0.25rem 1rem; border-radius: 50px; font-size: 0.9rem;">Live Analytics</span>
    </div>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Real-time project intelligence & performance metrics</p>
</div>
""", unsafe_allow_html=True)

# --- Status Bar ---
if st.session_state.last_update:
    timestamp = st.session_state.last_update.strftime("%a, %d %b %Y • %I:%M:%S %p")
    st.markdown(f"""
    <div class="status-bar">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span class="live-badge">LIVE</span>
            <span style="color: #475569;">🔄 {timestamp} PKT</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; color: #22c55e;">
            <span>●</span><span>Auto-refresh {REFRESH_INTERVAL}s</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- KPI Dashboard ---
if st.session_state.data_sheets and not st.session_state.show_original:
    df_p, df_w, df_c, df_r, df_t = [st.session_state.data_sheets[s] for s in SHEETS]
    
    metrics = [
        (f"{len(df_p)}", "Total Projects", f"{len(df_p[df_p['Status']=='Active']) if 'Status' in df_p.columns else 0} Active"),
        (f"{df_w['Hours Worked'].sum():.0f}" if not df_w.empty else "0", "Total Hours", "Work Log"),
        (f"${df_c['Monthly Salary'].sum():,.0f}" if not df_c.empty else "$0", "Monthly Salary", "Employee Cost"),
        (f"{len(df_t)}", "Total Tasks", f"{len(df_t[df_t['Status']!='Done']) if 'Status' in df_t.columns else 0} Pending"),
        (f"{len(df_r)}", "Resources", "Active Links")
    ]
    
    cols = st.columns(5)
    for i, (value, label, sub) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
                <div style="font-size:0.7rem; color:#64748b; margin-top:0.25rem;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

# --- Main Layout ---
if st.session_state.show_original:
    st.markdown("### 📋 Source Data View")
    if st.button("← Dashboard", type="primary", use_container_width=True):
        st.session_state.show_original = False
        st.rerun()
    
    tabs = st.tabs(SHEETS)
    for i, (sheet, df) in enumerate(st.session_state.data_sheets.items()):
        with tabs[i]:
            st.dataframe(df, use_container_width=True, hide_index=True)
            if not df.empty:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(f"📥 Download CSV", csv, f"{sheet}.csv", "text/csv")
else:
    left, right = st.columns([1, 4])
    
    with left:
        st.markdown('<div class="nav-card">', unsafe_allow_html=True)
        st.markdown("##### 📊 Views")
        
        tabs = {'PROJECT_MASTER': ('📁', 'Projects'), 'DAILY_WORK_LOG': ('📝', 'Work Log'),
                'EMPLOYEE_COST': ('💰', 'Costs'), 'RESOURCE_LINKS': ('🔗', 'Resources'),
                'TASK_PLAN': ('✅', 'Tasks')}
        
        for key, (icon, label) in tabs.items():
            if st.button(f"{icon} {label}", key=f"nav_{key}", use_container_width=True,
                        type="primary" if st.session_state.selected_tab == key else "secondary"):
                st.session_state.selected_tab = key
                st.rerun()
        
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("📋 Source Data", use_container_width=True):
            st.session_state.show_original = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with right:
        tab = st.session_state.selected_tab
        st.markdown(f"### {tabs[tab][1]} Analytics")
        
        if st.session_state.data_sheets:
            df = st.session_state.data_sheets[tab]
            
            if df.empty:
                st.info("📭 No data available")
            else:
                # Project Master
                if tab == 'PROJECT_MASTER':
                    with st.container():
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        cols = st.columns(3)
                        filters = {}
                        for i, (col, options) in enumerate({'Company': 'Company name', 'Status': 'Status', 'Quarter': 'Quarter'}.items()):
                            if options in df.columns:
                                with cols[i]:
                                    filters[col] = st.selectbox(col, ['All'] + list(df[options].unique()))
                        
                        filtered = df.copy()
                        for key, col in {'Company': 'Company name', 'Status': 'Status', 'Quarter': 'Quarter'}.items():
                            if filters.get(key, 'All') != 'All' and col in df.columns:
                                filtered = filtered[filtered[col] == filters[key]]
                        
                        if 'Company name' in filtered.columns and 'Budget' in filtered.columns:
                            fig = px.bar(filtered.groupby('Company name')['Budget'].sum().reset_index(), 
                                       x='Company name', y='Budget', title="Budget Distribution",
                                       color_discrete_sequence=['#3b82f6'])
                            fig.update_layout(plot_bgcolor='white', height=300, margin=dict(t=30))
                            st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        with st.expander("📋 Project Details", expanded=True):
                            st.dataframe(filtered, use_container_width=True, hide_index=True)
                
                # Tasks
                elif tab == 'TASK_PLAN':
                    with st.container():
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        cols = st.columns(2)
                        filters = {}
                        for i, (col, options) in enumerate({'Priority': 'Priority', 'Status': 'Status'}.items()):
                            if options in df.columns:
                                with cols[i]:
                                    filters[col] = st.selectbox(col, ['All'] + list(df[options].unique()))
                        
                        filtered = df.copy()
                        for key, col in {'Priority': 'Priority', 'Status': 'Status'}.items():
                            if filters.get(key, 'All') != 'All' and col in df.columns:
                                filtered = filtered[filtered[col] == filters[key]]
                        
                        if 'Owner (Team / Client)' in filtered.columns:
                            fig = px.pie(filtered['Owner (Team / Client)'].value_counts().reset_index(),
                                       values='count', names='index', title="Task Ownership",
                                       color_discrete_sequence=px.colors.qualitative.Set3)
                            fig.update_layout(height=300, margin=dict(t=30))
                            st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        with st.expander("📋 Task Details", expanded=True):
                            st.dataframe(filtered, use_container_width=True, hide_index=True)
                
                # Work Log
                elif tab == 'DAILY_WORK_LOG':
                    with st.container():
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        if 'Employee Name' in df.columns:
                            emp = st.selectbox('Employee', ['All'] + list(df['Employee Name'].unique()))
                            filtered = df if emp == 'All' else df[df['Employee Name'] == emp]
                            
                            if 'Hours Worked' in filtered.columns:
                                chart = filtered.groupby('Employee Name')['Hours Worked'].sum().reset_index()
                                fig = px.bar(chart, x='Employee Name', y='Hours Worked', title="Hours by Employee",
                                           color_discrete_sequence=['#3b82f6'])
                                fig.update_layout(plot_bgcolor='white', height=300, margin=dict(t=30))
                                st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        with st.expander("📋 Work Log Details", expanded=True):
                            st.dataframe(filtered if 'filtered' in locals() else df, use_container_width=True, hide_index=True)
                
                # Employee Cost
                elif tab == 'EMPLOYEE_COST':
                    if 'Role' in df.columns and 'Monthly Salary' in df.columns:
                        with st.container():
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            chart = df.groupby('Role')['Monthly Salary'].sum().reset_index()
                            fig = px.bar(chart, x='Role', y='Monthly Salary', title="Salary by Role",
                                       color='Role', color_discrete_sequence=px.colors.qualitative.Set3)
                            fig.update_layout(plot_bgcolor='white', height=350, margin=dict(t=30))
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    with st.expander("📋 Cost Details", expanded=True):
                        st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Resources
                elif tab == 'RESOURCE_LINKS':
                    st.info("🔗 Resource links available in table below")
                    with st.expander("📋 Resource Details", expanded=True):
                        st.dataframe(df, use_container_width=True, hide_index=True)

# --- Auto-refresh ---
st.markdown(f'<meta http-equiv="refresh" content="{REFRESH_INTERVAL}">', unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #94a3b8; font-size: 0.75rem;'>Project Pulse • Real-time Analytics • v2.0</div>", unsafe_allow_html=True)
