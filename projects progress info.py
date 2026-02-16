import streamlit as st
import pandas as pd
import plotly.express as px
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
for key in ['selected_tab', 'last_update', 'data_sheets', 'show_original', 'filters']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'selected_tab' else 'PROJECT_MASTER'
        if key == 'filters':
            st.session_state.filters = {}

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
    
    /* Sub Scorecards */
    .sub-scorecard-container { background: white; border-radius: 16px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .sub-scorecard { background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid #e2e8f0; }
    .sub-value { font-size: 1.5rem; font-weight: 700; color: #3b82f6; }
    .sub-label { font-size: 0.8rem; color: #64748b; margin-top: 0.25rem; }
    
    /* Navigation */
    .nav-card { background: white; border-radius: 16px; padding: 1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    .stButton > button { width: 100%; text-align: left; border-radius: 12px; margin: 0.25rem 0; border: 1px solid #e2e8f0; background: white; transition: all 0.2s; }
    .stButton > button:hover { transform: translateX(5px); border-color: #3b82f6; background: #f8fafc; }
    
    /* Charts */
    .chart-container { background: white; border-radius: 16px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    
    /* Filters */
    .filter-section { background: white; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; border: 1px solid #e2e8f0; }
    
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

# --- Top Level KPI Dashboard ---
if st.session_state.data_sheets and not st.session_state.show_original:
    df_p, df_w, df_c, df_r, df_t = [st.session_state.data_sheets[s] for s in SHEETS]
    
    metrics = [
        (f"{len(df_p)}", "Total Projects", f"{len(df_p[df_p['Status']=='Active']) if 'Status' in df_p.columns else 0} Active"),
        (f"{df_w['Hours Worked'].sum():.0f}" if not df_w.empty else "0", "Total Hours", f"{len(df_w['Employee Name'].unique()) if 'Employee Name' in df_w.columns else 0} Employees"),
        (f"${df_c['Monthly Salary'].sum():,.0f}" if not df_c.empty else "$0", "Monthly Salary", f"{len(df_c)} Employees"),
        (f"{len(df_t)}", "Total Tasks", f"{len(df_t[df_t['Status']=='Done']) if 'Status' in df_t.columns else 0} Completed"),
        (f"{len(df_r)}", "Resources", f"{len(df_r['Category'].unique()) if 'Category' in df_r.columns else 0} Categories")
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
                st.session_state.filters = {}  # Reset filters on tab change
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
                # Filter Section
                with st.expander("🔍 Filters", expanded=True):
                    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
                    filters_applied = False
                    
                    if tab == 'PROJECT_MASTER':
                        cols = st.columns(3)
                        filter_cols = {}
                        if 'Company name' in df.columns:
                            with cols[0]:
                                filter_cols['Company'] = st.selectbox('Company', ['All'] + list(df['Company name'].unique()))
                                if filter_cols['Company'] != 'All': filters_applied = True
                        if 'Status' in df.columns:
                            with cols[1]:
                                filter_cols['Status'] = st.selectbox('Status', ['All'] + list(df['Status'].unique()))
                                if filter_cols['Status'] != 'All': filters_applied = True
                        if 'Quarter' in df.columns:
                            with cols[2]:
                                filter_cols['Quarter'] = st.selectbox('Quarter', ['All'] + list(df['Quarter'].unique()))
                                if filter_cols['Quarter'] != 'All': filters_applied = True
                    
                    elif tab == 'TASK_PLAN':
                        cols = st.columns(3)
                        filter_cols = {}
                        if 'Priority' in df.columns:
                            with cols[0]:
                                filter_cols['Priority'] = st.selectbox('Priority', ['All'] + list(df['Priority'].unique()))
                                if filter_cols['Priority'] != 'All': filters_applied = True
                        if 'Status' in df.columns:
                            with cols[1]:
                                filter_cols['Status'] = st.selectbox('Status', ['All'] + list(df['Status'].unique()))
                                if filter_cols['Status'] != 'All': filters_applied = True
                        if 'Owner (Team / Client)' in df.columns:
                            with cols[2]:
                                filter_cols['Owner'] = st.selectbox('Owner', ['All'] + list(df['Owner (Team / Client)'].unique()))
                                if filter_cols['Owner'] != 'All': filters_applied = True
                    
                    elif tab == 'DAILY_WORK_LOG':
                        cols = st.columns(2)
                        filter_cols = {}
                        if 'Employee Name' in df.columns:
                            with cols[0]:
                                filter_cols['Employee'] = st.selectbox('Employee', ['All'] + list(df['Employee Name'].unique()))
                                if filter_cols['Employee'] != 'All': filters_applied = True
                        if 'Project' in df.columns:
                            with cols[1]:
                                filter_cols['Project'] = st.selectbox('Project', ['All'] + list(df['Project'].unique()))
                                if filter_cols['Project'] != 'All': filters_applied = True
                    
                    elif tab == 'EMPLOYEE_COST':
                        cols = st.columns(2)
                        filter_cols = {}
                        if 'Role' in df.columns:
                            with cols[0]:
                                filter_cols['Role'] = st.selectbox('Role', ['All'] + list(df['Role'].unique()))
                                if filter_cols['Role'] != 'All': filters_applied = True
                        if 'Department' in df.columns:
                            with cols[1]:
                                filter_cols['Department'] = st.selectbox('Department', ['All'] + list(df['Department'].unique()))
                                if filter_cols['Department'] != 'All': filters_applied = True
                    
                    elif tab == 'RESOURCE_LINKS':
                        cols = st.columns(2)
                        filter_cols = {}
                        if 'Category' in df.columns:
                            with cols[0]:
                                filter_cols['Category'] = st.selectbox('Category', ['All'] + list(df['Category'].unique()))
                                if filter_cols['Category'] != 'All': filters_applied = True
                        if 'Type' in df.columns:
                            with cols[1]:
                                filter_cols['Type'] = st.selectbox('Type', ['All'] + list(df['Type'].unique()))
                                if filter_cols['Type'] != 'All': filters_applied = True
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Apply filters
                filtered = df.copy()
                if filters_applied:
                    for key, col in filter_cols.items():
                        if col != 'All':
                            if key == 'Company' and 'Company name' in df.columns:
                                filtered = filtered[filtered['Company name'] == col]
                            elif key == 'Owner' and 'Owner (Team / Client)' in df.columns:
                                filtered = filtered[filtered['Owner (Team / Client)'] == col]
                            elif key in ['Priority', 'Status', 'Quarter', 'Employee', 'Project', 'Role', 'Department', 'Category', 'Type']:
                                col_name = 'Employee Name' if key == 'Employee' else key
                                if col_name in df.columns:
                                    filtered = filtered[filtered[col_name] == col]
                
                # Sub Scorecards based on filtered data
                st.markdown('<div class="sub-scorecard-container">', unsafe_allow_html=True)
                st.markdown("##### 📊 Current View Metrics")
                
                if tab == 'PROJECT_MASTER':
                    cols = st.columns(4)
                    with cols[0]:
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{len(filtered)}</div>
                            <div class="sub-label">Projects</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        active = len(filtered[filtered['Status']=='Active']) if 'Status' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{active}</div>
                            <div class="sub-label">Active</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[2]:
                        budget = filtered['Budget'].sum() if 'Budget' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">${budget:,.0f}</div>
                            <div class="sub-label">Total Budget</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[3]:
                        companies = filtered['Company name'].nunique() if 'Company name' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{companies}</div>
                            <div class="sub-label">Companies</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                elif tab == 'TASK_PLAN':
                    cols = st.columns(4)
                    with cols[0]:
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{len(filtered)}</div>
                            <div class="sub-label">Tasks</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        pending = len(filtered[filtered['Status']!='Done']) if 'Status' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{pending}</div>
                            <div class="sub-label">Pending</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[2]:
                        completed = len(filtered[filtered['Status']=='Done']) if 'Status' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{completed}</div>
                            <div class="sub-label">Completed</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[3]:
                        high_priority = len(filtered[filtered['Priority']=='High']) if 'Priority' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{high_priority}</div>
                            <div class="sub-label">High Priority</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                elif tab == 'DAILY_WORK_LOG':
                    cols = st.columns(4)
                    with cols[0]:
                        total_hours = filtered['Hours Worked'].sum() if 'Hours Worked' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{total_hours:.0f}</div>
                            <div class="sub-label">Total Hours</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        employees = filtered['Employee Name'].nunique() if 'Employee Name' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{employees}</div>
                            <div class="sub-label">Employees</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[2]:
                        avg_hours = filtered['Hours Worked'].mean() if 'Hours Worked' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{avg_hours:.1f}</div>
                            <div class="sub-label">Avg Hours</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[3]:
                        days = filtered['Date'].nunique() if 'Date' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{days}</div>
                            <div class="sub-label">Days Logged</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                elif tab == 'EMPLOYEE_COST':
                    cols = st.columns(4)
                    with cols[0]:
                        employees = len(filtered)
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{employees}</div>
                            <div class="sub-label">Employees</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        total_salary = filtered['Monthly Salary'].sum() if 'Monthly Salary' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">${total_salary:,.0f}</div>
                            <div class="sub-label">Monthly Salary</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[2]:
                        avg_salary = filtered['Monthly Salary'].mean() if 'Monthly Salary' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">${avg_salary:,.0f}</div>
                            <div class="sub-label">Avg Salary</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[3]:
                        roles = filtered['Role'].nunique() if 'Role' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{roles}</div>
                            <div class="sub-label">Roles</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                elif tab == 'RESOURCE_LINKS':
                    cols = st.columns(4)
                    with cols[0]:
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{len(filtered)}</div>
                            <div class="sub-label">Resources</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[1]:
                        categories = filtered['Category'].nunique() if 'Category' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{categories}</div>
                            <div class="sub-label">Categories</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[2]:
                        types = filtered['Type'].nunique() if 'Type' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{types}</div>
                            <div class="sub-label">Types</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[3]:
                        st.markdown(f"""
                        <div class="sub-scorecard">
                            <div class="sub-value">{'Active'}</div>
                            <div class="sub-label">Status</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Charts
                with st.container():
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    
                    if tab == 'PROJECT_MASTER' and 'Company name' in filtered.columns and 'Budget' in filtered.columns:
                        chart_data = filtered.groupby('Company name')['Budget'].sum().reset_index()
                        fig = px.bar(chart_data, x='Company name', y='Budget', title="Budget by Company",
                                   color_discrete_sequence=['#3b82f6'])
                        fig.update_layout(plot_bgcolor='white', height=300, margin=dict(t=30))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif tab == 'TASK_PLAN' and 'Owner (Team / Client)' in filtered.columns:
                        chart_data = filtered['Owner (Team / Client)'].value_counts().reset_index()
                        chart_data.columns = ['Owner', 'Count']
                        fig = px.pie(chart_data, values='Count', names='Owner', title="Tasks by Owner",
                                   color_discrete_sequence=px.colors.qualitative.Set3)
                        fig.update_layout(height=300, margin=dict(t=30))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif tab == 'DAILY_WORK_LOG' and 'Employee Name' in filtered.columns and 'Hours Worked' in filtered.columns:
                        chart_data = filtered.groupby('Employee Name')['Hours Worked'].sum().reset_index()
                        fig = px.bar(chart_data, x='Employee Name', y='Hours Worked', title="Hours by Employee",
                                   color_discrete_sequence=['#3b82f6'])
                        fig.update_layout(plot_bgcolor='white', height=300, margin=dict(t=30))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif tab == 'EMPLOYEE_COST' and 'Role' in filtered.columns and 'Monthly Salary' in filtered.columns:
                        chart_data = filtered.groupby('Role')['Monthly Salary'].sum().reset_index()
                        fig = px.bar(chart_data, x='Role', y='Monthly Salary', title="Salary by Role",
                                   color='Role', color_discrete_sequence=px.colors.qualitative.Set3)
                        fig.update_layout(plot_bgcolor='white', height=300, margin=dict(t=30))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif tab == 'RESOURCE_LINKS' and 'Category' in filtered.columns:
                        chart_data = filtered['Category'].value_counts().reset_index()
                        chart_data.columns = ['Category', 'Count']
                        fig = px.pie(chart_data, values='Count', names='Category', title="Resources by Category",
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig.update_layout(height=300, margin=dict(t=30))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Details
                with st.expander("📋 Detailed View", expanded=True):
                    st.dataframe(filtered, use_container_width=True, hide_index=True)

# --- Auto-refresh ---
st.markdown(f'<meta http-equiv="refresh" content="{REFRESH_INTERVAL}">', unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #94a3b8; font-size: 0.75rem;'>Project Pulse • Real-time Analytics • v3.0</div>", unsafe_allow_html=True)
