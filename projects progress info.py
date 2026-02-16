import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import pytz

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

# --- Custom CSS ---
st.markdown("""
<style>
    .header-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem 2rem; border-radius: 20px; margin-bottom: 1rem; color: white; display: flex; align-items: center; justify-content: space-between; }
    .header-title { margin: 0; font-size: 2rem; }
    .header-subtitle { margin: 0.2rem 0 0 0; display: flex; align-items: center; gap: 0.5rem; }
    .live-badge { background: #ff4444; color: white; padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
    .timestamp { color: white; font-size: 0.9rem; display: flex; align-items: center; gap: 0.5rem; }
    .pk-badge { background: #2c3e50; color: white; padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.75rem; }
    .header-scorecards { display: flex; gap: 1rem; }
    .header-scorecard { background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 12px; text-align: center; min-width: 90px; backdrop-filter: blur(10px); }
    .header-scorecard-value { font-size: 1.2rem; font-weight: 700; color: white; }
    .header-scorecard-label { font-size: 0.7rem; color: rgba(255,255,255,0.9); text-transform: uppercase; }
    .header-scorecard-sub { font-size: 0.65rem; color: rgba(255,255,255,0.8); }
    .secondary-scorecards { background: #f8f9fa; padding: 1rem; border-radius: 15px; margin: 1rem 0; border: 1px solid #dee2e6; }
    .scorecard { background: white; padding: 0.8rem; border-radius: 10px; text-align: center; border-left: 4px solid #667eea; }
    .scorecard-value { font-size: 1.5rem; font-weight: 700; color: #2d3748; }
    .scorecard-label { font-size: 0.8rem; color: #718096; }
    .tab-container { background: white; border-radius: 15px; padding: 1rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    .stButton > button { width: 100%; text-align: left; border-radius: 10px; margin: 0.2rem 0; }
    .reset-btn { background: #ff4444; color: white; }
    #MainMenu, footer, .stDeployButton { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

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
            <div class="header-scorecard"><div class="header-scorecard-value">{metrics['total_projects']}</div><div class="header-scorecard-label">Projects</div><div class="header-scorecard-sub">{metrics['active_projects']} Active</div></div>
            <div class="header-scorecard"><div class="header-scorecard-value">{metrics['total_hours']:.0f}</div><div class="header-scorecard-label">Hours</div><div class="header-scorecard-sub">Work Log</div></div>
            <div class="header-scorecard"><div class="header-scorecard-value">${metrics['total_salary']:,.0f}</div><div class="header-scorecard-label">Salary</div><div class="header-scorecard-sub">Monthly</div></div>
            <div class="header-scorecard"><div class="header-scorecard-value">{metrics['total_tasks']}</div><div class="header-scorecard-label">Tasks</div><div class="header-scorecard-sub">{metrics['pending_tasks']} Pending</div></div>
            <div class="header-scorecard"><div class="header-scorecard-value">{metrics['total_resources']}</div><div class="header-scorecard-label">Resources</div><div class="header-scorecard-sub">Links</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="header-container">
        <div><h1 class="header-title">📊 Project Pulse</h1><p class="header-subtitle">Original Sheets View</p></div>
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
            
            # Filters and Secondary Scorecards
            with st.expander("🔍 Filters", expanded=True):
                cols = st.columns([3, 1])
                with cols[0]:
                    if current == 'PROJECT_MASTER':
                        filter_cols = st.columns(3)
                        with filter_cols[0]:
                            companies = ['All'] + list(df['Company name'].unique()) if 'Company name' in df.columns else ['All']
                            st.session_state.filters['company'] = st.selectbox('Company', companies, key='f_comp')
                        with filter_cols[1]:
                            statuses = ['All'] + list(df['Status'].unique()) if 'Status' in df.columns else ['All']
                            st.session_state.filters['status'] = st.selectbox('Status', statuses, key='f_stat')
                        with filter_cols[2]:
                            quarters = ['All'] + list(df['Quarter'].unique()) if 'Quarter' in df.columns else ['All']
                            st.session_state.filters['quarter'] = st.selectbox('Quarter', quarters, key='f_quart')
                    
                    elif current == 'TASK_PLAN':
                        filter_cols = st.columns(3)
                        with filter_cols[0]:
                            priorities = ['All'] + list(df['Priority'].unique()) if 'Priority' in df.columns else ['All']
                            st.session_state.filters['priority'] = st.selectbox('Priority', priorities, key='f_pri')
                        with filter_cols[1]:
                            statuses = ['All'] + list(df['Status'].unique()) if 'Status' in df.columns else ['All']
                            st.session_state.filters['status'] = st.selectbox('Status', statuses, key='f_stat')
                        with filter_cols[2]:
                            owners = ['All'] + list(df['Owner (Team / Client)'].unique()) if 'Owner (Team / Client)' in df.columns else ['All']
                            st.session_state.filters['owner'] = st.selectbox('Owner', owners, key='f_own')
                    
                    elif current == 'DAILY_WORK_LOG':
                        employees = ['All'] + list(df['Employee Name'].unique()) if 'Employee Name' in df.columns else ['All']
                        st.session_state.filters['employee'] = st.selectbox('Employee', employees, key='f_emp')
                    
                    elif current == 'EMPLOYEE_COST':
                        roles = ['All'] + list(df['Role'].unique()) if 'Role' in df.columns else ['All']
                        st.session_state.filters['role'] = st.selectbox('Role', roles, key='f_role')
                
                with cols[1]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🔄 Reset Filters", use_container_width=True):
                        st.session_state.filters = {}
                        st.rerun()
            
            # Apply filters
            filtered = df.copy()
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
            
            # Secondary Scorecards
            if not filtered.empty:
                st.markdown('<div class="secondary-scorecards">', unsafe_allow_html=True)
                st.markdown("##### 📊 Filtered View Metrics")
                
                if current == 'PROJECT_MASTER':
                    cols = st.columns(4)
                    with cols[0]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered)}</div><div class='scorecard-label'>Projects</div></div>", unsafe_allow_html=True)
                    with cols[1]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>${filtered['Budget'].sum():,.0f}</div><div class='scorecard-label'>Total Budget</div></div>" if 'Budget' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Budget</div></div>", unsafe_allow_html=True)
                    with cols[2]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered[filtered['Status']=='Active'])}</div><div class='scorecard-label'>Active</div></div>" if 'Status' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Active</div></div>", unsafe_allow_html=True)
                    with cols[3]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Company name'].nunique()}</div><div class='scorecard-label'>Companies</div></div>" if 'Company name' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Companies</div></div>", unsafe_allow_html=True)
                
                elif current == 'TASK_PLAN':
                    cols = st.columns(4)
                    with cols[0]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered)}</div><div class='scorecard-label'>Tasks</div></div>", unsafe_allow_html=True)
                    with cols[1]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered[filtered['Status']=='Done'])}</div><div class='scorecard-label'>Completed</div></div>" if 'Status' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Completed</div></div>", unsafe_allow_html=True)
                    with cols[2]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered[filtered['Priority']=='High'])}</div><div class='scorecard-label'>High Priority</div></div>" if 'Priority' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>High Priority</div></div>", unsafe_allow_html=True)
                    with cols[3]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Owner (Team / Client)'].nunique()}</div><div class='scorecard-label'>Owners</div></div>" if 'Owner (Team / Client)' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Owners</div></div>", unsafe_allow_html=True)
                
                elif current == 'DAILY_WORK_LOG':
                    cols = st.columns(4)
                    with cols[0]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered)}</div><div class='scorecard-label'>Entries</div></div>", unsafe_allow_html=True)
                    with cols[1]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Hours Worked'].sum():.1f}</div><div class='scorecard-label'>Total Hours</div></div>" if 'Hours Worked' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Hours</div></div>", unsafe_allow_html=True)
                    with cols[2]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Hours Worked'].mean():.1f}</div><div class='scorecard-label'>Avg Hours</div></div>" if 'Hours Worked' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Avg Hours</div></div>", unsafe_allow_html=True)
                    with cols[3]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Employee Name'].nunique()}</div><div class='scorecard-label'>Employees</div></div>" if 'Employee Name' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Employees</div></div>", unsafe_allow_html=True)
                
                elif current == 'EMPLOYEE_COST':
                    cols = st.columns(4)
                    with cols[0]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered)}</div><div class='scorecard-label'>Employees</div></div>", unsafe_allow_html=True)
                    with cols[1]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>${filtered['Monthly Salary'].sum():,.0f}</div><div class='scorecard-label'>Total Salary</div></div>" if 'Monthly Salary' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>$0</div><div class='scorecard-label'>Salary</div></div>", unsafe_allow_html=True)
                    with cols[2]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>${filtered['Monthly Salary'].mean():,.0f}</div><div class='scorecard-label'>Avg Salary</div></div>" if 'Monthly Salary' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>$0</div><div class='scorecard-label'>Avg Salary</div></div>", unsafe_allow_html=True)
                    with cols[3]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Role'].nunique()}</div><div class='scorecard-label'>Roles</div></div>" if 'Role' in filtered.columns else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Roles</div></div>", unsafe_allow_html=True)
                
                elif current == 'RESOURCE_LINKS':
                    cols = st.columns(3)
                    with cols[0]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered)}</div><div class='scorecard-label'>Resources</div></div>", unsafe_allow_html=True)
                    with cols[1]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Type'].nunique() if 'Type' in filtered.columns else 0}</div><div class='scorecard-label'>Types</div></div>", unsafe_allow_html=True)
                    with cols[2]: st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Category'].nunique() if 'Category' in filtered.columns else 0}</div><div class='scorecard-label'>Categories</div></div>", unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Charts
                if current == 'PROJECT_MASTER' and 'Company name' in filtered.columns and 'Budget' in filtered.columns:
                    fig = px.bar(filtered.groupby('Company name')['Budget'].sum().reset_index(), x='Company name', y='Budget', title="Budget by Company", color_discrete_sequence=['#667eea'])
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif current == 'TASK_PLAN' and 'Owner (Team / Client)' in filtered.columns:
                    fig = px.pie(filtered['Owner (Team / Client)'].value_counts().reset_index(), values='count', names='Owner (Team / Client)', title="Tasks by Owner", color_discrete_sequence=px.colors.sequential.Viridis)
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif current == 'DAILY_WORK_LOG' and 'Employee Name' in filtered.columns and 'Hours Worked' in filtered.columns:
                    fig = px.bar(filtered.groupby('Employee Name')['Hours Worked'].sum().reset_index(), x='Employee Name', y='Hours Worked', title="Hours by Employee", color_discrete_sequence=['#667eea'])
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif current == 'EMPLOYEE_COST' and 'Role' in filtered.columns and 'Monthly Salary' in filtered.columns:
                    fig = px.bar(filtered.groupby('Role')['Monthly Salary'].sum().reset_index(), x='Role', y='Monthly Salary', title="Salary by Role", color_discrete_sequence=['#667eea'])
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                with st.expander("📋 Details", expanded=True):
                    st.dataframe(filtered, use_container_width=True, hide_index=True)
        else:
            st.info("📭 No data available")

# --- Auto-refresh and Footer ---
st.markdown(f'<meta http-equiv="refresh" content="{REFRESH_INTERVAL}">', unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #999; font-size: 0.8rem;'>Project Pulse • Auto-refreshes every {REFRESH_INTERVAL}s</div>", unsafe_allow_html=True)
