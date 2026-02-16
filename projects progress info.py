import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import pytz
import time

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Project Pulse",
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
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'data_sheets' not in st.session_state:
    st.session_state.data_sheets = None
if 'show_original' not in st.session_state:
    st.session_state.show_original = False
if 'filters' not in st.session_state:
    st.session_state.filters = {
        'PROJECT_MASTER': {'company': 'All', 'status': 'All', 'quarter': 'All'},
        'TASK_PLAN': {'priority': 'All', 'status': 'All'},
        'DAILY_WORK_LOG': {'employee': 'All'},
        'EMPLOYEE_COST': {},
        'RESOURCE_LINKS': {}
    }

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
if last_update:
    st.session_state.last_update = last_update
    st.session_state.data_sheets = data_sheets

# --- Custom CSS for Modern Look ---
st.markdown("""
<style>
    /* Modern gradient header */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .header-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .header-right {
        display: flex;
        align-items: center;
        gap: 2rem;
    }
    
    .header-title {
        margin: 0;
        font-size: 2rem;
    }
    
    .header-subtitle {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Status bar inside header */
    .header-status {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 50px;
        display: flex;
        align-items: center;
        gap: 1rem;
        backdrop-filter: blur(10px);
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
        color: white;
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
    
    /* Scorecards banner inside header */
    .header-scorecards {
        display: flex;
        gap: 1rem;
    }
    
    .header-scorecard {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 12px;
        text-align: center;
        min-width: 100px;
        backdrop-filter: blur(10px);
        border-left: 4px solid rgba(255,255,255,0.5);
    }
    
    .header-scorecard-value {
        font-size: 1.2rem;
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
    
    /* Secondary Scorecards banner */
    .scorecard-banner {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .scorecard {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        border-left: 4px solid #667eea;
    }
    
    .scorecard:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .scorecard-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        line-height: 1.2;
    }
    
    .scorecard-label {
        font-size: 0.85rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Filter section styling */
    .filter-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .reset-button {
        background: linear-gradient(135deg, #f56565 0%, #c53030 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        border: none;
        cursor: pointer;
        font-size: 0.9rem;
        transition: all 0.2s;
    }
    
    .reset-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(245, 101, 101, 0.3);
    }
    
    /* Modern tabs */
    .tab-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .nav-header {
        padding: 0.5rem 1rem;
        color: #4a5568;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button {
        width: 100%;
        text-align: left;
        border-radius: 10px;
        margin: 0.25rem 0;
        border: none;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateX(5px);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        border: 1px solid #f0f0f0;
        font-weight: 500;
    }
    
    /* View original button */
    .view-original-btn {
        margin-top: 1rem;
        padding: 0.5rem;
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        border-radius: 10px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .view-original-btn:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(72, 187, 120, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- Header with Integrated Status and Scorecards ---
if st.session_state.data_sheets and not st.session_state.show_original:
    df_projects = st.session_state.data_sheets['PROJECT_MASTER']
    df_work = st.session_state.data_sheets['DAILY_WORK_LOG']
    df_cost = st.session_state.data_sheets['EMPLOYEE_COST']
    df_tasks = st.session_state.data_sheets['TASK_PLAN']
    df_resources = st.session_state.data_sheets['RESOURCE_LINKS']
    
    # Calculate metrics
    total_projects = len(df_projects) if not df_projects.empty else 0
    active_projects = len(df_projects[df_projects['Status'] == 'Active']) if not df_projects.empty and 'Status' in df_projects.columns else 0
    total_hours = df_work['Hours Worked'].sum() if not df_work.empty else 0
    total_salary = df_cost['Monthly Salary'].sum() if not df_cost.empty and 'Monthly Salary' in df_cost.columns else 0
    total_tasks = len(df_tasks) if not df_tasks.empty else 0
    pending_tasks = len(df_tasks[df_tasks['Status'] != 'Done']) if not df_tasks.empty and 'Status' in df_tasks.columns else 0
    total_resources = len(df_resources) if not df_resources.empty else 0
    
    timestamp_str = st.session_state.last_update.strftime("%a, %d %b, %Y, %I:%M:%S %p") if st.session_state.last_update else ""
    
    st.markdown(f"""
    <div class="header-container">
        <div class="header-left">
            <div>
                <h1 class="header-title">📊 Project Pulse</h1>
                <p class="header-subtitle">
                    <span class="live-badge">LIVE</span>
                    <span class="timestamp">
                        <span>🔄 {timestamp_str}</span>
                        <span class="pk-badge">PKT</span>
                    </span>
                </p>
            </div>
        </div>
        <div class="header-scorecards">
            <div class="header-scorecard">
                <div class="header-scorecard-value">{total_projects}</div>
                <div class="header-scorecard-label">Projects</div>
                <div class="header-scorecard-sub">{active_projects} Active</div>
            </div>
            <div class="header-scorecard">
                <div class="header-scorecard-value">{total_hours:.0f}</div>
                <div class="header-scorecard-label">Hours</div>
                <div class="header-scorecard-sub">Work Log</div>
            </div>
            <div class="header-scorecard">
                <div class="header-scorecard-value">${total_salary:,.0f}</div>
                <div class="header-scorecard-label">Salary</div>
                <div class="header-scorecard-sub">Monthly</div>
            </div>
            <div class="header-scorecard">
                <div class="header-scorecard-value">{total_tasks}</div>
                <div class="header-scorecard-label">Tasks</div>
                <div class="header-scorecard-sub">{pending_tasks} Pending</div>
            </div>
            <div class="header-scorecard">
                <div class="header-scorecard-value">{total_resources}</div>
                <div class="header-scorecard-label">Resources</div>
                <div class="header-scorecard-sub">Links</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Simple header for original sheets view
    st.markdown("""
    <div class="header-container">
        <div class="header-left">
            <div>
                <h1 class="header-title">📊 Project Pulse</h1>
                <p class="header-subtitle">Original Sheets View</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Main Layout ---
if st.session_state.show_original:
    # Show Original Sheets View
    st.markdown("### 📋 Original Sheets Data")
    
    if st.button("← Back to Dashboard", type="primary"):
        st.session_state.show_original = False
        st.rerun()
    
    tabs = st.tabs(list(st.session_state.data_sheets.keys()))
    for i, (sheet_name, df) in enumerate(st.session_state.data_sheets.items()):
        with tabs[i]:
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Download button for each sheet
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {sheet_name} as CSV",
                data=csv,
                file_name=f"{sheet_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
else:
    # Normal Dashboard View
    left_col, right_col = st.columns([1, 4])

    # --- Left Column: Navigation ---
    with left_col:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        st.markdown('<div class="nav-header">📋 Navigation</div>', unsafe_allow_html=True)
        
        tabs = {
            'PROJECT_MASTER': ('📁', 'Projects'),
            'DAILY_WORK_LOG': ('📝', 'Work Log'),
            'EMPLOYEE_COST': ('💰', 'Costs'),
            'RESOURCE_LINKS': ('🔗', 'Resources'),
            'TASK_PLAN': ('✅', 'Tasks')
        }
        
        for tab_key, (icon, label) in tabs.items():
            if st.button(
                f"{icon} {label}", 
                key=f"nav_{tab_key}", 
                use_container_width=True,
                type="secondary" if st.session_state.selected_tab != tab_key else "primary"
            ):
                st.session_state.selected_tab = tab_key
                st.rerun()
        
        # View Original Sheets Option (below Tasks)
        st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
        if st.button("📋 View Original Sheets", key="view_original", use_container_width=True, type="secondary"):
            st.session_state.show_original = True
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Right Column: Content ---
    with right_col:
        current_tab = st.session_state.selected_tab
        st.markdown(f"### {tabs[current_tab][1]} Dashboard")
        
        if st.session_state.data_sheets:
            df = st.session_state.data_sheets[current_tab]
            
            if df.empty:
                st.info("📭 No data available")
            else:
                # Initialize filtered dataframe
                filtered = df.copy()
                
                # Filters and Secondary Scorecards based on current tab
                if current_tab == 'PROJECT_MASTER':
                    # Filter section with reset button
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    with col1:
                        companies = ['All'] + list(df['Company name'].unique()) if 'Company name' in df.columns else ['All']
                        company = st.selectbox('Company', companies, 
                                             index=0 if st.session_state.filters[current_tab]['company'] == 'All' 
                                             else companies.index(st.session_state.filters[current_tab]['company']) 
                                             if st.session_state.filters[current_tab]['company'] in companies else 0,
                                             key='filter_company')
                        st.session_state.filters[current_tab]['company'] = company
                    with col2:
                        statuses = ['All'] + list(df['Status'].unique()) if 'Status' in df.columns else ['All']
                        status = st.selectbox('Status', statuses,
                                            index=0 if st.session_state.filters[current_tab]['status'] == 'All'
                                            else statuses.index(st.session_state.filters[current_tab]['status'])
                                            if st.session_state.filters[current_tab]['status'] in statuses else 0,
                                            key='filter_status')
                        st.session_state.filters[current_tab]['status'] = status
                    with col3:
                        quarters = ['All'] + list(df['Quarter'].unique()) if 'Quarter' in df.columns else ['All']
                        quarter = st.selectbox('Quarter', quarters,
                                             index=0 if st.session_state.filters[current_tab]['quarter'] == 'All'
                                             else quarters.index(st.session_state.filters[current_tab]['quarter'])
                                             if st.session_state.filters[current_tab]['quarter'] in quarters else 0,
                                             key='filter_quarter')
                        st.session_state.filters[current_tab]['quarter'] = quarter
                    with col4:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🔄 Reset", key="reset_projects", use_container_width=True):
                            st.session_state.filters[current_tab] = {'company': 'All', 'status': 'All', 'quarter': 'All'}
                            st.rerun()
                    
                    # Apply filters
                    if company != 'All': filtered = filtered[filtered['Company name'] == company]
                    if status != 'All': filtered = filtered[filtered['Status'] == status]
                    if quarter != 'All': filtered = filtered[filtered['Quarter'] == quarter]
                    
                    # Secondary Scorecards for Projects
                    st.markdown('<div class="scorecard-banner">', unsafe_allow_html=True)
                    st.markdown("##### 📊 Filtered Projects Overview")
                    cols = st.columns(4)
                    
                    with cols[0]:
                        total_filtered = len(filtered)
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">{total_filtered}</div>
                            <div class="scorecard-label">Filtered Projects</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[1]:
                        active_filtered = len(filtered[filtered['Status'] == 'Active']) if 'Status' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">{active_filtered}</div>
                            <div class="scorecard-label">Active Projects</div>
                            <div style="font-size:0.8rem; color:#48bb78;">{((active_filtered/total_filtered*100) if total_filtered>0 else 0):.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[2]:
                        total_budget = filtered['Budget'].sum() if 'Budget' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">${total_budget:,.0f}</div>
                            <div class="scorecard-label">Total Budget</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[3]:
                        avg_budget = filtered['Budget'].mean() if 'Budget' in filtered.columns and len(filtered) > 0 else 0
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">${avg_budget:,.0f}</div>
                            <div class="scorecard-label">Avg Budget</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Chart
                    if 'Company name' in filtered.columns and 'Budget' in filtered.columns:
                        chart_data = filtered.groupby('Company name')['Budget'].sum().reset_index()
                        fig = px.bar(chart_data, x='Company name', y='Budget', 
                                   title="Budget by Company (Filtered)",
                                   color_discrete_sequence=['#667eea'])
                        fig.update_layout(plot_bgcolor='white', height=350, margin=dict(t=30))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    with st.expander("📋 Details", expanded=True):
                        st.dataframe(filtered, use_container_width=True, hide_index=True)
                
                # Tasks Tab
                elif current_tab == 'TASK_PLAN':
                    # Filter section with reset button
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        priorities = ['All'] + list(df['Priority'].unique()) if 'Priority' in df.columns else ['All']
                        priority = st.selectbox('Priority', priorities,
                                              index=0 if st.session_state.filters[current_tab]['priority'] == 'All'
                                              else priorities.index(st.session_state.filters[current_tab]['priority'])
                                              if st.session_state.filters[current_tab]['priority'] in priorities else 0,
                                              key='task_priority')
                        st.session_state.filters[current_tab]['priority'] = priority
                    with col2:
                        statuses = ['All'] + list(df['Status'].unique()) if 'Status' in df.columns else ['All']
                        status = st.selectbox('Status', statuses,
                                            index=0 if st.session_state.filters[current_tab]['status'] == 'All'
                                            else statuses.index(st.session_state.filters[current_tab]['status'])
                                            if st.session_state.filters[current_tab]['status'] in statuses else 0,
                                            key='task_status')
                        st.session_state.filters[current_tab]['status'] = status
                    with col3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🔄 Reset", key="reset_tasks", use_container_width=True):
                            st.session_state.filters[current_tab] = {'priority': 'All', 'status': 'All'}
                            st.rerun()
                    
                    # Apply filters
                    if priority != 'All': filtered = filtered[filtered['Priority'] == priority]
                    if status != 'All': filtered = filtered[filtered['Status'] == status]
                    
                    # Secondary Scorecards for Tasks
                    st.markdown('<div class="scorecard-banner">', unsafe_allow_html=True)
                    st.markdown("##### 📊 Filtered Tasks Overview")
                    cols = st.columns(4)
                    
                    with cols[0]:
                        total_filtered = len(filtered)
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">{total_filtered}</div>
                            <div class="scorecard-label">Filtered Tasks</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[1]:
                        completed = len(filtered[filtered['Status'] == 'Done']) if 'Status' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">{completed}</div>
                            <div class="scorecard-label">Completed</div>
                            <div style="font-size:0.8rem; color:#48bb78;">{((completed/total_filtered*100) if total_filtered>0 else 0):.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[2]:
                        high_priority = len(filtered[filtered['Priority'] == 'High']) if 'Priority' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">{high_priority}</div>
                            <div class="scorecard-label">High Priority</div>
                            <div style="font-size:0.8rem; color:#f56565;">Critical</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[3]:
                        in_progress = len(filtered[filtered['Status'] == 'In Progress']) if 'Status' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">{in_progress}</div>
                            <div class="scorecard-label">In Progress</div>
                            <div style="font-size:0.8rem; color:#f59e0b;">Active</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Chart
                    if 'Owner (Team / Client)' in filtered.columns:
                        chart_data = filtered['Owner (Team / Client)'].value_counts().reset_index()
                        chart_data.columns = ['Owner', 'Count']
                        fig = px.pie(chart_data, values='Count', names='Owner',
                                   title="Tasks by Owner (Filtered)",
                                   color_discrete_sequence=px.colors.sequential.Viridis)
                        fig.update_layout(height=350, margin=dict(t=30))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    with st.expander("📋 Details", expanded=True):
                        st.dataframe(filtered, use_container_width=True, hide_index=True)
                
                # Work Log Tab
                elif current_tab == 'DAILY_WORK_LOG':
                    # Filter section with reset button
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        employees = ['All'] + list(df['Employee Name'].unique()) if 'Employee Name' in df.columns else ['All']
                        employee = st.selectbox('Employee', employees,
                                              index=0 if st.session_state.filters[current_tab]['employee'] == 'All'
                                              else employees.index(st.session_state.filters[current_tab]['employee'])
                                              if st.session_state.filters[current_tab]['employee'] in employees else 0,
                                              key='work_employee')
                        st.session_state.filters[current_tab]['employee'] = employee
                    with col2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🔄 Reset", key="reset_work", use_container_width=True):
                            st.session_state.filters[current_tab] = {'employee': 'All'}
                            st.rerun()
                    
                    # Apply filters
                    if employee != 'All': filtered = filtered[filtered['Employee Name'] == employee]
                    
                    # Secondary Scorecards for Work Log
                    st.markdown('<div class="scorecard-banner">', unsafe_allow_html=True)
                    st.markdown("##### 📊 Filtered Work Log Overview")
                    cols = st.columns(4)
                    
                    with cols[0]:
                        total_entries = len(filtered)
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">{total_entries}</div>
                            <div class="scorecard-label">Total Entries</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[1]:
                        total_hours_filtered = filtered['Hours Worked'].sum() if 'Hours Worked' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">{total_hours_filtered:.1f}</div>
                            <div class="scorecard-label">Total Hours</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[2]:
                        avg_hours = filtered['Hours Worked'].mean() if 'Hours Worked' in filtered.columns and len(filtered) > 0 else 0
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">{avg_hours:.1f}</div>
                            <div class="scorecard-label">Avg Hours/Day</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[3]:
                        unique_employees = filtered['Employee Name'].nunique() if 'Employee Name' in filtered.columns else 0
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">{unique_employees}</div>
                            <div class="scorecard-label">Employees</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Chart
                    if 'Employee Name' in filtered.columns:
                        chart_data = filtered.groupby('Employee Name')['Hours Worked'].sum().reset_index()
                        fig = px.bar(chart_data, x='Employee Name', y='Hours Worked',
                                   title="Hours by Employee (Filtered)",
                                   color_discrete_sequence=['#667eea'])
                        fig.update_layout(height=350, margin=dict(t=30))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    with st.expander("📋 Details", expanded=True):
                        st.dataframe(filtered, use_container_width=True, hide_index=True)
                
                # Employee Cost Tab
                elif current_tab == 'EMPLOYEE_COST':
                    # Secondary Scorecards for Employee Cost
                    st.markdown('<div class="scorecard-banner">', unsafe_allow_html=True)
                    st.markdown("##### 📊 Employee Cost Overview")
                    cols = st.columns(4)
                    
                    with cols[0]:
                        total_employees = len(df)
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">{total_employees}</div>
                            <div class="scorecard-label">Total Employees</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[1]:
                        total_salary_filtered = df['Monthly Salary'].sum() if 'Monthly Salary' in df.columns else 0
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">${total_salary_filtered:,.0f}</div>
                            <div class="scorecard-label">Monthly Salary</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[2]:
                        avg_salary = df['Monthly Salary'].mean() if 'Monthly Salary' in df.columns and len(df) > 0 else 0
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">${avg_salary:,.0f}</div>
                            <div class="scorecard-label">Avg Salary</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[3]:
                        unique_roles = df['Role'].nunique() if 'Role' in df.columns else 0
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">{unique_roles}</div>
                            <div class="scorecard-label">Unique Roles</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Chart
                    if 'Role' in df.columns and 'Monthly Salary' in df.columns:
                        chart_data = df.groupby('Role')['Monthly Salary'].sum().reset_index()
                        fig = px.bar(chart_data, x='Role', y='Monthly Salary',
                                   title="Salary by Role",
                                   color_discrete_sequence=['#667eea'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Data table
                    with st.expander("📋 Details", expanded=True):
                        st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Resource Links Tab
                elif current_tab == 'RESOURCE_LINKS':
                    # Secondary Scorecards for Resources
                    st.markdown('<div class="scorecard-banner">', unsafe_allow_html=True)
                    st.markdown("##### 📊 Resources Overview")
                    cols = st.columns(3)
                    
                    with cols[0]:
                        total_resources_filtered = len(df)
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">{total_resources_filtered}</div>
                            <div class="scorecard-label">Total Resources</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with cols[1]:
                        # Count by type if available
                        if 'Type' in df.columns:
                            types = df['Type'].nunique()
                            st.markdown(f"""
                            <div class="scorecard">
                                <div class="scorecard-value">{types}</div>
                                <div class="scorecard-label">Resource Types</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="scorecard">
                                <div class="scorecard-value">-</div>
                                <div class="scorecard-label">Resource Types</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with cols[2]:
                        # Most recent if date column exists
                        st.markdown(f"""
                        <div class="scorecard">
                            <div class="scorecard-value">🔗</div>
                            <div class="scorecard-label">Click to View</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.info("🔗 Click on links in the table")
                    with st.expander("📋 Details", expanded=True):
                        st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.error("Failed to load data. Please check your connection.")

# --- Auto-refresh using meta tag ---
st.markdown(f"""
    <meta http-equiv="refresh" content="{REFRESH_INTERVAL}">
""", unsafe_allow_html=True)

# --- Minimal Footer ---
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #999; font-size: 0.8rem;'>"
    f"Project Pulse • Auto-refreshes every {REFRESH_INTERVAL}s</div>",
    unsafe_allow_html=True
)
