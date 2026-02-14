import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Dynamic Project Dashboard")
st.title("📊 Vertical Tab Dashboard with Dynamic Content Panels")

# --- Data Loading Function (Cached for Performance) ---
@st.cache_data
def load_data_from_gsheet():
    """Loads data from the published Google Sheet Excel URL."""
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQFttuVQlH84hCC-brrcJFa6eyrMeyc25Aqm_dLgfpuEBr0WCdc4OTKKZVK2Y6IfOoPdQFbYmSdrSYP/pub?output=xlsx"
    try:
        # Load all sheets from the Excel file
        # Note: Sheet names must match exactly or be updated here
        xl = pd.ExcelFile(url)
        sheets = {
            'PROJECT_MASTER': pd.read_excel(xl, 'PROJECT_MASTER'),
            'DAILY_WORK_LOG': pd.read_excel(xl, 'DAILY_WORK_LOG'),
            'EMPLOYEE_COST': pd.read_excel(xl, 'EMPLOYEE_COST'),
            'RESOURCE_LINKS': pd.read_excel(xl, 'RESOURCE_LINKS'),
            'TASK_PLAN': pd.read_excel(xl, 'TASK PLAN + RESPONSIBILITY') # Adjust sheet name if needed
        }
        # Basic data cleaning: Convert date columns, fill NaNs
        for name, df in sheets.items():
            for col in df.columns:
                if 'Date' in col or 'date' in col or 'Start' in col or 'End' in col:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except: pass
        st.success("Data loaded successfully!")
        return sheets
    except Exception as e:
        st.error(f"Error loading data: {e}. Please check the URL and sheet names.")
        # Return empty dataframes as fallback
        return {name: pd.DataFrame() for name in ['PROJECT_MASTER', 'DAILY_WORK_LOG', 'EMPLOYEE_COST', 'RESOURCE_LINKS', 'TASK_PLAN']}

# --- Load Data ---
data_sheets = load_data_from_gsheet()

# --- Custom CSS for Vertical Tabs Look ---
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        text-align: left;
        background-color: transparent;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #f0f2f6;
        border: none;
    }
    .stButton > button:focus {
        background-color: #e0e3e9;
        border: none;
        box-shadow: none;
        outline: 2px solid #4e8cff;
    }
    div[data-testid="column"]:first-child {
        background-color: #f9f9f9;
        padding: 20px 10px;
        border-radius: 10px;
    }
    h3 {
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Create Two Main Columns: Left for Icons, Right for Content ---
left_col, right_col = st.columns([1, 5])

# --- Left Column: Vertical Icon Tabs (Navigation) ---
with left_col:
    st.markdown("### Navigation")
    st.divider()

    # Use session state to track the selected tab
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = 'PROJECT_MASTER'

    # Define icons and labels for tabs
    tabs = {
        'PROJECT_MASTER': ('📁', 'Project Master'),
        'DAILY_WORK_LOG': ('📝', 'Daily Work Log'),
        'EMPLOYEE_COST': ('💰', 'Employee Cost'),
        'RESOURCE_LINKS': ('🔗', 'Resource Links'),
        'TASK_PLAN': ('✅', 'Task Plan')
    }

    # Create buttons that act as tabs
    for tab_key, (icon, label) in tabs.items():
        button_label = f"{icon} {label}"
        if st.button(button_label, key=f"btn_{tab_key}", use_container_width=True):
            st.session_state.selected_tab = tab_key

    st.divider()
    st.caption("Dynamic Content Panels")

# --- Right Column: Dynamic Content Based on Selected Tab ---
with right_col:
    st.markdown(f"### {tabs[st.session_state.selected_tab][1]} Dashboard")

    # --- Panel 1: PROJECT_MASTER ---
    if st.session_state.selected_tab == 'PROJECT_MASTER':
        df = data_sheets['PROJECT_MASTER']
        if not df.empty:
            # Filters Row
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'Company name' in df.columns:
                    companies = ['All'] + list(df['Company name'].unique())
                    selected_company = st.selectbox('Filter by Company', companies)
            with col2:
                if 'Quarter' in df.columns:
                    quarters = ['All'] + list(df['Quarter'].unique())
                    selected_quarter = st.selectbox('Filter by Quarter', quarters)
            with col3:
                if 'Status' in df.columns:
                    statuses = ['All'] + list(df['Status'].unique())
                    selected_status = st.selectbox('Filter by Status', statuses)

            # Apply filters
            filtered_df = df.copy()
            if selected_company != 'All':
                filtered_df = filtered_df[filtered_df['Company name'] == selected_company]
            if selected_quarter != 'All':
                filtered_df = filtered_df[filtered_df['Quarter'] == selected_quarter]
            if selected_status != 'All':
                filtered_df = filtered_df[filtered_df['Status'] == selected_status]

            # Scorecards
            st.subheader("Key Metrics")
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("Total Projects", len(filtered_df))
            with mc2:
                if 'Budget' in filtered_df.columns:
                    total_budget = filtered_df['Budget'].sum()
                    st.metric("Total Budget", f"${total_budget:,.0f}")
            with mc3:
                if 'Account Manager' in filtered_df.columns:
                    unique_ams = filtered_df['Account Manager'].nunique()
                    st.metric("Account Managers", unique_ams)
            with mc4:
                if 'Status' in filtered_df.columns:
                    active_count = len(filtered_df[filtered_df['Status'] == 'Active'])
                    st.metric("Active Projects", active_count)

            # Graph
            st.subheader("Budget by Company")
            if 'Company name' in filtered_df.columns and 'Budget' in filtered_df.columns:
                budget_by_company = filtered_df.groupby('Company name')['Budget'].sum().reset_index()
                fig = px.bar(budget_by_company, x='Company name', y='Budget', title="Budget Distribution")
                st.plotly_chart(fig, use_container_width=True)

            # Data Table
            st.subheader("Project Master Data")
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        else:
            st.warning("PROJECT_MASTER data is empty or could not be loaded.")

    # --- Panel 2: DAILY_WORK_LOG ---
    elif st.session_state.selected_tab == 'DAILY_WORK_LOG':
        df = data_sheets['DAILY_WORK_LOG']
        if not df.empty:
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                if 'Employee Name' in df.columns:
                    employees = ['All'] + list(df['Employee Name'].unique())
                    selected_emp = st.selectbox('Filter by Employee', employees)
            with col2:
                if 'Date' in df.columns:
                    min_date = df['Date'].min().date() if pd.notna(df['Date'].min()) else datetime.today().date()
                    max_date = df['Date'].max().date() if pd.notna(df['Date'].max()) else datetime.today().date()
                    date_range = st.date_input("Date Range", [min_date, max_date])

            # Apply filters (simplified)
            filtered_df = df.copy()
            if selected_emp != 'All':
                filtered_df = filtered_df[filtered_df['Employee Name'] == selected_emp]

            # Scorecards
            st.subheader("Work Log Summary")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Total Hours", f"{filtered_df['Hours Worked'].sum():.1f}")
            with sc2:
                total_cost = (filtered_df['Hours Worked'] * filtered_df['Employee Hourly Rate']).sum()
                st.metric("Total Cost", f"${total_cost:,.0f}")
            with sc3:
                st.metric("Total Entries", len(filtered_df))

            # Graph: Hours by Employee
            if 'Employee Name' in filtered_df.columns and 'Hours Worked' in filtered_df.columns:
                hours_by_emp = filtered_df.groupby('Employee Name')['Hours Worked'].sum().reset_index()
                fig = px.pie(hours_by_emp, values='Hours Worked', names='Employee Name', title="Work Hours Distribution")
                st.plotly_chart(fig, use_container_width=True)

            # Data Table
            st.subheader("Daily Work Log Data")
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        else:
            st.warning("DAILY_WORK_LOG data is empty or could not be loaded.")

    # --- Panel 3: EMPLOYEE_COST ---
    elif st.session_state.selected_tab == 'EMPLOYEE_COST':
        df = data_sheets['EMPLOYEE_COST']
        if not df.empty:
            st.subheader("Employee Cost Overview")
            # Simple table and bar chart
            if not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)

                if 'Role' in df.columns and 'Monthly Salary' in df.columns:
                    cost_by_role = df.groupby('Role')['Monthly Salary'].sum().reset_index()
                    fig = px.bar(cost_by_role, x='Role', y='Monthly Salary', title="Monthly Salary by Role")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("EMPLOYEE_COST data is empty or could not be loaded.")

    # --- Panel 4: RESOURCE_LINKS ---
    elif st.session_state.selected_tab == 'RESOURCE_LINKS':
        df = data_sheets['RESOURCE_LINKS']
        if not df.empty:
            st.subheader("Resource Links & Project Info")
            # Display as a table with clickable links if possible
            # For display purposes, just show the dataframe
            st.info("Tip: You can click on links in the table if they are properly formatted as URLs.")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("RESOURCE_LINKS data is empty or could not be loaded.")

    # --- Panel 5: TASK PLAN + RESPONSIBILITY ---
    elif st.session_state.selected_tab == 'TASK_PLAN':
        df = data_sheets['TASK_PLAN']
        if not df.empty:
            st.subheader("Task Plan & Responsibility")

            # Filters
            col1, col2 = st.columns(2)
            with col1:
                if 'Priority' in df.columns:
                    priorities = ['All'] + list(df['Priority'].unique())
                    selected_priority = st.selectbox('Filter by Priority', priorities)
            with col2:
                if 'Status' in df.columns:
                    statuses = ['All'] + list(df['Status'].unique())
                    selected_status = st.selectbox('Filter by Status', statuses)

            # Apply filters
            filtered_df = df.copy()
            if selected_priority != 'All':
                filtered_df = filtered_df[filtered_df['Priority'] == selected_priority]
            if selected_status != 'All':
                filtered_df = filtered_df[filtered_df['Status'] == selected_status]

            # Scorecards
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Total Tasks", len(filtered_df))
            with sc2:
                if 'Status' in filtered_df.columns:
                    pending = len(filtered_df[filtered_df['Status'] != 'Done'])
                    st.metric("Pending Tasks", pending)
            with sc3:
                if 'Priority' in filtered_df.columns:
                    high = len(filtered_df[filtered_df['Priority'] == 'High'])
                    st.metric("High Priority", high)

            # Graph: Tasks by Owner
            if 'Owner (Team / Client)' in filtered_df.columns:
                owner_count = filtered_df['Owner (Team / Client)'].value_counts().reset_index()
                owner_count.columns = ['Owner', 'Count']
                fig = px.pie(owner_count, values='Count', names='Owner', title="Tasks by Owner")
                st.plotly_chart(fig, use_container_width=True)

            # Data Table
            st.subheader("Task Details")
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        else:
            st.warning("TASK_PLAN data is empty or could not be loaded.")

# --- Optional: Footer ---
st.divider()
st.caption("Dynamic Dashboard - Data updates automatically from Google Sheet.")
