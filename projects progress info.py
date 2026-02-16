import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pytz
import numpy as np

st.set_page_config(layout="wide", page_title="WK Agency Dashboard", page_icon="📊", initial_sidebar_state="collapsed")

REFRESH_INTERVAL = 5
PAKISTAN_TZ = pytz.timezone('Asia/Karachi')
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQFttuVQlH84hCC-brrcJFa6eyrMeyc25Aqm_dLgfpuEBr0WCdc4OTKKZVK2Y6IfOoPdQFbYmSdrSYP/pub?output=xlsx"

# Initialize session state
for key in ['page', 'selected_quarter', 'selected_project', 'filters']:
    if key not in st.session_state:
        if key == 'page': st.session_state[key] = 'Executive Overview'
        elif key == 'selected_quarter': st.session_state[key] = 'All'
        elif key == 'selected_project': st.session_state[key] = None
        elif key == 'filters': st.session_state[key] = {}

@st.cache_data(ttl=REFRESH_INTERVAL)
def load_data():
    try:
        xl = pd.ExcelFile(DATA_URL)
        project_master = pd.read_excel(xl, 'PROJECT_MASTER')
        work_log = pd.read_excel(xl, 'DAILY_WORK_LOG')
        employee_cost = pd.read_excel(xl, 'EMPLOYEE_COST')
        resource_links = pd.read_excel(xl, 'RESOURCE_LINKS')
        task_plan = pd.read_excel(xl, 'TASK PLAN + RESPONSIBILITY')
        
        # Clean data - fill NaN values
        for df in [project_master, work_log, employee_cost, resource_links, task_plan]:
            if not df.empty:
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].fillna('Unknown')
                for col in df.select_dtypes(include=[np.number]).columns:
                    df[col] = df[col].fillna(0)
        
        # Calculate hourly rates if columns exist
        if not employee_cost.empty and 'Monthly Salary' in employee_cost.columns:
            employee_cost['Hourly Rate'] = employee_cost['Monthly Salary'] / 160
            employee_cost['Hourly Rate'] = employee_cost['Hourly Rate'].fillna(0)
        
        # Merge work log with hourly rates if both exist and have required columns
        if not work_log.empty and not employee_cost.empty:
            if 'Employee Name' in work_log.columns and 'Employee Name' in employee_cost.columns:
                work_log = work_log.merge(employee_cost[['Employee Name', 'Hourly Rate']], on='Employee Name', how='left')
                work_log['Hourly Rate'] = work_log['Hourly Rate'].fillna(0)
                if 'Hours Worked' in work_log.columns:
                    work_log['Task Cost'] = work_log['Hours Worked'] * work_log['Hourly Rate']
                else:
                    work_log['Task Cost'] = 0
            else:
                work_log['Task Cost'] = 0
        else:
            if not work_log.empty:
                work_log['Task Cost'] = 0
        
        return {
            'PROJECT_MASTER': project_master,
            'DAILY_WORK_LOG': work_log,
            'EMPLOYEE_COST': employee_cost,
            'RESOURCE_LINKS': resource_links,
            'TASK_PLAN': task_plan
        }, datetime.now(PAKISTAN_TZ)
    except Exception as e:
        st.error(f"⚠️ Data load failed: {str(e)}")
        return {name: pd.DataFrame() for name in ['PROJECT_MASTER', 'DAILY_WORK_LOG', 'EMPLOYEE_COST', 'RESOURCE_LINKS', 'TASK_PLAN']}, None

data_sheets, last_update = load_data()

# Custom CSS
st.markdown("""
<style>
    .main-header { background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1rem; }
    .kpi-card { background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; border-left: 4px solid #1e3c72; margin: 0.5rem 0; }
    .kpi-value { font-size: 1.8rem; font-weight: bold; color: #1e3c72; }
    .kpi-label { color: #666; font-size: 0.9rem; text-transform: uppercase; }
    .nav-button { margin: 0.2rem 0; }
    .filter-section { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #dee2e6; }
    .resource-link { padding: 0.5rem; background: #f8f9fa; border-radius: 5px; margin: 0.2rem 0; }
    .resource-link a { color: #1e3c72; text-decoration: none; }
    .resource-link:hover { background: #e9ecef; }
</style>
""", unsafe_allow_html=True)

# Navigation
st.markdown("<div class='main-header'><h1>📊 WK Agency Internal Performance Dashboard</h1><p>Real-time Agency Analytics</p></div>", unsafe_allow_html=True)

pages = ['Executive Overview', 'Quarterly Performance', 'Employee Performance', 'Project Detail View']
cols = st.columns(len(pages))
for i, page in enumerate(pages):
    with cols[i]:
        if st.button(page, use_container_width=True, type="primary" if st.session_state.page == page else "secondary"):
            st.session_state.page = page
            st.rerun()

# Load data with safe access
df_p = data_sheets['PROJECT_MASTER'] if not data_sheets['PROJECT_MASTER'].empty else pd.DataFrame()
df_w = data_sheets['DAILY_WORK_LOG'] if not data_sheets['DAILY_WORK_LOG'].empty else pd.DataFrame()
df_c = data_sheets['EMPLOYEE_COST'] if not data_sheets['EMPLOYEE_COST'].empty else pd.DataFrame()
df_r = data_sheets['RESOURCE_LINKS'] if not data_sheets['RESOURCE_LINKS'].empty else pd.DataFrame()
df_t = data_sheets['TASK_PLAN'] if not data_sheets['TASK_PLAN'].empty else pd.DataFrame()

# Safe column checking function
def has_column(df, col_name):
    return col_name in df.columns if not df.empty else False

# Calculate project metrics safely
if not df_p.empty:
    # Ensure required columns exist
    if 'Project ID' not in df_p.columns:
        df_p['Project ID'] = range(1, len(df_p) + 1)
    
    # Initialize cost columns
    df_p['Total Cost'] = 0
    df_p['Total Hours'] = 0
    df_p['Profit'] = 0
    df_p['Profit Margin'] = 0
    
    # Calculate project costs from work log if available
    if not df_w.empty and has_column(df_w, 'Project ID') and has_column(df_w, 'Task Cost'):
        project_costs = df_w.groupby('Project ID')['Task Cost'].sum().reset_index()
        project_costs.columns = ['Project ID', 'Total Cost']
        df_p = df_p.merge(project_costs, on='Project ID', how='left', suffixes=('', '_new'))
        if 'Total Cost_new' in df_p.columns:
            df_p['Total Cost'] = df_p['Total Cost_new'].fillna(0)
            df_p.drop('Total Cost_new', axis=1, inplace=True)
    
    # Calculate project hours
    if not df_w.empty and has_column(df_w, 'Project ID') and has_column(df_w, 'Hours Worked'):
        project_hours = df_w.groupby('Project ID')['Hours Worked'].sum().reset_index()
        project_hours.columns = ['Project ID', 'Total Hours']
        df_p = df_p.merge(project_hours, on='Project ID', how='left', suffixes=('', '_new'))
        if 'Total Hours_new' in df_p.columns:
            df_p['Total Hours'] = df_p['Total Hours_new'].fillna(0)
            df_p.drop('Total Hours_new', axis=1, inplace=True)
    
    # Calculate profit
    if has_column(df_p, 'Budget'):
        df_p['Profit'] = df_p['Budget'] - df_p['Total Cost']
        df_p['Profit Margin'] = (df_p['Profit'] / df_p['Budget'] * 100).fillna(0)
    
    # Calculate duration
    if has_column(df_p, 'Start Date') and has_column(df_p, 'End Date'):
        df_p['Start Date'] = pd.to_datetime(df_p['Start Date'], errors='coerce')
        df_p['End Date'] = pd.to_datetime(df_p['End Date'], errors='coerce')
        df_p['Duration'] = (df_p['End Date'] - df_p['Start Date']).dt.days.fillna(0)

# Calculate employee metrics safely
employee_metrics = pd.DataFrame()
if not df_w.empty and has_column(df_w, 'Employee Name'):
    emp_group = df_w.groupby('Employee Name')
    emp_dict = {'Employee Name': emp_group.groups.keys()}
    
    if has_column(df_w, 'Hours Worked'):
        emp_dict['Total Hours'] = emp_group['Hours Worked'].sum().values
    else:
        emp_dict['Total Hours'] = [0] * len(emp_group)
    
    if has_column(df_w, 'Task Cost'):
        emp_dict['Total Cost'] = emp_group['Task Cost'].sum().values
    else:
        emp_dict['Total Cost'] = [0] * len(emp_group)
    
    if has_column(df_w, 'Project ID'):
        emp_dict['Projects Contributed'] = emp_group['Project ID'].nunique().values
    else:
        emp_dict['Projects Contributed'] = [0] * len(emp_group)
    
    if has_column(df_w, 'Billable'):
        emp_dict['Billable Hours'] = emp_group.apply(lambda x: (x['Billable'] == 'Yes').sum() if 'Billable' in x.columns else 0).values
    else:
        emp_dict['Billable Hours'] = [0] * len(emp_group)
    
    employee_metrics = pd.DataFrame(emp_dict)
    
    if not employee_metrics.empty:
        employee_metrics['Billable %'] = (employee_metrics['Billable Hours'] / employee_metrics['Total Hours'] * 100).fillna(0)
        employee_metrics['Avg Hours per Project'] = (employee_metrics['Total Hours'] / employee_metrics['Projects Contributed']).fillna(0)

# PAGE 1: Executive Overview
if st.session_state.page == 'Executive Overview':
    st.markdown("## 📈 Executive Overview")
    
    if not df_p.empty:
        # Safely calculate KPIs
        total_active = len(df_p[df_p['Status'] == 'Active']) if has_column(df_p, 'Status') else 0
        total_completed = len(df_p[df_p['Status'] == 'Completed']) if has_column(df_p, 'Status') else 0
        total_clients = df_p['Client Name'].nunique() if has_column(df_p, 'Client Name') else 0
        total_revenue = df_p['Budget'].sum() if has_column(df_p, 'Budget') else 0
        total_cost = df_p['Total Cost'].sum() if has_column(df_p, 'Total Cost') else 0
        total_profit = total_revenue - total_cost
        total_hours = df_p['Total Hours'].sum() if has_column(df_p, 'Total Hours') else 0
        avg_duration = df_p['Duration'].mean() if has_column(df_p, 'Duration') else 0
        
        # KPI Row
        kpi_cols = st.columns(8)
        kpis = [
            (total_active, "Active Projects"),
            (total_completed, "Completed"),
            (total_clients, "Clients"),
            (f"${total_revenue:,.0f}", "Revenue"),
            (f"${total_cost:,.0f}", "Cost"),
            (f"${total_profit:,.0f}", "Profit"),
            (f"{total_hours:,.0f}", "Hours"),
            (f"{avg_duration:.1f}", "Avg Duration")
        ]
        
        for col, (value, label) in zip(kpi_cols, kpis):
            with col:
                st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{value}</div><div class='kpi-label'>{label}</div></div>", unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue vs Cost
            if has_column(df_p, 'Project ID') and has_column(df_p, 'Budget') and has_column(df_p, 'Total Cost'):
                chart_data = df_p[['Project ID', 'Budget', 'Total Cost']].head(10).melt(id_vars=['Project ID'], var_name='Type', value_name='Amount')
                fig = px.bar(chart_data, x='Project ID', y='Amount', color='Type', barmode='group',
                            title="Revenue vs Cost (Top 10 Projects)",
                            color_discrete_map={'Budget': '#1e3c72', 'Total Cost': '#dc3545'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Hours per Project
            if has_column(df_p, 'Project ID') and has_column(df_p, 'Total Hours'):
                hours_data = df_p[['Project ID', 'Total Hours']].sort_values('Total Hours', ascending=False).head(10)
                fig2 = px.bar(hours_data, x='Project ID', y='Total Hours', title="Top 10 Projects by Hours",
                            color='Total Hours', color_continuous_scale='blues')
                st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Employee Contribution
            if not employee_metrics.empty and has_column(employee_metrics, 'Employee Name') and has_column(employee_metrics, 'Total Hours'):
                emp_data = employee_metrics[['Employee Name', 'Total Hours']].sort_values('Total Hours', ascending=False).head(10)
                fig3 = px.bar(emp_data, x='Employee Name', y='Total Hours', title="Top 10 Employees by Hours",
                            color='Total Hours', color_continuous_scale='greens')
                st.plotly_chart(fig3, use_container_width=True)
            
            # Quarterly Revenue
            if has_column(df_p, 'Quarter') and has_column(df_p, 'Budget'):
                quarterly = df_p.groupby('Quarter')['Budget'].sum().reset_index()
                fig4 = px.pie(quarterly, values='Budget', names='Quarter', title="Revenue by Quarter", hole=0.4)
                st.plotly_chart(fig4, use_container_width=True)

# PAGE 2: Quarterly Performance
elif st.session_state.page == 'Quarterly Performance':
    st.markdown("## 📅 Quarterly Performance View")
    
    if has_column(df_p, 'Quarter'):
        quarters = ['All'] + sorted(df_p['Quarter'].unique().tolist())
        selected_quarter = st.selectbox("Select Quarter", quarters, key='quarter_select')
        
        # Filter data
        if selected_quarter != 'All':
            df_p_filtered = df_p[df_p['Quarter'] == selected_quarter]
            if not df_w.empty and has_column(df_w, 'Project ID'):
                df_w_filtered = df_w[df_w['Project ID'].isin(df_p_filtered['Project ID'])]
            else:
                df_w_filtered = df_w
        else:
            df_p_filtered = df_p
            df_w_filtered = df_w
        
        if not df_p_filtered.empty:
            # Calculate metrics
            total_revenue = df_p_filtered['Budget'].sum() if has_column(df_p_filtered, 'Budget') else 0
            total_cost = df_p_filtered['Total Cost'].sum() if has_column(df_p_filtered, 'Total Cost') else 0
            total_profit = total_revenue - total_cost
            total_hours = df_p_filtered['Total Hours'].sum() if has_column(df_p_filtered, 'Total Hours') else 0
            projects_count = len(df_p_filtered)
            completed = len(df_p_filtered[df_p_filtered['Status'] == 'Completed']) if has_column(df_p_filtered, 'Status') else 0
            completion_rate = (completed / projects_count * 100) if projects_count > 0 else 0
            
            # Top employee
            top_employee = "N/A"
            top_hours = 0
            if not df_w_filtered.empty and has_column(df_w_filtered, 'Employee Name') and has_column(df_w_filtered, 'Hours Worked'):
                emp_sum = df_w_filtered.groupby('Employee Name')['Hours Worked'].sum()
                if not emp_sum.empty:
                    top_employee = emp_sum.idxmax()
                    top_hours = emp_sum.max()
            
            # KPI Row
            kpi_cols = st.columns(6)
            kpis = [
                (projects_count, "Projects"),
                (f"${total_revenue:,.0f}", "Revenue"),
                (f"${total_cost:,.0f}", "Cost"),
                (f"${total_profit:,.0f}", "Profit"),
                (f"{total_hours:,.0f}", "Hours"),
                (f"{completion_rate:.1f}%", "Completion")
            ]
            
            for col, (value, label) in zip(kpi_cols, kpis):
                with col:
                    st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{value}</div><div class='kpi-label'>{label}</div></div>", unsafe_allow_html=True)
            
            st.info(f"🏆 Top Contributor: **{top_employee}** with {top_hours:,.0f} hours")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                if not df_w_filtered.empty and has_column(df_w_filtered, 'Employee Name') and has_column(df_w_filtered, 'Hours Worked'):
                    emp_contrib = df_w_filtered.groupby('Employee Name')['Hours Worked'].sum().reset_index()
                    fig = px.bar(emp_contrib, x='Employee Name', y='Hours Worked', 
                                title=f"Employee Contribution {selected_quarter}",
                                color='Hours Worked', color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if has_column(df_p_filtered, 'Project ID') and has_column(df_p_filtered, 'Budget') and has_column(df_p_filtered, 'Total Cost'):
                    chart_data = df_p_filtered[['Project ID', 'Budget', 'Total Cost']].head(10).melt(id_vars=['Project ID'], var_name='Type', value_name='Amount')
                    fig2 = px.bar(chart_data, x='Project ID', y='Amount', color='Type', barmode='group',
                                 title=f"Revenue vs Cost {selected_quarter}",
                                 color_discrete_map={'Budget': '#2ecc71', 'Total Cost': '#e74c3c'})
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Projects table
            st.markdown("### 📋 Projects Detail")
            display_cols = ['Project ID', 'Client Name', 'Total Hours', 'Total Cost', 'Budget', 'Profit', 'Duration', 'Status']
            available_cols = [col for col in display_cols if col in df_p_filtered.columns]
            
            if available_cols:
                display_df = df_p_filtered[available_cols].copy()
                # Format currency columns
                for col in ['Budget', 'Total Cost', 'Profit']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "$0")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

# PAGE 3: Employee Performance
elif st.session_state.page == 'Employee Performance':
    st.markdown("## 👥 Employee Performance")
    
    if not employee_metrics.empty:
        employees = ['All'] + sorted(employee_metrics['Employee Name'].unique().tolist())
        selected_employee = st.selectbox("Select Employee", employees)
        
        if selected_employee != 'All':
            emp_data = employee_metrics[employee_metrics['Employee Name'] == selected_employee].iloc[0]
            
            # Employee KPI cards
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{emp_data['Total Hours']:.0f}</div><div class='kpi-label'>Total Hours</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{emp_data['Billable %']:.1f}%</div><div class='kpi-label'>Billable %</div></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{emp_data['Projects Contributed']:.0f}</div><div class='kpi-label'>Projects</div></div>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<div class='kpi-card'><div class='kpi-value'>${emp_data['Total Cost']:,.0f}</div><div class='kpi-label'>Cost Generated</div></div>", unsafe_allow_html=True)
            with col5:
                st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{emp_data['Avg Hours per Project']:.1f}</div><div class='kpi-label'>Avg Hrs/Project</div></div>", unsafe_allow_html=True)
            
            # Employee's work log
            if not df_w.empty and has_column(df_w, 'Employee Name'):
                emp_work = df_w[df_w['Employee Name'] == selected_employee]
                if has_column(emp_work, 'Date'):
                    emp_work = emp_work.sort_values('Date', ascending=False)
                st.markdown("#### 📝 Recent Work")
                st.dataframe(emp_work.head(10), use_container_width=True, hide_index=True)
        else:
            # All employees comparison
            col1, col2 = st.columns(2)
            
            with col1:
                if has_column(employee_metrics, 'Employee Name') and has_column(employee_metrics, 'Total Hours'):
                    fig = px.bar(employee_metrics, x='Employee Name', y='Total Hours', 
                                title="Contribution by Hours",
                                color='Total Hours', color_continuous_scale='blues')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if has_column(employee_metrics, 'Employee Name') and has_column(employee_metrics, 'Total Cost'):
                    fig2 = px.bar(employee_metrics, x='Employee Name', y='Total Cost', 
                                 title="Contribution by Cost",
                                 color='Total Cost', color_continuous_scale='reds')
                    st.plotly_chart(fig2, use_container_width=True)

# PAGE 4: Project Detail View
elif st.session_state.page == 'Project Detail View':
    st.markdown("## 🔍 Project Detail View")
    
    if not df_p.empty and has_column(df_p, 'Project ID'):
        projects = ['Select a project'] + sorted(df_p['Project ID'].unique().tolist())
        selected_project = st.selectbox("Select Project ID", projects)
        
        if selected_project != 'Select a project':
            project_data = df_p[df_p['Project ID'] == selected_project].iloc[0]
            
            # Project Information
            st.markdown("### 📋 Project Information")
            info_cols = st.columns(4)
            
            with info_cols[0]:
                st.markdown(f"**Client Name:** {project_data.get('Client Name', 'N/A')}")
                st.markdown(f"**Account Manager:** {project_data.get('Account Manager', 'N/A')}")
                st.markdown(f"**Project Manager:** {project_data.get('Project Manager', 'N/A')}")
            
            with info_cols[1]:
                st.markdown(f"**Quarter:** {project_data.get('Quarter', 'N/A')}")
                st.markdown(f"**Start Date:** {project_data.get('Start Date', 'N/A')}")
                st.markdown(f"**End Date:** {project_data.get('End Date', 'N/A')}")
            
            with info_cols[2]:
                st.markdown(f"**Duration:** {project_data.get('Duration', 0):.0f} days")
                st.markdown(f"**Budget:** ${project_data.get('Budget', 0):,.0f}")
                st.markdown(f"**Total Hours:** {project_data.get('Total Hours', 0):.0f}")
            
            with info_cols[3]:
                st.markdown(f"**Total Cost:** ${project_data.get('Total Cost', 0):,.0f}")
                st.markdown(f"**Profit:** ${project_data.get('Profit', 0):,.0f}")
                status = project_data.get('Status', 'N/A')
                st.markdown(f"**Status:** {status}")
            
            # Task Breakdown
            st.markdown("### ✅ Task Breakdown")
            if not df_w.empty and has_column(df_w, 'Project ID'):
                project_tasks = df_w[df_w['Project ID'] == selected_project].copy()
                if not project_tasks.empty:
                    display_cols = []
                    if has_column(project_tasks, 'Task'): display_cols.append('Task')
                    if has_column(project_tasks, 'Employee Name'): display_cols.append('Employee')
                    if has_column(project_tasks, 'Hours Worked'): display_cols.append('Hours')
                    if has_column(project_tasks, 'Task Cost'): 
                        project_tasks['Cost'] = project_tasks['Task Cost'].apply(lambda x: f"${x:,.2f}")
                        display_cols.append('Cost')
                    if has_column(project_tasks, 'Billable'): display_cols.append('Billable')
                    
                    if display_cols:
                        st.dataframe(project_tasks[display_cols], use_container_width=True, hide_index=True)
                else:
                    st.info("No task data available for this project")
            
            # Resource Links
            st.markdown("### 🔗 Project Resources")
            if not df_r.empty and has_column(df_r, 'Project ID'):
                project_resources = df_r[df_r['Project ID'] == selected_project]
                if not project_resources.empty:
                    for _, row in project_resources.iterrows():
                        for col in df_r.columns:
                            if col != 'Project ID' and pd.notna(row[col]) and row[col] != 'Unknown':
                                st.markdown(f"<div class='resource-link'>🔗 <a href='{row[col]}' target='_blank'>{col}</a></div>", unsafe_allow_html=True)
                else:
                    st.info("No resources available for this project")
            else:
                st.info("Resource links not available")

# Footer
if last_update:
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; color: #666; padding: 1rem;'>Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')} PKT | Auto-refreshes every {REFRESH_INTERVAL}s</div>", unsafe_allow_html=True)
