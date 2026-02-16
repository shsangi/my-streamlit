import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pytz

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
        
        # Calculate hourly rates
        if not employee_cost.empty and 'Monthly Salary' in employee_cost.columns:
            employee_cost['Hourly Rate'] = employee_cost['Monthly Salary'] / 160
        
        # Merge work log with hourly rates to calculate task costs
        if not work_log.empty and not employee_cost.empty:
            work_log = work_log.merge(employee_cost[['Employee Name', 'Hourly Rate']], on='Employee Name', how='left')
            work_log['Task Cost'] = work_log['Hours Worked'] * work_log['Hourly Rate']
        
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

# Custom CSS for professional UI
st.markdown("""
<style>
    .main-header { background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1rem; }
    .kpi-card { background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; border-left: 4px solid #1e3c72; }
    .kpi-value { font-size: 1.8rem; font-weight: bold; color: #1e3c72; }
    .kpi-label { color: #666; font-size: 0.9rem; text-transform: uppercase; }
    .nav-button { margin: 0.2rem 0; }
    .filter-section { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #dee2e6; }
    .project-card { background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e0e0e0; margin: 0.5rem 0; cursor: pointer; }
    .project-card:hover { border-color: #1e3c72; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .resource-link { padding: 0.5rem; background: #f8f9fa; border-radius: 5px; margin: 0.2rem 0; }
    .resource-link a { color: #1e3c72; text-decoration: none; }
    .resource-link:hover { background: #e9ecef; }
    .status-active { color: #28a745; font-weight: bold; }
    .status-completed { color: #6c757d; }
    .status-in-progress { color: #ffc107; }
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

# Load data into variables
df_p = data_sheets['PROJECT_MASTER'] if not data_sheets['PROJECT_MASTER'].empty else pd.DataFrame()
df_w = data_sheets['DAILY_WORK_LOG'] if not data_sheets['DAILY_WORK_LOG'].empty else pd.DataFrame()
df_c = data_sheets['EMPLOYEE_COST'] if not data_sheets['EMPLOYEE_COST'].empty else pd.DataFrame()
df_r = data_sheets['RESOURCE_LINKS'] if not data_sheets['RESOURCE_LINKS'].empty else pd.DataFrame()
df_t = data_sheets['TASK_PLAN'] if not data_sheets['TASK_PLAN'].empty else pd.DataFrame()

# Calculate derived metrics
if not df_p.empty and not df_w.empty:
    # Project costs from work log
    project_costs = df_w.groupby('Project ID')['Task Cost'].sum().reset_index()
    project_costs.columns = ['Project ID', 'Total Cost']
    
    # Merge with project master
    df_p = df_p.merge(project_costs, on='Project ID', how='left')
    df_p['Total Cost'] = df_p['Total Cost'].fillna(0)
    df_p['Profit'] = df_p['Budget'] - df_p['Total Cost']
    df_p['Profit Margin'] = (df_p['Profit'] / df_p['Budget'] * 100).fillna(0)
    
    # Project hours
    project_hours = df_w.groupby('Project ID')['Hours Worked'].sum().reset_index()
    project_hours.columns = ['Project ID', 'Total Hours']
    df_p = df_p.merge(project_hours, on='Project ID', how='left')
    df_p['Total Hours'] = df_p['Total Hours'].fillna(0)
    
    # Duration calculation
    if 'Start Date' in df_p.columns and 'End Date' in df_p.columns:
        df_p['Start Date'] = pd.to_datetime(df_p['Start Date'])
        df_p['End Date'] = pd.to_datetime(df_p['End Date'])
        df_p['Duration'] = (df_p['End Date'] - df_p['Start Date']).dt.days

# Employee metrics
if not df_w.empty and not df_c.empty:
    employee_metrics = df_w.groupby('Employee Name').agg({
        'Hours Worked': 'sum',
        'Task Cost': 'sum',
        'Project ID': 'nunique',
        'Billable': lambda x: (x == 'Yes').sum() if 'Billable' in df_w.columns else 0
    }).reset_index()
    employee_metrics.columns = ['Employee Name', 'Total Hours', 'Total Cost', 'Projects Contributed', 'Billable Hours']
    employee_metrics['Billable %'] = (employee_metrics['Billable Hours'] / employee_metrics['Total Hours'] * 100).fillna(0)
    employee_metrics['Avg Hours per Project'] = employee_metrics['Total Hours'] / employee_metrics['Projects Contributed']

# PAGE 1: Executive Overview
if st.session_state.page == 'Executive Overview':
    st.markdown("## 📈 Executive Overview")
    
    # KPI Scorecards
    if not df_p.empty:
        total_active = len(df_p[df_p['Status'] == 'Active']) if 'Status' in df_p.columns else 0
        total_completed = len(df_p[df_p['Status'] == 'Completed']) if 'Status' in df_p.columns else 0
        total_clients = df_p['Client Name'].nunique() if 'Client Name' in df_p.columns else 0
        total_revenue = df_p['Budget'].sum() if 'Budget' in df_p.columns else 0
        total_cost = df_p['Total Cost'].sum()
        total_profit = total_revenue - total_cost
        total_hours = df_w['Hours Worked'].sum() if not df_w.empty else 0
        avg_duration = df_p['Duration'].mean() if 'Duration' in df_p.columns else 0
        
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7, kpi8 = st.columns(8)
        with kpi1: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total_active}</div><div class='kpi-label'>Active Projects</div></div>", unsafe_allow_html=True)
        with kpi2: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total_completed}</div><div class='kpi-label'>Completed</div></div>", unsafe_allow_html=True)
        with kpi3: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total_clients}</div><div class='kpi-label'>Clients</div></div>", unsafe_allow_html=True)
        with kpi4: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>${total_revenue:,.0f}</div><div class='kpi-label'>Revenue</div></div>", unsafe_allow_html=True)
        with kpi5: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>${total_cost:,.0f}</div><div class='kpi-label'>Cost</div></div>", unsafe_allow_html=True)
        with kpi6: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>${total_profit:,.0f}</div><div class='kpi-label'>Profit</div></div>", unsafe_allow_html=True)
        with kpi7: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total_hours:,.0f}</div><div class='kpi-label'>Billable Hours</div></div>", unsafe_allow_html=True)
        with kpi8: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{avg_duration:.1f}</div><div class='kpi-label'>Avg Duration</div></div>", unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            # Revenue vs Cost per project
            chart_data = df_p[['Project ID', 'Budget', 'Total Cost']].melt(id_vars=['Project ID'], var_name='Type', value_name='Amount')
            fig = px.bar(chart_data, x='Project ID', y='Amount', color='Type', barmode='group', 
                        title="Revenue vs Cost per Project", color_discrete_map={'Budget': '#1e3c72', 'Total Cost': '#dc3545'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Hours per Project
            hours_data = df_p[['Project ID', 'Total Hours']].sort_values('Total Hours', ascending=False).head(10)
            fig2 = px.bar(hours_data, x='Project ID', y='Total Hours', title="Top 10 Projects by Hours", color='Total Hours', color_continuous_scale='blues')
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            # Employee Contribution
            if not df_w.empty:
                emp_hours = df_w.groupby('Employee Name')['Hours Worked'].sum().sort_values(ascending=False).head(10).reset_index()
                fig3 = px.bar(emp_hours, x='Employee Name', y='Hours Worked', title="Top Employees by Hours", color='Hours Worked', color_continuous_scale='greens')
                st.plotly_chart(fig3, use_container_width=True)
            
            # Quarterly Revenue Breakdown
            if 'Quarter' in df_p.columns:
                quarterly = df_p.groupby('Quarter')['Budget'].sum().reset_index()
                fig4 = px.pie(quarterly, values='Budget', names='Quarter', title="Revenue by Quarter", hole=0.4)
                st.plotly_chart(fig4, use_container_width=True)

# PAGE 2: Quarterly Performance
elif st.session_state.page == 'Quarterly Performance':
    st.markdown("## 📅 Quarterly Performance View")
    
    # Quarter filter
    if 'Quarter' in df_p.columns:
        quarters = ['All'] + sorted(df_p['Quarter'].unique().tolist())
        selected_quarter = st.selectbox("Select Quarter", quarters, key='quarter_select')
        
        # Filter data by quarter
        if selected_quarter != 'All':
            df_p_filtered = df_p[df_p['Quarter'] == selected_quarter]
            df_w_filtered = df_w[df_w['Project ID'].isin(df_p_filtered['Project ID'])] if not df_w.empty else df_w
        else:
            df_p_filtered = df_p
            df_w_filtered = df_w
        
        if not df_p_filtered.empty:
            # KPI row
            total_revenue = df_p_filtered['Budget'].sum()
            total_cost = df_p_filtered['Total Cost'].sum()
            total_profit = total_revenue - total_cost
            total_hours = df_p_filtered['Total Hours'].sum()
            projects_count = len(df_p_filtered)
            completed = len(df_p_filtered[df_p_filtered['Status'] == 'Completed']) if 'Status' in df_p_filtered.columns else 0
            completion_rate = (completed / projects_count * 100) if projects_count > 0 else 0
            
            # Most contributing employee
            if not df_w_filtered.empty:
                top_employee = df_w_filtered.groupby('Employee Name')['Hours Worked'].sum().idxmax()
                top_hours = df_w_filtered.groupby('Employee Name')['Hours Worked'].sum().max()
            else:
                top_employee, top_hours = "N/A", 0
            
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            with k1: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{projects_count}</div><div class='kpi-label'>Projects</div></div>", unsafe_allow_html=True)
            with k2: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>${total_revenue:,.0f}</div><div class='kpi-label'>Revenue</div></div>", unsafe_allow_html=True)
            with k3: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>${total_cost:,.0f}</div><div class='kpi-label'>Cost</div></div>", unsafe_allow_html=True)
            with k4: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>${total_profit:,.0f}</div><div class='kpi-label'>Profit</div></div>", unsafe_allow_html=True)
            with k5: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{total_hours:,.0f}</div><div class='kpi-label'>Hours</div></div>", unsafe_allow_html=True)
            with k6: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{completion_rate:.1f}%</div><div class='kpi-label'>Completion</div></div>", unsafe_allow_html=True)
            
            st.info(f"🏆 Top Contributor: **{top_employee}** with {top_hours:,.0f} hours")
            
            # Charts
            col1, col2 = st.columns(2)
            with col1:
                # Employee Contribution for selected quarter
                if not df_w_filtered.empty:
                    emp_contrib = df_w_filtered.groupby('Employee Name')['Hours Worked'].sum().reset_index()
                    fig = px.bar(emp_contrib, x='Employee Name', y='Hours Worked', title=f"Employee Contribution {selected_quarter}", 
                                color='Hours Worked', color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Revenue vs Cost for selected quarter
                chart_data = df_p_filtered[['Project ID', 'Budget', 'Total Cost']].melt(id_vars=['Project ID'], var_name='Type', value_name='Amount')
                fig2 = px.bar(chart_data, x='Project ID', y='Amount', color='Type', barmode='group', 
                             title=f"Revenue vs Cost {selected_quarter}", color_discrete_map={'Budget': '#2ecc71', 'Total Cost': '#e74c3c'})
                st.plotly_chart(fig2, use_container_width=True)
            
            # Projects table
            st.markdown("### 📋 Projects Detail")
            display_cols = ['Project ID', 'Client Name', 'Total Hours', 'Total Cost', 'Budget', 'Profit', 'Duration', 'Status']
            display_df = df_p_filtered[[col for col in display_cols if col in df_p_filtered.columns]].copy()
            if 'Profit' in display_df.columns:
                display_df['Profit'] = display_df['Profit'].apply(lambda x: f"${x:,.0f}")
            if 'Budget' in display_df.columns:
                display_df['Budget'] = display_df['Budget'].apply(lambda x: f"${x:,.0f}")
            if 'Total Cost' in display_df.columns:
                display_df['Total Cost'] = display_df['Total Cost'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

# PAGE 3: Employee Performance
elif st.session_state.page == 'Employee Performance':
    st.markdown("## 👥 Employee Performance")
    
    if not df_w.empty and 'employee_metrics' in locals():
        # Employee selector
        employees = ['All'] + sorted(employee_metrics['Employee Name'].unique().tolist())
        selected_employee = st.selectbox("Select Employee", employees)
        
        if selected_employee != 'All':
            emp_data = employee_metrics[employee_metrics['Employee Name'] == selected_employee].iloc[0]
            
            # Employee KPI cards
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{emp_data['Total Hours']:.0f}</div><div class='kpi-label'>Total Hours</div></div>", unsafe_allow_html=True)
            with col2: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{emp_data['Billable %']:.1f}%</div><div class='kpi-label'>Billable %</div></div>", unsafe_allow_html=True)
            with col3: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{emp_data['Projects Contributed']:.0f}</div><div class='kpi-label'>Projects</div></div>", unsafe_allow_html=True)
            with col4: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>${emp_data['Total Cost']:,.0f}</div><div class='kpi-label'>Cost Generated</div></div>", unsafe_allow_html=True)
            with col5: st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{emp_data['Avg Hours per Project']:.1f}</div><div class='kpi-label'>Avg Hrs/Project</div></div>", unsafe_allow_html=True)
            
            # Employee's work log
            emp_work = df_w[df_w['Employee Name'] == selected_employee].sort_values('Date', ascending=False) if 'Date' in df_w.columns else df_w[df_w['Employee Name'] == selected_employee]
            st.markdown("#### 📝 Recent Work")
            st.dataframe(emp_work.head(10), use_container_width=True, hide_index=True)
        else:
            # All employees comparison
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(employee_metrics, x='Employee Name', y='Total Hours', title="Contribution by Hours",
                            color='Total Hours', color_continuous_scale='blues')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig2 = px.bar(employee_metrics, x='Employee Name', y='Total Cost', title="Contribution by Cost",
                             color='Total Cost', color_continuous_scale='reds')
                st.plotly_chart(fig2, use_container_width=True)
            
            # Quarterly contribution breakdown
            if 'Quarter' in df_p.columns and not df_w.empty:
                df_w_quarter = df_w.merge(df_p[['Project ID', 'Quarter']], on='Project ID', how='left')
                quarter_emp = df_w_quarter.groupby(['Quarter', 'Employee Name'])['Hours Worked'].sum().reset_index()
                fig3 = px.bar(quarter_emp, x='Quarter', y='Hours Worked', color='Employee Name', 
                             title="Employee Hours by Quarter", barmode='group')
                st.plotly_chart(fig3, use_container_width=True)

# PAGE 4: Project Detail View
elif st.session_state.page == 'Project Detail View':
    st.markdown("## 🔍 Project Detail View")
    
    if not df_p.empty:
        # Project selector
        projects = ['Select a project'] + sorted(df_p['Project ID'].unique().tolist())
        selected_project = st.selectbox("Select Project ID", projects)
        
        if selected_project != 'Select a project':
            project_data = df_p[df_p['Project ID'] == selected_project].iloc[0]
            
            # Project Information Section
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
                status_class = "status-active" if status == "Active" else "status-completed" if status == "Completed" else "status-in-progress"
                st.markdown(f"**Status:** <span class='{status_class}'>{status}</span>", unsafe_allow_html=True)
            
            # Task Breakdown
            st.markdown("### ✅ Task Breakdown")
            if not df_w.empty:
                project_tasks = df_w[df_w['Project ID'] == selected_project].copy()
                if not project_tasks.empty:
                    # Merge with employee cost for rates
                    project_tasks = project_tasks.merge(df_c[['Employee Name', 'Monthly Salary']], on='Employee Name', how='left')
                    display_tasks = project_tasks[['Task', 'Employee Name', 'Hours Worked', 'Task Cost', 'Billable']].copy()
                    display_tasks.columns = ['Task', 'Employee', 'Hours', 'Cost', 'Billable']
                    display_tasks['Cost'] = display_tasks['Cost'].apply(lambda x: f"${x:,.2f}")
                    st.dataframe(display_tasks, use_container_width=True, hide_index=True)
                else:
                    st.info("No task data available for this project")
            
            # Resource Links Section
            st.markdown("### 🔗 Project Resources")
            if not df_r.empty:
                project_resources = df_r[df_r['Project ID'] == selected_project]
                if not project_resources.empty:
                    resource_types = {
                        'Company Website': '🌐',
                        'Staging URL': '🖥️',
                        'Shared Folder': '📁',
                        'Meeting Notes': '📝',
                        'Creative Brief': '📄',
                        'Figma Design': '🎨',
                        'Deliverables': '📦',
                        'Timeline': '⏰'
                    }
                    
                    for _, row in project_resources.iterrows():
                        for resource_type, emoji in resource_types.items():
                            if resource_type in row.index and pd.notna(row[resource_type]):
                                st.markdown(f"<div class='resource-link'>{emoji} <a href='{row[resource_type]}' target='_blank'>{resource_type}</a></div>", unsafe_allow_html=True)
                else:
                    st.info("No resources available for this project")
            else:
                st.info("Resource links not available")
            
            # Timeline/Progress
            st.markdown("### 📊 Project Timeline")
            if 'Start Date' in project_data.index and 'End Date' in project_data.index:
                start = pd.to_datetime(project_data['Start Date'])
                end = pd.to_datetime(project_data['End Date'])
                today = datetime.now()
                
                total_days = (end - start).days
                elapsed_days = (today - start).days if today > start else 0
                progress = min(100, max(0, (elapsed_days / total_days * 100))) if total_days > 0 else 0
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=progress,
                    title={'text': "Project Progress"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [None, 100]},
                          'bar': {'color': "#1e3c72"},
                          'steps': [
                              {'range': [0, 50], 'color': "lightgray"},
                              {'range': [50, 100], 'color': "gray"}],
                          'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 90}}))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)

# Footer with last update
if last_update:
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; color: #666;'>Last updated: {last_update.strftime('%Y-%m-%d %H:%M:%S')} PKT | Auto-refreshes every {REFRESH_INTERVAL}s</div>", unsafe_allow_html=True)
