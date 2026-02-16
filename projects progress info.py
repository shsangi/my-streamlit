import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pytz

st.set_page_config(layout="wide", page_title="Project Pulse", page_icon="📊", initial_sidebar_state="collapsed")

REFRESH_INTERVAL = 5
PAKISTAN_TZ = pytz.timezone('Asia/Karachi')
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQFttuVQlH84hCC-brrcJFa6eyrMeyc25Aqm_dLgfpuEBr0WCdc4OTKKZVK2Y6IfOoPdQFbYmSdrSYP/pub?output=xlsx"

for key in ['selected_tab', 'last_update', 'data_sheets', 'show_original', 'filters']:
    if key not in st.session_state:
        if key == 'selected_tab': st.session_state[key] = 'PROJECT_MASTER'
        elif key == 'filters': st.session_state[key] = {}
        else: st.session_state[key] = None if key != 'show_original' else False

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

data_sheets, last_update = load_data()
if last_update:
    st.session_state.last_update = last_update
    st.session_state.data_sheets = data_sheets

st.markdown("""
<style>
    .header-container { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); padding: 1rem 2rem; border-radius: 16px; margin-bottom: 1rem; color: white; display: flex; align-items: center; justify-content: space-between; }
    .header-title { margin: 0; font-size: 2rem; font-weight: 700; }
    .header-subtitle { margin: 0.2rem 0 0 0; display: flex; align-items: center; gap: 0.5rem; }
    .live-badge { background: #ef4444; color: white; padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.75rem; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
    .timestamp { color: white; font-size: 0.9rem; display: flex; align-items: center; gap: 0.5rem; }
    .pk-badge { background: rgba(255,255,255,0.2); color: white; padding: 0.2rem 0.7rem; border-radius: 20px; }
    .header-scorecards { display: flex; gap: 1rem; }
    .header-scorecard { background: rgba(255,255,255,0.15); padding: 0.5rem 1rem; border-radius: 12px; text-align: center; min-width: 90px; backdrop-filter: blur(10px); }
    .header-scorecard-value { font-size: 1.2rem; font-weight: 700; }
    .header-scorecard-label { font-size: 0.7rem; opacity: 0.9; text-transform: uppercase; }
    .scorecard { background: #f9fafb; padding: 0.8rem; border-radius: 10px; text-align: center; border-left: 4px solid #6366f1; }
    .scorecard-value { font-size: 1.5rem; font-weight: 700; color: #1f2937; }
    .scorecard-label { font-size: 0.8rem; color: #6b7280; }
    .tab-container { background: white; border-radius: 12px; padding: 1rem; border: 1px solid #e5e7eb; }
    .stButton > button { border-radius: 8px; }
    .table-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; background: #f9fafb; padding: 0.5rem; border-radius: 8px; }
    .table-shape { background: white; padding: 0.2rem 0.8rem; border-radius: 15px; font-size: 0.8rem; color: #6b7280; border: 1px solid #e5e7eb; }
    .chart-container { background: white; padding: 0.5rem; border-radius: 12px; border: 1px solid #e5e7eb; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

def create_chart(data, chart_type, title):
    if chart_type == 'bar' and len(data) > 0:
        fig = px.bar(data, x=data.columns[0], y=data.columns[1], title=title, 
                     color=data.columns[1], color_continuous_scale='viridis', text=data.columns[1])
        fig.update_traces(texttemplate='%{text:.0f}', textposition='outside', marker_line_width=1, marker_line_color='white')
        fig.update_layout(height=300, margin=dict(t=30, l=30, r=30, b=30), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        return fig
    elif chart_type == 'pie' and len(data) > 0:
        fig = px.pie(data, values=data.columns[1], names=data.columns[0], title=title, hole=0.4,
                     color_discrete_sequence=px.colors.sequential.Viridis)
        fig.update_layout(height=300, margin=dict(t=30, l=30, r=30, b=30), showlegend=False)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    return None

if st.session_state.data_sheets and not st.session_state.show_original:
    df_p, df_w, df_c, df_t, df_r = [st.session_state.data_sheets[s] for s in ['PROJECT_MASTER', 'DAILY_WORK_LOG', 'EMPLOYEE_COST', 'TASK_PLAN', 'RESOURCE_LINKS']]
    metrics = {
        'total_projects': len(df_p), 'active_projects': len(df_p[df_p['Status']=='Active']) if not df_p.empty and 'Status' in df_p.columns else 0,
        'total_hours': df_w['Hours Worked'].sum() if not df_w.empty else 0,
        'total_salary': df_c['Monthly Salary'].sum() if not df_c.empty and 'Monthly Salary' in df_c.columns else 0,
        'total_tasks': len(df_t), 'pending_tasks': len(df_t[df_t['Status']!='Done']) if not df_t.empty and 'Status' in df_t.columns else 0,
        'total_resources': len(df_r)
    }
    timestamp = st.session_state.last_update.strftime("%a, %d %b, %I:%M:%S %p") if st.session_state.last_update else ""
    st.markdown(f"""
    <div class="header-container"><div><h1 class="header-title">📊 Project Pulse</h1><p class="header-subtitle"><span class="live-badge">LIVE</span><span class="timestamp">🔄 {timestamp} <span class="pk-badge">PKT</span></span></p></div>
    <div class="header-scorecards">{''.join([f"<div class='header-scorecard'><div class='header-scorecard-value'>{metrics[k]}</div><div class='header-scorecard-label'>{k.replace('_',' ').title()}</div></div>" for k in ['total_projects','total_hours','total_salary','total_tasks','total_resources']])}</div></div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<div class='header-container'><div><h1 class='header-title'>📊 Project Pulse</h1><p>Original Sheets View</p></div></div>", unsafe_allow_html=True)

if st.session_state.show_original:
    st.markdown("### 📋 Original Sheets")
    if st.button("← Back to Dashboard"): st.session_state.show_original = False; st.rerun()
    tabs = st.tabs(list(st.session_state.data_sheets.keys()))
    for i, (sheet_name, df) in enumerate(st.session_state.data_sheets.items()):
        with tabs[i]:
            st.markdown(f"<div class='table-header'><span>📄 {sheet_name.replace('_',' ').title()}</span><span class='table-shape'>{df.shape[0]} rows × {df.shape[1]} cols</span></div>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
else:
    lcol, rcol = st.columns([1, 4])
    with lcol:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        tabs = {'PROJECT_MASTER':'📁 Projects','DAILY_WORK_LOG':'📝 Work Log','EMPLOYEE_COST':'💰 Costs','RESOURCE_LINKS':'🔗 Resources','TASK_PLAN':'✅ Tasks'}
        for key, label in tabs.items():
            if st.button(label, key=f"nav_{key}", use_container_width=True, type="secondary" if st.session_state.selected_tab != key else "primary"):
                st.session_state.selected_tab = key; st.session_state.filters = {}; st.rerun()
        if st.button("📋 Original Sheets", use_container_width=True): st.session_state.show_original = True; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with rcol:
        current = st.session_state.selected_tab
        st.markdown(f"### {tabs[current]}")
        
        if st.session_state.data_sheets and not st.session_state.data_sheets[current].empty:
            df = st.session_state.data_sheets[current].copy()
            for col in df.select_dtypes(include=['object']).columns: df[col] = df[col].fillna('Unknown')
            
            if current != 'RESOURCE_LINKS':
                with st.expander("🔍 Filters", expanded=False):
                    cols = st.columns([3,1])
                    with cols[0]:
                        if current == 'PROJECT_MASTER':
                            fc = st.columns(3)
                            with fc[0]: st.session_state.filters['company'] = st.selectbox('Company', ['All']+sorted(df['Company name'].unique()) if 'Company name' in df else ['All'])
                            with fc[1]: st.session_state.filters['status'] = st.selectbox('Status', ['All']+sorted(df['Status'].unique()) if 'Status' in df else ['All'])
                            with fc[2]: st.session_state.filters['quarter'] = st.selectbox('Quarter', ['All']+sorted(df['Quarter'].unique()) if 'Quarter' in df else ['All'])
                        elif current == 'TASK_PLAN':
                            fc = st.columns(3)
                            with fc[0]: st.session_state.filters['priority'] = st.selectbox('Priority', ['All']+sorted(df['Priority'].unique()) if 'Priority' in df else ['All'])
                            with fc[1]: st.session_state.filters['status'] = st.selectbox('Status', ['All']+sorted(df['Status'].unique()) if 'Status' in df else ['All'])
                            with fc[2]: st.session_state.filters['owner'] = st.selectbox('Owner', ['All']+sorted(df['Owner (Team / Client)'].unique()) if 'Owner (Team / Client)' in df else ['All'])
                        elif current == 'DAILY_WORK_LOG':
                            st.session_state.filters['employee'] = st.selectbox('Employee', ['All']+sorted(df['Employee Name'].unique()) if 'Employee Name' in df else ['All'])
                        elif current == 'EMPLOYEE_COST':
                            st.session_state.filters['role'] = st.selectbox('Role', ['All']+sorted(df['Role'].unique()) if 'Role' in df else ['All'])
                    with cols[1]:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🔄 Reset", use_container_width=True): st.session_state.filters = {}; st.rerun()
            
            filtered = df.copy()
            if current != 'RESOURCE_LINKS':
                if current == 'PROJECT_MASTER':
                    if st.session_state.filters.get('company','All')!='All': filtered = filtered[filtered['Company name']==st.session_state.filters['company']]
                    if st.session_state.filters.get('status','All')!='All': filtered = filtered[filtered['Status']==st.session_state.filters['status']]
                    if st.session_state.filters.get('quarter','All')!='All': filtered = filtered[filtered['Quarter']==st.session_state.filters['quarter']]
                elif current == 'TASK_PLAN':
                    if st.session_state.filters.get('priority','All')!='All': filtered = filtered[filtered['Priority']==st.session_state.filters['priority']]
                    if st.session_state.filters.get('status','All')!='All': filtered = filtered[filtered['Status']==st.session_state.filters['status']]
                    if st.session_state.filters.get('owner','All')!='All': filtered = filtered[filtered['Owner (Team / Client)']==st.session_state.filters['owner']]
                elif current == 'DAILY_WORK_LOG' and st.session_state.filters.get('employee','All')!='All':
                    filtered = filtered[filtered['Employee Name']==st.session_state.filters['employee']]
                elif current == 'EMPLOYEE_COST' and st.session_state.filters.get('role','All')!='All':
                    filtered = filtered[filtered['Role']==st.session_state.filters['role']]
            
            if not filtered.empty:
                cols = st.columns(4)
                if current == 'PROJECT_MASTER':
                    cols[0].markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered)}</div><div class='scorecard-label'>Projects</div></div>", unsafe_allow_html=True)
                    cols[1].markdown(f"<div class='scorecard'><div class='scorecard-value'>${filtered['Budget'].sum():,.0f}</div><div class='scorecard-label'>Budget</div></div>" if 'Budget' in filtered else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Budget</div></div>", unsafe_allow_html=True)
                    cols[2].markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered[filtered['Status']=='Active'])}</div><div class='scorecard-label'>Active</div></div>" if 'Status' in filtered else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Active</div></div>", unsafe_allow_html=True)
                    cols[3].markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Company name'].nunique()}</div><div class='scorecard-label'>Companies</div></div>" if 'Company name' in filtered else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Companies</div></div>", unsafe_allow_html=True)
                    
                    if 'Company name' in filtered and 'Budget' in filtered:
                        chart_data = filtered.groupby('Company name')['Budget'].sum().reset_index()
                        fig = create_chart(chart_data, 'bar', "Budget by Company")
                        if fig: st.plotly_chart(fig, use_container_width=True)
                
                elif current == 'TASK_PLAN':
                    cols[0].markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered)}</div><div class='scorecard-label'>Tasks</div></div>", unsafe_allow_html=True)
                    cols[1].markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered[filtered['Status']=='Done'])}</div><div class='scorecard-label'>Completed</div></div>" if 'Status' in filtered else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Completed</div></div>", unsafe_allow_html=True)
                    cols[2].markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered[filtered['Priority']=='High'])}</div><div class='scorecard-label'>High Priority</div></div>" if 'Priority' in filtered else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>High Priority</div></div>", unsafe_allow_html=True)
                    cols[3].markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Owner (Team / Client)'].nunique()}</div><div class='scorecard-label'>Owners</div></div>" if 'Owner (Team / Client)' in filtered else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Owners</div></div>", unsafe_allow_html=True)
                    
                    if 'Owner (Team / Client)' in filtered:
                        chart_data = filtered['Owner (Team / Client)'].value_counts().reset_index()
                        fig = create_chart(chart_data, 'pie', "Tasks by Owner")
                        if fig: st.plotly_chart(fig, use_container_width=True)
                
                elif current == 'DAILY_WORK_LOG':
                    cols[0].markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered)}</div><div class='scorecard-label'>Entries</div></div>", unsafe_allow_html=True)
                    cols[1].markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Hours Worked'].sum():.1f}</div><div class='scorecard-label'>Total Hours</div></div>" if 'Hours Worked' in filtered else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Hours</div></div>", unsafe_allow_html=True)
                    cols[2].markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Hours Worked'].mean():.1f}</div><div class='scorecard-label'>Avg Hours</div></div>" if 'Hours Worked' in filtered else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Avg Hours</div></div>", unsafe_allow_html=True)
                    cols[3].markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Employee Name'].nunique()}</div><div class='scorecard-label'>Employees</div></div>" if 'Employee Name' in filtered else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Employees</div></div>", unsafe_allow_html=True)
                    
                    if 'Employee Name' in filtered and 'Hours Worked' in filtered:
                        chart_data = filtered.groupby('Employee Name')['Hours Worked'].sum().reset_index()
                        fig = create_chart(chart_data, 'bar', "Hours by Employee")
                        if fig: st.plotly_chart(fig, use_container_width=True)
                
                elif current == 'EMPLOYEE_COST':
                    cols[0].markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered)}</div><div class='scorecard-label'>Employees</div></div>", unsafe_allow_html=True)
                    cols[1].markdown(f"<div class='scorecard'><div class='scorecard-value'>${filtered['Monthly Salary'].sum():,.0f}</div><div class='scorecard-label'>Total Salary</div></div>" if 'Monthly Salary' in filtered else "<div class='scorecard'><div class='scorecard-value'>$0</div><div class='scorecard-label'>Salary</div></div>", unsafe_allow_html=True)
                    cols[2].markdown(f"<div class='scorecard'><div class='scorecard-value'>${filtered['Monthly Salary'].mean():,.0f}</div><div class='scorecard-label'>Avg Salary</div></div>" if 'Monthly Salary' in filtered else "<div class='scorecard'><div class='scorecard-value'>$0</div><div class='scorecard-label'>Avg Salary</div></div>", unsafe_allow_html=True)
                    cols[3].markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Role'].nunique()}</div><div class='scorecard-label'>Roles</div></div>" if 'Role' in filtered else "<div class='scorecard'><div class='scorecard-value'>0</div><div class='scorecard-label'>Roles</div></div>", unsafe_allow_html=True)
                    
                    if 'Role' in filtered and 'Monthly Salary' in filtered:
                        chart_data = filtered.groupby('Role')['Monthly Salary'].sum().reset_index()
                        fig = create_chart(chart_data, 'bar', "Salary by Role")
                        if fig: st.plotly_chart(fig, use_container_width=True)
                
                elif current == 'RESOURCE_LINKS':
                    cols[0].markdown(f"<div class='scorecard'><div class='scorecard-value'>{len(filtered)}</div><div class='scorecard-label'>Resources</div></div>", unsafe_allow_html=True)
                    cols[1].markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Type'].nunique() if 'Type' in filtered else 0}</div><div class='scorecard-label'>Types</div></div>", unsafe_allow_html=True)
                    cols[2].markdown(f"<div class='scorecard'><div class='scorecard-value'>{filtered['Category'].nunique() if 'Category' in filtered else 0}</div><div class='scorecard-label'>Categories</div></div>", unsafe_allow_html=True)
                
                with st.expander("📋 Details", expanded=True):
                    st.markdown(f"<div class='table-header'><span>📄 {current.replace('_',' ').title()}</span><span class='table-shape'>{filtered.shape[0]} rows × {filtered.shape[1]} cols</span></div>", unsafe_allow_html=True)
                    st.dataframe(filtered, use_container_width=True, hide_index=True)
        else:
            st.info("📭 No data available")

st.markdown(f'<meta http-equiv="refresh" content="{REFRESH_INTERVAL}">', unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center;color:#999;font-size:0.8rem;padding:0.5rem;'>Project Pulse • Auto-refreshes every {REFRESH_INTERVAL}s</div>", unsafe_allow_html=True)
