import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import pytz

st.set_page_config(layout="wide", page_title="Project Pulse", page_icon="📊", initial_sidebar_state="collapsed")

REFRESH_INTERVAL, PAKISTAN_TZ = 5, pytz.timezone('Asia/Karachi')
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQFttuVQlH84hCC-brrcJFa6eyrMeyc25Aqm_dLgfpuEBr0WCdc4OTKKZVK2Y6IfOoPdQFbYmSdrSYP/pub?output=xlsx"

for key in ['tab', 'data', 'original', 'filters']:
    if key not in st.session_state:
        st.session_state[key] = 'PROJECT_MASTER' if key == 'tab' else ({} if key == 'filters' else None if key != 'original' else False)

@st.cache_data(ttl=REFRESH_INTERVAL)
def load_data():
    try:
        xl = pd.ExcelFile(DATA_URL)
        sheets = {name: pd.read_excel(xl, sheet) for name, sheet in 
                 [('PROJECT_MASTER','PROJECT_MASTER'), ('DAILY_WORK_LOG','DAILY_WORK_LOG'),
                  ('EMPLOYEE_COST','EMPLOYEE_COST'), ('RESOURCE_LINKS','RESOURCE_LINKS'),
                  ('TASK_PLAN','TASK PLAN + RESPONSIBILITY')]}
        return sheets, datetime.now(PAKISTAN_TZ)
    except:
        return {name: pd.DataFrame() for name in ['PROJECT_MASTER','DAILY_WORK_LOG','EMPLOYEE_COST','RESOURCE_LINKS','TASK_PLAN']}, None

data, last = load_data()
if last: st.session_state.update({'data': data, 'last': last})

st.markdown("""
<style>
.header-container{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:1rem 2rem;border-radius:20px;margin-bottom:1rem;color:white;display:flex;align-items:center;justify-content:space-between}
.header-title{margin:0;font-size:2rem}
.live-badge{background:#ff4444;color:white;padding:0.2rem 0.7rem;border-radius:20px;font-size:0.75rem;animation:pulse 2s infinite}
@keyframes pulse{0%{opacity:1}50%{opacity:0.7}100%{opacity:1}}
.timestamp{color:white;font-size:0.9rem}
.pk-badge{background:#2c3e50;color:white;padding:0.2rem 0.7rem;border-radius:20px}
.header-scorecards{display:flex;gap:1rem}
.header-scorecard{background:rgba(255,255,255,0.2);padding:0.5rem 1rem;border-radius:12px;text-align:center;min-width:90px}
.header-scorecard-value{font-size:1.2rem;font-weight:700}
.secondary-scorecards{background:#f8f9fa;padding:1rem;border-radius:15px;margin:1rem 0;border:1px solid #dee2e6}
.scorecard{background:white;padding:0.8rem;border-radius:10px;text-align:center;border-left:4px solid #667eea}
.scorecard-value{font-size:1.5rem;font-weight:700;color:#2d3748}
.table-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem}
.table-shape{background:#e2e8f0;padding:0.2rem 0.8rem;border-radius:15px;font-size:0.8rem}
.stButton>button{width:100%;text-align:left}
#MainMenu,footer,.stDeployButton{visibility:hidden}
</style>
""", unsafe_allow_html=True)

if st.session_state.data and not st.session_state.original:
    df = {k: v for k, v in st.session_state.data.items()}
    ts = st.session_state.last.strftime("%a, %d %b, %Y, %I:%M:%S %p")
    st.markdown(f"""
    <div class="header-container">
        <div><h1 class="header-title">📊 Project Pulse</h1><p><span class="live-badge">LIVE</span> <span class="timestamp">🔄 {ts} <span class="pk-badge">PKT</span></span></p></div>
        <div class="header-scorecards">
            <div class="header-scorecard"><div class="header-scorecard-value">{len(df['PROJECT_MASTER'])}</div><div>Projects</div><div>{len(df['PROJECT_MASTER'][df['PROJECT_MASTER']['Status']=='Active']) if 'Status' in df['PROJECT_MASTER'] else 0} Active</div></div>
            <div class="header-scorecard"><div class="header-scorecard-value">{df['DAILY_WORK_LOG']['Hours Worked'].sum() if not df['DAILY_WORK_LOG'].empty else 0:.0f}</div><div>Hours</div></div>
            <div class="header-scorecard"><div class="header-scorecard-value">${df['EMPLOYEE_COST']['Monthly Salary'].sum() if not df['EMPLOYEE_COST'].empty else 0:,.0f}</div><div>Salary</div></div>
            <div class="header-scorecard"><div class="header-scorecard-value">{len(df['TASK_PLAN'])}</div><div>Tasks</div><div>{len(df['TASK_PLAN'][df['TASK_PLAN']['Status']!='Done']) if 'Status' in df['TASK_PLAN'] else 0} Pending</div></div>
            <div class="header-scorecard"><div class="header-scorecard-value">{len(df['RESOURCE_LINKS'])}</div><div>Resources</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<div class='header-container'><div><h1 class='header-title'>📊 Project Pulse</h1><p>Original Sheets View</p></div></div>", unsafe_allow_html=True)

if st.session_state.original:
    st.markdown("### 📋 Original Sheets Data")
    if st.button("← Back to Dashboard"): st.session_state.original = False; st.rerun()
    tabs = st.tabs(list(st.session_state.data.keys()))
    for i, (name, df) in enumerate(st.session_state.data.items()):
        with tabs[i]:
            st.markdown(f"<div class='table-header'><span>📄 {name}</span><span class='table-shape'>{df.shape[0]} rows × {df.shape[1]} cols</span></div>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(f"📥 Download", df.to_csv(index=False).encode(), f"{name}.csv")
else:
    l, r = st.columns([1, 4])
    with l:
        st.markdown("<div style='background:white;padding:1rem;border-radius:15px'>")
        st.markdown("📋 Navigation")
        tabs = {'PROJECT_MASTER':'📁 Projects','DAILY_WORK_LOG':'📝 Work Log','EMPLOYEE_COST':'💰 Costs','RESOURCE_LINKS':'🔗 Resources','TASK_PLAN':'✅ Tasks'}
        for k, v in tabs.items():
            if st.button(v, key=k, use_container_width=True, type="secondary" if st.session_state.tab != k else "primary"):
                st.session_state.tab = k; st.session_state.filters = {}; st.rerun()
        st.markdown("<hr>")
        if st.button("📋 View Original Sheets", use_container_width=True): st.session_state.original = True; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    with r:
        cur, df = st.session_state.tab, st.session_state.data[st.session_state.tab].copy()
        st.markdown(f"### {tabs[cur]}")
        
        if not df.empty:
            if cur != 'RESOURCE_LINKS':
                with st.expander("🔍 Filters"):
                    cols = st.columns([3, 1])
                    with cols[0]:
                        if cur == 'PROJECT_MASTER':
                            c1, c2, c3 = st.columns(3)
                            with c1: f1 = st.selectbox('Company', ['All'] + [x for x in df['Company name'].dropna().unique() if pd.notna(x)], key='comp')
                            with c2: f2 = st.selectbox('Status', ['All'] + [x for x in df['Status'].dropna().unique() if pd.notna(x)], key='stat')
                            with c3: f3 = st.selectbox('Quarter', ['All'] + [x for x in df['Quarter'].dropna().unique() if pd.notna(x)], key='quart')
                            st.session_state.filters = {'company':f1,'status':f2,'quarter':f3}
                        elif cur == 'TASK_PLAN':
                            c1, c2, c3 = st.columns(3)
                            with c1: f1 = st.selectbox('Priority', ['All'] + [x for x in df['Priority'].dropna().unique() if pd.notna(x)], key='pri')
                            with c2: f2 = st.selectbox('Status', ['All'] + [x for x in df['Status'].dropna().unique() if pd.notna(x)], key='stat2')
                            with c3: f3 = st.selectbox('Owner', ['All'] + [x for x in df['Owner (Team / Client)'].dropna().unique() if pd.notna(x)], key='own')
                            st.session_state.filters = {'priority':f1,'status':f2,'owner':f3}
                        elif cur == 'DAILY_WORK_LOG':
                            f1 = st.selectbox('Employee', ['All'] + [x for x in df['Employee Name'].dropna().unique() if pd.notna(x)])
                            st.session_state.filters = {'employee':f1}
                        elif cur == 'EMPLOYEE_COST':
                            f1 = st.selectbox('Role', ['All'] + [x for x in df['Role'].dropna().unique() if pd.notna(x)])
                            st.session_state.filters = {'role':f1}
                    with cols[1]:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🔄 Reset", use_container_width=True): st.session_state.filters = {}; st.rerun()
            else:
                st.info("🔗 Resource Links - No filters")
            
            # Apply filters
            filtered = df.copy()
            if cur == 'PROJECT_MASTER' and st.session_state.filters:
                if st.session_state.filters.get('company','All')!='All': filtered = filtered[filtered['Company name']==st.session_state.filters['company']]
                if st.session_state.filters.get('status','All')!='All': filtered = filtered[filtered['Status']==st.session_state.filters['status']]
                if st.session_state.filters.get('quarter','All')!='All': filtered = filtered[filtered['Quarter']==st.session_state.filters['quarter']]
            elif cur == 'TASK_PLAN' and st.session_state.filters:
                if st.session_state.filters.get('priority','All')!='All': filtered = filtered[filtered['Priority']==st.session_state.filters['priority']]
                if st.session_state.filters.get('status','All')!='All': filtered = filtered[filtered['Status']==st.session_state.filters['status']]
                if st.session_state.filters.get('owner','All')!='All': filtered = filtered[filtered['Owner (Team / Client)']==st.session_state.filters['owner']]
            elif cur == 'DAILY_WORK_LOG' and st.session_state.filters.get('employee','All')!='All':
                filtered = filtered[filtered['Employee Name']==st.session_state.filters['employee']]
            elif cur == 'EMPLOYEE_COST' and st.session_state.filters.get('role','All')!='All':
                filtered = filtered[filtered['Role']==st.session_state.filters['role']]
            
            # Metrics
            st.markdown("<div class='secondary-scorecards'><h5>📊 Metrics</h5><div style='display:grid;grid-template-columns:repeat(4,1fr);gap:1rem'>", unsafe_allow_html=True)
            metrics = {
                'PROJECT_MASTER': [(len(filtered),'Projects'), (filtered['Budget'].sum() if 'Budget' in filtered else 0,'Budget $'),
                                   (len(filtered[filtered['Status']=='Active']) if 'Status' in filtered else 0,'Active'),
                                   (filtered['Company name'].nunique() if 'Company name' in filtered else 0,'Companies')],
                'TASK_PLAN': [(len(filtered),'Tasks'), (len(filtered[filtered['Status']=='Done']) if 'Status' in filtered else 0,'Done'),
                             (len(filtered[filtered['Priority']=='High']) if 'Priority' in filtered else 0,'High Priority'),
                             (filtered['Owner (Team / Client)'].nunique() if 'Owner (Team / Client)' in filtered else 0,'Owners')],
                'DAILY_WORK_LOG': [(len(filtered),'Entries'), (filtered['Hours Worked'].sum() if 'Hours Worked' in filtered else 0,'Total Hrs'),
                                  (filtered['Hours Worked'].mean() if 'Hours Worked' in filtered else 0,'Avg Hrs'),
                                  (filtered['Employee Name'].nunique() if 'Employee Name' in filtered else 0,'Employees')],
                'EMPLOYEE_COST': [(len(filtered),'Employees'), (filtered['Monthly Salary'].sum() if 'Monthly Salary' in filtered else 0,'Total $'),
                                 (filtered['Monthly Salary'].mean() if 'Monthly Salary' in filtered else 0,'Avg $'),
                                 (filtered['Role'].nunique() if 'Role' in filtered else 0,'Roles')],
                'RESOURCE_LINKS': [(len(filtered),'Resources'), (filtered['Type'].nunique() if 'Type' in filtered else 0,'Types'),
                                  (filtered['Category'].nunique() if 'Category' in filtered else 0,'Categories'), (0,'')]
            }
            for v, l in metrics[cur][:4]:
                st.markdown(f"<div class='scorecard'><div class='scorecard-value'>{v if isinstance(v,str) else f'{v:,.0f}'}</div><div>{l}</div></div>", unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Charts
            if cur == 'PROJECT_MASTER' and 'Company name' in filtered and 'Budget' in filtered:
                st.plotly_chart(px.bar(filtered.groupby('Company name')['Budget'].sum().reset_index(), x='Company name', y='Budget'), use_container_width=True)
            elif cur == 'TASK_PLAN' and 'Owner (Team / Client)' in filtered:
                st.plotly_chart(px.pie(filtered['Owner (Team / Client)'].value_counts().reset_index(), values='count', names='Owner (Team / Client)'), use_container_width=True)
            elif cur == 'DAILY_WORK_LOG' and 'Employee Name' in filtered and 'Hours Worked' in filtered:
                st.plotly_chart(px.bar(filtered.groupby('Employee Name')['Hours Worked'].sum().reset_index(), x='Employee Name', y='Hours Worked'), use_container_width=True)
            elif cur == 'EMPLOYEE_COST' and 'Role' in filtered and 'Monthly Salary' in filtered:
                st.plotly_chart(px.bar(filtered.groupby('Role')['Monthly Salary'].sum().reset_index(), x='Role', y='Monthly Salary'), use_container_width=True)
            
            # Table
            with st.expander("📋 Details", expanded=True):
                st.markdown(f"<div class='table-header'><span>📄 {cur}</span><span class='table-shape'>{filtered.shape[0]} rows × {filtered.shape[1]} cols</span></div>", unsafe_allow_html=True)
                st.dataframe(filtered, use_container_width=True, hide_index=True)

st.markdown(f'<meta http-equiv="refresh" content="{REFRESH_INTERVAL}"><hr><div style="text-align:center;color:#999">Project Pulse • Auto-refreshes every {REFRESH_INTERVAL}s</div>', unsafe_allow_html=True)
