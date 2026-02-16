# --- Custom CSS with mobile-responsive header ---
st.markdown("""
<style>
    /* Modern gradient header - mobile responsive */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Desktop layout */
    .header-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }
    
    .header-title-section {
        flex: 1;
        min-width: 250px;
    }
    
    .header-title {
        margin: 0;
        font-size: 1.8rem;
        line-height: 1.2;
    }
    
    .header-subtitle {
        margin: 0.3rem 0 0 0;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    
    .live-badge {
        background: #ff4444;
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        animation: pulse 2s infinite;
        white-space: nowrap;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .timestamp {
        color: white;
        font-size: 0.8rem;
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.3rem;
        background: rgba(255,255,255,0.1);
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
    }
    
    .pk-badge {
        background: #2c3e50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 500;
        white-space: nowrap;
    }
    
    /* Scorecards section - scrollable on mobile */
    .header-scorecards {
        display: flex;
        gap: 0.5rem;
        overflow-x: auto;
        padding: 0.3rem 0;
        max-width: 100%;
        scrollbar-width: thin;
        scrollbar-color: rgba(255,255,255,0.5) rgba(255,255,255,0.1);
        -webkit-overflow-scrolling: touch;
    }
    
    .header-scorecards::-webkit-scrollbar {
        height: 4px;
    }
    
    .header-scorecards::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    
    .header-scorecards::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.5);
        border-radius: 10px;
    }
    
    .header-scorecard {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 0.8rem;
        border-radius: 12px;
        text-align: center;
        min-width: 70px;
        backdrop-filter: blur(10px);
        border-left: 3px solid rgba(255,255,255,0.5);
        flex-shrink: 0;
    }
    
    .header-scorecard-value {
        font-size: 1rem;
        font-weight: 700;
        color: white;
        line-height: 1.2;
        white-space: nowrap;
    }
    
    .header-scorecard-label {
        font-size: 0.6rem;
        color: rgba(255,255,255,0.9);
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    .header-scorecard-sub {
        font-size: 0.55rem;
        color: rgba(255,255,255,0.8);
        white-space: nowrap;
    }
    
    /* Responsive breakpoints */
    @media (max-width: 768px) {
        .header-content {
            flex-direction: column;
            align-items: flex-start;
        }
        
        .header-title-section {
            width: 100%;
        }
        
        .header-title {
            font-size: 1.5rem;
        }
        
        .header-subtitle {
            font-size: 0.8rem;
        }
        
        .header-scorecards {
            width: 100%;
            margin-top: 0.5rem;
        }
        
        .header-scorecard {
            min-width: 65px;
            padding: 0.4rem 0.6rem;
        }
        
        .header-scorecard-value {
            font-size: 0.9rem;
        }
    }
    
    @media (max-width: 480px) {
        .header-title {
            font-size: 1.3rem;
        }
        
        .timestamp {
            font-size: 0.7rem;
        }
        
        .header-scorecard {
            min-width: 60px;
            padding: 0.3rem 0.5rem;
        }
        
        .header-scorecard-value {
            font-size: 0.8rem;
        }
        
        .header-scorecard-label {
            font-size: 0.55rem;
        }
        
        .header-scorecard-sub {
            font-size: 0.5rem;
        }
    }
    
    /* Rest of your existing CSS remains the same */
    .secondary-scorecards { background: #f8f9fa; padding: 1rem; border-radius: 15px; margin: 1rem 0; border: 1px solid #dee2e6; }
    .scorecard { background: white; padding: 0.8rem; border-radius: 10px; text-align: center; border-left: 4px solid #667eea; }
    .scorecard-value { font-size: 1.5rem; font-weight: 700; color: #2d3748; }
    .scorecard-label { font-size: 0.8rem; color: #718096; }
    .tab-container { background: white; border-radius: 15px; padding: 1rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    .stButton > button { width: 100%; text-align: left; border-radius: 10px; margin: 0.2rem 0; }
    .table-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem; }
    .table-name { font-weight: 600; color: #2d3748; font-size: 1rem; }
    .table-shape { background: #e2e8f0; padding: 0.2rem 0.8rem; border-radius: 15px; font-size: 0.8rem; color: #4a5568; white-space: nowrap; }
    .no-filters-msg { background: #e2e8f0; padding: 0.5rem 1rem; border-radius: 10px; color: #4a5568; font-size: 0.9rem; text-align: center; }
    #MainMenu, footer, .stDeployButton { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Header with Integrated Status and Scorecards (Mobile Responsive) ---
if st.session_state.data_sheets and not st.session_state.show_original:
    df_p = st.session_state.data_sheets['PROJECT_MASTER']
    df_w = st.session_state.data_sheets['DAILY_WORK_LOG']
    df_c = st.session_state.data_sheets['EMPLOYEE_COST']
    df_t = st.session_state.data_sheets['TASK_PLAN']
    df_r = st.session_state.data_sheets['RESOURCE_LINKS']
    
    # Calculate metrics
    total_projects = len(df_p) if not df_p.empty else 0
    active_projects = len(df_p[df_p['Status'] == 'Active']) if not df_p.empty and 'Status' in df_p.columns else 0
    total_hours = df_w['Hours Worked'].sum() if not df_w.empty else 0
    total_salary = df_c['Monthly Salary'].sum() if not df_c.empty and 'Monthly Salary' in df_c.columns else 0
    total_tasks = len(df_t) if not df_t.empty else 0
    pending_tasks = len(df_t[df_t['Status'] != 'Done']) if not df_t.empty and 'Status' in df_t.columns else 0
    total_resources = len(df_r) if not df_r.empty else 0
    
    timestamp_str = st.session_state.last_update.strftime("%a, %d %b, %Y, %I:%M:%S %p") if st.session_state.last_update else ""
    
    st.markdown(f"""
    <div class="header-container">
        <div class="header-content">
            <div class="header-title-section">
                <h1 class="header-title">📊 Project Pulse</h1>
                <div class="header-subtitle">
                    <span class="live-badge">LIVE</span>
                    <span class="timestamp">
                        <span>🔄 {timestamp_str}</span>
                        <span class="pk-badge">PKT</span>
                    </span>
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
    </div>
    """, unsafe_allow_html=True)
