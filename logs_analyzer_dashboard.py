import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pytz
import io

# Format duration function
def format_duration(seconds):
    if pd.isna(seconds):
        return ""
    seconds = int(seconds)
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# Get Ghana time
def get_ghana_time():
    ghana_tz = pytz.timezone('Africa/Accra')
    return datetime.now(ghana_tz)

# Process data function (same as before)
def process_data(df, start_date=None, end_date=None, selected_devices=None):
    try:
        df = df.copy()
        
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            df = df[(df['Record Time'] >= start_date) & (df['Record Time'] <= end_date + timedelta(days=1))]
        
        if selected_devices and len(selected_devices) > 0:
            df = df[df['Device Name'].isin(selected_devices)]
        
        if df.empty:
            empty_summary = pd.DataFrame(columns=[
                'Device', 'Current_Status', 'Last_Offline_Time', 
                'Total_DownTime_Events', 'Current_Downtime_Duration', 
                'Total_Downtime_Duration'
            ])
            empty_downtime = pd.DataFrame(columns=[
                'Device', 'Offline_Time', 'Online_Time', 
                'Downtime_Duration', 'Downtime_Status'
            ])
            return empty_summary, empty_downtime, get_ghana_time()
        
        current_time = pd.Timestamp.now(tz='Africa/Accra')
        
        df_downtime = (
            df
            .assign(next_status=df.groupby('Device Name')['status'].shift(-1),
                    next_time=df.groupby('Device Name')['Record Time'].shift(-1),
                    prev_status=df.groupby('Device Name')['status'].shift(1))
            .loc[lambda x: x['status'] == 'offline']
            .assign(
                Downtime_Seconds=lambda x: np.where(
                    x['next_status'] == 'online',
                    (x['next_time'] - x['Record Time']).dt.total_seconds(),
                    np.nan
                ),
                Downtime_Status=lambda x: np.where(
                    x['next_status'] == 'online',
                    'Completed',
                    np.where(
                        x['prev_status'] == 'online',
                        'Ongoing',
                        'Intermediate'
                    )
                )
            )
            .rename(columns={'Record Time': 'Offline_Time', 'next_time': 'Online_Time',
                             'Device Name': 'Device'})
            [['Device', 'Offline_Time', 'Online_Time', 'Downtime_Seconds', 'Downtime_Status']]
        )
        
        if df_downtime.empty:
            empty_summary = pd.DataFrame(columns=[
                'Device', 'Current_Status', 'Last_Offline_Time', 
                'Total_DownTime_Events', 'Current_Downtime_Duration', 
                'Total_Downtime_Duration'
            ])
            return empty_summary, df_downtime, current_time
        
        mask = (df_downtime['Online_Time'].notna()) & (df_downtime['Downtime_Status'] == 'Ongoing')
        df_downtime.loc[mask, 'Downtime_Status'] = 'Completed'
        
        def recalculate_downtime(row):
            try:
                if row['Downtime_Status'] == 'Completed':
                    return row['Downtime_Seconds'] if not pd.isna(row['Downtime_Seconds']) else 0
                elif pd.notna(row['Online_Time']):
                    return (row['Online_Time'] - row['Offline_Time']).total_seconds()
                else:
                    offline_time = row['Offline_Time']
                    if isinstance(offline_time, pd.Timestamp):
                        if offline_time.tz is not None:
                            offline_time = offline_time.tz_localize(None)
                        return (current_time.tz_localize(None) - offline_time).total_seconds()
                    else:
                        offline_time = pd.to_datetime(offline_time)
                        return (current_time.tz_localize(None) - offline_time).total_seconds()
            except Exception as e:
                return 0
        
        df_downtime = df_downtime.copy()
        df_downtime.loc[:, 'Downtime_Seconds'] = df_downtime.apply(recalculate_downtime, axis=1)
        df_downtime.loc[:, 'Downtime_Duration'] = df_downtime['Downtime_Seconds'].apply(format_duration)
        
        analysis_time = pd.Timestamp.now(tz='Africa/Accra')
        
        if df_downtime['Device'].nunique() == 0:
            empty_summary = pd.DataFrame(columns=[
                'Device', 'Current_Status', 'Last_Offline_Time', 
                'Total_DownTime_Events', 'Current_Downtime_Duration', 
                'Total_Downtime_Duration'
            ])
            return empty_summary, df_downtime, analysis_time
        
        summary = (
            df_downtime.groupby('Device')
            .agg({
                'Offline_Time': 'last',
                'Online_Time': 'last',
                'Downtime_Seconds': ['count', 'sum'],
                'Downtime_Status': lambda x: (x == 'Ongoing').sum()
            })
            .reset_index()
        )
        
        if summary.empty:
            empty_summary = pd.DataFrame(columns=[
                'Device', 'Current_Status', 'Last_Offline_Time', 
                'Total_DownTime_Events', 'Current_Downtime_Duration', 
                'Total_Downtime_Duration'
            ])
            return empty_summary, df_downtime, analysis_time
        
        summary.columns = [
            'Device', 'Last_Offline_Time', 'Last_Online_Time',
            'Total_DownTime_Events', 'Total_Downtime_Seconds', 'Ongoing_Count'
        ]
        
        summary['Total_Downtime_Seconds'] = pd.to_numeric(summary['Total_Downtime_Seconds'], errors='coerce').fillna(0).round(0)
        
        def calculate_current_downtime(row):
            try:
                if row['Ongoing_Count'] > 0:
                    offline_time = row['Last_Offline_Time']
                    if isinstance(offline_time, pd.Timestamp):
                        if offline_time.tz is not None:
                            offline_time = offline_time.tz_localize(None)
                        return (analysis_time.tz_localize(None) - offline_time).total_seconds()
                    else:
                        offline_time = pd.to_datetime(offline_time)
                        return (analysis_time.tz_localize(None) - offline_time).total_seconds()
                else:
                    return np.nan
            except:
                return np.nan
        
        summary['Current_Downtime_Seconds'] = summary.apply(calculate_current_downtime, axis=1).round(0)
        
        ongoing_mask = summary['Ongoing_Count'] > 0
        summary.loc[ongoing_mask, 'Total_Downtime_Seconds'] = np.maximum(
            summary.loc[ongoing_mask, 'Total_Downtime_Seconds'],
            summary.loc[ongoing_mask, 'Current_Downtime_Seconds'].fillna(0)
        )
        
        summary['Current_Downtime_Duration'] = summary['Current_Downtime_Seconds'].apply(lambda x: format_duration(x) if not pd.isna(x) else "")
        summary['Total_Downtime_Duration'] = summary['Total_Downtime_Seconds'].apply(format_duration)
        summary['Current_Status'] = np.where(summary['Ongoing_Count'] > 0, '🔴 Offline', '✔️ Online')
        
        df_downtime['Downtime_Status'] = np.where(df_downtime['Downtime_Status'] == 'Ongoing', '🔴 Ongoing', '✔️ Completed')
        
        df_downtime_display = df_downtime[['Device', 'Offline_Time', 'Online_Time', 'Downtime_Duration', 'Downtime_Status']]
        
        return summary, df_downtime_display, analysis_time
    
    except Exception as e:
        st.error(f"Error in process_data: {str(e)}")
        empty_summary = pd.DataFrame(columns=[
            'Device', 'Current_Status', 'Last_Offline_Time', 
            'Total_DownTime_Events', 'Current_Downtime_Duration', 
            'Total_Downtime_Duration'
        ])
        empty_downtime = pd.DataFrame(columns=[
            'Device', 'Offline_Time', 'Online_Time', 
            'Downtime_Duration', 'Downtime_Status'
        ])
        return empty_summary, empty_downtime, get_ghana_time()

# Mobile-friendly CSS
def inject_mobile_css():
    st.markdown("""
    <style>
    /* Mobile-first responsive design */
    @media (max-width: 768px) {
        /* Hide Streamlit header elements */
        header { visibility: hidden; }
        .st-emotion-cache-1avcm0n { display: none; }
        
        /* Better mobile spacing */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        /* Mobile-friendly tables */
        .dataframe {
            font-size: 12px !important;
        }
        
        .dataframe td, .dataframe th {
            padding: 4px 6px !important;
        }
        
        /* Full-width buttons on mobile */
        .stButton > button {
            width: 100%;
        }
        
        /* Compact metrics */
        .stMetric {
            padding: 8px !important;
            min-height: 70px;
        }
        
        .stMetric label {
            font-size: 11px !important;
        }
        
        .stMetric div[data-testid="stMetricValue"] {
            font-size: 16px !important;
        }
        
        /* Collapse sidebar by default on mobile */
        [data-testid="stSidebar"] {
            min-width: 0px !important;
            max-width: 100vw !important;
            transform: translateX(-100%);
        }
        
        [data-testid="stSidebar"][aria-expanded="true"] {
            transform: translateX(0);
        }
        
        /* Mobile-friendly select boxes */
        .stSelectbox, .stDateInput, .stMultiselect {
            font-size: 14px !important;
        }
        
        /* Touch-friendly elements */
        button, [role="button"], input, select, textarea {
            min-height: 44px !important;
        }
        
        /* Hide desktop-only elements */
        .desktop-only {
            display: none !important;
        }
    }
    
    /* Desktop styles */
    @media (min-width: 769px) {
        .mobile-only {
            display: none !important;
        }
    }
    
    /* Common improvements */
    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .status-online {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-offline {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .timestamp {
        font-size: 12px;
        color: #666;
        text-align: center;
        padding: 8px;
        background: #f8f9fa;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    
    /* Tab styling */
    .mobile-tab {
        display: inline-block;
        padding: 10px 15px;
        margin: 2px;
        border-radius: 8px;
        background: #f0f2f6;
        cursor: pointer;
        text-align: center;
        flex: 1;
    }
    
    .mobile-tab.active {
        background: #4B6FFF;
        color: white;
        font-weight: bold;
    }
    
    .tab-content {
        padding-top: 15px;
    }
    
    /* Scrollable containers for mobile */
    .scrollable-table {
        overflow-x: auto;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit App
def main():
    # Mobile-first configuration
    st.set_page_config(
        page_title="Device Downtime Mobile",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Inject mobile CSS
    inject_mobile_css()
    
    # Initialize session state
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "dashboard"
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'downtime' not in st.session_state:
        st.session_state.downtime = None
    
    # Get Ghana time
    ghana_time = get_ghana_time()
    current_time_str = ghana_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    
    # Mobile header with time
    st.markdown(f'<div class="timestamp">⏰ Ghana Time: {current_time_str}</div>', unsafe_allow_html=True)
    
    # Mobile Tabs Navigation (Simulated bottom menu)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Dashboard", use_container_width=True, 
                    type="primary" if st.session_state.current_tab == "dashboard" else "secondary"):
            st.session_state.current_tab = "dashboard"
            st.rerun()
    
    with col2:
        if st.button("📋 Summary", use_container_width=True,
                    type="primary" if st.session_state.current_tab == "summary" else "secondary"):
            st.session_state.current_tab = "summary"
            st.rerun()
    
    with col3:
        if st.button("🔍 Downtime", use_container_width=True,
                    type="primary" if st.session_state.current_tab == "downtime" else "secondary"):
            st.session_state.current_tab = "downtime"
            st.rerun()
    
    with col4:
        if st.button("⚙️ Filters", use_container_width=True,
                    type="primary" if st.session_state.current_tab == "filters" else "secondary"):
            st.session_state.current_tab = "filters"
            st.rerun()
    
    st.divider()
    
    # Show content based on current tab
    if st.session_state.current_tab == "dashboard":
        show_dashboard()
    elif st.session_state.current_tab == "summary":
        show_summary()
    elif st.session_state.current_tab == "downtime":
        show_downtime()
    elif st.session_state.current_tab == "filters":
        show_filters()
    else:
        show_dashboard()

def show_dashboard():
    """Show dashboard view"""
    st.markdown("<h3 style='text-align: center;'>📱 Device Dashboard</h3>", unsafe_allow_html=True)
    
    if not st.session_state.processed:
        st.info("👋 Welcome! Go to ⚙️ Filters tab to upload a CSV file")
        return
    
    summary = st.session_state.summary
    downtime = st.session_state.downtime
    
    # Quick stats
    if not summary.empty:
        total_online = len(summary[summary['Current_Status'] == '✔️ Online'])
        total_offline = len(summary[summary['Current_Status'] == '🔴 Offline'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", len(summary))
        with col2:
            st.metric("✅ Online", total_online)
        with col3:
            st.metric("🔴 Offline", total_offline)
    
    st.divider()
    
    # Recent activity
    st.subheader("Recent Status")
    if not summary.empty:
        # Show top devices
        display_df = summary[['Device', 'Current_Status', 'Current_Downtime_Duration']].head(8)
        
        # Color coding
        def color_cells(val):
            if val == '✔️ Online':
                return 'background-color: #d4edda; color: #155724;'
            elif val == '🔴 Offline':
                return 'background-color: #f8d7da; color: #721c24;'
            return ''
        
        styled_df = display_df.style.map(color_cells, subset=['Current_Status'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Recent downtime
    if not downtime.empty:
        st.divider()
        st.subheader("Recent Downtime")
        recent_events = downtime.head(5)[['Device', 'Offline_Time', 'Downtime_Status']]
        st.dataframe(recent_events, use_container_width=True, hide_index=True)

def show_summary():
    """Show summary view"""
    st.markdown("<h3 style='text-align: center;'>📋 Device Summary</h3>", unsafe_allow_html=True)
    
    if not st.session_state.processed:
        st.info("No data loaded. Go to ⚙️ Filters tab first.")
        return
    
    summary = st.session_state.summary
    
    if summary.empty:
        st.warning("No summary data available")
        return
    
    # Filter options
    col1, col2 = st.columns([2, 1])
    with col1:
        status_filter = st.selectbox(
            "Filter by status",
            ["All", "Online", "Offline"],
            label_visibility="collapsed"
        )
    
    # Apply filter
    if status_filter == "Online":
        filtered = summary[summary['Current_Status'] == '✔️ Online']
    elif status_filter == "Offline":
        filtered = summary[summary['Current_Status'] == '🔴 Offline']
    else:
        filtered = summary
    
    # Display
    display_cols = ['Device', 'Current_Status', 'Total_DownTime_Events', 
                   'Current_Downtime_Duration', 'Total_Downtime_Duration']
    
    st.markdown('<div class="scrollable-table">', unsafe_allow_html=True)
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.caption(f"Showing {len(filtered)} devices")
    
    # Download button
    if not filtered.empty:
        excel_buffer = io.BytesIO()
        filtered[display_cols].to_excel(excel_buffer, index=False)
        
        st.download_button(
            label="📥 Download Summary",
            data=excel_buffer.getvalue(),
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True
        )

def show_downtime():
    """Show downtime view"""
    st.markdown("<h3 style='text-align: center;'>🔍 Downtime Events</h3>", unsafe_allow_html=True)
    
    if not st.session_state.processed:
        st.info("No data loaded. Go to ⚙️ Filters tab first.")
        return
    
    downtime = st.session_state.downtime
    
    if downtime.empty:
        st.warning("No downtime events found")
        return
    
    # Filter options
    col1, col2 = st.columns([2, 1])
    with col1:
        status_filter = st.selectbox(
            "Filter events",
            ["All", "Completed", "Ongoing"],
            label_visibility="collapsed"
        )
    
    # Apply filter
    if status_filter == "Completed":
        filtered = downtime[downtime['Downtime_Status'] == '✔️ Completed']
    elif status_filter == "Ongoing":
        filtered = downtime[downtime['Downtime_Status'] == '🔴 Ongoing']
    else:
        filtered = downtime
    
    # Display
    st.markdown('<div class="scrollable-table">', unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.caption(f"Showing {len(filtered)} events")
    
    # Download button
    if not filtered.empty:
        excel_buffer = io.BytesIO()
        filtered.to_excel(excel_buffer, index=False)
        
        st.download_button(
            label="📥 Download Downtime",
            data=excel_buffer.getvalue(),
            file_name=f"downtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True
        )

def show_filters():
    """Show filters/upload view"""
    st.markdown("<h3 style='text-align: center;'>⚙️ Upload & Filters</h3>", unsafe_allow_html=True)
    
    # File upload section
    st.subheader("📁 Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        key="file_uploader",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing data..."):
                df = pd.read_csv(uploaded_file)
                
                # Data preprocessing
                df['Record Time Format'] = pd.to_datetime(
                    df['Record Time'], 
                    dayfirst=True,
                    errors='coerce'
                )
                df['Record Time'] = df['Record Time Format']
                df = df.drop(columns=['Record Time Format'], errors='ignore')
                
                # Filter encoding records
                df = df[df['Type'].str.contains('encoding', case=False, na=False)]
                
                # Create status column
                df['status'] = 'unknown'
                df.loc[df['Type'].str.contains('online', case=False, na=False), 'status'] = 'online'
                df.loc[df['Type'].str.contains('offline', case=False, na=False), 'status'] = 'offline'
                df = df.drop(columns=['Type'], errors='ignore')
                
                df = df.sort_values(by=['Device Name', 'Record Time'], ascending=[True, True])
                
                # Store in session state
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                # Auto-process with default settings
                min_date = df['Record Time'].min().date()
                max_date = df['Record Time'].max().date()
                
                summary, downtime, _ = process_data(
                    df.copy(),
                    pd.to_datetime(min_date),
                    pd.to_datetime(max_date),
                    []  # All devices
                )
                
                st.session_state.summary = summary
                st.session_state.downtime = downtime
                st.session_state.processed = True
                
                st.success(f"✅ File loaded! {len(df)} records processed")
                st.info(f"📱 Found {len(summary)} devices")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Filters section (only if data is loaded)
    if st.session_state.data_loaded and st.session_state.df is not None:
        st.divider()
        st.subheader("🔍 Apply Filters")
        
        df = st.session_state.df
        
        # Date range
        min_date = df['Record Time'].min().date()
        max_date = df['Record Time'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From", min_date, key="start_date")
        with col2:
            end_date = st.date_input("To", max_date, key="end_date")
        
        # Device filter
        all_devices = sorted(df['Device Name'].unique())
        selected_devices = st.multiselect(
            "Select devices (empty for all)",
            all_devices,
            default=[],
            key="device_filter"
        )
        
        # Apply filters button
        if st.button("🔄 Apply Filters", use_container_width=True, type="primary"):
            with st.spinner("Applying filters..."):
                summary, downtime, _ = process_data(
                    df.copy(),
                    pd.to_datetime(start_date),
                    pd.to_datetime(end_date),
                    selected_devices
                )
                
                st.session_state.summary = summary
                st.session_state.downtime = downtime
                st.session_state.processed = True
                
                st.success("✅ Filters applied!")
                st.rerun()
        
        # Clear data button
        if st.button("🗑️ Clear All Data", use_container_width=True, type="secondary"):
            for key in ['df', 'summary', 'downtime', 'processed', 'data_loaded']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Help section
    with st.expander("📋 CSV Format Help"):
        st.code("""
Required columns:
- Record Time: DD-MM-YYYY HH:MM:SS
- Device Name: Device identifier
- Type: Contains 'encoding online' or 'encoding offline'

Example:
Record Time,Device Name,Type
01-11-2023 10:00:00,Device1,encoding online
01-11-2023 10:05:00,Device1,encoding offline
01-11-2023 10:10:00,Device1,encoding online
        """)

if __name__ == "__main__":
    main()
