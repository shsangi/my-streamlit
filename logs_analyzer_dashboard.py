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

# Get Ghana time (Africa/Accra - GMT/UTC+0)
def get_ghana_time():
    # Ghana uses GMT/UTC (Accra time - no daylight saving)
    ghana_tz = pytz.timezone('Africa/Accra')
    return datetime.now(ghana_tz)

# Process data function with error handling
def process_data(df, start_date=None, end_date=None, selected_devices=None):
    try:
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Filter by date range if provided
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            df = df[(df['Record Time'] >= start_date) & (df['Record Time'] <= end_date + timedelta(days=1))]
        
        # Filter by selected devices if provided
        if selected_devices and len(selected_devices) > 0:
            df = df[df['Device Name'].isin(selected_devices)]
        
        # Check if we have data after filtering
        if df.empty:
            # Return empty dataframes with correct structure
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
        
        # Process downtime - use Ghana time consistently
        current_time = pd.Timestamp.now(tz='Africa/Accra')
        
        df_downtime = (
            df
            .assign(next_status=df.groupby('Device Name')['status'].shift(-1),
                    next_time=df.groupby('Device Name')['Record Time'].shift(-1),
                    prev_status=df.groupby('Device Name')['status'].shift(1))
            
            # Find all offline periods
            .loc[lambda x: x['status'] == 'offline']
            
            # Calculate downtime
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
        
        # Check if we have downtime data
        if df_downtime.empty:
            # Create empty summary with correct structure
            empty_summary = pd.DataFrame(columns=[
                'Device', 'Current_Status', 'Last_Offline_Time', 
                'Total_DownTime_Events', 'Current_Downtime_Duration', 
                'Total_Downtime_Duration'
            ])
            # Return empty dataframes
            return empty_summary, df_downtime, current_time
        
        # Fix misclassified records
        mask = (df_downtime['Online_Time'].notna()) & (df_downtime['Downtime_Status'] == 'Ongoing')
        df_downtime.loc[mask, 'Downtime_Status'] = 'Completed'
        
        # Recalculate downtime with error handling
        def recalculate_downtime(row):
            try:
                if row['Downtime_Status'] == 'Completed':
                    return row['Downtime_Seconds'] if not pd.isna(row['Downtime_Seconds']) else 0
                elif pd.notna(row['Online_Time']):
                    return (row['Online_Time'] - row['Offline_Time']).total_seconds()
                else:
                    # Calculate exact difference from offline time to current Ghana time
                    offline_time = row['Offline_Time']
                    if isinstance(offline_time, pd.Timestamp):
                        # Ensure offline_time has no timezone for consistent calculation
                        if offline_time.tz is not None:
                            offline_time = offline_time.tz_localize(None)
                        return (current_time.tz_localize(None) - offline_time).total_seconds()
                    else:
                        # Handle string or other datetime formats
                        offline_time = pd.to_datetime(offline_time)
                        return (current_time.tz_localize(None) - offline_time).total_seconds()
            except Exception as e:
                return 0
        
        # Safely assign Downtime_Seconds
        df_downtime = df_downtime.copy()
        df_downtime.loc[:, 'Downtime_Seconds'] = df_downtime.apply(recalculate_downtime, axis=1)
        df_downtime.loc[:, 'Downtime_Duration'] = df_downtime['Downtime_Seconds'].apply(format_duration)
        
        # Create summary with Ghana time
        analysis_time = pd.Timestamp.now(tz='Africa/Accra')
        
        # Check if we have data to group
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
        
        # Handle case where summary might be empty
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
        
        # Convert to appropriate types with error handling
        summary['Total_Downtime_Seconds'] = pd.to_numeric(summary['Total_Downtime_Seconds'], errors='coerce').fillna(0).round(0)
        
        # Calculate current downtime accurately using Ghana time
        def calculate_current_downtime(row):
            try:
                if row['Ongoing_Count'] > 0:
                    offline_time = row['Last_Offline_Time']
                    if isinstance(offline_time, pd.Timestamp):
                        # Ensure offline_time has no timezone for consistent calculation
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
        
        # Ensure Total >= Current for ongoing devices
        ongoing_mask = summary['Ongoing_Count'] > 0
        summary.loc[ongoing_mask, 'Total_Downtime_Seconds'] = np.maximum(
            summary.loc[ongoing_mask, 'Total_Downtime_Seconds'],
            summary.loc[ongoing_mask, 'Current_Downtime_Seconds'].fillna(0)
        )
        
        # Format durations with error handling
        summary['Current_Downtime_Duration'] = summary['Current_Downtime_Seconds'].apply(lambda x: format_duration(x) if not pd.isna(x) else "")
        summary['Total_Downtime_Duration'] = summary['Total_Downtime_Seconds'].apply(format_duration)
        summary['Current_Status'] = np.where(summary['Ongoing_Count'] > 0, '🔴 Offline', '✔️ Online')
        
        # Format downtime status with emojis and remove Downtime_Seconds column
        df_downtime['Downtime_Status'] = np.where(df_downtime['Downtime_Status'] == 'Ongoing', '🔴 Ongoing', '✔️ Completed')
        
        # Select only needed columns for downtime table (remove Downtime_Seconds)
        df_downtime_display = df_downtime[['Device', 'Offline_Time', 'Online_Time', 'Downtime_Duration', 'Downtime_Status']]
        
        return summary, df_downtime_display, analysis_time
    
    except Exception as e:
        st.error(f"Error in process_data: {str(e)}")
        # Return empty dataframes on error
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

# Add custom CSS for mobile bottom menu
def add_bottom_menu_css():
    st.markdown("""
    <style>
    /* Mobile bottom menu styling */
    @media (max-width: 768px) {
        /* Hide default sidebar on mobile */
        [data-testid="stSidebar"] {
            display: none;
        }
        
        /* Style for bottom navigation menu */
        .bottom-menu {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #0e1117;
            border-top: 1px solid #2b313e;
            padding: 10px 5px;
            z-index: 9999;
            display: flex;
            justify-content: space-around;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.3);
        }
        
        .bottom-menu-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 8px 12px;
            color: #d1d5db;
            text-decoration: none;
            border-radius: 8px;
            transition: all 0.2s;
            min-width: 70px;
            font-size: 12px;
            cursor: pointer;
            border: none;
            background: none;
            font-family: inherit;
        }
        
        .bottom-menu-item:hover {
            background-color: #262730;
            color: white;
        }
        
        .bottom-menu-item.active {
            background-color: #262730;
            color: #4ade80;
            font-weight: bold;
        }
        
        .bottom-menu-icon {
            font-size: 20px;
            margin-bottom: 4px;
        }
        
        /* Add padding to main content to prevent bottom menu overlap */
        .main .block-container {
            padding-bottom: 80px !important;
        }
        
        /* Adjust columns layout for mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        
        /* Make tables scroll horizontally on mobile */
        .stDataFrame {
            overflow-x: auto;
        }
        
        /* Adjust metric cards for mobile */
        [data-testid="stMetric"] {
            min-width: 100px;
        }
        
        /* Adjust buttons for mobile */
        .stButton button {
            width: 100%;
        }
        
        /* Adjust date inputs for mobile */
        .stDateInput {
            width: 100%;
        }
        
        /* Make select boxes full width on mobile */
        .stSelectbox {
            width: 100%;
        }
        
        /* Adjust multiselect for mobile */
        .stMultiSelect {
            width: 100%;
        }
        
        /* Hide sections by default, show only active one */
        .page-section {
            display: none;
        }
        
        .page-section.active {
            display: block;
        }
    }
    
    /* Desktop styles */
    @media (min-width: 769px) {
        .bottom-menu {
            display: none;
        }
        
        /* Show sidebar on desktop */
        [data-testid="stSidebar"] {
            display: block;
        }
        
        /* On desktop, show all sections */
        .page-section {
            display: block !important;
        }
    }
    
    /* General improvements for mobile */
    .stDataFrame table {
        font-size: 12px;
    }
    
    /* Improve mobile touch targets */
    .stButton button, .stDownloadButton button, .stSelectbox div {
        min-height: 44px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for current page
def init_session_state():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'summary'

# Streamlit App
def main():
    st.set_page_config(
        page_title="Device Downtime Report",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for mobile
    add_bottom_menu_css()
    
    # Initialize session state
    init_session_state()
    
    # Get Ghana time
    ghana_time = get_ghana_time()
    current_time_str = ghana_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    
    # Header with timestamp
    st.markdown(
        f'<div style="text-align: right; font-size: 0.8em; color: #666; margin-bottom: 10px;">'
        f'⏰ Analysis time (Ghana/Accra): {current_time_str}'
        f'</div>',
        unsafe_allow_html=True
    )
    
    st.title("📊 Device Downtime Report")
    
    # Initialize session state for status filters
    if 'summary_status_filter' not in st.session_state:
        st.session_state.summary_status_filter = "All"
    if 'downtime_status_filter' not in st.session_state:
        st.session_state.downtime_status_filter = "All"
    if 'analysis_time' not in st.session_state:
        st.session_state.analysis_time = None
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Report Controls")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'], 
                                         on_change=lambda: st.session_state.update({
                                             "data_loaded": False, 
                                             "processed": False,
                                             "last_error": None,
                                             "summary_status_filter": "All",
                                             "downtime_status_filter": "All",
                                             "analysis_time": None,
                                             "current_page": "summary"
                                         }))
        
        # Auto-process when file is uploaded
        if uploaded_file is not None and not st.session_state.data_loaded:
            try:
                with st.spinner("Loading and processing data..."):
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
                    
                    # Store the processed dataframe in session state
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    
                    # Auto-process with default settings
                    min_date = df['Record Time'].min().date()
                    max_date = df['Record Time'].max().date()
                    all_devices = sorted(df['Device Name'].unique())
                    
                    # Process with ALL devices selected
                    summary, downtime, analysis_time = process_data(
                        df.copy(),
                        pd.to_datetime(min_date),
                        pd.to_datetime(max_date),
                        []
                    )
                    
                    # Store in session state
                    st.session_state.summary = summary
                    st.session_state.downtime = downtime
                    st.session_state.analysis_time = analysis_time
                    st.session_state.processed = True
                    
                    st.success(f"File loaded and processed successfully!")
                    st.info(f"Total records: {len(df)}")
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.session_state.last_error = str(e)
        
        # Show filters and refresh button if data is loaded
        if 'df' in st.session_state and st.session_state.data_loaded:
            df = st.session_state.df
            
            # Date range selector with validation
            st.subheader("📅 Date Range Filter")
            min_date = df['Record Time'].min().date()
            max_date = df['Record Time'].max().date()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
            
            # Validate date range
            if start_date > end_date:
                st.error("❌ Start date must be before end date!")
                start_date, end_date = end_date, start_date
            
            # Device filter
            st.subheader("📱 Device Filter")
            all_devices = sorted(df['Device Name'].unique())
            
            selected_devices = st.multiselect(
                "Select devices (empty = all)",
                all_devices,
                default=[]
            )
            
            st.write("---")
            
            # Refresh Report button
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Refresh", use_container_width=True):
                    try:
                        with st.spinner("Updating report with new filters..."):
                            summary, downtime, analysis_time = process_data(
                                df.copy(),
                                pd.to_datetime(start_date),
                                pd.to_datetime(end_date),
                                selected_devices
                            )
                            
                            if summary.empty and downtime.empty:
                                st.warning("⚠️ No data found for the selected filters.")
                            else:
                                st.session_state.summary = summary
                                st.session_state.downtime = downtime
                                st.session_state.analysis_time = analysis_time
                                st.session_state.processed = True
                                st.success("✅ Refreshed!")
                            
                    except Exception as e:
                        st.error(f"❌ Error updating report: {str(e)}")
                        st.session_state.last_error = str(e)
            
            with col2:
                if st.button("🗑️ Reset", type="secondary", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            
            # Show current filter info
            st.caption(f"📅 Date range: {start_date} to {end_date}")
            if selected_devices:
                st.caption(f"📱 Showing {len(selected_devices)} of {len(all_devices)} devices")
            else:
                st.caption(f"📱 Showing all {len(all_devices)} devices")
        
        elif uploaded_file is None:
            st.warning("📁 Please upload a CSV file to begin")
    
    # Main content area - Different sections based on current page
    if 'processed' in st.session_state and st.session_state.processed:
        summary = st.session_state.summary
        downtime = st.session_state.downtime
        
        # Check if we have data
        if summary.empty and downtime.empty:
            st.warning("⚠️ No data found for the selected filters. Please adjust your criteria in the sidebar.")
        else:
            # SUMMARY PAGE
            with st.container():
                st.markdown(f'<div class="page-section {"active" if st.session_state.current_page == "summary" else ""}">', unsafe_allow_html=True)
                
                # Calculate online and offline counts
                total_online = len(summary[summary['Current_Status'] == '✔️ Online']) if not summary.empty else 0
                total_offline = len(summary[summary['Current_Status'] == '🔴 Offline']) if not summary.empty else 0
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Total Devices",
                        value=len(summary) if not summary.empty else 0,
                        delta=None
                    )
                with col2:
                    st.metric(
                        label="Total ✔️ Online",
                        value=total_online,
                        delta=None
                    )
                with col3:
                    st.metric(
                        label="Total 🔴 Offline",
                        value=total_offline,
                        delta=None
                    )
                
                st.divider()
                
                if not summary.empty:
                    # Create display summary
                    display_summary = summary[['Device', 'Current_Status', 'Last_Offline_Time', 
                                               'Total_DownTime_Events', 'Current_Downtime_Duration', 
                                               'Total_Downtime_Duration']].copy()
                    
                    # Status filter for summary table
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader(f"📋 Summary Table ({display_summary.shape[0]} rows × {display_summary.shape[1]} cols)")
                    
                    with col2:
                        summary_status_options = ["All", "✔️ Online", "🔴 Offline"]
                        selected_summary_status = st.selectbox(
                            "Filter by Status",
                            options=summary_status_options,
                            index=summary_status_options.index(st.session_state.summary_status_filter) if st.session_state.summary_status_filter in summary_status_options else 0,
                            key="summary_status_filter_select",
                            label_visibility="collapsed"
                        )
                        
                        st.session_state.summary_status_filter = selected_summary_status
                    
                    # Apply status filter
                    if st.session_state.summary_status_filter != "All":
                        filtered_summary = display_summary[display_summary['Current_Status'] == st.session_state.summary_status_filter].copy()
                        st.caption(f"📊 Showing {len(filtered_summary)} devices with status: {st.session_state.summary_status_filter}")
                        st.dataframe(filtered_summary, use_container_width=True)
                    else:
                        st.dataframe(display_summary, use_container_width=True)
                else:
                    st.info("ℹ️ No summary data available for the selected filters.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # DOWNTIME PAGE
            with st.container():
                st.markdown(f'<div class="page-section {"active" if st.session_state.current_page == "downtime" else ""}">', unsafe_allow_html=True)
                
                if not downtime.empty:
                    st.subheader(f"🔍 Downtime Events ({downtime.shape[0]} rows × {downtime.shape[1]} cols)")
                    
                    # Status filter for downtime table
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write("")
                    
                    with col2:
                        downtime_status_options = ["All", "✔️ Completed", "🔴 Ongoing"]
                        selected_downtime_status = st.selectbox(
                            "Filter by Status",
                            options=downtime_status_options,
                            index=downtime_status_options.index(st.session_state.downtime_status_filter) if st.session_state.downtime_status_filter in downtime_status_options else 0,
                            key="downtime_status_filter_select",
                            label_visibility="collapsed"
                        )
                        
                        st.session_state.downtime_status_filter = selected_downtime_status
                    
                    # Apply status filter
                    if st.session_state.downtime_status_filter != "All":
                        if st.session_state.downtime_status_filter == "✔️ Completed":
                            filtered_downtime = downtime[downtime['Downtime_Status'] == '✔️ Completed'].copy()
                        else:  # "🔴 Ongoing"
                            filtered_downtime = downtime[downtime['Downtime_Status'] == '🔴 Ongoing'].copy()
                        
                        st.caption(f"📊 Showing {len(filtered_downtime)} events with status: {st.session_state.downtime_status_filter}")
                        st.dataframe(filtered_downtime, use_container_width=True)
                    else:
                        st.dataframe(downtime, use_container_width=True)
                else:
                    st.info("ℹ️ No downtime events found for the selected filters.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # DOWNLOADS PAGE
            with st.container():
                st.markdown(f'<div class="page-section {"active" if st.session_state.current_page == "downloads" else ""}">', unsafe_allow_html=True)
                
                st.subheader("📥 Download Reports")
                
                if not summary.empty:
                    # Prepare summary data for download
                    if st.session_state.summary_status_filter != "All":
                        download_summary = summary[summary['Current_Status'] == st.session_state.summary_status_filter]
                    else:
                        download_summary = summary
                    
                    # Create Excel file for summary
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        download_summary[['Device', 'Current_Status', 'Last_Offline_Time', 
                                         'Total_DownTime_Events', 'Current_Downtime_Duration', 
                                         'Total_Downtime_Duration']].to_excel(writer, sheet_name='Summary', index=False)
                    excel_data = excel_buffer.getvalue()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Summary Excel",
                            data=excel_data,
                            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            use_container_width=True
                        )
                
                if not downtime.empty:
                    # Prepare downtime data for download
                    if st.session_state.downtime_status_filter != "All":
                        if st.session_state.downtime_status_filter == "✔️ Completed":
                            download_downtime = downtime[downtime['Downtime_Status'] == '✔️ Completed']
                        else:  # "🔴 Ongoing"
                            download_downtime = downtime[downtime['Downtime_Status'] == '🔴 Ongoing']
                    else:
                        download_downtime = downtime
                    
                    # Create Excel file for downtime
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        download_downtime.to_excel(writer, sheet_name='Downtime Events', index=False)
                    excel_data = excel_buffer.getvalue()
                    
                    col1, col2 = st.columns(2)
                    with col2:
                        st.download_button(
                            label="📥 Download Downtime Excel",
                            data=excel_data,
                            file_name=f"downtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            use_container_width=True
                        )
                
                st.divider()
                
                # Download all data button
                if not summary.empty and not downtime.empty:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        summary[['Device', 'Current_Status', 'Last_Offline_Time', 
                                'Total_DownTime_Events', 'Current_Downtime_Duration', 
                                'Total_Downtime_Duration']].to_excel(writer, sheet_name='Summary', index=False)
                        downtime.to_excel(writer, sheet_name='Downtime Events', index=False)
                    excel_data = excel_buffer.getvalue()
                    
                    st.download_button(
                        label="📦 Download Complete Report",
                        data=excel_data,
                        file_name=f"complete_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # HELP PAGE
        with st.container():
            st.markdown(f'<div class="page-section {"active" if st.session_state.current_page == "help" else ""}">', unsafe_allow_html=True)
            
            st.info("👈 Please upload a CSV file using the sidebar controls to generate reports.")
            
            with st.expander("📋 Expected CSV Format"):
                st.code("""
Required columns:
- Record Time: Timestamp (DD-MM-YYYY HH:MM:SS)
- Device Name: Device identifier
- Type: Should contain 'encoding' and either 'online' or 'offline'

Example:
Record Time,Device Name,Type
01-11-2023 10:00:00,Device1,encoding online
01-11-2023 10:05:00,Device1,encoding offline
01-11-2023 10:10:00,Device1,encoding online
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # FILTERS PAGE (Sidebar controls visible on mobile)
    with st.container():
        st.markdown(f'<div class="page-section {"active" if st.session_state.current_page == "filters" else ""}">', unsafe_allow_html=True)
        
        st.subheader("⚙️ Filter Settings")
        
        if 'df' in st.session_state and st.session_state.data_loaded:
            df = st.session_state.df
            
            # Date range selector
            min_date = df['Record Time'].min().date()
            max_date = df['Record Time'].max().date()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date, key="mobile_start")
            with col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date, key="mobile_end")
            
            if start_date > end_date:
                st.error("❌ Start date must be before end date!")
                start_date, end_date = end_date, start_date
            
            # Device filter
            all_devices = sorted(df['Device Name'].unique())
            selected_devices = st.multiselect(
                "Select devices (empty = all)",
                all_devices,
                default=[]
            )
            
            if st.button("🔄 Apply Filters", use_container_width=True):
                try:
                    with st.spinner("Applying filters..."):
                        summary, downtime, analysis_time = process_data(
                            df.copy(),
                            pd.to_datetime(start_date),
                            pd.to_datetime(end_date),
                            selected_devices
                        )
                        
                        if summary.empty and downtime.empty:
                            st.warning("⚠️ No data found for the selected filters.")
                        else:
                            st.session_state.summary = summary
                            st.session_state.downtime = downtime
                            st.session_state.analysis_time = analysis_time
                            st.session_state.processed = True
                            st.session_state.current_page = "summary"
                            st.success("✅ Filters applied!")
                            st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error applying filters: {str(e)}")
        else:
            st.info("📁 Upload a CSV file first to access filter settings.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom menu for mobile
    st.markdown("""
    <div class="bottom-menu">
        <button class="bottom-menu-item %s" onclick="window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'summary'}, '*')">
            <div class="bottom-menu-icon">📊</div>
            <div>Summary</div>
        </button>
        <button class="bottom-menu-item %s" onclick="window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'downtime'}, '*')">
            <div class="bottom-menu-icon">🔍</div>
            <div>Downtime</div>
        </button>
        <button class="bottom-menu-item %s" onclick="window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'downloads'}, '*')">
            <div class="bottom-menu-icon">📥</div>
            <div>Downloads</div>
        </button>
        <button class="bottom-menu-item %s" onclick="window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'filters'}, '*')">
            <div class="bottom-menu-icon">⚙️</div>
            <div>Filters</div>
        </button>
        <button class="bottom-menu-item %s" onclick="window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'help'}, '*')">
            <div class="bottom-menu-icon">❓</div>
            <div>Help</div>
        </button>
    </div>
    """ % (
        "active" if st.session_state.current_page == "summary" else "",
        "active" if st.session_state.current_page == "downtime" else "",
        "active" if st.session_state.current_page == "downloads" else "",
        "active" if st.session_state.current_page == "filters" else "",
        "active" if st.session_state.current_page == "help" else ""
    ), unsafe_allow_html=True)
    
    # JavaScript to handle bottom menu clicks
    st.markdown("""
    <script>
    // Listen for messages from bottom menu
    window.addEventListener('message', function(event) {
        if (event.data.type === 'streamlit:setComponentValue') {
            // Update the current page in Streamlit session state
            const page = event.data.value;
            
            // Send message back to Streamlit to update session state
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: page
            }, '*');
            
            // Update active state in the menu
            document.querySelectorAll('.bottom-menu-item').forEach(item => {
                item.classList.remove('active');
                if (item.textContent.includes(page.charAt(0).toUpperCase() + page.slice(1))) {
                    item.classList.add('active');
                }
            });
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Create a hidden component to trigger rerun when page changes
    page_changed = st.checkbox("Page changed", value=False, key="page_changed", label_visibility="collapsed")
    
    # Create buttons to change pages (hidden, used by JavaScript)
    cols = st.columns(5)
    with cols[0]:
        if st.button("Go to Summary", key="goto_summary", type="secondary", use_container_width=True):
            st.session_state.current_page = "summary"
            st.rerun()
    with cols[1]:
        if st.button("Go to Downtime", key="goto_downtime", type="secondary", use_container_width=True):
            st.session_state.current_page = "downtime"
            st.rerun()
    with cols[2]:
        if st.button("Go to Downloads", key="goto_downloads", type="secondary", use_container_width=True):
            st.session_state.current_page = "downloads"
            st.rerun()
    with cols[3]:
        if st.button("Go to Filters", key="goto_filters", type="secondary", use_container_width=True):
            st.session_state.current_page = "filters"
            st.rerun()
    with cols[4]:
        if st.button("Go to Help", key="goto_help", type="secondary", use_container_width=True):
            st.session_state.current_page = "help"
            st.rerun()

if __name__ == "__main__":
    main()
