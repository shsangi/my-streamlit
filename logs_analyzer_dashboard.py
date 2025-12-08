import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pytz
import io
import time

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

# Mobile-friendly CSS for bottom menu
def inject_custom_css():
    st.markdown("""
    <style>
    /* Mobile-specific styles */
    @media (max-width: 768px) {
        /* Hide unnecessary elements on mobile */
        .stApp header {
            display: none !important;
        }
        
        /* Main content padding for bottom menu */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 70px;
        }
        
        /* Bottom Navigation Menu */
        .bottom-menu {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-around;
            align-items: center;
            z-index: 1000;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }
        
        .menu-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 8px 5px;
            flex: 1;
            text-decoration: none;
            color: #666;
            font-size: 12px;
            transition: all 0.2s;
            height: 100%;
        }
        
        .menu-item.active {
            color: #FF4B4B;
            font-weight: 600;
        }
        
        .menu-item:hover {
            background-color: #f5f5f5;
        }
        
        .menu-icon {
            font-size: 18px;
            margin-bottom: 2px;
        }
        
        /* Adjust sidebar for mobile */
        [data-testid="stSidebar"] {
            min-width: 100% !important;
            max-width: 100% !important;
        }
        
        /* Compact tables for mobile */
        .dataframe {
            font-size: 12px !important;
        }
        
        /* Smaller buttons on mobile */
        .stButton button {
            width: 100% !important;
            font-size: 14px !important;
            padding: 8px 16px !important;
        }
        
        /* Smaller metrics for mobile */
        .stMetric {
            padding: 10px !important;
        }
        
        .stMetric label {
            font-size: 12px !important;
        }
        
        .stMetric div[data-testid="stMetricValue"] {
            font-size: 18px !important;
        }
        
        /* Hide some elements on mobile */
        .st-emotion-cache-1v0mbdj {
            display: none;
        }
    }
    
    /* Desktop styles */
    @media (min-width: 769px) {
        .bottom-menu {
            display: none !important;
        }
    }
    
    /* Common styles */
    .stSelectbox, .stDateInput, .stMultiselect {
        font-size: 14px !important;
    }
    
    /* Status colors */
    .status-online {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-offline {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Timestamp display */
    .timestamp {
        font-size: 12px;
        color: #666;
        text-align: center;
        padding: 8px;
        background: #f8f9fa;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Bottom Navigation Menu Component
def bottom_navigation(current_tab="Dashboard"):
    menu_html = f"""
    <div class="bottom-menu">
        <a href="?tab=dashboard" class="menu-item {'active' if current_tab == 'dashboard' else ''}">
            <div class="menu-icon">📊</div>
            <div>Dashboard</div>
        </a>
        <a href="?tab=summary" class="menu-item {'active' if current_tab == 'summary' else ''}">
            <div class="menu-icon">📋</div>
            <div>Summary</div>
        </a>
        <a href="?tab=downtime" class="menu-item {'active' if current_tab == 'downtime' else ''}">
            <div class="menu-icon">🔍</div>
            <div>Downtime</div>
        </a>
        <a href="?tab=filters" class="menu-item {'active' if current_tab == 'filters' else ''}">
            <div class="menu-icon">⚙️</div>
            <div>Filters</div>
        </a>
    </div>
    """
    st.markdown(menu_html, unsafe_allow_html=True)

# Streamlit App
def main():
    # Mobile-first page config
    st.set_page_config(
        page_title="Device Downtime Report",
        layout="wide",
        initial_sidebar_state="collapsed"  # Collapse sidebar on mobile
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # Get current tab from URL
    query_params = st.query_params
    current_tab = query_params.get("tab", ["dashboard"])[0]
    
    # Get Ghana time
    ghana_time = get_ghana_time()
    current_time_str = ghana_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    
    # Display timestamp
    st.markdown(f'<div class="timestamp">⏰ Ghana Time: {current_time_str}</div>', unsafe_allow_html=True)
    
    # Initialize session state
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
    if 'analysis_time' not in st.session_state:
        st.session_state.analysis_time = None
    
    # Mobile Header
    if current_tab != "filters":
        st.title("📱 Device Downtime")
    
    # File upload handler (only in filters tab)
    if current_tab == "filters":
        handle_file_upload()
    
    # Show appropriate content based on current tab
    if current_tab == "dashboard" and st.session_state.processed:
        show_dashboard()
    elif current_tab == "summary" and st.session_state.processed:
        show_summary()
    elif current_tab == "downtime" and st.session_state.processed:
        show_downtime()
    elif current_tab == "filters":
        # Filters tab is handled separately
        pass
    else:
        # Show welcome/upload prompt
        if current_tab != "filters":
            st.info("👈 Tap ⚙️ in bottom menu to upload CSV file")
            with st.expander("📋 Expected CSV Format", expanded=False):
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
    
    # Add bottom navigation menu
    bottom_navigation(current_tab)

def handle_file_upload():
    """Handle file upload in filters tab"""
    st.title("⚙️ Filters & Upload")
    
    # File upload section
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            with st.spinner("Loading data..."):
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
                
                # Process with ALL devices selected
                summary, downtime, analysis_time = process_data(
                    df.copy(),
                    pd.to_datetime(min_date),
                    pd.to_datetime(max_date),
                    []  # Empty list means ALL devices
                )
                
                # Store in session state
                st.session_state.summary = summary
                st.session_state.downtime = downtime
                st.session_state.analysis_time = analysis_time
                st.session_state.processed = True
                
                st.success(f"✅ File loaded successfully! ({len(df)} records)")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Filters section (only show if data is loaded)
    if st.session_state.data_loaded and st.session_state.df is not None:
        st.divider()
        st.subheader("🔍 Filter Options")
        
        df = st.session_state.df
        
        # Date range selector
        min_date = df['Record Time'].min().date()
        max_date = df['Record Time'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Validate date range
        if start_date > end_date:
            st.error("Start date must be before end date!")
            start_date, end_date = end_date, start_date
        
        # Device filter
        all_devices = sorted(df['Device Name'].unique())
        selected_devices = st.multiselect(
            "Select devices (empty = all)",
            all_devices,
            default=[],
            help="Select specific devices or leave empty for all"
        )
        
        # Apply filters button
        if st.button("🔄 Apply Filters", use_container_width=True, type="primary"):
            with st.spinner("Applying filters..."):
                summary, downtime, analysis_time = process_data(
                    df.copy(),
                    pd.to_datetime(start_date),
                    pd.to_datetime(end_date),
                    selected_devices
                )
                
                # Update session state
                st.session_state.summary = summary
                st.session_state.downtime = downtime
                st.session_state.analysis_time = analysis_time
                st.session_state.processed = True
                
                st.success("✅ Filters applied!")
                time.sleep(1)
                st.rerun()
        
        # Reset button
        if st.button("🗑️ Clear Data", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Current filter info
        st.caption(f"📅 Date range: {start_date} to {end_date}")
        if selected_devices:
            st.caption(f"📱 {len(selected_devices)} of {len(all_devices)} devices selected")
        else:
            st.caption(f"📱 All {len(all_devices)} devices selected")

def show_dashboard():
    """Show dashboard tab"""
    summary = st.session_state.summary
    downtime = st.session_state.downtime
    
    # Calculate stats
    total_devices = len(summary) if not summary.empty else 0
    total_online = len(summary[summary['Current_Status'] == '✔️ Online']) if not summary.empty else 0
    total_offline = len(summary[summary['Current_Status'] == '🔴 Offline']) if not summary.empty else 0
    
    # Stats in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total", total_devices)
    with col2:
        st.metric("Online", total_online, delta=None)
    with col3:
        st.metric("Offline", total_offline, delta=None)
    
    st.divider()
    
    # Quick status overview
    st.subheader("📱 Device Status Overview")
    if not summary.empty:
        # Show first 10 devices for mobile
        display_summary = summary[['Device', 'Current_Status', 'Current_Downtime_Duration']].head(10)
        
        # Apply status colors
        def color_status(val):
            if val == '✔️ Online':
                return 'color: #28a745; font-weight: bold;'
            elif val == '🔴 Offline':
                return 'color: #dc3545; font-weight: bold;'
            return ''
        
        styled_df = display_summary.style.map(color_status, subset=['Current_Status'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        if len(summary) > 10:
            st.caption(f"Showing 10 of {len(summary)} devices")
    else:
        st.info("No data available")
    
    # Recent downtime events
    st.divider()
    st.subheader("🔄 Recent Downtime Events")
    if not downtime.empty:
        recent_downtime = downtime[['Device', 'Offline_Time', 'Downtime_Status']].head(5)
        st.dataframe(recent_downtime, use_container_width=True, hide_index=True)
    else:
        st.info("No downtime events")

def show_summary():
    """Show summary tab"""
    summary = st.session_state.summary
    
    if summary.empty:
        st.warning("No summary data available")
        return
    
    st.subheader("📋 Device Summary")
    
    # Status filter
    status_options = ["All", "✔️ Online", "🔴 Offline"]
    selected_status = st.selectbox("Filter by status", status_options, label_visibility="collapsed")
    
    # Apply filter
    if selected_status == "All":
        filtered_summary = summary
    else:
        filtered_summary = summary[summary['Current_Status'] == selected_status]
    
    # Display table
    display_summary = filtered_summary[['Device', 'Current_Status', 'Last_Offline_Time', 
                                        'Total_DownTime_Events', 'Current_Downtime_Duration', 
                                        'Total_Downtime_Duration']].copy()
    
    st.dataframe(display_summary, use_container_width=True, hide_index=True)
    
    st.caption(f"Showing {len(filtered_summary)} devices")
    
    # Download button
    if not filtered_summary.empty:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            filtered_summary.to_excel(writer, sheet_name='Summary', index=False)
        excel_data = excel_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Summary",
            data=excel_data,
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True
        )

def show_downtime():
    """Show downtime events tab"""
    downtime = st.session_state.downtime
    
    if downtime.empty:
        st.warning("No downtime events available")
        return
    
    st.subheader("🔍 Downtime Events")
    
    # Status filter
    status_options = ["All", "✔️ Completed", "🔴 Ongoing"]
    selected_status = st.selectbox("Filter by event status", status_options, label_visibility="collapsed")
    
    # Apply filter
    if selected_status == "All":
        filtered_downtime = downtime
    elif selected_status == "✔️ Completed":
        filtered_downtime = downtime[downtime['Downtime_Status'] == '✔️ Completed']
    else:  # "🔴 Ongoing"
        filtered_downtime = downtime[downtime['Downtime_Status'] == '🔴 Ongoing']
    
    # Display table
    st.dataframe(filtered_downtime, use_container_width=True, hide_index=True)
    
    st.caption(f"Showing {len(filtered_downtime)} events")
    
    # Download button
    if not filtered_downtime.empty:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            filtered_downtime.to_excel(writer, sheet_name='Downtime', index=False)
        excel_data = excel_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Downtime",
            data=excel_data,
            file_name=f"downtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True
        )

if __name__ == "__main__":
    main()
