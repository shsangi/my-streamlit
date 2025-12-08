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
        /* Hide sidebar on mobile if needed */
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
        }
        
        .bottom-menu-item:hover {
            background-color: #262730;
            color: white;
        }
        
        .bottom-menu-item.active {
            background-color: #262730;
            color: #4ade80;
        }
        
        .bottom-menu-icon {
            font-size: 20px;
            margin-bottom: 4px;
        }
        
        /* Add padding to main content to prevent bottom menu overlap */
        .main .block-container {
            padding-bottom: 80px !important;
        }
        
        /* Make buttons full width on mobile */
        .stButton > button {
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
        
        /* Make columns stack on mobile */
        [data-testid="column"] {
            width: 100% !important;
        }
    }
    
    /* Desktop styles for bottom menu */
    @media (min-width: 769px) {
        /* Show bottom menu on desktop too */
        .bottom-menu {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background-color: rgba(14, 17, 23, 0.95);
            border: 1px solid #2b313e;
            border-radius: 12px;
            padding: 10px 20px;
            z-index: 9999;
            display: flex;
            justify-content: space-around;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            width: auto;
            min-width: 400px;
        }
        
        .bottom-menu-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 10px 15px;
            color: #d1d5db;
            text-decoration: none;
            border-radius: 8px;
            transition: all 0.2s;
            min-width: 80px;
            font-size: 13px;
            cursor: pointer;
            margin: 0 5px;
        }
        
        .bottom-menu-item:hover {
            background-color: #262730;
            color: white;
        }
        
        .bottom-menu-item.active {
            background-color: #262730;
            color: #4ade80;
        }
        
        .bottom-menu-icon {
            font-size: 22px;
            margin-bottom: 4px;
        }
        
        /* Add padding to main content to prevent overlap */
        .main .block-container {
            padding-bottom: 100px !important;
        }
        
        /* Show sidebar on desktop */
        [data-testid="stSidebar"] {
            display: block;
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
    
    /* Hide elements with hide-mobile class on mobile */
    @media (max-width: 768px) {
        .hide-mobile {
            display: none !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Bottom menu component
def render_bottom_menu():
    st.markdown("""
    <div class="bottom-menu">
        <div class="bottom-menu-item" onclick="setView('home')" id="menu-home">
            <div class="bottom-menu-icon">🏠</div>
            <div>Home</div>
        </div>
        <div class="bottom-menu-item" onclick="setView('summary')" id="menu-summary">
            <div class="bottom-menu-icon">📊</div>
            <div>Summary</div>
        </div>
        <div class="bottom-menu-item" onclick="setView('events')" id="menu-events">
            <div class="bottom-menu-icon">🔍</div>
            <div>Events</div>
        </div>
    </div>
    
    <script>
    function setView(view) {
        // Update active menu item
        document.querySelectorAll('.bottom-menu-item').forEach(item => {
            item.classList.remove('active');
        });
        document.getElementById('menu-' + view).classList.add('active');
        
        // Store the selected view
        localStorage.setItem('selectedView', view);
        
        // Trigger Streamlit rerun
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: view
        }, '*');
    }
    
    // Initialize active state on page load
    document.addEventListener('DOMContentLoaded', function() {
        const savedView = localStorage.getItem('selectedView') || 'home';
        document.getElementById('menu-' + savedView).classList.add('active');
    });
    </script>
    """, unsafe_allow_html=True)

# Render filters section
def render_filters_section(df, start_date, end_date, selected_devices, all_devices):
    with st.container():
        st.subheader("⚙️ Filters")
        
        # Date range selector with validation
        col1, col2 = st.columns(2)
        with col1:
            new_start_date = st.date_input("Start Date", start_date, 
                                         min_value=start_date, max_value=end_date,
                                         key="filter_start_date")
        with col2:
            new_end_date = st.date_input("End Date", end_date, 
                                       min_value=start_date, max_value=end_date,
                                       key="filter_end_date")
        
        # Device filter
        new_selected_devices = st.multiselect(
            "Select devices (empty = all)",
            all_devices,
            default=selected_devices,
            key="filter_devices"
        )
        
        # Apply filters button
        if st.button("Apply Filters", type="primary", use_container_width=True):
            st.session_state.start_date = new_start_date
            st.session_state.end_date = new_end_date
            st.session_state.selected_devices = new_selected_devices
            st.session_state.apply_filters = True
            st.rerun()
        
        # Clear filters button
        if st.button("Clear Filters", type="secondary", use_container_width=True):
            st.session_state.selected_devices = []
            st.session_state.apply_filters = True
            st.rerun()

# Render home view
def render_home_view(summary, downtime, current_view):
    if current_view != "home":
        return
    
    st.header("🏠 Home")
    
    # Show metrics
    total_online = len(summary[summary['Current_Status'] == '✔️ Online']) if not summary.empty else 0
    total_offline = len(summary[summary['Current_Status'] == '🔴 Offline']) if not summary.empty else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Devices", len(summary) if not summary.empty else 0)
    with col2:
        st.metric("✔️ Online", total_online)
    with col3:
        st.metric("🔴 Offline", total_offline)
    
    # Summary section
    if not summary.empty:
        st.subheader(f"📋 Summary ({summary.shape[0]} rows)")
        
        # Create display summary
        display_summary = summary[['Device', 'Current_Status', 'Last_Offline_Time', 
                                   'Total_DownTime_Events', 'Current_Downtime_Duration', 
                                   'Total_Downtime_Duration']].copy()
        
        # Status filter for summary
        summary_status_options = ["All", "✔️ Online", "🔴 Offline"]
        selected_summary_status = st.selectbox(
            "Filter by Status",
            options=summary_status_options,
            index=0,
            key="home_summary_filter"
        )
        
        # Apply filter
        if selected_summary_status != "All":
            filtered_summary = display_summary[display_summary['Current_Status'] == selected_summary_status].copy()
            st.dataframe(filtered_summary, use_container_width=True)
        else:
            st.dataframe(display_summary, use_container_width=True)
        
        # Download button for summary
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            display_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        st.download_button(
            label="📥 Download Summary",
            data=excel_buffer.getvalue(),
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True
        )
    
    # Downtime events section
    if not downtime.empty:
        st.subheader(f"🔍 Downtime Events ({downtime.shape[0]} rows)")
        
        # Status filter for downtime
        downtime_status_options = ["All", "✔️ Completed", "🔴 Ongoing"]
        selected_downtime_status = st.selectbox(
            "Filter by Status",
            options=downtime_status_options,
            index=0,
            key="home_downtime_filter"
        )
        
        # Apply filter
        if selected_downtime_status != "All":
            status_to_filter = '✔️ Completed' if selected_downtime_status == "✔️ Completed" else '🔴 Ongoing'
            filtered_downtime = downtime[downtime['Downtime_Status'] == status_to_filter].copy()
            st.dataframe(filtered_downtime, use_container_width=True)
        else:
            st.dataframe(downtime, use_container_width=True)
        
        # Download button for downtime
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            downtime.to_excel(writer, sheet_name='Downtime Events', index=False)
        
        st.download_button(
            label="📥 Download Events",
            data=excel_buffer.getvalue(),
            file_name=f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True
        )

# Render summary view
def render_summary_view(summary, current_view):
    if current_view != "summary":
        return
    
    st.header("📊 Summary View")
    
    if not summary.empty:
        # Create display summary
        display_summary = summary[['Device', 'Current_Status', 'Last_Offline_Time', 
                                   'Total_DownTime_Events', 'Current_Downtime_Duration', 
                                   'Total_Downtime_Duration']].copy()
        
        st.subheader(f"Summary Table ({display_summary.shape[0]} rows × {display_summary.shape[1]} cols)")
        
        # Status filter
        summary_status_options = ["All", "✔️ Online", "🔴 Offline"]
        selected_summary_status = st.selectbox(
            "Filter by Status",
            options=summary_status_options,
            index=0,
            key="summary_view_filter"
        )
        
        # Apply filter
        if selected_summary_status != "All":
            filtered_summary = display_summary[display_summary['Current_Status'] == selected_summary_status].copy()
            st.dataframe(filtered_summary, use_container_width=True)
            download_data = filtered_summary
        else:
            st.dataframe(display_summary, use_container_width=True)
            download_data = display_summary
        
        # Download button
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            download_data.to_excel(writer, sheet_name='Summary', index=False)
        
        st.download_button(
            label="📥 Download Summary Excel",
            data=excel_buffer.getvalue(),
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True
        )
    else:
        st.info("No summary data available.")

# Render events view
def render_events_view(downtime, current_view):
    if current_view != "events":
        return
    
    st.header("🔍 Events View")
    
    if not downtime.empty:
        st.subheader(f"Downtime Events ({downtime.shape[0]} rows × {downtime.shape[1]} cols)")
        
        # Status filter
        downtime_status_options = ["All", "✔️ Completed", "🔴 Ongoing"]
        selected_downtime_status = st.selectbox(
            "Filter by Status",
            options=downtime_status_options,
            index=0,
            key="events_view_filter"
        )
        
        # Apply filter
        if selected_downtime_status != "All":
            status_to_filter = '✔️ Completed' if selected_downtime_status == "✔️ Completed" else '🔴 Ongoing'
            filtered_downtime = downtime[downtime['Downtime_Status'] == status_to_filter].copy()
            st.dataframe(filtered_downtime, use_container_width=True)
            download_data = filtered_downtime
        else:
            st.dataframe(downtime, use_container_width=True)
            download_data = downtime
        
        # Download button
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            download_data.to_excel(writer, sheet_name='Downtime Events', index=False)
        
        st.download_button(
            label="📥 Download Events Excel",
            data=excel_buffer.getvalue(),
            file_name=f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True
        )
    else:
        st.info("No downtime events available.")

# Streamlit App
def main():
    st.set_page_config(
        page_title="Device Downtime Report",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for mobile
    add_bottom_menu_css()
    
    # Get Ghana time
    ghana_time = get_ghana_time()
    current_time_str = ghana_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    
    # Add timestamp display at the top with small font - shows Ghana time
    st.markdown(
        f'<div style="text-align: right; font-size: 0.8em; color: #666; margin-bottom: 10px;">'
        f'⏰ Analysis time (Ghana/Accra): {current_time_str}'
        f'</div>',
        unsafe_allow_html=True
    )
    
    st.title("📊 Device Downtime Report")
    
    # Initialize session state
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "home"
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'apply_filters' not in st.session_state:
        st.session_state.apply_filters = False
    
    # Handle view change from JavaScript
    if 'view' in st.query_params:
        st.session_state.current_view = st.query_params['view']
    
    # Create a container for the main content
    main_container = st.container()
    
    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'], 
                                         help="Upload your device downtime CSV file")
        
        if uploaded_file is not None:
            try:
                if 'df' not in st.session_state or st.session_state.data_loaded == False:
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
                        
                        # Set default date range
                        min_date = df['Record Time'].min().date()
                        max_date = df['Record Time'].max().date()
                        st.session_state.start_date = min_date
                        st.session_state.end_date = max_date
                        st.session_state.selected_devices = []
                        
                        st.success(f"✅ File loaded successfully!")
                        st.info(f"Total records: {len(df)}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Show filters section in sidebar when data is loaded
        if 'df' in st.session_state and st.session_state.data_loaded:
            df = st.session_state.df
            
            # Initialize date range if not set
            if 'start_date' not in st.session_state:
                min_date = df['Record Time'].min().date()
                max_date = df['Record Time'].max().date()
                st.session_state.start_date = min_date
                st.session_state.end_date = max_date
                st.session_state.selected_devices = []
            
            # Get all devices
            all_devices = sorted(df['Device Name'].unique())
            
            # Render filters section
            render_filters_section(
                df, 
                st.session_state.start_date, 
                st.session_state.end_date,
                st.session_state.selected_devices,
                all_devices
            )
            
            # Process data when filters are applied
            if st.session_state.get('apply_filters', False):
                with st.spinner("Applying filters..."):
                    summary, downtime, analysis_time = process_data(
                        df.copy(),
                        pd.to_datetime(st.session_state.start_date),
                        pd.to_datetime(st.session_state.end_date),
                        st.session_state.selected_devices
                    )
                    
                    st.session_state.summary = summary
                    st.session_state.downtime = downtime
                    st.session_state.analysis_time = analysis_time
                    st.session_state.processed = True
                    st.session_state.apply_filters = False
                    
                    st.success("✅ Filters applied!")
        else:
            st.info("Upload a CSV file to begin")
    
    # Main content area
    with main_container:
        # Check if we have processed data
        if 'processed' in st.session_state and st.session_state.processed:
            summary = st.session_state.summary
            downtime = st.session_state.downtime
            current_view = st.session_state.current_view
            
            # Check if we have data
            if summary.empty and downtime.empty:
                st.warning("⚠️ No data found for the selected filters. Please adjust your criteria in the sidebar.")
            else:
                # Render the appropriate view based on current_view
                render_home_view(summary, downtime, current_view)
                render_summary_view(summary, current_view)
                render_events_view(downtime, current_view)
        else:
            # Initial state or no file uploaded
            st.info("👈 Please upload a CSV file using the sidebar to generate reports.")
            
            # Display sample of expected format
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
    
    # Add bottom menu (always visible)
    render_bottom_menu()
    
    # JavaScript to handle menu clicks and communicate with Streamlit
    st.markdown("""
    <script>
    // Function to handle menu clicks
    function handleMenuClick(view) {
        // Update URL with view parameter
        const url = new URL(window.location);
        url.searchParams.set('view', view);
        window.history.pushState({}, '', url);
        
        // Trigger Streamlit rerun
        window.location.reload();
    }
    
    // Attach click handlers to menu items
    document.addEventListener('DOMContentLoaded', function() {
        // Get current view from URL or default to 'home'
        const urlParams = new URLSearchParams(window.location.search);
        const currentView = urlParams.get('view') || 'home';
        
        // Update active menu item
        document.querySelectorAll('.bottom-menu-item').forEach(item => {
            item.classList.remove('active');
            if (item.id === 'menu-' + currentView) {
                item.classList.add('active');
            }
            
            // Update click handlers
            const view = item.id.replace('menu-', '');
            item.onclick = function() { handleMenuClick(view); };
        });
    });
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
