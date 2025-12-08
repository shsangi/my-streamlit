import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pytz
import io 
import base64

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

# Add custom CSS for mobile
def add_custom_css():
    st.markdown("""
    <style>
    /* Mobile bottom menu */
    .mobile-bottom-menu {
        display: flex;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #0e1117;
        border-top: 1px solid #2b313e;
        padding: 8px 0;
        z-index: 9999;
        justify-content: space-around;
    }
    
    .mobile-menu-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 8px 12px;
        color: #8b949e;
        text-decoration: none;
        border-radius: 8px;
        font-size: 11px;
        cursor: pointer;
        transition: all 0.2s;
        flex: 1;
        max-width: 20%;
    }
    
    .mobile-menu-item:hover {
        background: #1c2128;
        color: #ffffff;
    }
    
    .mobile-menu-item.active {
        color: #58a6ff;
        background: #1c2128;
    }
    
    .mobile-menu-icon {
        font-size: 18px;
        margin-bottom: 4px;
    }
    
    /* Hide content by default */
    .mobile-content {
        display: none;
        padding-bottom: 70px; /* Space for bottom menu */
    }
    
    .mobile-content.active {
        display: block;
    }
    
    /* Hide sidebar on mobile */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        
        .main .block-container {
            padding-bottom: 80px !important;
        }
        
        /* Show mobile content sections */
        .mobile-content-section {
            padding: 15px;
        }
        
        /* Make buttons mobile-friendly */
        .stButton > button {
            width: 100%;
            min-height: 44px;
            font-size: 16px;
        }
        
        /* Make inputs mobile-friendly */
        .stDateInput, .stSelectbox, .stMultiSelect {
            width: 100%;
        }
        
        /* Adjust table for mobile */
        .stDataFrame {
            font-size: 12px;
        }
        
        /* Adjust metrics for mobile */
        [data-testid="stMetric"] {
            padding: 10px;
            margin-bottom: 10px;
        }
    }
    
    /* Desktop styles */
    @media (min-width: 769px) {
        .mobile-bottom-menu {
            display: none !important;
        }
        
        .desktop-content {
            display: block;
        }
        
        /* Show sidebar on desktop */
        section[data-testid="stSidebar"] {
            display: block !important;
        }
    }
    
    /* Make download buttons full width on mobile */
    @media (max-width: 768px) {
        .stDownloadButton > button {
            width: 100% !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Create mobile bottom menu
def create_mobile_menu():
    st.markdown("""
    <div class="mobile-bottom-menu" id="mobileMenu">
        <div class="mobile-menu-item active" onclick="showMobileSection('dashboard')">
            <div class="mobile-menu-icon">📊</div>
            <div>Dashboard</div>
        </div>
        <div class="mobile-menu-item" onclick="showMobileSection('summary')">
            <div class="mobile-menu-icon">📋</div>
            <div>Summary</div>
        </div>
        <div class="mobile-menu-item" onclick="showMobileSection('downtime')">
            <div class="mobile-menu-icon">🔍</div>
            <div>Downtime</div>
        </div>
        <div class="mobile-menu-item" onclick="showMobileSection('filters')">
            <div class="mobile-menu-icon">⚙️</div>
            <div>Filters</div>
        </div>
        <div class="mobile-menu-item" onclick="showMobileSection('downloads')">
            <div class="mobile-menu-icon">📥</div>
            <div>Download</div>
        </div>
    </div>
    
    <script>
    function showMobileSection(sectionId) {
        // Hide all content sections
        document.querySelectorAll('.mobile-content-section').forEach(section => {
            section.style.display = 'none';
        });
        
        // Show selected section
        const selectedSection = document.getElementById(sectionId + '-section');
        if (selectedSection) {
            selectedSection.style.display = 'block';
        }
        
        // Update active menu item
        document.querySelectorAll('.mobile-menu-item').forEach(item => {
            item.classList.remove('active');
        });
        event.currentTarget.classList.add('active');
        
        // Store in session state
        sessionStorage.setItem('mobileSection', sectionId);
    }
    
    // Initialize on load
    document.addEventListener('DOMContentLoaded', function() {
        const savedSection = sessionStorage.getItem('mobileSection') || 'dashboard';
        const sectionToShow = document.getElementById(savedSection + '-section');
        const menuItem = document.querySelector(`.mobile-menu-item[onclick*="${savedSection}"]`);
        
        if (sectionToShow) {
            document.querySelectorAll('.mobile-content-section').forEach(section => {
                section.style.display = 'none';
            });
            sectionToShow.style.display = 'block';
        }
        
        if (menuItem) {
            document.querySelectorAll('.mobile-menu-item').forEach(item => {
                item.classList.remove('active');
            });
            menuItem.classList.add('active');
        }
    });
    
    // Detect mobile and show/hide sections accordingly
    function checkMobileView() {
        const isMobile = window.innerWidth <= 768;
        const mobileMenu = document.getElementById('mobileMenu');
        if (mobileMenu) {
            if (isMobile) {
                mobileMenu.style.display = 'flex';
                // Hide desktop content
                document.querySelectorAll('.desktop-content').forEach(el => {
                    el.style.display = 'none';
                });
            } else {
                mobileMenu.style.display = 'none';
                // Show desktop content
                document.querySelectorAll('.desktop-content').forEach(el => {
                    el.style.display = 'block';
                });
            }
        }
    }
    
    window.addEventListener('resize', checkMobileView);
    window.addEventListener('load', checkMobileView);
    </script>
    """, unsafe_allow_html=True)

# Mobile content sections
def mobile_dashboard_section(summary, downtime, analysis_time):
    """Dashboard view for mobile"""
    if summary.empty and downtime.empty:
        st.warning("⚠️ No data found. Please upload a CSV file and apply filters.")
        return
    
    # Ghana time display
    ghana_time = get_ghana_time()
    current_time_str = ghana_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    st.caption(f"⏰ Ghana Time: {current_time_str}")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Devices", len(summary) if not summary.empty else 0)
    with col2:
        total_online = len(summary[summary['Current_Status'] == '✔️ Online']) if not summary.empty else 0
        st.metric("✅ Online", total_online)
    with col3:
        total_offline = len(summary[summary['Current_Status'] == '🔴 Offline']) if not summary.empty else 0
        st.metric("🔴 Offline", total_offline)
    
    # Quick stats
    if not summary.empty:
        st.subheader("📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Downtime Events", 
                     summary['Total_DownTime_Events'].sum() if 'Total_DownTime_Events' in summary.columns else 0)
        with col2:
            ongoing_count = len(summary[summary['Current_Status'] == '🔴 Offline'])
            st.metric("Currently Offline", ongoing_count)
    
    # Recent downtime
    if not downtime.empty:
        st.subheader("🔍 Recent Downtime Events")
        recent_downtime = downtime.sort_values('Offline_Time', ascending=False).head(5)
        st.dataframe(recent_downtime, use_container_width=True)

def mobile_summary_section(summary):
    """Summary table view for mobile"""
    if summary.empty:
        st.info("ℹ️ No summary data available")
        return
    
    display_summary = summary[['Device', 'Current_Status', 'Last_Offline_Time', 
                               'Total_DownTime_Events', 'Current_Downtime_Duration', 
                               'Total_Downtime_Duration']].copy()
    
    # Status filter
    status_filter = st.selectbox(
        "Filter Status",
        ["All", "✔️ Online", "🔴 Offline"],
        key="mobile_summary_filter"
    )
    
    if status_filter != "All":
        filtered = display_summary[display_summary['Current_Status'] == status_filter]
        st.dataframe(filtered, use_container_width=True)
        st.caption(f"Showing {len(filtered)} devices")
    else:
        st.dataframe(display_summary, use_container_width=True)
        st.caption(f"Showing all {len(display_summary)} devices")

def mobile_downtime_section(downtime):
    """Downtime events view for mobile"""
    if downtime.empty:
        st.info("ℹ️ No downtime events found")
        return
    
    # Status filter
    status_filter = st.selectbox(
        "Filter Status",
        ["All", "✔️ Completed", "🔴 Ongoing"],
        key="mobile_downtime_filter"
    )
    
    if status_filter != "All":
        if status_filter == "✔️ Completed":
            filtered = downtime[downtime['Downtime_Status'] == '✔️ Completed']
        else:
            filtered = downtime[downtime['Downtime_Status'] == '🔴 Ongoing']
        st.dataframe(filtered, use_container_width=True)
        st.caption(f"Showing {len(filtered)} events")
    else:
        st.dataframe(downtime, use_container_width=True)
        st.caption(f"Showing all {len(downtime)} events")

def mobile_filters_section():
    """Filters and controls for mobile"""
    st.subheader("⚙️ Filters & Controls")
    
    # File upload
    uploaded_file = st.file_uploader("📁 Upload CSV File", type=['csv'], key="mobile_upload")
    
    if uploaded_file is not None:
        try:
            # Load and process data
            df = pd.read_csv(uploaded_file)
            
            # Data preprocessing
            df['Record Time'] = pd.to_datetime(df['Record Time'], dayfirst=True, errors='coerce')
            df = df[df['Type'].str.contains('encoding', case=False, na=False)]
            
            # Create status column
            df['status'] = 'unknown'
            df.loc[df['Type'].str.contains('online', case=False, na=False), 'status'] = 'online'
            df.loc[df['Type'].str.contains('offline', case=False, na=False), 'status'] = 'offline'
            df = df.sort_values(by=['Device Name', 'Record Time'], ascending=[True, True])
            
            # Store in session state
            st.session_state.mobile_df = df
            st.session_state.mobile_data_loaded = True
            
            # Date range
            min_date = df['Record Time'].min().date()
            max_date = df['Record Time'].max().date()
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", min_date, key="mobile_start")
            with col2:
                end_date = st.date_input("End Date", max_date, key="mobile_end")
            
            if start_date > end_date:
                st.error("❌ Start date must be before end date!")
                start_date, end_date = end_date, start_date
            
            # Device filter
            all_devices = sorted(df['Device Name'].unique())
            selected_devices = st.multiselect(
                "Select Devices (empty for all)",
                all_devices,
                key="mobile_devices"
            )
            
            # Process button
            if st.button("🔄 Apply Filters & Process", use_container_width=True):
                with st.spinner("Processing..."):
                    summary, downtime, analysis_time = process_data(
                        df.copy(),
                        pd.to_datetime(start_date),
                        pd.to_datetime(end_date),
                        selected_devices
                    )
                    
                    st.session_state.mobile_summary = summary
                    st.session_state.mobile_downtime = downtime
                    st.session_state.mobile_analysis_time = analysis_time
                    st.session_state.mobile_processed = True
                    
                    if summary.empty and downtime.empty:
                        st.warning("⚠️ No data found for selected filters")
                    else:
                        st.success("✅ Data processed successfully!")
                        
                        # Switch to dashboard view
                        st.markdown("""
                        <script>
                        setTimeout(() => {
                            showMobileSection('dashboard');
                        }, 100);
                        </script>
                        """, unsafe_allow_html=True)
            
            # Reset button
            if st.button("🗑️ Reset All", type="secondary", use_container_width=True):
                for key in ['mobile_df', 'mobile_data_loaded', 'mobile_summary', 
                          'mobile_downtime', 'mobile_processed']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.info("Please upload a CSV file to get started")

def mobile_downloads_section(summary, downtime):
    """Downloads section for mobile"""
    st.subheader("📥 Download Reports")
    
    if 'mobile_processed' not in st.session_state or not st.session_state.mobile_processed:
        st.info("ℹ️ Please process data first in the Filters section")
        return
    
    if summary.empty and downtime.empty:
        st.warning("⚠️ No data available to download")
        return
    
    # Create download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if not summary.empty:
            # Summary Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                summary.to_excel(writer, sheet_name='Summary', index=False)
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="📥 Summary",
                data=excel_data,
                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True
            )
    
    with col2:
        if not downtime.empty:
            # Downtime Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                downtime.to_excel(writer, sheet_name='Downtime', index=False)
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                label="📥 Downtime",
                data=excel_data,
                file_name=f"downtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True
            )
    
    # Combined download
    if not summary.empty and not downtime.empty:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            summary.to_excel(writer, sheet_name='Summary', index=False)
            downtime.to_excel(writer, sheet_name='Downtime', index=False)
        excel_data = excel_buffer.getvalue()
        
        st.download_button(
            label="📥 Combined Report",
            data=excel_data,
            file_name=f"full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            use_container_width=True
        )

# Main Streamlit App
def main():
    st.set_page_config(
        page_title="Device Downtime Report",
        layout="wide",
        initial_sidebar_state="collapsed"  # Start with collapsed sidebar on mobile
    )
    
    # Add custom CSS
    add_custom_css()
    
    # Initialize session states for mobile
    if 'mobile_processed' not in st.session_state:
        st.session_state.mobile_processed = False
    if 'mobile_data_loaded' not in st.session_state:
        st.session_state.mobile_data_loaded = False
    
    # Title with responsive design
    st.title("📊 Device Downtime Report")
    
    # Ghana time display (desktop only)
    ghana_time = get_ghana_time()
    current_time_str = ghana_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    
    # Create mobile content sections (hidden by default, shown via JS)
    with st.container():
        # Dashboard section
        st.markdown('<div class="mobile-content-section" id="dashboard-section" style="display: block;">', unsafe_allow_html=True)
        if 'mobile_processed' in st.session_state and st.session_state.mobile_processed:
            mobile_dashboard_section(
                st.session_state.mobile_summary,
                st.session_state.mobile_downtime,
                st.session_state.get('mobile_analysis_time', ghana_time)
            )
        else:
            st.info("👈 Please go to Filters section to upload and process data")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Summary section
        st.markdown('<div class="mobile-content-section" id="summary-section" style="display: none;">', unsafe_allow_html=True)
        if 'mobile_processed' in st.session_state and st.session_state.mobile_processed:
            mobile_summary_section(st.session_state.mobile_summary)
        else:
            st.info("ℹ️ No data available. Please process data first.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Downtime section
        st.markdown('<div class="mobile-content-section" id="downtime-section" style="display: none;">', unsafe_allow_html=True)
        if 'mobile_processed' in st.session_state and st.session_state.mobile_processed:
            mobile_downtime_section(st.session_state.mobile_downtime)
        else:
            st.info("ℹ️ No data available. Please process data first.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Filters section
        st.markdown('<div class="mobile-content-section" id="filters-section" style="display: none;">', unsafe_allow_html=True)
        mobile_filters_section()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Downloads section
        st.markdown('<div class="mobile-content-section" id="downloads-section" style="display: none;">', unsafe_allow_html=True)
        if 'mobile_processed' in st.session_state and st.session_state.mobile_processed:
            mobile_downloads_section(
                st.session_state.mobile_summary,
                st.session_state.mobile_downtime
            )
        else:
            st.info("ℹ️ No data available. Please process data first.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Desktop sidebar (hidden on mobile via CSS)
    with st.sidebar:
        st.header("Desktop Controls")
        
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'], 
                                         on_change=lambda: st.session_state.update({
                                             "data_loaded": False, 
                                             "processed": False,
                                             "last_error": None,
                                             "summary_status_filter": "All",
                                             "downtime_status_filter": "All",
                                             "analysis_time": None
                                         }))
        
        # ... (rest of desktop sidebar code remains the same as your original)
        # [Keep all your existing desktop sidebar code here]
        # This part remains unchanged from your original code
    
    # Create mobile bottom menu (will be shown/hidden via CSS/JS)
    create_mobile_menu()
    
    # Add JavaScript to detect mobile and switch view
    st.markdown("""
    <script>
    // Check on load and resize
    function checkView() {
        if (window.innerWidth <= 768) {
            // Mobile view
            document.querySelectorAll('.desktop-content').forEach(el => {
                el.style.display = 'none';
            });
            document.getElementById('mobileMenu').style.display = 'flex';
        } else {
            // Desktop view
            document.querySelectorAll('.desktop-content').forEach(el => {
                el.style.display = 'block';
            });
            document.getElementById('mobileMenu').style.display = 'none';
        }
    }
    
    window.addEventListener('load', checkView);
    window.addEventListener('resize', checkView);
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
