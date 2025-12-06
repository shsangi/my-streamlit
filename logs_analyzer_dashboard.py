import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pytz
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Advanced Device Downtime Analyzer",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ==================== THEME MANAGEMENT ====================
def apply_theme_settings():
    """Apply theme settings from session state."""
    if 'theme' not in st.session_state:
        st.session_state.theme = 'system'
    
    # Apply theme via CSS
    theme_css = """
    <style>
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    """
    
    if st.session_state.theme == 'dark':
        theme_css += """
        :root {
            --background-color: #0E1117;
            --text-color: #FAFAFA;
        }
        """
    elif st.session_state.theme == 'light':
        theme_css += """
        :root {
            --background-color: #FFFFFF;
            --text-color: #31333F;
        }
        """
    else:  # system - uses Streamlit's default
        theme_css += """
        :root {
            --background-color: inherit;
            --text-color: inherit;
        }
        """
    
    st.markdown(theme_css, unsafe_allow_html=True)

# Apply theme at the start
apply_theme_settings()

# ==================== HELPER FUNCTIONS ====================
def format_duration(seconds):
    """Format seconds into human-readable duration."""
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

def get_ghana_time():
    """Get current Ghana time."""
    ghana_tz = pytz.timezone('Africa/Accra')
    return datetime.now(ghana_tz)

def create_download_link(df, filename, text):
    """Create a download link for DataFrame."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{text}</a>'

def calculate_uptime_percentage(downtime_seconds, total_seconds):
    """Calculate uptime percentage."""
    if total_seconds == 0:
        return 100.0
    uptime_seconds = total_seconds - downtime_seconds
    return (uptime_seconds / total_seconds) * 100

# ==================== DATA PROCESSING ====================
def process_data(df, start_date=None, end_date=None, selected_devices=None):
    """Process data with enhanced analytics."""
    try:
        df = df.copy()
        
        # Filter by date range
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            df = df[(df['Record Time'] >= start_date) & (df['Record Time'] <= end_date + timedelta(days=1))]
        
        # Filter by selected devices
        if selected_devices and len(selected_devices) > 0:
            df = df[df['Device Name'].isin(selected_devices)]
        
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), get_ghana_time()
        
        # Process downtime
        current_time = pd.Timestamp.now(tz='Africa/Accra')
        
        df_downtime = (
            df
            .assign(
                next_status=df.groupby('Device Name')['status'].shift(-1),
                next_time=df.groupby('Device Name')['Record Time'].shift(-1),
                prev_status=df.groupby('Device Name')['status'].shift(1)
            )
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
                    np.where(x['prev_status'] == 'online', 'Ongoing', 'Intermediate')
                )
            )
            .rename(columns={'Record Time': 'Offline_Time', 'next_time': 'Online_Time', 'Device Name': 'Device'})
            [['Device', 'Offline_Time', 'Online_Time', 'Downtime_Seconds', 'Downtime_Status']]
        )
        
        if df_downtime.empty:
            return pd.DataFrame(), pd.DataFrame(), current_time
        
        # Fix misclassified records
        mask = (df_downtime['Online_Time'].notna()) & (df_downtime['Downtime_Status'] == 'Ongoing')
        df_downtime.loc[mask, 'Downtime_Status'] = 'Completed'
        
        # Calculate downtime
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
            except:
                return 0
        
        df_downtime = df_downtime.copy()
        df_downtime.loc[:, 'Downtime_Seconds'] = df_downtime.apply(recalculate_downtime, axis=1)
        df_downtime.loc[:, 'Downtime_Duration'] = df_downtime['Downtime_Seconds'].apply(format_duration)
        
        # Create summary with enhanced metrics
        analysis_time = pd.Timestamp.now(tz='Africa/Accra')
        
        summary = (
            df_downtime.groupby('Device')
            .agg({
                'Offline_Time': ['count', 'first', 'last'],
                'Online_Time': 'last',
                'Downtime_Seconds': 'sum',
                'Downtime_Status': lambda x: (x == 'Ongoing').sum()
            })
            .reset_index()
        )
        
        summary.columns = [
            'Device', 'Total_Events', 'First_Offline', 'Last_Offline',
            'Last_Online', 'Total_Downtime_Seconds', 'Ongoing_Count'
        ]
        
        # Calculate enhanced metrics
        summary['Total_Downtime_Seconds'] = pd.to_numeric(summary['Total_Downtime_Seconds'], errors='coerce').fillna(0).round(0)
        
        # Current downtime for ongoing devices
        def calculate_current_downtime(row):
            try:
                if row['Ongoing_Count'] > 0:
                    offline_time = row['Last_Offline']
                    if isinstance(offline_time, pd.Timestamp):
                        if offline_time.tz is not None:
                            offline_time = offline_time.tz_localize(None)
                        return (analysis_time.tz_localize(None) - offline_time).total_seconds()
                    else:
                        offline_time = pd.to_datetime(offline_time)
                        return (analysis_time.tz_localize(None) - offline_time).total_seconds()
                else:
                    return 0
            except:
                return 0
        
        summary['Current_Downtime_Seconds'] = summary.apply(calculate_current_downtime, axis=1).round(0)
        summary['Current_Downtime_Duration'] = summary['Current_Downtime_Seconds'].apply(format_duration)
        summary['Total_Downtime_Duration'] = summary['Total_Downtime_Seconds'].apply(format_duration)
        summary['Current_Status'] = np.where(summary['Ongoing_Count'] > 0, '🔴 Offline', '✔️ Online')
        
        # Calculate uptime percentage
        total_period_seconds = (analysis_time - summary['First_Offline'].min()).total_seconds()
        summary['Uptime_Percentage'] = summary.apply(
            lambda row: calculate_uptime_percentage(row['Total_Downtime_Seconds'], total_period_seconds),
            axis=1
        ).round(2)
        
        # Calculate MTBF (Mean Time Between Failures)
        summary['MTBF_Hours'] = np.where(
            summary['Total_Events'] > 1,
            (total_period_seconds - summary['Total_Downtime_Seconds']) / (summary['Total_Events'] - 1) / 3600,
            0
        ).round(2)
        
        # Format downtime status
        df_downtime['Downtime_Status'] = np.where(
            df_downtime['Downtime_Status'] == 'Ongoing', '🔴 Ongoing', '✔️ Completed'
        )
        
        df_downtime_display = df_downtime[['Device', 'Offline_Time', 'Online_Time', 'Downtime_Duration', 'Downtime_Status']]
        
        return summary, df_downtime_display, analysis_time
        
    except Exception as e:
        st.error(f"Error in process_data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), get_ghana_time()

# ==================== VISUALIZATION FUNCTIONS ====================
def create_downtime_trend_chart(downtime_df):
    """Create downtime trend chart."""
    if downtime_df.empty:
        return None
    
    # Create daily downtime summary
    downtime_df['Date'] = pd.to_datetime(downtime_df['Offline_Time']).dt.date
    daily_summary = downtime_df.groupby('Date').size().reset_index(name='Downtime_Events')
    
    fig = px.line(
        daily_summary,
        x='Date',
        y='Downtime_Events',
        title='📈 Daily Downtime Events Trend',
        labels={'Date': 'Date', 'Downtime_Events': 'Number of Downtime Events'},
        markers=True
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
        height=300
    )
    
    return fig

def create_device_status_chart(summary_df):
    """Create device status distribution chart."""
    if summary_df.empty:
        return None
    
    status_counts = summary_df['Current_Status'].value_counts()
    
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title='📊 Device Status Distribution',
        color_discrete_sequence=['#1cc88a', '#e74a3b'],
        hole=0.3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False, height=300)
    
    return fig

def create_top_downtime_chart(summary_df):
    """Create chart for top devices by downtime."""
    if summary_df.empty or len(summary_df) < 2:
        return None
    
    top_devices = summary_df.nlargest(10, 'Total_Downtime_Seconds')
    
    fig = px.bar(
        top_devices,
        x='Device',
        y='Total_Downtime_Seconds',
        title='🏆 Top 10 Devices by Total Downtime',
        labels={'Total_Downtime_Seconds': 'Total Downtime (seconds)', 'Device': 'Device'},
        color='Total_Downtime_Seconds',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        xaxis_tickangle=-45
    )
    
    return fig

def create_uptime_gauge(uptime_percentage):
    """Create uptime percentage gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=uptime_percentage,
        title={'text': "📈 Overall Uptime %"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 90], 'color': "lightgray"},
                {'range': [90, 95], 'color': "gray"},
                {'range': [95, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 95
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

# ==================== MAIN APPLICATION ====================
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    .theme-button {
        width: 100%;
        margin: 2px 0;
        border-radius: 5px;
    }
    .theme-button-active {
        background-color: #4CAF50 !important;
        color: white !important;
        border: 2px solid #45a049 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with time
    ghana_time = get_ghana_time()
    current_time_str = ghana_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    
    col1, col2, col3 = st.columns([2, 3, 2])
    with col1:
        st.markdown(f'<div style="text-align: left; font-size: 0.9em; color: #666;">⏰ Ghana Time: {current_time_str}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<h1 class="main-header">📊 Advanced Device Downtime Analyzer</h1>', unsafe_allow_html=True)
    with col3:
        if st.button("🔄 Auto-refresh", key="refresh_btn"):
            st.rerun()
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'summary_status_filter' not in st.session_state:
        st.session_state.summary_status_filter = "All"
    if 'downtime_status_filter' not in st.session_state:
        st.session_state.downtime_status_filter = "All"
    if 'analysis_time' not in st.session_state:
        st.session_state.analysis_time = None
    if 'theme' not in st.session_state:
        st.session_state.theme = 'system'
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("### ⚙️ Controls Panel")
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload CSV File",
            type=['csv'],
            help="Upload your device logs CSV file"
        )
        
        if uploaded_file is not None and not st.session_state.data_loaded:
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
                    
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    
                    # Auto-process
                    min_date = df['Record Time'].min().date()
                    max_date = df['Record Time'].max().date()
                    
                    summary, downtime, analysis_time = process_data(
                        df.copy(),
                        pd.to_datetime(min_date),
                        pd.to_datetime(max_date),
                        []
                    )
                    
                    st.session_state.summary = summary
                    st.session_state.downtime = downtime
                    st.session_state.analysis_time = analysis_time
                    st.session_state.processed = True
                    
                    st.success("✅ Data loaded successfully!")
                    st.info(f"📊 {len(df)} records processed")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        # Advanced Filters Section
        if 'df' in st.session_state and st.session_state.data_loaded:
            st.markdown("---")
            st.markdown("### 🔍 Advanced Filters")
            
            df = st.session_state.df
            
            # Date Range with Quick Select
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "📅 Start Date",
                    df['Record Time'].min().date(),
                    help="Select start date for analysis"
                )
            with col2:
                end_date = st.date_input(
                    "📅 End Date",
                    df['Record Time'].max().date(),
                    help="Select end date for analysis"
                )
            
            # Device Filter with Search
            all_devices = sorted(df['Device Name'].unique())
            device_search = st.text_input("🔎 Search Devices", "", help="Type to filter devices")
            
            if device_search:
                filtered_devices = [d for d in all_devices if device_search.lower() in d.lower()]
            else:
                filtered_devices = all_devices
            
            selected_devices = st.multiselect(
                "📱 Select Devices",
                filtered_devices,
                default=[],
                help="Select specific devices (empty = all)"
            )
            
            # Duration Filter
            st.markdown("#### ⏱️ Duration Filter")
            min_duration, max_duration = st.slider(
                "Filter by downtime duration (hours)",
                0, 720, (0, 720),
                help="Filter events by duration range"
            )
            
            # Action Buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Apply Filters", use_container_width=True):
                    with st.spinner("Applying filters..."):
                        summary, downtime, analysis_time = process_data(
                            st.session_state.df.copy(),
                            pd.to_datetime(start_date),
                            pd.to_datetime(end_date),
                            selected_devices
                        )
                        st.session_state.summary = summary
                        st.session_state.downtime = downtime
                        st.session_state.analysis_time = analysis_time
                        st.success("Filters applied!")
                        
            with col2:
                if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            
            # Quick Stats
            st.markdown("---")
            st.markdown("### 📈 Quick Stats")
            if st.session_state.processed:
                summary = st.session_state.summary
                if not summary.empty:
                    st.metric("Total Devices", len(summary))
                    st.metric("Online Devices", len(summary[summary['Current_Status'] == '✔️ Online']))
                    st.metric("Offline Devices", len(summary[summary['Current_Status'] == '🔴 Offline']))
        
        # ==================== THEME SETTINGS ====================
        st.markdown("---")
        
        # Theme Settings Button
        theme_expander = st.expander("🎨 Theme Settings", expanded=False)
        
        with theme_expander:
            st.markdown("**Select Theme:**")
            
            # Create three columns for theme buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Light theme button
                light_active = st.session_state.theme == 'light'
                light_btn = st.button(
                    "☀️ Light",
                    key="theme_light",
                    help="Light theme",
                    use_container_width=True
                )
                if light_btn:
                    st.session_state.theme = 'light'
                    apply_theme_settings()
                    st.rerun()
                
                # Add visual indicator for active theme
                if light_active:
                    st.markdown('<div style="text-align: center; color: green; font-size: 0.8em;">✓ Active</div>', unsafe_allow_html=True)
            
            with col2:
                # Dark theme button
                dark_active = st.session_state.theme == 'dark'
                dark_btn = st.button(
                    "🌙 Dark",
                    key="theme_dark",
                    help="Dark theme",
                    use_container_width=True
                )
                if dark_btn:
                    st.session_state.theme = 'dark'
                    apply_theme_settings()
                    st.rerun()
                
                # Add visual indicator for active theme
                if dark_active:
                    st.markdown('<div style="text-align: center; color: green; font-size: 0.8em;">✓ Active</div>', unsafe_allow_html=True)
            
            with col3:
                # System theme button
                system_active = st.session_state.theme == 'system'
                system_btn = st.button(
                    "💻 System",
                    key="theme_system",
                    help="Use system default theme",
                    use_container_width=True
                )
                if system_btn:
                    st.session_state.theme = 'system'
                    apply_theme_settings()
                    st.rerun()
                
                # Add visual indicator for active theme
                if system_active:
                    st.markdown('<div style="text-align: center; color: green; font-size: 0.8em;">✓ Active</div>', unsafe_allow_html=True)
            
            # Theme description
            st.markdown("---")
            st.markdown("**Theme Info:**")
            if st.session_state.theme == 'light':
                st.info("☀️ **Light Theme** - Clean and bright interface")
            elif st.session_state.theme == 'dark':
                st.info("🌙 **Dark Theme** - Easy on the eyes in low light")
            else:
                st.info("💻 **System Theme** - Follows your system settings")
    
    # ==================== MAIN CONTENT ====================
    if 'processed' in st.session_state and st.session_state.processed:
        summary = st.session_state.summary
        downtime = st.session_state.downtime
        
        if summary.empty and downtime.empty:
            st.warning("⚠️ No data found for the selected filters.")
        else:
            # ==================== DASHBOARD METRICS ====================
            st.markdown("## 📊 Dashboard Overview")
            
            # Calculate overall metrics
            total_devices = len(summary) if not summary.empty else 0
            online_devices = len(summary[summary['Current_Status'] == '✔️ Online']) if not summary.empty else 0
            offline_devices = len(summary[summary['Current_Status'] == '🔴 Offline']) if not summary.empty else 0
            total_downtime_seconds = summary['Total_Downtime_Seconds'].sum() if not summary.empty else 0
            total_downtime_formatted = format_duration(total_downtime_seconds)
            
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                <div>Total Devices</div>
                <div class="metric-value">{total_devices}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #1cc88a 0%, #13855c 100%);">
                <div>✔️ Online</div>
                <div class="metric-value">{online_devices}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #e74a3b 0%, #be2617 100%);">
                <div>🔴 Offline</div>
                <div class="metric-value">{offline_devices}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #f6c23e 0%, #d19c0f 100%);">
                <div>Total Downtime</div>
                <div class="metric-value">{total_downtime_formatted}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # ==================== VISUALIZATIONS ====================
            st.markdown("## 📈 Analytics & Visualizations")
            
            if not summary.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Uptime Gauge
                    overall_uptime = calculate_uptime_percentage(
                        total_downtime_seconds,
                        total_devices * 24 * 3600  # Approximate total period
                    )
                    fig_gauge = create_uptime_gauge(overall_uptime)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    # Status Distribution
                    fig_pie = create_device_status_chart(summary)
                    if fig_pie:
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                col3, col4 = st.columns(2)
                
                with col3:
                    # Downtime Trend
                    if not downtime.empty:
                        fig_trend = create_downtime_trend_chart(downtime)
                        if fig_trend:
                            st.plotly_chart(fig_trend, use_container_width=True)
                
                with col4:
                    # Top Downtime Devices
                    fig_top = create_top_downtime_chart(summary)
                    if fig_top:
                        st.plotly_chart(fig_top, use_container_width=True)
            
            # ==================== ENHANCED SUMMARY TABLE ====================
            st.markdown("## 📋 Enhanced Device Summary")
            
            if not summary.empty:
                # Filter dropdown
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    summary_status_options = ["All", "✔️ Online", "🔴 Offline"]
                    selected_summary_status = st.selectbox(
                        "Filter by Status",
                        options=summary_status_options,
                        index=summary_status_options.index(st.session_state.summary_status_filter),
                        key="summary_status_filter_select"
                    )
                    st.session_state.summary_status_filter = selected_summary_status
                
                with col2:
                    # Sort options
                    sort_by = st.selectbox(
                        "Sort by",
                        ["Device", "Status", "Total Downtime", "Uptime %", "MTBF"],
                        key="summary_sort"
                    )
                
                with col3:
                    # Items per page
                    items_per_page = st.selectbox("Rows", [10, 25, 50, 100], key="summary_rows")
                
                # Apply filters
                if st.session_state.summary_status_filter != "All":
                    display_summary = summary[summary['Current_Status'] == st.session_state.summary_status_filter].copy()
                else:
                    display_summary = summary.copy()
                
                # Apply sorting
                sort_columns = {
                    "Device": "Device",
                    "Status": "Current_Status",
                    "Total Downtime": "Total_Downtime_Seconds",
                    "Uptime %": "Uptime_Percentage",
                    "MTBF": "MTBF_Hours"
                }
                display_summary = display_summary.sort_values(
                    by=sort_columns[sort_by],
                    ascending=(sort_by not in ["Total Downtime", "MTBF"])
                )
                
                # Pagination
                total_pages = max(1, len(display_summary) // items_per_page + (1 if len(display_summary) % items_per_page else 0))
                page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="summary_page")
                start_idx = (page_number - 1) * items_per_page
                end_idx = min(page_number * items_per_page, len(display_summary))
                
                # Display table
                st.dataframe(
                    display_summary.iloc[start_idx:end_idx][[
                        'Device', 'Current_Status', 'Total_Events',
                        'Current_Downtime_Duration', 'Total_Downtime_Duration',
                        'Uptime_Percentage', 'MTBF_Hours', 'First_Offline', 'Last_Offline'
                    ]],
                    use_container_width=True,
                    column_config={
                        "Device": "Device",
                        "Current_Status": "Status",
                        "Total_Events": "Events",
                        "Current_Downtime_Duration": "Current Downtime",
                        "Total_Downtime_Duration": "Total Downtime",
                        "Uptime_Percentage": st.column_config.NumberColumn("Uptime %", format="%.2f%%"),
                        "MTBF_Hours": st.column_config.NumberColumn("MTBF (hrs)", format="%.1f"),
                        "First_Offline": "First Offline",
                        "Last_Offline": "Last Offline"
                    }
                )
                
                # Export buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv = display_summary.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary CSV",
                        data=csv,
                        file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv',
                        use_container_width=True
                    )
                with col2:
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        display_summary.to_excel(writer, index=False, sheet_name='Summary')
                    st.download_button(
                        label="📥 Download Summary Excel",
                        data=buffer.getvalue(),
                        file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True
                    )
                with col3:
                    if st.button("📊 Generate Report", use_container_width=True):
                        st.info("Report generation feature coming soon!")
            
            # ==================== ENHANCED DOWNTIME EVENTS ====================
            st.markdown("## 🔍 Detailed Downtime Events")
            
            if not downtime.empty:
                # Enhanced filtering for downtime
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    downtime_status_options = ["All", "✔️ Completed", "🔴 Ongoing"]
                    selected_downtime_status = st.selectbox(
                        "Filter Events by Status",
                        options=downtime_status_options,
                        index=downtime_status_options.index(st.session_state.downtime_status_filter),
                        key="downtime_status_filter_select"
                    )
                    st.session_state.downtime_status_filter = selected_downtime_status
                
                with col2:
                    # Device filter for events
                    event_devices = sorted(downtime['Device'].unique())
                    selected_event_device = st.multiselect(
                        "Filter by Device",
                        event_devices,
                        default=[],
                        key="event_device_filter"
                    )
                
                with col3:
                    event_items_per_page = st.selectbox("Rows", [10, 25, 50, 100], key="downtime_rows")
                
                # Apply filters
                if st.session_state.downtime_status_filter != "All":
                    if st.session_state.downtime_status_filter == "✔️ Completed":
                        filtered_downtime = downtime[downtime['Downtime_Status'] == '✔️ Completed'].copy()
                    else:
                        filtered_downtime = downtime[downtime['Downtime_Status'] == '🔴 Ongoing'].copy()
                else:
                    filtered_downtime = downtime.copy()
                
                if selected_event_device:
                    filtered_downtime = filtered_downtime[filtered_downtime['Device'].isin(selected_event_device)]
                
                # Display table
                st.dataframe(
                    filtered_downtime,
                    use_container_width=True,
                    column_config={
                        "Device": "Device",
                        "Offline_Time": "Offline Time",
                        "Online_Time": "Online Time",
                        "Downtime_Duration": "Duration",
                        "Downtime_Status": "Status"
                    }
                )
                
                # Export buttons for events
                col1, col2 = st.columns(2)
                with col1:
                    csv_downtime = filtered_downtime.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Events CSV",
                        data=csv_downtime,
                        file_name=f"downtime_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv',
                        use_container_width=True
                    )
                with col2:
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        filtered_downtime.to_excel(writer, index=False, sheet_name='Downtime Events')
                    st.download_button(
                        label="📥 Download Events Excel",
                        data=buffer.getvalue(),
                        file_name=f"downtime_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True
                    )
            
            # ==================== ADVANCED ANALYTICS ====================
            st.markdown("## 🧠 Advanced Analytics")
            
            if not summary.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "📈 Overall Uptime %",
                        f"{overall_uptime:.2f}%",
                        delta=f"{(overall_uptime - 95):+.2f}%" if overall_uptime else None,
                        delta_color="normal" if overall_uptime >= 95 else "inverse"
                    )
                
                with col2:
                    avg_mtbf = summary['MTBF_Hours'].mean()
                    st.metric(
                        "⏱️ Average MTBF",
                        f"{avg_mtbf:.1f} hours",
                        help="Mean Time Between Failures"
                    )
                
                with col3:
                    if not downtime.empty:
                        avg_duration = downtime['Downtime_Duration'].apply(
                            lambda x: sum(int(t) * 60**i for i, t in enumerate(reversed(x.split(':')))) if ':' in x else 0
                        ).mean() / 60  # Convert to minutes
                        st.metric(
                            "🕒 Avg. Downtime Duration",
                            f"{avg_duration:.1f} minutes"
                        )
                
                # Performance Insights
                st.markdown("### 💡 Performance Insights")
                
                if offline_devices > 0:
                    st.warning(f"⚠️ **{offline_devices} device(s) currently offline** - Consider immediate attention.")
                
                if not summary.empty:
                    worst_device = summary.loc[summary['Total_Downtime_Seconds'].idxmax()]
                    best_device = summary.loc[summary['Uptime_Percentage'].idxmax()]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"🚨 **Most problematic device:** {worst_device['Device']} ({format_duration(worst_device['Total_Downtime_Seconds'])})")
                    with col2:
                        st.success(f"🏆 **Best performing device:** {best_device['Device']} ({best_device['Uptime_Percentage']:.2f}% uptime)")
    
    else:
        # Initial state
        st.markdown("## 👋 Welcome to Advanced Device Downtime Analyzer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            ### 📋 How to use:
            1. **Upload** your CSV file using the sidebar
            2. **Apply filters** to refine your analysis
            3. **Explore** interactive charts and tables
            4. **Download** reports in CSV or Excel format
            5. **Monitor** real-time device status
            
            ### 🚀 Key Features:
            - 📊 Interactive visualizations
            - 🔍 Advanced filtering options
            - 📈 Real-time analytics
            - 📥 Multiple export formats
            - 🎯 Performance insights
            """)
        
        with col2:
            st.markdown("""
            ### 📁 Expected CSV Format:
            ```csv
            Record Time,Device Name,Type
            DD-MM-YYYY HH:MM:SS,Device1,encoding online
            DD-MM-YYYY HH:MM:SS,Device1,encoding offline
            DD-MM-YYYY HH:MM:SS,Device2,encoding online
            ```
            
            ### ⚠️ Requirements:
            - CSV file with proper headers
            - Timestamps in DD-MM-YYYY HH:MM:SS format
            - 'encoding online' or 'encoding offline' in Type column
            """)
            
            with st.expander("📋 Sample Data"):
                sample_data = pd.DataFrame({
                    'Record Time': ['01-11-2023 10:00:00', '01-11-2023 10:05:00', '01-11-2023 10:10:00'],
                    'Device Name': ['Device1', 'Device1', 'Device2'],
                    'Type': ['encoding online', 'encoding offline', 'encoding online']
                })
                st.dataframe(sample_data)

if __name__ == "__main__":
    main()
