import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Vertical Tab Dashboard",
    layout="wide"
)

# -----------------------------
# LOAD DATA FROM GOOGLE SHEET
# -----------------------------
@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQFttuVQlH84hCC-brrcJFa6eyrMeyc25Aqm_dLgfpuEBr0WCdc4OTKKZVK2Y6IfOoPdQFbYmSdrSYP/pub?output=xlsx"
    df = pd.read_excel(url, engine="openpyxl")
    return df

df = load_data()

# Ensure correct column names exist
df.columns = df.columns.str.strip()

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("📊 Navigation")

menu = st.sidebar.radio(
    "Go to",
    [
        "Project Master",
        "Daily Work Log",
        "Employee Cost",
        "Resource Links",
        "Task Plan"
    ]
)

# -----------------------------
# PROJECT MASTER DASHBOARD
# -----------------------------
if menu == "Project Master":

    st.title("📁 PROJECT_MASTER Dashboard")

    # =============================
    # FILTER SECTION
    # =============================
    st.sidebar.subheader("🔎 Filters")

    status_filter = st.sidebar.multiselect(
        "Filter by Status",
        options=df["Status"].unique(),
        default=df["Status"].unique()
    )

    quarter_filter = st.sidebar.multiselect(
        "Filter by Quarter",
        options=df["Quarter"].unique(),
        default=df["Quarter"].unique()
    )

    search_term = st.sidebar.text_input("Search Project ID / Company")

    filtered_df = df[
        (df["Status"].isin(status_filter)) &
        (df["Quarter"].isin(quarter_filter))
    ]

    if search_term:
        filtered_df = filtered_df[
            filtered_df.astype(str)
            .apply(lambda row: row.str.contains(search_term, case=False).any(), axis=1)
        ]

    # =============================
    # KPI SCORECARDS
    # =============================
    total_budget = filtered_df["Budget"].sum()
    total_projects = filtered_df["Project ID"].nunique()
    in_progress = filtered_df[filtered_df["Status"] == "In Progress"].shape[0]
    completion_rate = round(
        (filtered_df[filtered_df["Status"] == "Completed"].shape[0] /
         max(total_projects, 1)) * 100, 2
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Projects", total_projects)
    col2.metric("Total Budget", f"${total_budget:,.0f}")
    col3.metric("In Progress", in_progress)
    col4.metric("Completion Rate", f"{completion_rate}%")

    st.divider()

    # =============================
    # GRAPHS
    # =============================
    col1, col2 = st.columns(2)

    # Budget by Quarter
    budget_by_quarter = filtered_df.groupby("Quarter")["Budget"].sum().reset_index()
    fig_budget = px.bar(
        budget_by_quarter,
        x="Quarter",
        y="Budget",
        title="Budget by Quarter",
        color="Quarter"
    )
    col1.plotly_chart(fig_budget, use_container_width=True)

    # Status Distribution
    status_dist = filtered_df["Status"].value_counts().reset_index()
    status_dist.columns = ["Status", "Count"]

    fig_status = px.pie(
        status_dist,
        names="Status",
        values="Count",
        title="Project Status Distribution",
        hole=0.4
    )
    col2.plotly_chart(fig_status, use_container_width=True)

    st.divider()

    # =============================
    # DATA TABLE
    # =============================
    st.subheader("📋 Project Data Table")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=500
    )

# -----------------------------
# DAILY WORK LOG
# -----------------------------
elif menu == "Daily Work Log":
    st.title("📅 DAILY_WORK_LOG")
    st.info("Add separate sheet in Google Sheets with this name.")
    st.dataframe(df)

# -----------------------------
# EMPLOYEE COST
# -----------------------------
elif menu == "Employee Cost":
    st.title("💰 EMPLOYEE_COST")
    st.info("Add separate sheet in Google Sheets with this name.")
    st.dataframe(df)

# -----------------------------
# RESOURCE LINKS
# -----------------------------
elif menu == "Resource Links":
    st.title("🔗 RESOURCE_LINKS")
    st.info("Add separate sheet in Google Sheets with this name.")
    st.dataframe(df)

# -----------------------------
# TASK PLAN
# -----------------------------
elif menu == "Task Plan":
    st.title("📋 TASK_PLAN + RESPONSIBILITY")
    st.info("Add separate sheet in Google Sheets with this name.")
    st.dataframe(df)
