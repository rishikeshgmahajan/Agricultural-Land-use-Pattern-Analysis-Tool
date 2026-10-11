import os

import pandas as pd
import streamlit as st
from data_preprocessor import preprocess_data

# 1. Page configuration
st.set_page_config(
    page_title="AgriScope | Land-Use Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Theme and styling
st.markdown(
    """
    <style>
    :root {
        --forest: #214E34;
        --forest-dark: #173A27;
        --sage: #E7EFE5;
        --cream: #F7F6EF;
        --paper: #FFFFFF;
        --gold: #D6A84F;
        --ink: #25352B;
        --muted: #6D7B70;
        --line: #E1E7DE;
    }

    .stApp {
        background: var(--cream);
        color: var(--ink);
    }

    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--forest-dark);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #F7F8F2;
    }

    .sidebar-brand {
        padding: 0.5rem 0 1rem 0;
    }

    .brand-icon {
        font-size: 2.2rem;
        line-height: 1.2;
    }

    .brand-name {
        color: #FFFFFF;
        font-size: 1.35rem;
        font-weight: 750;
        letter-spacing: 0.02em;
        margin-top: 0.3rem;
    }

    .brand-tagline {
        color: #C5D6C8;
        font-size: 0.82rem;
        margin-top: 0.2rem;
    }

    .sidebar-status {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 0.8rem;
        margin-top: 0.8rem;
        font-size: 0.82rem;
    }

    /* Sidebar navigation buttons */
    section[data-testid="stSidebar"] .stButton {
        margin: 0.12rem 0;
    }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        background: rgba(255,255,255,0.02);
        color: #F7F8F2;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 0.72rem 0.8rem;
        font-weight: 600;
        font-size: 0.96rem;
        letter-spacing: 0.01em;
        transition: all 0.15s ease;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.08);
        color: #FFFFFF;
        border-color: rgba(255,255,255,0.11);
    }

    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(180deg, rgba(214,168,79,0.18), rgba(214,168,79,0.08));
        color: #FFFFFF;
        border: 1px solid rgba(214,168,79,0.52);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
    }

    /* Headings */
    h1, h2, h3 {
        color: var(--forest-dark) !important;
        letter-spacing: -0.025em;
    }

    h1 {
        font-weight: 760 !important;
    }

    /* Top header */
    .topline {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 650;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .page-description {
        color: var(--muted);
        font-size: 1rem;
        margin-top: -0.5rem;
        margin-bottom: 1.2rem;
    }

    /* Dashboard hero */
    .hero {
        background: linear-gradient(115deg, #214E34 0%, #2F6845 70%, #47764C 100%);
        border-radius: 18px;
        padding: 2rem 2.1rem;
        color: white;
        margin: 0.4rem 0 1.4rem 0;
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "🌾";
        position: absolute;
        right: 2rem;
        top: 0.25rem;
        font-size: 6.5rem;
        opacity: 0.13;
        transform: rotate(-12deg);
    }

    .hero-kicker {
        color: #DCE9D9;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 700;
        margin-bottom: 0.65rem;
    }

    .hero-title {
        color: white !important;
        font-size: 2rem;
        line-height: 1.2;
        font-weight: 760;
        max-width: 750px;
        margin: 0;
    }

    .hero-text {
        color: #E8F0E7;
        font-size: 0.98rem;
        line-height: 1.6;
        max-width: 720px;
        margin-top: 0.8rem;
        margin-bottom: 0;
    }

    /* Section labels and cards */
    .section-eyebrow {
        color: #66816B;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }

    .info-card {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1.15rem 1.2rem;
        min-height: 135px;
        box-shadow: 0 3px 12px rgba(35, 62, 40, 0.035);
    }

    .info-card-title {
        color: var(--forest-dark);
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .info-card-text {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 1rem 1.1rem;
        box-shadow: 0 3px 12px rgba(35, 62, 40, 0.035);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: var(--forest-dark);
        font-weight: 750;
    }

    /* Inputs and buttons */
    div[data-baseweb="select"] > div {
        border-radius: 9px;
        border-color: var(--line);
        background: white;
    }

    .stButton button {
        border-radius: 9px;
        font-weight: 650;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid var(--line);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        color: var(--forest) !important;
        border-bottom: 3px solid var(--gold) !important;
    }

    /* Notices */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Footer */
    .app-footer {
        color: #7C887D;
        font-size: 0.78rem;
        border-top: 1px solid var(--line);
        margin-top: 2.5rem;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. Data loading
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "classification-of-area.csv",
)


@st.cache_data
def load_data():
    return preprocess_data(pd.read_csv(DATA_PATH))


def get_data():
    try:
        return load_data()
    except FileNotFoundError:
        return None
    except Exception as error:
        st.error(f"Unable to load the dataset: {error}")
        return None


df = get_data()

# 4. Sidebar navigation
def set_page(page_name):
    st.session_state.page = page_name


if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

pages = [
        ("⌂", "Dashboard"),
        ("▤", "Data Explorer"),
        ("↗", "Land-Use Trends"),
        ("▦", "Statistics & Forecasting"),
        ("ⓘ", "About the Project"),
    ]
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">🌱</div>
            <div class="brand-name">AgriScope</div>
            <div class="brand-tagline">Indian Agricultural Land-Use Analysis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        """
        <div style="color:#BFD2C2; font-size:0.72rem;
                    font-weight:700; letter-spacing:0.12em;
                    margin-bottom:0.5rem;">
            WORKSPACE
        </div>
        """,
        unsafe_allow_html=True,
    )

    for icon, page_name in pages:
        button_label = f"{icon}   {page_name}"
        button_type = (
            "primary" if st.session_state.page == page_name else "secondary"
        )

        st.button(
            button_label,
            key=f"nav_{page_name}",
            type=button_type,
            use_container_width=True,
            on_click=set_page,
            args=(page_name,),
        )

    st.divider()

    if df is not None:
        st.markdown(
            f"""
            <div class="sidebar-status">
                <div style="color:#C5D6C8; font-size:0.72rem;
                            text-transform:uppercase; letter-spacing:0.08em;">
                    Dataset status
                </div>
                <div style="font-size:0.95rem; font-weight:700; margin-top:0.25rem;">
                    ● Data loaded
                </div>
                <div style="color:#D1DED3; margin-top:0.25rem;">
                    {len(df):,} source records
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="sidebar-status">
                <div style="color:#C5D6C8; font-size:0.72rem;
                            text-transform:uppercase; letter-spacing:0.08em;">
                    Dataset status
                </div>
                <div style="font-size:0.95rem; font-weight:700; margin-top:0.25rem;">
                    ○ CSV not found
                </div>
                <div style="color:#D1DED3; margin-top:0.25rem;">
                    Add the dataset beside app.py
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

page = st.session_state.page

# 5. Shared page helpers
def page_header(kicker, title, description):
    st.markdown(
        f'<div class="topline">{kicker}</div>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(
        f'<div class="page-description">{description}</div>',
        unsafe_allow_html=True,
    )


def info_card(title, description):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-title">{title}</div>
            <div class="info-card-text">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_data_unavailable():
    st.warning(
        "The dataset could not be found. Put "
        "`classification-of-area.csv` in the project folder, "
        "then restart the app."
    )


def show_footer():
    st.markdown(
        """
        <div class="app-footer">
            AgriScope · Student Capstone Project · Agricultural Land-Use Patterns in India
        </div>
        """,
        unsafe_allow_html=True,
    )

# 6. Dashboard
def dashboard_page():
    page_header(
        "PROJECT OVERVIEW",
        "Agricultural Land-Use Analysis",
        "A data-driven view of how land-use patterns change across India.",
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Explore · Understand · Forecast</div>
            <div class="hero-title">
                Understanding India's changing agricultural landscape
            </div>
            <p class="hero-text">
                Explore land-use records, compare patterns across regions,
                and examine historical trends through statistical analysis
                and forecasting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df is None:
        show_data_unavailable()
        show_footer()
        return

    st.markdown(
        '<div class="section-eyebrow">Dataset at a glance</div>',
        unsafe_allow_html=True,
    )
    st.subheader("The data behind the analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Source records", f"{len(df):,}")

    with col2:
        state_count = (
            df["state_name"].nunique()
            if "state_name" in df.columns
            else 0
        )
        st.metric("States & UTs", f"{state_count:,}")

    with col3:
        year_count = (
            df["year"].nunique()
            if "year" in df.columns
            else 0
        )
        st.metric("Reporting years", f"{year_count:,}")

    st.write("")
    st.markdown(
        '<div class="section-eyebrow">Explore the workspace</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Choose an area to explore")

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        info_card(
            "🗂️ Data Explorer",
            "Browse the source records and filter by land-use measure "
            "and state or union territory.",
        )
        st.write("")
        info_card(
            "📈 Land-Use Trends",
            "Explore long-term changes and compare land-use patterns "
            "across different regions.",
        )

    with col_right:
        info_card(
            "🔬 Statistics & Forecasting",
            "Review statistical summaries, trend models, forecast "
            "estimates, and model evaluation.",
        )
        st.write("")
        info_card(
            "🌿 About the Project",
            "Read about the project goals, dataset, methods, and "
            "the analysis modules.",
        )

    show_footer()

# 7. Data Explorer
def data_explorer_page():
    page_header(
        "DATA WORKSPACE",
        "Data Explorer",
        "Inspect the source dataset and narrow records by category or region.",
    )

    if df is None:
        show_data_unavailable()
        return

    st.markdown(
        '<div class="section-eyebrow">Source data</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Dataset preview")

    st.caption(
        f"Displaying up to 100 rows from the loaded CSV · "
        f"{len(df):,} records total"
    )
    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.write("")
    st.markdown(
        '<div class="section-eyebrow">Filter and inspect</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Explore records")

    filtered = df.copy()
    col1, col2 = st.columns(2)

    with col1:
        if "land_use_metrics" in filtered.columns:
            metrics = sorted(
                filtered["land_use_metrics"]
                .dropna()
                .astype(str)
                .unique()
            )
            selected_metric = st.selectbox(
                "Land-use metric",
                ["All metrics"] + metrics,
            )

            if selected_metric != "All metrics":
                filtered = filtered[
                    filtered["land_use_metrics"].astype(str)
                    == selected_metric
                ]

    with col2:
        if "state_name" in filtered.columns:
            states = sorted(
                filtered["state_name"]
                .dropna()
                .astype(str)
                .unique()
            )
            selected_state = st.selectbox(
                "State / UT",
                ["All states / UTs"] + states,
            )

            if selected_state != "All states / UTs":
                filtered = filtered[
                    filtered["state_name"].astype(str)
                    == selected_state
                ]

    st.caption(
        f"{len(filtered):,} records match the selected filters."
    )
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    show_footer()

# 8. Land-Use Trends
def trends_page():
    page_header(
        "VISUAL ANALYTICS",
        "Land-Use Trends",
        "Visual comparisons of historical land-use patterns across years and regions.",
    )

    if df is None:
        show_data_unavailable()
        return

    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Trend analysis workspace</div>
            <div class="info-card-text">
                This section is reserved for the charts and visual summaries
                produced by the team's visualization module.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Planned visualizations")

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        info_card(
            "Annual land-use trends",
            "Line charts showing changes in selected measures over time.",
        )
        st.write("")
        info_card(
            "Regional comparisons",
            "Compare selected measures across states and union territories.",
        )

    with col_right:
        info_card(
            "Land-use composition",
            "Visual breakdowns of the available land-use categories.",
        )
        st.write("")
        info_card(
            "District-level patterns",
            "Explore differences between districts where the data supports it.",
        )

    st.write("")
    st.info(
        "Charts will be connected here after `eda_visualizer.py` "
        "has its first usable functions."
    )

    # Integration point for the visualization module:
    # Import the chart functions and display the returned figures here.

    show_footer()

# 9. Statistics & Forecasting
def forecasting_page():
    page_header(
        "STATISTICAL ANALYSIS",
        "Statistics & Forecasting",
        "Examine historical changes and explore possible future trajectories.",
    )

    metrics = sorted(
        df["land_use_metrics"].dropna().astype(str).unique()
    )
    if not metrics:
        st.warning("No land-use metrics are available for analysis.")
        return

    st.markdown(
        '<div class="section-eyebrow">Forecast controls</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Choose a forecast series")

    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox("Land-use measure", metrics)
    with col2:
        horizon = st.selectbox(
            "Forecast horizon",
            [5, 10],
            format_func=lambda years: f"{years} years",
        )

    annual_data = (
        df[df["land_use_metrics"].astype(str) == metric]
        .dropna(subset=["start_year", "area"])
        .groupby("start_year", as_index=False)["area"]
        .sum()
        .rename(columns={"start_year": "year", "area": "value"})
    )

    if annual_data.empty:
        st.warning("There is not enough valid data for this metric.")
        return

    try:
        from stat_forecasting import (
            analyze_trend,
            descriptive_summary,
            forecast_annual,
            interpret_trend,
            rolling_origin_validation,
        )
    except ImportError as error:
        st.error(
            "Statistics and forecasting dependencies are unavailable: "
            f"{error}"
        )
        return

    summary = descriptive_summary(annual_data)
    trend = analyze_trend(annual_data)
    forecast_result = forecast_annual(
        annual_data,
        periods=horizon,
        validation_size=5,
    )

    tab1, tab2, tab3 = st.tabs(
        ["Statistical Summary", "Forecast Results", "Model Evaluation"]
    )

    with tab1:
        st.subheader("Statistical summary")
        summary_columns = st.columns(4)
        summary_columns[0].metric("Observations", summary["observations"])
        summary_columns[1].metric("Average area", f"{summary['mean']:,.2f}")
        summary_columns[2].metric("Minimum", f"{summary['minimum']:,.2f}")
        summary_columns[3].metric("Maximum", f"{summary['maximum']:,.2f}")
        st.dataframe(
            pd.DataFrame([summary]),
            use_container_width=True,
            hide_index=True,
        )
        st.write(interpret_trend(trend)["summary"])

    with tab2:
        st.subheader("Forecast table")
        if forecast_result is None:
            st.warning("Not enough annual observations for forecasting.")
        else:
            st.write("Selected model:", forecast_result["selected_model"])
            st.dataframe(
                forecast_result["forecast"],
                use_container_width=True,
                hide_index=True,
            )
            st.info(forecast_result["interpretation"])

    with tab3:
        st.subheader("Model performance")
        validation = rolling_origin_validation(annual_data, test_size=5)
        if validation is None:
            st.warning("Not enough annual observations for model validation.")
        else:
            error_table = pd.DataFrame([
                {
                    "model": "Last Value Baseline",
                    **validation["baseline_errors"],
                },
                {
                    "model": "Holt Linear Trend",
                    **validation["holt_errors"],
                },
            ])
            st.dataframe(
                error_table,
                use_container_width=True,
                hide_index=True,
            )

    show_footer()

# 10. About the Project
def about_page():
    page_header(
        "PROJECT INFORMATION",
        "About AgriScope",
        "An academic project focused on agricultural land-use patterns in India.",
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Our objective</div>
            <div class="hero-title">
                Turning land-use records into meaningful analysis
            </div>
            <p class="hero-text">
                The project examines changes in agricultural land use across
                Indian states and districts using data preparation,
                visualization, statistical methods, and forecasting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Project modules")

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        info_card(
            "01 · Data preprocessing",
            "Data loading, cleaning, filtering, aggregation, and preparation "
            "of analysis-ready tables.",
        )
        st.write("")
        info_card(
            "02 · Visualization",
            "Charts and visual comparisons that help reveal changes and "
            "differences in land-use patterns.",
        )

    with col_right:
        info_card(
            "03 · Statistics & forecasting",
            "Statistical summaries, suitable tests, trend modelling, "
            "forecasting, and model validation.",
        )
        st.write("")
        info_card(
            "04 · Application interface",
            "A shared interactive interface for browsing the dataset "
            "and presenting the team's analysis.",
        )

    st.write("")
    st.subheader("Dataset information")
    st.write(
        "The project uses the supplied `classification-of-area.csv` file. "
        "The final report should document the data source, coverage, "
        "variable definitions, and relevant limitations."
    )

    show_footer()

# 11. Page routing
if page == "Dashboard":
    dashboard_page()

elif page == "Data Explorer":
    data_explorer_page()

elif page == "Land-Use Trends":
    trends_page()

elif page == "Statistics & Forecasting":
    forecasting_page()

elif page == "About the Project":
    about_page()