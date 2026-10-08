import pandas as pd
from data_preprocessor import *
from stat_forecasting import *

# DISPLAY HELPERS
def header(title):
    print()
    print("=" * 65)
    print(title)
    print("=" * 65)


def subheader(title):
    print()
    print("-" * 65)
    print(title)
    print("-" * 65)


def show_numbered_list(items):
    for number, item in enumerate(items, start=1):
        print(f"{number}. {item}")


# DATA LOADING
def load_application_data():
    """
    Load the two datasets required by different application modules.

    member2_data:
        Cleaned dataset containing state/district information.

    member3_data:
        Annual aggregated dataset used by statistics and forecasting.
    """

    print("Loading project data...")

    member2_data = create_member2_dataset()
    member3_data = create_member3_dataset()

    print(
        f"Loaded {len(member2_data):,} cleaned records "
        f"and {len(member3_data):,} annual metric records."
    )

    return member2_data, member3_data

# DATASET MENU
def dataset_menu(cleaned_data):
    while True:

        header("DATASET")

        print("1. View first records")
        print("2. View last records")
        print("3. Dataset information")
        print("4. View available metrics")
        print("5. View available states")
        print("0. Back")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":

            subheader("FIRST 10 RECORDS")

            print(cleaned_data.head(10).to_string(index=False))

        elif choice == "2":

            subheader("LAST 10 RECORDS")

            print(cleaned_data.tail(10).to_string(index=False))

        elif choice == "3":

            subheader("DATASET INFORMATION")

            print(f"Rows    : {cleaned_data.shape[0]:,}")
            print(f"Columns : {cleaned_data.shape[1]}")

            print("\nColumns:")

            for column in cleaned_data.columns:
                print(f"  - {column}")

            print("\nMissing values:")

            missing = cleaned_data.isna().sum()

            for column, count in missing.items():
                print(f"  {column}: {count:,}")

        elif choice == "4":

            subheader("AVAILABLE LAND-USE METRICS")

            metrics = (
                cleaned_data["land_use_metrics"]
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

            for metric in metrics:
                print(f"- {metric}")

        elif choice == "5":

            subheader("AVAILABLE STATES / UNION TERRITORIES")

            states = (
                cleaned_data["state_name"]
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

            for state in states:
                print(f"- {state}")

        elif choice == "0":
            return

        else:
            print("Invalid choice. Please select one of the available options.")

# METRIC SELECTION
def select_metric(annual_data):

    metrics = [
        metric
        for metric in annual_data["land_use_metrics"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    ]

    if not metrics:
        print("No metrics are available.")
        return None

    subheader("SELECT LAND-USE METRIC")

    show_numbered_list(metrics)

    print("0. Back")

    choice = input("\nEnter choice: ").strip()

    if choice == "0":
        return None

    if not choice.isdigit():
        print("Invalid choice.")
        return None

    number = int(choice)

    if number < 1 or number > len(metrics):
        print("Invalid choice.")
        return None

    return metrics[number - 1]


def get_metric_series(annual_data, metric):

    metric_data = annual_data[
        annual_data["land_use_metrics"] == metric
    ][["start_year", "area"]].copy()

    metric_data = metric_data.rename(
        columns={
            "start_year": "year",
            "area": "value",
        }
    )

    return metric_data.sort_values("year").reset_index(drop=True)


# STATISTICAL ANALYSIS
def statistical_analysis_menu(annual_data):

    while True:

        header("STATISTICAL ANALYSIS")

        metric = select_metric(annual_data)

        if metric is None:
            return

        statistical_metric_menu(annual_data, metric)


def statistical_metric_menu(annual_data, metric):

    while True:

        subheader(metric.upper())

        print("1. Descriptive Statistics")
        print("2. Trend Analysis")
        print("3. Year-over-Year Changes")
        print("4. Period Comparison")
        print("5. Correlation Analysis")
        print("0. Back")

        choice = input("\nEnter choice: ").strip()

        series = get_metric_series(annual_data, metric)

        if choice == "1":
            show_descriptive_statistics(series)

        elif choice == "2":
            show_trend_analysis(series)

        elif choice == "3":
            show_yearly_changes(series)

        elif choice == "4":
            show_period_comparison(series)

        elif choice == "5":
            show_correlation_analysis(annual_data, metric)

        elif choice == "0":
            return

        else:
            print("Invalid choice. Please select a valid option.")

# DESCRIPTIVE STATISTICS
def show_descriptive_statistics(series):

    subheader("DESCRIPTIVE STATISTICS")

    result = descriptive_summary(series)

    print(f"Observations        : {result['observations']}")
    print(f"Mean                : {result['mean']:,.2f}")
    print(f"Median              : {result['median']:,.2f}")
    print(f"Minimum             : {result['minimum']:,.2f}")
    print(f"Maximum             : {result['maximum']:,.2f}")
    print(f"Standard Deviation  : {result['standard_deviation']:,.2f}")
    print(f"First Year          : {result['first_year']}")
    print(f"Last Year           : {result['last_year']}")

    print("\nInterpretation:")

    interpretation = interpret_descriptive_summary(result)

    print(interpretation)

# TREND ANALYSIS
def show_trend_analysis(series):

    subheader("TREND ANALYSIS")

    result = analyze_trend(series)

    print(f"Slope          : {result['slope']:,.2f}")
    print(f"Intercept      : {result['intercept']:,.2f}")
    print(f"R²             : {result['r_squared']:.4f}")
    print(f"P-value        : {result['p_value']:.6f}")
    print(f"Standard Error : {result['standard_error']:,.2f}")
    print(f"Observations   : {result['observations']}")

    print("\nInterpretation:")

    interpretation = interpret_trend(result)

    print(interpretation["summary"])

# YEAR-OVER-YEAR CHANGE
def show_yearly_changes(series):

    subheader("YEAR-OVER-YEAR CHANGES")

    # Calculate yearly changes
    result = calculate_yearly_change(series)

    if result.empty:

        print("Year-over-year changes could not be calculated.")

        return

    # Show the latest 10 years
    print(
        result[
            [
                "year",
                "value",
                "change",
                "percentage_change"
            ]
        ]
        .tail(10)
        .to_string(index=False)
    )

    # Find largest increase and decrease
    interpretation = interpret_yearly_changes(series)

    if interpretation is None:

        print("\nNo year-over-year changes are available.")

        return

    print("\nLargest Increase")

    print(
        f"Year        : "
        f"{interpretation['largest_increase_year']}"
    )

    print(
        f"Change      : "
        f"{interpretation['largest_increase']:,.2f}"
    )

    print(
        f"Percentage   : "
        f"{interpretation['largest_increase_percentage']:.2f}%"
    )

    print("\nLargest Decrease")

    print(
        f"Year        : "
        f"{interpretation['largest_decrease_year']}"
    )

    print(
        f"Change      : "
        f"{interpretation['largest_decrease']:,.2f}"
    )

    print(
        f"Percentage   : "
        f"{interpretation['largest_decrease_percentage']:.2f}%"
    )

    print("\nInterpretation:")

    print(interpretation["summary"])

# PERIOD COMPARISON
def show_period_comparison(series):

    subheader("EARLY PERIOD VS LATER PERIOD")

    result = compare_period_trends(series)

    if result is None:
        print(
            "There are not enough yearly observations "
            "for this comparison."
        )
        return

    print(
        f"First Period  : "
        f"{result['first_period'][0]}–"
        f"{result['first_period'][1]}"
    )

    print(
        f"Second Period : "
        f"{result['second_period'][0]}–"
        f"{result['second_period'][1]}"
    )

    print(
        f"First Mean    : "
        f"{result['first_mean']:,.2f}"
    )

    print(
        f"Second Mean   : "
        f"{result['second_mean']:,.2f}"
    )

    print(
        f"Difference    : "
        f"{result['difference']:,.2f}"
    )

    print(
        f"Percentage Change : "
        f"{result['percentage_change']:.2f}%"
    )

    print("\nInterpretation:")

    print(interpret_period_comparison(result))


# CORRELATION ANALYSIS
def show_correlation_analysis(annual_data, metric):

    subheader("CORRELATION ANALYSIS")

    metric_data = annual_data[
        annual_data["land_use_metrics"] == metric
    ].copy()

    numeric_columns = (
        metric_data
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    # Do not use year for correlation selection
    numeric_columns = [
        column
        for column in numeric_columns
        if column != "start_year"
    ]

    if len(numeric_columns) < 2:

        print(
            "There are not enough numerical variables "
            "for correlation analysis."
        )

        return

    print("Available variables:")

    for number, column in enumerate(numeric_columns, start=1):
        print(f"{number}. {column}")

    print("0. Back")

    first_choice = input("\nSelect first variable: ").strip()

    if first_choice == "0":
        return

    if not first_choice.isdigit():
        print("Invalid choice.")
        return

    first_number = int(first_choice)

    if first_number < 1 or first_number > len(numeric_columns):
        print("Invalid choice.")
        return

    second_choice = input("Select second variable: ").strip()

    if second_choice == "0":
        return

    if not second_choice.isdigit():
        print("Invalid choice.")
        return

    second_number = int(second_choice)

    if second_number < 1 or second_number > len(numeric_columns):
        print("Invalid choice.")
        return

    first_variable = numeric_columns[first_number - 1]
    second_variable = numeric_columns[second_number - 1]

    if first_variable == second_variable:

        print("Please select two different variables.")

        return

    result = analyze_correlation(
        metric_data,
        first_variable,
        second_variable
    )

    if result is None:

        print(
            "Not enough valid observations "
            "for correlation analysis."
        )

        return

    print()
    print(f"Variable 1   : {first_variable}")
    print(f"Variable 2   : {second_variable}")
    print(f"Correlation  : {result['correlation']:.4f}")
    print(f"P-value      : {result['p_value']:.6f}")
    print(f"Observations : {result['observations']}")

    print("\nInterpretation:")

    print(interpret_correlation(result))


# FORECASTING
def forecasting_menu(annual_data):

    while True:

        header("FORECASTING")

        metric = select_metric(annual_data)

        if metric is None:
            return

        forecasting_metric_menu(annual_data, metric)


def forecasting_metric_menu(annual_data, metric):

    while True:

        subheader(metric.upper())

        print("1. Model Evaluation")
        print("2. Forecast Future Years")
        print("0. Back")

        choice = input("\nEnter choice: ").strip()

        series = get_metric_series(annual_data, metric)

        if choice == "1":

            show_model_evaluation(series)

        elif choice == "2":

            show_forecast(series)

        elif choice == "0":

            return

        else:

            print("Invalid choice.")


# MODEL EVALUATION
def show_model_evaluation(series):

    subheader("MODEL EVALUATION")

    result = rolling_origin_validation(
        series,
        test_size=5,
    )

    if result is None:
        print(
            "Not enough historical observations "
            "for model validation."
        )
        return

    print(
        f"Training Period : "
        f"{result['train_start']}–{result['train_end']}"
    )

    print(
        f"Validation Period : "
        f"{result['test_start']}–{result['test_end']}"
    )

    print("\nLast-Value Baseline")

    print(
        f"MAE  : "
        f"{result['baseline_errors']['mae']:,.2f}"
    )

    print(
        f"RMSE : "
        f"{result['baseline_errors']['rmse']:,.2f}"
    )

    print(
        f"MAPE : "
        f"{result['baseline_errors']['mape']:.2f}%"
    )

    print("\nHolt Linear Trend")

    print(
        f"MAE  : "
        f"{result['holt_errors']['mae']:,.2f}"
    )

    print(
        f"RMSE : "
        f"{result['holt_errors']['rmse']:,.2f}"
    )

    print(
        f"MAPE : "
        f"{result['holt_errors']['mape']:.2f}%"
    )

    print("\nInterpretation:")

    print(interpret_model_evaluation(result))


# FORECAST
def show_forecast(series):

    subheader("FUTURE FORECAST")

    print("Forecast horizon:")

    print("1. 3 years")
    print("2. 5 years")
    print("3. 10 years")
    print("0. Back")

    choice = input("\nEnter choice: ").strip()

    horizons = {
        "1": 3,
        "2": 5,
        "3": 10,
    }

    if choice == "0":
        return

    if choice not in horizons:
        print("Invalid choice.")
        return

    periods = horizons[choice]

    result = forecast_annual(
        series,
        periods=periods,
    )

    if result is None:
        print(
            "Not enough historical observations "
            "for forecasting."
        )
        return

    subheader("FORECAST RESULTS")

    print(
        f"Selected Model : "
        f"{result['selected_model']}"
    )

    print()

    print(
        result["forecast"].to_string(index=False)
    )

    print("\nInterpretation:")

    print(result["interpretation"])

# ============================================================
# STATE / DISTRICT ANALYSIS
# ============================================================

def state_district_menu(cleaned_data):

    while True:

        header("STATE / DISTRICT ANALYSIS")

        print("1. State Analysis")
        print("2. District Analysis")
        print("0. Back")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":

            state_analysis(cleaned_data)

        elif choice == "2":

            district_analysis(cleaned_data)

        elif choice == "0":

            return

        else:

            print("Invalid choice.")


# STATE ANALYSIS
def state_analysis(cleaned_data):

    states = (
        cleaned_data["state_name"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    subheader("SELECT STATE")

    show_numbered_list(states)

    print("0. Back")

    choice = input("\nEnter choice: ").strip()

    if choice == "0":
        return

    if not choice.isdigit():
        print("Invalid choice.")
        return

    number = int(choice)

    if number < 1 or number > len(states):
        print("Invalid choice.")
        return

    selected_state = states[number - 1]

    state_data = cleaned_data[
        cleaned_data["state_name"] == selected_state
    ].copy()

    state_analysis_menu(state_data, selected_state)


def state_analysis_menu(state_data, state_name):

    while True:

        subheader(f"STATE: {state_name}")

        print("1. Dataset records")
        print("2. Available districts")
        print("3. Metric summary")
        print("0. Back")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":

            print(
                state_data.head(10)
                .to_string(index=False)
            )

        elif choice == "2":

            districts = (
                state_data["district_name"]
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

            print("\nDistricts:")

            for district in districts:
                print(f"- {district}")

        elif choice == "3":

            summary = (
                state_data
                .groupby("land_use_metrics")["area"]
                .sum()
                .sort_values(ascending=False)
            )

            print("\nMetric summary:")

            print(summary.to_string())

        elif choice == "0":

            return

        else:

            print("Invalid choice.")


# DISTRICT ANALYSIS
def district_analysis(cleaned_data):

    states = (
        cleaned_data["state_name"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    subheader("SELECT STATE")

    show_numbered_list(states)

    print("0. Back")

    state_choice = input("\nEnter choice: ").strip()

    if state_choice == "0":
        return

    if not state_choice.isdigit():
        print("Invalid choice.")
        return

    state_number = int(state_choice)

    if state_number < 1 or state_number > len(states):
        print("Invalid choice.")
        return

    selected_state = states[state_number - 1]

    state_data = cleaned_data[
        cleaned_data["state_name"] == selected_state
    ]

    districts = (
        state_data["district_name"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    subheader(f"SELECT DISTRICT - {selected_state}")

    show_numbered_list(districts)

    print("0. Back")

    district_choice = input("\nEnter choice: ").strip()

    if district_choice == "0":
        return

    if not district_choice.isdigit():
        print("Invalid choice.")
        return

    district_number = int(district_choice)

    if district_number < 1 or district_number > len(districts):
        print("Invalid choice.")
        return

    selected_district = districts[district_number - 1]

    district_data = state_data[
        state_data["district_name"] == selected_district
    ]

    district_analysis_menu(
        district_data,
        selected_state,
        selected_district,
    )


def district_analysis_menu(
    district_data,
    state_name,
    district_name,
):

    while True:

        subheader(
            f"DISTRICT: {district_name}, {state_name}"
        )

        print("1. View records")
        print("2. Metric summary")
        print("3. Year-wise data")
        print("0. Back")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":

            print(
                district_data.head(10)
                .to_string(index=False)
            )

        elif choice == "2":

            summary = (
                district_data
                .groupby("land_use_metrics")["area"]
                .sum()
                .sort_values(ascending=False)
            )

            print("\nMetric summary:")

            print(summary.to_string())

        elif choice == "3":

            year_data = (
                district_data[
                    ["year", "land_use_metrics", "area"]
                ]
                .sort_values("year")
            )

            print(
                year_data.to_string(index=False)
            )

        elif choice == "0":

            return

        else:

            print("Invalid choice.")


# EDA & VISUALIZATION
def eda_menu():

    header("EDA & VISUALIZATION")

    """
eda_visualizer.py
=================
EDA & visualization module for the Agricultural Land-Use Pattern Analysis Tool.

Uses seaborn + matplotlib. Works with the two datasets that terminal_app.py
already loads:

    cleaned_data -> create_member2_dataset()  (state / district level rows)
    annual_data  -> create_member3_dataset()  (national yearly totals per metric)

Every chart is saved as a PNG in the "eda_outputs" folder (next to this file)
and is also shown in a window when a display is available. On machines
without a display (SSH, servers) the charts are only saved.

Entry point used by terminal_app.py:

    eda_visualization_menu(cleaned_data, annual_data)

Area values in the source data are treated as hectares.
"""

import os
import sys
from pathlib import Path

import matplotlib

# Use a non-interactive backend when there is no display (headless / SSH).
if os.environ.get("MPLBACKEND") is None:
    _has_display = (
        sys.platform in ("win32", "darwin")
        or os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
    )
    if not _has_display:
        matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).with_name("eda_outputs")

CORE_METRICS = [
    "Reporting Area",
    "Net Area Sown",
    "Cropped Area",
    "Area Sown More Than Once",
]

CLASSIFICATIONS = [
    "Forests",
    "Not available for Cultivation",
    "Other Uncultivated Land Excluding Fallow Land",
    "Fallow Land",
]

MILLION = 1_000_000

sns.set_theme(style="whitegrid", context="notebook", palette="deep")


# ----------------------------------------------------------------------
# DISPLAY / INPUT HELPERS (same look as terminal_app.py)
# ----------------------------------------------------------------------
def _header(title):
    print()
    print("=" * 65)
    print(title)
    print("=" * 65)


def _subheader(title):
    print()
    print("-" * 65)
    print(title)
    print("-" * 65)


def _pick_from_list(items, title):
    """Show a numbered list and return the chosen item (None = back)."""
    if not items:
        print("Nothing is available to choose from.")
        return None

    _subheader(title)
    for number, item in enumerate(items, start=1):
        print(f"{number}. {item}")
    print("0. Back")

    choice = input("\nEnter choice: ").strip()

    if choice == "0":
        return None
    if not choice.isdigit() or not 1 <= int(choice) <= len(items):
        print("Invalid choice.")
        return None
    return items[int(choice) - 1]


def _ask_int(prompt, default, low, high):
    """Ask for an integer; Enter keeps the default."""
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    if raw.isdigit() and low <= int(raw) <= high:
        return int(raw)
    print(f"Invalid value, using {default}.")
    return default


def _ask_year(years, prompt="Year"):
    """Ask for a year from the available list; Enter selects the latest."""
    years = sorted(int(y) for y in years)
    latest = years[-1]
    raw = input(f"{prompt} ({years[0]}-{latest}) [{latest}]: ").strip()
    if raw == "":
        return latest
    if raw.isdigit() and int(raw) in years:
        return int(raw)
    print(f"Year not available, using {latest}.")
    return latest


def _finish(fig, filename):
    """Save the figure, show it if a display exists, then close it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {path}")

    if matplotlib.get_backend().lower() != "agg":
        print("Close the chart window to continue...")
        plt.show()
    plt.close(fig)


def _slug(text):
    return "".join(c.lower() if c.isalnum() else "_" for c in str(text)).strip("_")


def _million_formatter():
    return mticker.FuncFormatter(lambda v, _: f"{v / MILLION:,.0f}")


# ----------------------------------------------------------------------
# DATA HELPERS
# ----------------------------------------------------------------------
def _categories(cleaned_data):
    """Core metrics first, then classification sub-categories."""
    present = set(cleaned_data["category"].dropna().unique())
    core = [m for m in CORE_METRICS if m in present]
    others = sorted(present - set(core) - {"Unspecified"})
    return core + others


def _category_rows(cleaned_data, category):
    rows = cleaned_data[
        (cleaned_data["category"] == category)
        & cleaned_data["area"].notna()
        & cleaned_data["start_year"].notna()
    ].copy()
    rows["start_year"] = rows["start_year"].astype(int)
    return rows


def _yearly_totals(cleaned_data, category):
    """National total + number of reporting districts per year."""
    rows = _category_rows(cleaned_data, category)
    if rows.empty:
        return pd.DataFrame(columns=["start_year", "area", "districts"])
    return (
        rows.groupby("start_year")
        .agg(area=("area", "sum"), districts=("district_name", "nunique"))
        .reset_index()
    )


def _district_year_table(cleaned_data):
    """Wide table: one row per (year, state, district), one column per category."""
    rows = cleaned_data[
        cleaned_data["area"].notna() & cleaned_data["start_year"].notna()
    ].copy()
    rows["start_year"] = rows["start_year"].astype(int)
    wide = (
        rows.groupby(["start_year", "state_name", "district_name", "category"])["area"]
        .sum()
        .unstack("category")
        .reset_index()
    )
    wide.columns.name = None
    return wide


# ----------------------------------------------------------------------
# 1. NATIONAL TREND OF ONE CATEGORY
# ----------------------------------------------------------------------
def plot_category_trend(cleaned_data):
    category = _pick_from_list(_categories(cleaned_data), "SELECT LAND-USE CATEGORY")
    if category is None:
        return

    totals = _yearly_totals(cleaned_data, category)
    if totals.empty:
        print("No data available for this category.")
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    sns.lineplot(data=totals, x="start_year", y="area", marker="o", ax=ax1)
    ax1.yaxis.set_major_formatter(_million_formatter())
    ax1.set_ylabel("Area (million hectares)")
    ax1.set_xlabel("")
    ax1.set_title(f"{category}: reported total area by year")

    ax2.bar(totals["start_year"], totals["districts"], color="#8da0cb")
    ax2.set_ylabel("Districts\nreporting")
    ax2.set_xlabel("Year (start of agricultural year)")
    ax2.tick_params(axis="x", rotation=60)

    fig.text(
        0.01, 0.005,
        "Note: totals depend on how many districts reported each year (lower panel).",
        fontsize=8, style="italic",
    )
    _finish(fig, f"trend_{_slug(category)}.png")


# ----------------------------------------------------------------------
# 2. CORE METRICS TOGETHER
# ----------------------------------------------------------------------
def plot_core_metrics(annual_data):
    data = annual_data[annual_data["land_use_metrics"].isin(CORE_METRICS)].copy()
    if data.empty:
        print("No core metric data available.")
        return

    data["start_year"] = data["start_year"].astype(int)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(
        data=data, x="start_year", y="area",
        hue="land_use_metrics", hue_order=CORE_METRICS,
        style="land_use_metrics", markers=True, dashes=False, ax=ax,
    )
    ax.yaxis.set_major_formatter(_million_formatter())
    ax.set_xlabel("Year (start of agricultural year)")
    ax.set_ylabel("Area (million hectares)")
    ax.set_title("Core land-use metrics: national totals over time")
    ax.legend(title="Metric", loc="best")
    _finish(fig, "core_metrics_trend.png")


# ----------------------------------------------------------------------
# 3. LAND-USE COMPOSITION
# ----------------------------------------------------------------------
def plot_land_use_composition(cleaned_data):
    rows = cleaned_data[
        cleaned_data["area_classification"].isin(CLASSIFICATIONS)
        & cleaned_data["area"].notna()
        & cleaned_data["start_year"].notna()
    ].copy()
    if rows.empty:
        print("No classification data available.")
        return
    rows["start_year"] = rows["start_year"].astype(int)

    table = rows.pivot_table(
        index="start_year", columns="area_classification",
        values="area", aggfunc="sum", fill_value=0,
    )
    table = table[[c for c in CLASSIFICATIONS if c in table.columns]]
    share = table.div(table.sum(axis=1), axis=0) * 100

    latest = int(rows["start_year"].max())
    sub = (
        rows[rows["start_year"] == latest]
        .groupby("area_sub_classification")["area"]
        .sum()
        .sort_values()
    )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [3, 2]}
    )

    colors = sns.color_palette("deep", len(share.columns))
    ax1.stackplot(share.index, share.T.values, labels=share.columns, colors=colors)
    ax1.set_xlim(share.index.min(), share.index.max())
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Year (start of agricultural year)")
    ax1.set_ylabel("Share of classified area (%)")
    ax1.set_title("Land-use composition over time")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)

    ax2.barh(sub.index, sub.values / MILLION, color="#66a61e")
    ax2.set_xlabel("Area (million hectares)")
    ax2.set_title(f"Breakdown by sub-classification ({latest})")
    ax2.tick_params(axis="y", labelsize=8)

    _finish(fig, "land_use_composition.png")


# ----------------------------------------------------------------------
# 4. CROPPING PATTERN
# ----------------------------------------------------------------------
def plot_cropping_pattern(cleaned_data):
    wide = _district_year_table(cleaned_data)
    needed = ["Net Area Sown", "Cropped Area", "Reporting Area"]
    if any(col not in wide.columns for col in needed):
        print("Required metrics are not available.")
        return

    # Only districts that report all three values in the same year, so the
    # ratios are not distorted by districts that are missing.
    complete = wide.dropna(subset=needed)
    complete = complete[complete["Net Area Sown"] > 0]
    if complete.empty:
        print("Not enough complete records.")
        return

    yearly = complete.groupby("start_year")[needed].sum()
    intensity = yearly["Cropped Area"] / yearly["Net Area Sown"] * 100
    sown_share = yearly["Net Area Sown"] / yearly["Reporting Area"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    sns.lineplot(x=intensity.index, y=intensity.values,
                 marker="o", color="#d95f02", ax=ax1)
    ax1.set_title("Cropping intensity\n(Cropped Area / Net Area Sown x 100)")
    ax1.set_xlabel("Year (start of agricultural year)")
    ax1.set_ylabel("Percent")

    sns.lineplot(x=sown_share.index, y=sown_share.values,
                 marker="o", color="#1b9e77", ax=ax2)
    ax2.set_title("Net Area Sown as share of Reporting Area")
    ax2.set_xlabel("Year (start of agricultural year)")
    ax2.set_ylabel("Percent")

    fig.text(
        0.01, 0.005,
        "Based only on districts reporting all three metrics in that year.",
        fontsize=8, style="italic",
    )
    _finish(fig, "cropping_pattern.png")


# ----------------------------------------------------------------------
# 5. STATE COMPARISON (BAR)
# ----------------------------------------------------------------------
def plot_state_comparison(cleaned_data):
    category = _pick_from_list(_categories(cleaned_data), "SELECT LAND-USE CATEGORY")
    if category is None:
        return

    rows = _category_rows(cleaned_data, category)
    if rows.empty:
        print("No data available for this category.")
        return

    year = _ask_year(rows["start_year"].unique())
    top_n = _ask_int("Number of states to show", 15, 3, 36)

    by_state = (
        rows[rows["start_year"] == year]
        .groupby("state_name")["area"].sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    if by_state.empty:
        print("No data for that year.")
        return

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(by_state) + 2))
    sns.barplot(x=by_state.values / MILLION, y=by_state.index.tolist(),
                hue=by_state.index.tolist(), palette="viridis",
                legend=False, ax=ax)
    ax.set_xlabel("Area (million hectares)")
    ax.set_ylabel("")
    ax.set_title(f"{category}: top {len(by_state)} states, {year}")
    _finish(fig, f"states_{_slug(category)}_{year}.png")


# ----------------------------------------------------------------------
# 6. STATE TRENDS (LINES)
# ----------------------------------------------------------------------
def plot_state_trends(cleaned_data):
    category = _pick_from_list(_categories(cleaned_data), "SELECT LAND-USE CATEGORY")
    if category is None:
        return

    rows = _category_rows(cleaned_data, category)
    if rows.empty:
        print("No data available for this category.")
        return

    top_n = _ask_int("Number of states to compare", 6, 2, 10)
    latest = rows["start_year"].max()
    ranking = (
        rows[rows["start_year"] == latest]
        .groupby("state_name")["area"].sum()
        .sort_values(ascending=False)
    )
    chosen = ranking.head(top_n).index.tolist()

    trend = (
        rows[rows["state_name"].isin(chosen)]
        .groupby(["start_year", "state_name"])["area"].sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(data=trend, x="start_year", y="area", hue="state_name",
                 hue_order=chosen, marker="o", ax=ax)
    ax.yaxis.set_major_formatter(_million_formatter())
    ax.set_xlabel("Year (start of agricultural year)")
    ax.set_ylabel("Area (million hectares)")
    ax.set_title(f"{category}: largest {len(chosen)} states (ranked by {int(latest)})")
    ax.legend(title="State", fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    _finish(fig, f"state_trends_{_slug(category)}.png")


# ----------------------------------------------------------------------
# 7. STATE x YEAR HEATMAP
# ----------------------------------------------------------------------
def plot_state_heatmap(cleaned_data):
    category = _pick_from_list(_categories(cleaned_data), "SELECT LAND-USE CATEGORY")
    if category is None:
        return

    rows = _category_rows(cleaned_data, category)
    if rows.empty:
        print("No data available for this category.")
        return

    top_n = _ask_int("Number of states to show", 15, 3, 36)
    table = rows.pivot_table(index="state_name", columns="start_year",
                             values="area", aggfunc="sum")
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).head(top_n).index]
    table = table / MILLION

    fig, ax = plt.subplots(figsize=(14, 0.45 * len(table) + 3))
    sns.heatmap(table, cmap="YlGn", linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Million hectares"}, ax=ax)
    ax.set_xlabel("Year (start of agricultural year)")
    ax.set_ylabel("")
    ax.set_title(f"{category}: state x year (blank = no data)")
    _finish(fig, f"state_heatmap_{_slug(category)}.png")


# ----------------------------------------------------------------------
# 8. DISTRICT COMPARISON
# ----------------------------------------------------------------------
def plot_district_comparison(cleaned_data):
    states = sorted(cleaned_data["state_name"].dropna().unique().tolist())
    state = _pick_from_list(states, "SELECT STATE")
    if state is None:
        return

    category = _pick_from_list(_categories(cleaned_data), "SELECT LAND-USE CATEGORY")
    if category is None:
        return

    rows = _category_rows(cleaned_data, category)
    rows = rows[rows["state_name"] == state]
    if rows.empty:
        print("No data available for this state and category.")
        return

    year = _ask_year(rows["start_year"].unique())
    top_n = _ask_int("Number of districts to show", 15, 3, 40)

    by_district = (
        rows[rows["start_year"] == year]
        .groupby("district_name")["area"].sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    if by_district.empty:
        print("No data for that year.")
        return

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(by_district) + 2))
    sns.barplot(x=by_district.values / 1000, y=by_district.index.tolist(),
                hue=by_district.index.tolist(), palette="mako",
                legend=False, ax=ax)
    ax.set_xlabel("Area (thousand hectares)")
    ax.set_ylabel("")
    ax.set_title(f"{state}: top {len(by_district)} districts, {category} ({year})")
    _finish(fig, f"districts_{_slug(state)}_{_slug(category)}_{year}.png")


# ----------------------------------------------------------------------
# 9. DISTRIBUTION ACROSS DISTRICTS
# ----------------------------------------------------------------------
def plot_distribution(cleaned_data):
    category = _pick_from_list(_categories(cleaned_data), "SELECT LAND-USE CATEGORY")
    if category is None:
        return

    rows = _category_rows(cleaned_data, category)
    rows = rows[rows["area"] > 0]
    if rows.empty:
        print("No positive values available for this category.")
        return

    years = sorted(rows["start_year"].unique())
    shown_years = years[::5]
    if years[-1] not in shown_years:
        shown_years.append(years[-1])
    subset = rows[rows["start_year"].isin(shown_years)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(data=subset, x="start_year", y="area", color="#a6cee3",
                fliersize=2, ax=ax1)
    ax1.set_yscale("log")
    ax1.set_xlabel("Year (start of agricultural year)")
    ax1.set_ylabel("District area (hectares, log scale)")
    ax1.set_title(f"{category}: district-level spread")

    latest = subset[subset["start_year"] == shown_years[-1]]
    sns.histplot(latest["area"], log_scale=True, bins=35, kde=True,
                 color="#1f78b4", ax=ax2)
    ax2.set_xlabel("District area (hectares, log scale)")
    ax2.set_ylabel("Number of districts")
    ax2.set_title(f"Distribution in {shown_years[-1]}")

    _finish(fig, f"distribution_{_slug(category)}.png")


# ----------------------------------------------------------------------
# 10. CORRELATION HEATMAP
# ----------------------------------------------------------------------
def plot_correlation_heatmap(cleaned_data):
    wide = _district_year_table(cleaned_data)
    columns = [c for c in _categories(cleaned_data) if c in wide.columns]
    numeric = wide[columns]
    if numeric.dropna(how="all").shape[0] < 10 or len(columns) < 2:
        print("Not enough data for a correlation matrix.")
        return

    corr = numeric.corr(min_periods=10)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"label": "Pearson r"}, ax=ax)
    ax.set_title("Correlation between land-use categories\n(district-year level)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    _finish(fig, "correlation_heatmap.png")


# ----------------------------------------------------------------------
# 11. DATA QUALITY / MISSING VALUES
# ----------------------------------------------------------------------
def plot_missing_values(cleaned_data):
    rows = cleaned_data[cleaned_data["start_year"].notna()].copy()
    rows["start_year"] = rows["start_year"].astype(int)
    rows["missing"] = rows["area"].isna()

    by_category = rows.groupby("category")["missing"].mean().mul(100).sort_values()
    by_year = rows.groupby("start_year")["missing"].mean().mul(100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.barh(by_category.index, by_category.values, color="#fc8d62")
    ax1.set_xlabel("Missing area values (%)")
    ax1.set_title("Missing values by land-use category")
    ax1.tick_params(axis="y", labelsize=8)

    ax2.bar(by_year.index, by_year.values, color="#8da0cb")
    ax2.set_xlabel("Year (start of agricultural year)")
    ax2.set_ylabel("Missing area values (%)")
    ax2.set_title("Missing values by year")
    ax2.tick_params(axis="x", rotation=60)

    _finish(fig, "missing_values.png")


# ----------------------------------------------------------------------
# MENU (called from terminal_app.py)
# ----------------------------------------------------------------------
def eda_visualization_menu(cleaned_data, annual_data):
    """Interactive EDA & visualization menu."""

    actions = {
        "1": ("Trend of a land-use category (national)", lambda: plot_category_trend(cleaned_data)),
        "2": ("Core metrics comparison", lambda: plot_core_metrics(annual_data)),
        "3": ("Land-use composition", lambda: plot_land_use_composition(cleaned_data)),
        "4": ("Cropping pattern (intensity)", lambda: plot_cropping_pattern(cleaned_data)),
        "5": ("State comparison (bar chart)", lambda: plot_state_comparison(cleaned_data)),
        "6": ("State trends (line chart)", lambda: plot_state_trends(cleaned_data)),
        "7": ("State x year heatmap", lambda: plot_state_heatmap(cleaned_data)),
        "8": ("District comparison within a state", lambda: plot_district_comparison(cleaned_data)),
        "9": ("Distribution across districts", lambda: plot_distribution(cleaned_data)),
        "10": ("Correlation heatmap", lambda: plot_correlation_heatmap(cleaned_data)),
        "11": ("Missing values overview", lambda: plot_missing_values(cleaned_data)),
    }

    while True:
        _header("EDA & VISUALIZATION")

        for key, (label, _) in actions.items():
            print(f"{key}. {label}")
        print("0. Back")

        choice = input("\nEnter choice: ").strip()

        if choice == "0":
            return

        if choice not in actions:
            print("Invalid choice. Please select one of the available options.")
            continue

        try:
            actions[choice][1]()
        except (EOFError, KeyboardInterrupt):
            raise
        except Exception as error:  # keep the terminal app alive
            plt.close("all")
            print(f"\nCould not create the chart: {error}")


if __name__ == "__main__":
    from data_preprocessor import create_member2_dataset, create_member3_dataset

    print("Loading project data...")
    eda_visualization_menu(create_member2_dataset(), create_member3_dataset())




# ABOUT PROJECT
def about_project():

    header("ABOUT PROJECT")

    print("Agricultural Land-Use Pattern Analysis Tool")
    print()
    print("Project:")
    print("Analysis of Changing Agricultural Land-Use")
    print("Patterns Across Indian States and Districts")
    print()
    print("This console application provides:")
    print("- Processed dataset inspection")
    print("- EDA and visualization access")
    print("- Statistical analysis")
    print("- Time-series forecasting")
    print("- State-level analysis")
    print("- District-level analysis")
    print()
    print("Data source:")
    print("Ministry of Agriculture and Farmers Welfare")
    print("Department of Agriculture & Farmers Welfare")


# MAIN APPLICATION
def main():

    print()
    print("=" * 65)
    print("AGRICULTURAL LAND-USE PATTERN ANALYSIS TOOL")
    print("=" * 65)

    cleaned_data, annual_data = load_application_data()

    while True:

        header("MAIN MENU")

        print("1. Dataset")
        print("2. EDA & Visualization")
        print("3. Statistical Analysis")
        print("4. Forecasting")
        print("5. State / District Analysis")
        print("6. About Project")
        print("0. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":

            dataset_menu(cleaned_data)

        elif choice == "2":

            eda_menu()

        elif choice == "3":

            statistical_analysis_menu(annual_data)

        elif choice == "4":

            forecasting_menu(annual_data)

        elif choice == "5":

            state_district_menu(cleaned_data)

        elif choice == "6":

            about_project()

        elif choice == "0":

            print("\nExiting Agricultural Land-Use Pattern Analysis Tool.")

            break

        else:

            print("Invalid choice. Please select a valid option.")


if __name__ == "__main__":
    main()
