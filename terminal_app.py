import shutil

import pandas as pd
from data_preprocessor import *
from stat_forecasting import *

# DISPLAY HELPERS
def header(title):
    _print_box_title(title, "double")


def subheader(title):
    _print_box_title(title, "single")


def _print_box_title(title, style):
    width = max(36, min(shutil.get_terminal_size((100, 20)).columns - 2, 78))
    title = str(title)
    title = title[:width - 6]
    inner_width = width - 2

    horizontal = "=" if style == "double" else "-"

    print()
    print(f"+{horizontal * inner_width}+")
    print(f"|  {title:<{inner_width - 4}}  |")
    print(f"+{horizontal * inner_width}+")


def _format_cell(value):
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def print_table(data):
    if isinstance(data, pd.Series):
        data = data.to_frame().reset_index()
    elif not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if data.empty:
        print("  (no records)")
        return

    headers = [str(column) for column in data.columns]
    rows = [
        [_format_cell(value) for value in row]
        for row in data.itertuples(index=False, name=None)
    ]
    terminal_width = max(60, shutil.get_terminal_size((100, 20)).columns - 4)
    max_cell_width = max(10, min(36, (terminal_width - len(headers) * 3) // len(headers)))
    widths = [
        min(max_cell_width, max(len(headers[index]), *(len(row[index]) for row in rows)))
        for index in range(len(headers))
    ]

    def fit(value, width):
        if len(value) <= width:
            return value
        return value[:width - 3] + "..."

    def border(left, join, right):
        return left + join.join("-" * (width + 2) for width in widths) + right

    print(border("+", "+", "+"))
    print("| " + " | ".join(fit(value, width).ljust(width) for value, width in zip(headers, widths)) + " |")
    print(border("+", "+", "+"))
    for row in rows:
        print("| " + " | ".join(fit(value, width).ljust(width) for value, width in zip(row, widths)) + " |")
    print(border("+", "+", "+"))


def print_menu_options(options):
    print_table(
        pd.DataFrame(
            [(key, label) for key, label in options],
            columns=["KEY", "OPTION"],
        )
    )


def show_numbered_list(items):
    print_table(
        pd.DataFrame(
            [(number, item) for number, item in enumerate(items, start=1)],
            columns=["#", "OPTION"],
        )
    )


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

        print_menu_options([
            ("1", "View first records"),
            ("2", "View last records"),
            ("3", "Dataset information"),
            ("4", "View available metrics"),
            ("5", "View available states"),
            ("0", "Back"),
        ])

        choice = input("\nEnter choice: ").strip()

        if choice == "1":

            subheader("FIRST 10 RECORDS")

            print_table(cleaned_data.head(10))

        elif choice == "2":

            subheader("LAST 10 RECORDS")

            print_table(cleaned_data.tail(10))

        elif choice == "3":

            subheader("DATASET INFORMATION")

            missing = cleaned_data.isna().sum()
            print_table(
                pd.DataFrame({
                    "COLUMN": cleaned_data.columns,
                    "MISSING VALUES": missing.values,
                })
            )
            print(f"\nRows: {cleaned_data.shape[0]:,}  |  Columns: {cleaned_data.shape[1]}")

        elif choice == "4":

            subheader("AVAILABLE LAND-USE METRICS")

            metrics = (
                cleaned_data["land_use_metrics"]
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

            show_numbered_list(metrics)

        elif choice == "5":

            subheader("AVAILABLE STATES / UNION TERRITORIES")

            states = (
                cleaned_data["state_name"]
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

            show_numbered_list(states)

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

    print_menu_options([("0", "Back")])

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

        print_menu_options([
            ("1", "Descriptive Statistics"),
            ("2", "Trend Analysis"),
            ("3", "Year-over-Year Changes"),
            ("4", "Period Comparison"),
            ("5", "Correlation Analysis"),
            ("0", "Back"),
        ])

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

    print_table(pd.DataFrame([
        ("Observations", result["observations"]),
        ("Mean", result["mean"]),
        ("Median", result["median"]),
        ("Minimum", result["minimum"]),
        ("Maximum", result["maximum"]),
        ("Standard Deviation", result["standard_deviation"]),
        ("First Year", result["first_year"]),
        ("Last Year", result["last_year"]),
    ], columns=["MEASURE", "VALUE"]))

    print("\nInterpretation:")

    interpretation = interpret_descriptive_summary(result)

    print(interpretation)

# TREND ANALYSIS
def show_trend_analysis(series):

    subheader("TREND ANALYSIS")

    result = analyze_trend(series)

    print_table(pd.DataFrame([
        ("Slope", result["slope"]),
        ("Intercept", result["intercept"]),
        ("R-squared", f"{result['r_squared']:.4f}"),
        ("P-value", f"{result['p_value']:.6f}"),
        ("Standard Error", result["standard_error"]),
        ("Observations", result["observations"]),
    ], columns=["MEASURE", "VALUE"]))

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
    print_table(
        result[
            ["year", "value", "change", "percentage_change"]
        ].tail(10)
    )

    # Find largest increase and decrease
    interpretation = interpret_yearly_changes(series)

    if interpretation is None:

        print("\nNo year-over-year changes are available.")

        return

    print_table(pd.DataFrame([
        (
            "Largest increase",
            interpretation["largest_increase_year"],
            interpretation["largest_increase"],
            f"{interpretation['largest_increase_percentage']:.2f}%",
        ),
        (
            "Largest decrease",
            interpretation["largest_decrease_year"],
            interpretation["largest_decrease"],
            f"{interpretation['largest_decrease_percentage']:.2f}%",
        ),
    ], columns=["CHANGE", "YEAR", "AREA CHANGE", "PERCENTAGE"]))

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

    print_table(pd.DataFrame([
        ("First Period", f"{result['first_period'][0]}–{result['first_period'][1]}"),
        ("Second Period", f"{result['second_period'][0]}–{result['second_period'][1]}"),
        ("First Mean", result["first_mean"]),
        ("Second Mean", result["second_mean"]),
        ("Difference", result["difference"]),
        ("Percentage Change", f"{result['percentage_change']:.2f}%"),
    ], columns=["MEASURE", "VALUE"]))

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

    show_numbered_list(numeric_columns)
    print_menu_options([("0", "Back")])

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

    print_table(pd.DataFrame([
        ("Variable 1", first_variable),
        ("Variable 2", second_variable),
        ("Correlation", f"{result['correlation']:.4f}"),
        ("P-value", f"{result['p_value']:.6f}"),
        ("Observations", result["observations"]),
    ], columns=["MEASURE", "VALUE"]))

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

        print_menu_options([
            ("1", "Model Evaluation"),
            ("2", "Forecast Future Years"),
            ("0", "Back"),
        ])

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

    print_table(pd.DataFrame([
        ("Training Period", f"{result['train_start']}–{result['train_end']}"),
        ("Validation Period", f"{result['test_start']}–{result['test_end']}"),
        ("Baseline MAE", result["baseline_errors"]["mae"]),
        ("Baseline RMSE", result["baseline_errors"]["rmse"]),
        ("Baseline MAPE", f"{result['baseline_errors']['mape']:.2f}%"),
        ("Holt MAE", result["holt_errors"]["mae"]),
        ("Holt RMSE", result["holt_errors"]["rmse"]),
        ("Holt MAPE", f"{result['holt_errors']['mape']:.2f}%"),
    ], columns=["MEASURE", "VALUE"]))

    print("\nInterpretation:")

    print(interpret_model_evaluation(result))


# FORECAST
def show_forecast(series):

    subheader("FUTURE FORECAST")

    print_menu_options([
        ("1", "3 years"),
        ("2", "5 years"),
        ("3", "10 years"),
        ("0", "Back"),
    ])

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

    print_table(pd.DataFrame([
        ("Selected Model", result["selected_model"]),
    ], columns=["MEASURE", "VALUE"]))
    print_table(result["forecast"])

    print("\nInterpretation:")

    print(result["interpretation"])

# ============================================================
# STATE / DISTRICT ANALYSIS
# ============================================================

def state_district_menu(cleaned_data):

    while True:

        header("STATE / DISTRICT ANALYSIS")

        print_menu_options([
            ("1", "State Analysis"),
            ("2", "District Analysis"),
            ("0", "Back"),
        ])

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

    print_menu_options([("0", "Back")])

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

        print_menu_options([
            ("1", "Dataset records"),
            ("2", "Available districts"),
            ("3", "Metric summary"),
            ("0", "Back"),
        ])

        choice = input("\nEnter choice: ").strip()

        if choice == "1":

            print_table(state_data.head(10))

        elif choice == "2":

            districts = (
                state_data["district_name"]
                .dropna()
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

            show_numbered_list(districts)

        elif choice == "3":

            summary = (
                state_data
                .groupby("land_use_metrics")["area"]
                .sum()
                .sort_values(ascending=False)
            )

            print_table(summary)

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

    print_menu_options([("0", "Back")])

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

    print_menu_options([("0", "Back")])

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

        print_menu_options([
            ("1", "View records"),
            ("2", "Metric summary"),
            ("3", "Year-wise data"),
            ("0", "Back"),
        ])

        choice = input("\nEnter choice: ").strip()

        if choice == "1":

            print_table(district_data.head(10))

        elif choice == "2":

            summary = (
                district_data
                .groupby("land_use_metrics")["area"]
                .sum()
                .sort_values(ascending=False)
            )

            print_table(summary)

        elif choice == "3":

            year_data = (
                district_data[
                    ["year", "land_use_metrics", "area"]
                ]
                .sort_values("year")
            )

            print_table(year_data)

        elif choice == "0":

            return

        else:

            print("Invalid choice.")


# EDA & VISUALIZATION
def eda_menu():

    header("EDA & VISUALIZATION")

    print("EDA and visualization functions will be")
    print("connected here from eda_visualizer.py.")
    print_table(pd.DataFrame({
        "AVAILABLE SECTION": [
            "Trend visualization",
            "State comparison",
            "District comparison",
            "Land-use category visualization",
            "Cropping pattern visualization",
        ]
    }))



# ABOUT PROJECT
def about_project():

    header("ABOUT PROJECT")

    print_table(pd.DataFrame([
        ("Project", "Analysis of Changing Agricultural Land-Use Patterns Across Indian States and Districts"),
        ("Features", "Processed dataset inspection"),
        ("Features", "EDA and visualization access"),
        ("Features", "Statistical analysis"),
        ("Features", "Time-series forecasting"),
        ("Features", "State-level analysis"),
        ("Features", "District-level analysis"),
        ("Data source", "Ministry of Agriculture and Farmers Welfare"),
        ("Department", "Department of Agriculture & Farmers Welfare"),
    ], columns=["DETAIL", "DESCRIPTION"]))


# MAIN APPLICATION
def main():

    header("AGRICULTURAL LAND-USE PATTERN ANALYSIS TOOL")

    cleaned_data, annual_data = load_application_data()

    while True:

        header("MAIN MENU")

        print_menu_options([
            ("1", "Dataset"),
            ("2", "EDA & Visualization"),
            ("3", "Statistical Analysis"),
            ("4", "Forecasting"),
            ("5", "State / District Analysis"),
            ("6", "About Project"),
            ("0", "Exit"),
        ])

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