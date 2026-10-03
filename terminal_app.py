import pandas as pd

from data_preprocessor import *

from stat_forecasting import *

# ============================================================
# DISPLAY HELPERS
# ============================================================

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


# ============================================================
# DATA LOADING
# ============================================================

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


# ============================================================
# DATASET MENU
# ============================================================

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


# ============================================================
# METRIC SELECTION
# ============================================================

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


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

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


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

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

# ============================================================
# TREND ANALYSIS
# ============================================================

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

# ============================================================
# YEAR-OVER-YEAR CHANGE
# ============================================================

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
    
# ============================================================
# PERIOD COMPARISON
# ============================================================

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

# ============================================================
# CORRELATION ANALYSIS
# ============================================================

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

# ============================================================
# FORECASTING
# ============================================================

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


# ============================================================
# MODEL EVALUATION
# ============================================================

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

# ============================================================
# FORECAST
# ============================================================

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


# ============================================================
# STATE ANALYSIS
# ============================================================

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


# ============================================================
# DISTRICT ANALYSIS
# ============================================================

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


# ============================================================
# EDA
# ============================================================

def eda_menu():

    header("EDA & VISUALIZATION")

    print("EDA and visualization functions will be")
    print("connected here from eda_visualizer.py.")
    print()
    print("Available interface section:")
    print("- Trend visualization")
    print("- State comparison")
    print("- District comparison")
    print("- Land-use category visualization")
    print("- Cropping pattern visualization")


# ============================================================
# ABOUT
# ============================================================

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


# ============================================================
# MAIN APPLICATION
# ============================================================

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