import pandas as pd
from data_preprocessor import *
from stat_forecasting import *
#from statistics_forecasting_menu import *

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

# EDA & VISUALIZATION
def eda_menu(cleaned_data, annual_data):
    from eda_visualizer import eda_visualization_menu
    
    eda_visualization_menu(cleaned_data, annual_data)


# STATE / DISTRICT ANALYSIS
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
    about_project()
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

            eda_menu(cleaned_data, annual_data)

        elif choice == "3":

            #statistical_analysis_menu(annual_data)
            print("Statistical Analysis work will be available soon.")

        elif choice == "4":

            #forecasting_menu(annual_data)
            print("Forecasting work will be available soon.")

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