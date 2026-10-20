import os

import pandas as pd
from data_preprocessor import create_member2_dataset, create_member3_dataset
from statistics_forecasting_menu import * 

ASCII_UI = str.maketrans({
    "╭": "+",
    "╮": "+",
    "╰": "+",
    "╯": "+",
    "─": "-",
    "│": "|",
    "↩": "<-",
    "▸": ">",
    "•": "*",
    "═": "=",
    "┬": "+",
    "┴": "+",
    "├": "+",
    "┤": "+",
    "┼": "+",
})


def _print_line(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.translate(ASCII_UI))

# DISPLAY HELPERS
def header(title):
    _print_box_title(title, "double")


def subheader(title):
    _print_box_title(title, "single")


def _print_box_title(title, style):
    try:
        width = min(os.get_terminal_size().columns, 78)
    except OSError:
        width = 78
    width = max(36, width)
    title = str(title)[:width - 8]
    horizontal = "═" if style == "double" else "─"

    _print_line("")
    _print_line("╭" + horizontal * (width - 2) + "╮")
    _print_line("│  " + title.center(width - 6) + "  │")
    _print_line("╰" + horizontal * (width - 2) + "╯")


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
    try:
        terminal_width = max(60, os.get_terminal_size().columns - 4)
    except OSError:
        terminal_width = 76
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
        return left + join.join("─" * (width + 2) for width in widths) + right

    _print_line(border("╭", "┬", "╮"))
    _print_line("│ " + " │ ".join(fit(value, width).ljust(width) for value, width in zip(headers, widths)) + " │")
    _print_line(border("├", "┼", "┤"))
    for row in rows:
        _print_line("│ " + " │ ".join(fit(value, width).ljust(width) for value, width in zip(row, widths)) + " │")
    _print_line(border("╰", "┴", "╯"))


def print_menu_options(options):
    print_table(
        pd.DataFrame(
            [
                (key, f"{'↩' if key == '0' else '▸'}  {label}")
                for key, label in options
            ],
            columns=["KEY", "OPTION"],
        )
    )

def show_numbered_list(items):
    print_table(
        pd.DataFrame(
            [(number, f"•  {item}") for number, item in enumerate(items, start=1)],
            columns=["#", "OPTION"],
        )
    )


def _input_choice(menu_path):
    return input(f"\n{menu_path} > Enter choice: ").strip()


# DATA LOADING
def load_application_data():
    """
    Load the two datasets required by different application modules.

    member2_data:
        Cleaned dataset containing state/district information.

    member3_data:
        Annual aggregated dataset used by statistics and forecasting.
    """

    member2_data = create_member2_dataset()
    member3_data = create_member3_dataset()

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

        choice = _input_choice("Main Menu > Dataset")

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

# EDA & VISUALIZATION
def eda_menu(cleaned_data, annual_data):
    from eda_visualizer import eda_visualization_menu

    eda_visualization_menu(cleaned_data, annual_data)


# STATE / DISTRICT ANALYSIS
def state_district_menu(cleaned_data):

    while True:

        header("STATE / DISTRICT ANALYSIS")

        print_menu_options([
            ("1", "State Analysis"),
            ("2", "District Analysis"),
            ("0", "Back"),
        ])

        choice = _input_choice("Main Menu > State / District Analysis")

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

    choice = _input_choice("Main Menu > State / District Analysis > Select State")

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

        choice = _input_choice(
            f"Main Menu > State / District Analysis > State: {state_name}"
        )

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

    state_choice = _input_choice(
        "Main Menu > State / District Analysis > Select State"
    )

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

    district_choice = _input_choice(
        f"Main Menu > State / District Analysis > State: {selected_state} > Select District"
    )

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

        choice = _input_choice(
            f"Main Menu > State / District Analysis > "
            f"{district_name}, {state_name}"
        )

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

    header("AGRICULTURAL LAND-USE PATTERN ANALYSIS TOOL")
    cleaned_data, annual_data = load_application_data()

    while True:

        header("MAIN MENU")

        print_menu_options([
            ("1", "Dataset"),
            ("2", "EDA & Visualization (5 chart views)"),
            ("3", "Statistical Analysis"),
            ("4", "Forecasting"),
            ("5", "State / District Analysis"),
            ("6", "About Project"),
            ("0", "Exit"),
        ]) 

        choice = _input_choice("Main Menu")

        if choice == "1":

            dataset_menu(cleaned_data)

        elif choice == "2":

            eda_menu(cleaned_data, annual_data)

        elif choice == "3":
            statistical_analysis_menu(cleaned_data, annual_data)

        elif choice == "4":
            forecasting_menu(cleaned_data, annual_data)
    

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