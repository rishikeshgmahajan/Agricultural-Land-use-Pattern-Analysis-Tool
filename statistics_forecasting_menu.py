"""Console menus for statistical analysis and forecasting."""

import pandas as pd

import stat_forecasting as sf

ASCII_UI = str.maketrans({
    "╭": "+", "╮": "+", "╰": "+", "╯": "+", "─": "-", "│": "|",
    "═": "=", "↩": "<-", "▸": ">",
})


def _print_line(text=""):
    try:
        print(text)
    except UnicodeEncodeError:
        print(str(text).translate(ASCII_UI))


def _header(title):
    width = 66
    _print_line("")
    _print_line("╭" + "─" * (width - 2) + "╮")
    _print_line("│ " + title.center(width - 4) + " │")
    _print_line("╰" + "─" * (width - 2) + "╯")


def _show_options(options):
    for key, label in options:
        arrow = "↩" if key == "0" else "▸"
        _print_line(f"  {key}  {arrow} {label}")


def _ask(path):
    return input(f"\n{path} > Enter choice: ").strip()


def _ask_year(prompt, valid_years, default):
    """Ask for a year; Enter keeps the default."""
    if not valid_years:
        return None
    while True:
        answer = input(f"{prompt} (Enter = {default}): ").strip()
        if not answer:
            return default
        if answer.isdigit() and int(answer) in valid_years:
            return int(answer)
        _print_line(f"Please enter a year between {min(valid_years)} and {max(valid_years)}.")


def _pause():
    input("\nPress Enter to continue...")


 
# Statistical analysis menu
 
def _choose_states(cleaned_data):
    counts = sf.eligible_states(cleaned_data)
    top = counts.head(15)
    if top.empty:
        _print_line("No states have enough valid district data for comparison.")
        return None

    _print_line("\nStates with the most valid districts (latest usable year):")
    for number, (state, districts) in enumerate(top.items(), start=1):
        _print_line(f"  {number:>2}  {state} ({districts} districts)")
    _print_line("\nEnter 2 state numbers for a t-test, or 3 or more for ANOVA, separated by commas.")
    answer = input("Press Enter to use the 4 largest states: ").strip()
    if not answer:
        return list(top.index[:4])
    try:
        numbers = [int(part.strip()) for part in answer.split(",")]
    except ValueError:
        _print_line("Please enter numbers separated by commas.")
        return None
    if len(set(numbers)) < 2 or any(number < 1 or number > len(top) for number in numbers):
        _print_line("Choose at least two different numbers from the list.")
        return None
    return [top.index[number - 1] for number in dict.fromkeys(numbers)]


def statistical_analysis_menu(cleaned_data, annual_data):
    while True:
        _header("STATISTICAL ANALYSIS")
        _show_options([
            ("1", "Descriptive statistics & state ranking"),
            ("2", "Trend analysis"),
            ("3", "Early vs later period comparison"),
            ("4", "Year-to-year change analysis"),
            ("5", "Correlation analysis"),
            ("6", "State comparison (t-test / ANOVA)"),
            ("0", "Back"),
        ])
        choice = _ask("Main Menu > Statistical Analysis")

        if choice == "0":
            return
        elif choice == "1":
            sf.descriptive_statistics(cleaned_data, annual_data)
        elif choice == "2":
            sf.trend_analysis(cleaned_data)
        elif choice == "3":
            years = sf.usable_years(sf.district_metric_table(cleaned_data))
            if not years:
                _print_line("Not enough valid district data to compare years.")
                _pause()
                continue
            early = _ask_year("Early year (start of agricultural year)", years, years[0])
            late = _ask_year("Later year", years, years[-1])
            if early is None or late is None:
                _print_line("No usable years were found.")
            elif early >= late:
                _print_line("The later year must be after the early year.")
                _pause()
                continue
            else:
                sf.period_comparison(cleaned_data, early, late)
        elif choice == "4":
            sf.year_to_year_change(cleaned_data, annual_data)
        elif choice == "5":
            sf.correlation_analysis(cleaned_data)
        elif choice == "6":
            states = _choose_states(cleaned_data)
            if states:
                sf.compare_states(cleaned_data, states)
        else:
            _print_line("Please select one of the listed options.")
            continue
        _pause()


 
# Forecasting menu
 
def forecasting_menu(cleaned_data, annual_data=None):
    try:
        series, panel_states = sf.build_intensity_series(cleaned_data)
    except ValueError as error:
        _print_line(f"Cannot build the forecasting series: {error}")
        _pause()
        return

    while True:
        _header("FORECASTING")
        _print_line(f"Series: cropping intensity of {len(panel_states)} states passing the 80%-125% Net Area Sown stability screen")
        _show_options([
            ("1", "Time series object & subset"),
            ("2", "Stationarity test (ADF)"),
            ("3", "Smoothing (moving average)"),
            ("4", "Differencing"),
            ("5", "ARIMA model & forecast"),
            ("0", "Back"),
        ])
        choice = _ask("Main Menu > Forecasting")

        if choice == "0":
            return
        elif choice == "1":
            years = list(series.index.year)
            start = _ask_year("Subset start year", years, years[len(years) // 2])
            if start is not None:
                sf.time_series_overview(series, start)
        elif choice == "2":
            sf.stationarity_test(series)
        elif choice == "3":
            sf.smoothing_chart(series)
        elif choice == "4":
            sf.differencing_analysis(series)
        elif choice == "5":
            answer = input("Forecast horizon in years (Enter = 5): ").strip()
            horizon = int(answer) if answer.isdigit() and 1 <= int(answer) <= 10 else 5
            sf.arima_forecast(series, horizon=horizon)
        else:
            _print_line("Please select one of the listed options.")
            continue
        _pause()


if __name__ == "__main__":
    from data_preprocessor import create_member2_dataset, create_member3_dataset

    _print_line("Loading project data...")
    cleaned = create_member2_dataset()
    annual = create_member3_dataset()
    statistical_analysis_menu(cleaned, annual)
