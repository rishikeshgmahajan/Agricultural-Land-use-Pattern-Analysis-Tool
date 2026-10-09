"""
EDA and Visualization module for the Agricultural Land-Use Pattern Analysis Tool.

This module creates exploratory charts from the cleaned agricultural
land-use data.

The visualizations include:
1. Overall land-use trend
2. Comparison of land-use metrics
3. Land-use composition
4. Cropping intensity
5. State-wise comparison
6. Top and bottom states
7. State-wise trend
8. State-year heatmap
9. District-wise comparison
10. District-level distribution
11. Year-over-year change
12. Missing-values overview

The charts are displayed using Matplotlib and Seaborn and are also
saved in the eda_outputs folder.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_outputs")

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

# Conversion factor: 1 million hectares = 1,000,000 hectares.
MILLION = 1_000_000

sns.set_theme(style="whitegrid")


def _header(title):
    """Display the main heading of an EDA menu."""

    print()
    print("=" * 65)
    print(title)
    print("=" * 65)


def _subheader(title):
    """Display a smaller heading inside an EDA option."""

    print()
    print("-" * 65)
    print(title)
    print("-" * 65)


def _pick_from_list(items, title):
    """
    Display a numbered list and return the selected item.

    Returns None when the user chooses Back or enters an invalid choice.
    """

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

    if not choice.isdigit():
        print("Invalid choice.")
        return None

    number = int(choice)

    if number < 1 or number > len(items):
        print("Invalid choice.")
        return None

    return items[number - 1]


def _ask_int(prompt, default, low, high):
    """
    Ask the user for an integer within a specified range.

    Pressing Enter keeps the default value.
    """

    raw = input(f"{prompt} [{default}]: ").strip()

    if raw == "":
        return default

    if raw.isdigit():
        value = int(raw)

        if low <= value <= high:
            return value

    print(f"Invalid value. Using {default}.")
    return default


def _ask_year(years, prompt="Year"):
    """
    Ask the user to select one of the available years.

    Pressing Enter selects the latest available year.
    """

    years = sorted(int(year) for year in years)

    if not years:
        return None

    latest = years[-1]

    raw = input(
        f"{prompt} ({years[0]}-{latest}) [{latest}]: "
    ).strip()

    if raw == "":
        return latest

    if raw.isdigit() and int(raw) in years:
        return int(raw)

    print(f"Year not available. Using {latest}.")
    return latest


def _finish(fig, filename):
    """
    Save the generated chart inside the eda_outputs folder
    and close the figure.
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path = os.path.join(OUTPUT_DIR, filename)

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"\nChart saved to: {path}")

    plt.show()
    plt.close(fig)


def _slug(text):
    """
    Convert text into a simple filename-friendly format.
    """

    return "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(text)
    ).strip("_")


def get_categories(cleaned_data):
    """
    Return available land-use categories.

    Core agricultural metrics are shown first, followed by
    other available categories.
    """

    present = set(
        cleaned_data["category"]
        .dropna()
        .unique()
    )

    core = [
        metric
        for metric in CORE_METRICS
        if metric in present
    ]

    others = sorted(
        present
        - set(core)
        - {"Unspecified"}
    )

    return core + others


def get_category_data(cleaned_data, category):
    """
    Return valid records belonging to one selected category.
    """

    data = cleaned_data[
        (cleaned_data["category"] == category)
        & cleaned_data["area"].notna()
        & cleaned_data["start_year"].notna()
    ].copy()

    if data.empty:
        return data

    data["start_year"] = data["start_year"].astype(int)

    return data


def get_yearly_totals(cleaned_data, category):
    """
    Calculate yearly total area and number of reporting districts
    for a selected land-use category.
    """

    data = get_category_data(
        cleaned_data,
        category
    )

    if data.empty:
        return pd.DataFrame(
            columns=[
                "start_year",
                "area",
                "districts"
            ]
        )

    yearly = (
        data
        .groupby("start_year")
        .agg(
            area=("area", "sum"),
            districts=("district_name", "nunique")
        )
        .reset_index()
    )

    return yearly


def get_district_year_table(cleaned_data):
    """
    Convert the cleaned dataset into a district-year table.

    Each row represents one district in one year and each
    land-use category becomes a separate column.
    """

    data = cleaned_data[
        cleaned_data["area"].notna()
        & cleaned_data["start_year"].notna()
    ].copy()

    if data.empty:
        return pd.DataFrame()

    data["start_year"] = data["start_year"].astype(int)

    wide = (
        data
        .groupby(
            [
                "start_year",
                "state_name",
                "district_name",
                "category"
            ]
        )["area"]
        .sum()
        .unstack("category")
        .reset_index()
    )

    wide.columns.name = None

    return wide


# Line Chart: Year vs Selected Land-Use Category
def plot_category_trend(cleaned_data):
    """
    Shows how the area of one selected land-use category changes
    over the years.

    Why we are doing this:
    To identify whether the selected land-use category is increasing,
    decreasing, or fluctuating over time.

    What it compares/shows:
    The main chart compares yearly land-use area.
    The smaller chart below shows the number of districts reporting
    data in each year.

    How it is useful:
    It helps identify long-term land-use changes and also shows whether
    changes may be associated with differences in reporting coverage.
    """

    category = _pick_from_list(
        get_categories(cleaned_data),
        "SELECT LAND-USE CATEGORY"
    )

    if category is None:
        return

    yearly = get_yearly_totals(
        cleaned_data,
        category
    )

    if yearly.empty:
        print("No data available for this category.")
        return

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    sns.lineplot(
        data=yearly,
        x="start_year",
        y="area",
        marker="o",
        ax=ax1
    )

    ax1.set_ylabel("Area (million hectares)")
    ax1.set_xlabel("")
    ax1.set_title(
        f"{category}: Reported Total Area by Year"
    )

    ax1.set_yticklabels(
        [
            f"{value / MILLION:,.0f}"
            for value in ax1.get_yticks()
        ]
    )

    ax2.bar(
        yearly["start_year"],
        yearly["districts"]
    )

    ax2.set_ylabel("Districts\nReporting")
    ax2.set_xlabel(
        "Year (start of agricultural year)"
    )

    ax2.tick_params(
        axis="x",
        rotation=60
    )

    fig.text(
        0.01,
        0.005,
        "The lower chart shows the number of reporting districts.",
        fontsize=8,
        style="italic"
    )

    _finish(
        fig,
        f"trend_{_slug(category)}.png"
    )


# Line Chart: Year vs Core Agricultural Land-Use Metrics
def plot_core_metrics(annual_data):
    """
    Compares the major agricultural land-use metrics across the years.

    Why we are doing this:
    To understand how the main agricultural land-use measures have
    changed during the study period.

    What it compares/shows:
    It compares Reporting Area, Net Area Sown, Cropped Area,
    and Area Sown More Than Once over time.

    How it is useful:
    It helps identify similarities and differences between the major
    agricultural land-use trends.
    """

    data = annual_data[
        annual_data["land_use_metrics"].isin(
            CORE_METRICS
        )
    ].copy()

    if data.empty:
        print("No core metric data available.")
        return

    data["start_year"] = data["start_year"].astype(int)

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    sns.lineplot(
        data=data,
        x="start_year",
        y="area",
        hue="land_use_metrics",
        hue_order=CORE_METRICS,
        marker="o",
        ax=ax
    )

    ax.set_xlabel(
        "Year (start of agricultural year)"
    )

    ax.set_ylabel(
        "Area (million hectares)"
    )

    ax.set_title(
        "Core Land-Use Metrics: National Totals Over Time"
    )

    ax.legend(
        title="Metric",
        loc="best"
    )

    _finish(
        fig,
        "core_metrics_trend.png"
    )


# Stacked Area Chart: Year vs Land-Use Composition
# Bar Chart: Latest Year vs Land-Use Subcategories
def plot_land_use_composition(cleaned_data):
    """
    Shows how different land-use categories contribute to the
    overall land-use composition.

    Why we are doing this:
    To understand whether the relative composition of land use
    changes over the study period.

    What it compares/shows:
    The stacked area chart compares the percentage contribution
    of major land-use classifications across years.
    The bar chart shows the subcategory breakdown for the latest year.

    How it is useful:
    It helps identify shifts in the relative importance of different
    land-use categories instead of looking only at their total areas.
    """

    data = cleaned_data[
        cleaned_data["area_classification"].isin(
            CLASSIFICATIONS
        )
        & cleaned_data["area"].notna()
        & cleaned_data["start_year"].notna()
    ].copy()

    if data.empty:
        print("No classification data available.")
        return

    data["start_year"] = data["start_year"].astype(int)

    table = data.pivot_table(
        index="start_year",
        columns="area_classification",
        values="area",
        aggfunc="sum",
        fill_value=0
    )

    available_columns = [
        column
        for column in CLASSIFICATIONS
        if column in table.columns
    ]

    table = table[available_columns]

    share = (
        table
        .div(table.sum(axis=1), axis=0)
        * 100
    )

    latest_year = int(
        data["start_year"].max()
    )

    subcategories = (
        data[data["start_year"] == latest_year]
        .groupby("area_sub_classification")["area"]
        .sum()
        .sort_values()
    )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        gridspec_kw={"width_ratios": [3, 2]}
    )

    ax1.stackplot(
        share.index,
        share.T.values,
        labels=share.columns
    )

    ax1.set_ylim(0, 100)

    ax1.set_xlabel(
        "Year (start of agricultural year)"
    )

    ax1.set_ylabel(
        "Share of Classified Area (%)"
    )

    ax1.set_title(
        "Land-Use Composition Over Time"
    )

    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=8
    )

    ax2.barh(
        subcategories.index,
        subcategories.values / MILLION
    )

    ax2.set_xlabel(
        "Area (million hectares)"
    )

    ax2.set_title(
        f"Land-Use Subcategories in {latest_year}"
    )

    ax2.tick_params(
        axis="y",
        labelsize=8
    )

    _finish(
        fig,
        "land_use_composition.png"
    )


# Line Charts: Year vs Cropping Intensity and Net Area Sown Share
def plot_cropping_intensity(cleaned_data):
    """
    Shows changes in cropping intensity and the share of net area
    sown over the study period.

    Why we are doing this:
    To understand not only how much agricultural land exists, but
    also how intensively that land is being used.

    What it compares/shows:
    The first chart shows Cropping Intensity calculated as
    Cropped Area / Net Area Sown × 100.
    The second chart shows Net Area Sown as a percentage of
    Reporting Area.

    How it is useful:
    Together, these measures help distinguish changes in the amount
    of agricultural land from changes in how intensively the land
    is cultivated.
    """

    wide = get_district_year_table(
        cleaned_data
    )

    required = [
        "Net Area Sown",
        "Cropped Area",
        "Reporting Area"
    ]

    if any(
        column not in wide.columns
        for column in required
    ):
        print("Required metrics are not available.")
        return

    complete = wide.dropna(
        subset=required
    )

    complete = complete[
        complete["Net Area Sown"] > 0
    ]

    if complete.empty:
        print("Not enough complete records.")
        return

    yearly = (
        complete
        .groupby("start_year")[required]
        .sum()
    )

    cropping_intensity = (
        yearly["Cropped Area"]
        / yearly["Net Area Sown"]
        * 100
    )

    sown_share = (
        yearly["Net Area Sown"]
        / yearly["Reporting Area"]
        * 100
    )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 5.5)
    )

    sns.lineplot(
        x=cropping_intensity.index,
        y=cropping_intensity.values,
        marker="o",
        ax=ax1
    )

    ax1.set_title(
        "Cropping Intensity"
    )

    ax1.set_xlabel(
        "Year (start of agricultural year)"
    )

    ax1.set_ylabel(
        "Cropping Intensity (%)"
    )

    sns.lineplot(
        x=sown_share.index,
        y=sown_share.values,
        marker="o",
        ax=ax2
    )

    ax2.set_title(
        "Net Area Sown as Share of Reporting Area"
    )

    ax2.set_xlabel(
        "Year (start of agricultural year)"
    )

    ax2.set_ylabel(
        "Net Area Sown (%)"
    )

    fig.text(
        0.01,
        0.005,
        "Calculated using districts reporting all three required metrics.",
        fontsize=8,
        style="italic"
    )

    _finish(
        fig,
        "cropping_intensity.png"
    )


# Bar Chart: States vs Selected Land-Use Area for a Selected Year
def plot_state_comparison(cleaned_data):
    """
    Compares Indian states based on the area of a selected land-use
    category for one chosen year.

    Why we are doing this:
    To understand how land-use patterns differ between states.

    What it compares/shows:
    It compares the selected land-use area across states for
    the selected year.

    How it is useful:
    It helps identify states with relatively higher or lower amounts
    of the selected land-use category.
    """

    category = _pick_from_list(
        get_categories(cleaned_data),
        "SELECT LAND-USE CATEGORY"
    )

    if category is None:
        return

    data = get_category_data(
        cleaned_data,
        category
    )

    if data.empty:
        print("No data available.")
        return

    year = _ask_year(
        data["start_year"].unique()
    )

    top_n = _ask_int(
        "Number of states to show",
        15,
        3,
        36
    )

    state_values = (
        data[data["start_year"] == year]
        .groupby("state_name")["area"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    if state_values.empty:
        print("No data available for that year.")
        return

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        state_values.index[::-1],
        state_values.values[::-1] / MILLION
    )

    ax.set_xlabel(
        "Area (million hectares)"
    )

    ax.set_ylabel(
        "State"
    )

    ax.set_title(
        f"{category}: State Comparison ({year})"
    )

    _finish(
        fig,
        f"state_comparison_{_slug(category)}_{year}.png"
    )


# Bar Charts: Top and Bottom States vs Selected Land-Use Area
def plot_top_bottom_states(cleaned_data):
    """
    Identifies and compares states with the highest and lowest
    values for a selected land-use category.

    Why we are doing this:
    To identify states that stand out at the upper and lower
    ends of the land-use distribution.

    What it compares/shows:
    It compares the top states and bottom states based on the
    selected land-use area for a chosen year.

    How it is useful:
    This provides a quick state-level ranking and makes regional
    differences easier to interpret.
    """

    category = _pick_from_list(
        get_categories(cleaned_data),
        "SELECT LAND-USE CATEGORY"
    )

    if category is None:
        return

    data = get_category_data(
        cleaned_data,
        category
    )

    if data.empty:
        print("No data available.")
        return

    year = _ask_year(
        data["start_year"].unique()
    )

    top_n = _ask_int(
        "Number of top/bottom states",
        5,
        2,
        10
    )

    values = (
        data[data["start_year"] == year]
        .groupby("state_name")["area"]
        .sum()
        .sort_values(ascending=False)
    )

    if values.empty:
        print("No data available.")
        return

    top_states = values.head(top_n)
    bottom_states = values.tail(top_n).sort_values()

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    ax1.barh(
        top_states.index[::-1],
        top_states.values[::-1] / MILLION
    )

    ax1.set_title(
        f"Top {top_n} States"
    )

    ax1.set_xlabel(
        "Area (million hectares)"
    )

    ax2.barh(
        bottom_states.index,
        bottom_states.values / MILLION
    )

    ax2.set_title(
        f"Bottom {top_n} States"
    )

    ax2.set_xlabel(
        "Area (million hectares)"
    )

    fig.suptitle(
        f"{category}: Top and Bottom States ({year})"
    )

    _finish(
        fig,
        f"top_bottom_states_{_slug(category)}_{year}.png"
    )


# Line Chart: Year vs Selected Land-Use Area for Top States
def plot_state_trends(cleaned_data):
    """
    Shows how the selected land-use category changes over time
    for the leading states.

    Why we are doing this:
    A single-year state comparison cannot show whether a state has
    consistently remained high or whether its position has changed.

    What it compares/shows:
    It compares yearly land-use trends for the top states,
    ranked according to the latest available year.

    How it is useful:
    It helps determine whether leading states have followed similar
    trends or whether their land-use patterns have changed differently.
    """

    category = _pick_from_list(
        get_categories(cleaned_data),
        "SELECT LAND-USE CATEGORY"
    )

    if category is None:
        return

    data = get_category_data(
        cleaned_data,
        category
    )

    if data.empty:
        print("No data available.")
        return

    top_n = _ask_int(
        "Number of states to compare",
        6,
        2,
        10
    )

    latest_year = data["start_year"].max()

    ranking = (
        data[data["start_year"] == latest_year]
        .groupby("state_name")["area"]
        .sum()
        .sort_values(ascending=False)
    )

    selected_states = (
        ranking
        .head(top_n)
        .index
        .tolist()
    )

    trend = (
        data[
            data["state_name"].isin(
                selected_states
            )
        ]
        .groupby(
            ["start_year", "state_name"]
        )["area"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    sns.lineplot(
        data=trend,
        x="start_year",
        y="area",
        hue="state_name",
        marker="o",
        ax=ax
    )

    ax.set_xlabel(
        "Year (start of agricultural year)"
    )

    ax.set_ylabel(
        "Area (million hectares)"
    )

    ax.set_title(
        f"{category}: Land-Use Trends for Top States"
    )

    ax.legend(
        title="State",
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1, 0.5)
    )

    _finish(
        fig,
        f"state_trends_{_slug(category)}.png"
    )


# Heatmap: State vs Year for Selected Land-Use Category
def plot_state_heatmap(cleaned_data):
    """
    Displays selected land-use values for states across multiple years
    using colour intensity.

    Why we are doing this:
    To examine state-level changes across several years at the same time.

    What it compares/shows:
    States are shown on one axis and years on the other axis.
    Colour intensity represents the area of the selected land-use category.

    How it is useful:
    It helps quickly identify states with consistently high or low values
    and years where noticeable changes occurred.
    """

    category = _pick_from_list(
        get_categories(cleaned_data),
        "SELECT LAND-USE CATEGORY"
    )

    if category is None:
        return

    data = get_category_data(
        cleaned_data,
        category
    )

    if data.empty:
        print("No data available.")
        return

    top_n = _ask_int(
        "Number of states to show",
        15,
        3,
        36
    )

    table = data.pivot_table(
        index="state_name",
        columns="start_year",
        values="area",
        aggfunc="sum"
    )

    state_order = (
        table.sum(axis=1)
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    table = table.loc[state_order]

    table = table / MILLION

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    sns.heatmap(
        table,
        cmap="YlGn",
        linewidths=0.3,
        cbar_kws={
            "label": "Million hectares"
        },
        ax=ax
    )

    ax.set_xlabel(
        "Year (start of agricultural year)"
    )

    ax.set_ylabel(
        "State"
    )

    ax.set_title(
        f"{category}: State vs Year"
    )

    _finish(
        fig,
        f"state_heatmap_{_slug(category)}.png"
    )


# Bar Chart: Districts vs Selected Land-Use Area within a State
def plot_district_comparison(cleaned_data):
    """
    Compares districts within a selected state based on a selected
    land-use category.

    Why we are doing this:
    State-level analysis can hide important differences between
    individual districts.

    What it compares/shows:
    It compares the selected land-use area across districts for
    a chosen state and year.

    How it is useful:
    It helps identify districts with substantially different
    land-use patterns within the same state.
    """

    states = sorted(
        cleaned_data["state_name"]
        .dropna()
        .unique()
        .tolist()
    )

    state = _pick_from_list(
        states,
        "SELECT STATE"
    )

    if state is None:
        return

    category = _pick_from_list(
        get_categories(cleaned_data),
        "SELECT LAND-USE CATEGORY"
    )

    if category is None:
        return

    data = get_category_data(
        cleaned_data,
        category
    )

    data = data[
        data["state_name"] == state
    ]

    if data.empty:
        print(
            "No data available for this state and category."
        )
        return

    year = _ask_year(
        data["start_year"].unique()
    )

    top_n = _ask_int(
        "Number of districts to show",
        15,
        3,
        40
    )

    district_values = (
        data[data["start_year"] == year]
        .groupby("district_name")["area"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    if district_values.empty:
        print("No data available for that year.")
        return

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        district_values.index[::-1],
        district_values.values[::-1] / 1000
    )

    ax.set_xlabel(
        "Area (thousand hectares)"
    )

    ax.set_ylabel(
        "District"
    )

    ax.set_title(
        f"{state}: District Comparison for {category} ({year})"
    )

    _finish(
        fig,
        f"district_comparison_{_slug(state)}_{_slug(category)}_{year}.png"
    )


# Box Plot and Histogram: Distribution of District Land-Use Areas
def plot_district_distribution(cleaned_data):
    """
    Examines how the selected land-use area is distributed across
    districts.

    Why we are doing this:
    Total or average values can hide the variation between individual
    districts.

    What it compares/shows:
    The box plot shows the spread of district values across selected
    years. The histogram shows the distribution of district values
    in the latest available year.

    How it is useful:
    It helps identify variation, unusually high or low district values,
    and whether most districts have similar or very different
    land-use areas.
    """

    category = _pick_from_list(
        get_categories(cleaned_data),
        "SELECT LAND-USE CATEGORY"
    )

    if category is None:
        return

    data = get_category_data(
        cleaned_data,
        category
    )

    data = data[
        data["area"] > 0
    ]

    if data.empty:
        print(
            "No positive values available."
        )
        return

    years = sorted(
        data["start_year"].unique()
    )

    selected_years = years[::5]

    if years[-1] not in selected_years:
        selected_years.append(
            years[-1]
        )

    subset = data[
        data["start_year"].isin(
            selected_years
        )
    ]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    sns.boxplot(
        data=subset,
        x="start_year",
        y="area",
        ax=ax1
    )

    ax1.set_yscale("log")

    ax1.set_xlabel(
        "Year (start of agricultural year)"
    )

    ax1.set_ylabel(
        "District Area (hectares, log scale)"
    )

    ax1.set_title(
        f"{category}: District-Level Spread"
    )

    latest_year = selected_years[-1]

    latest_data = subset[
        subset["start_year"] == latest_year
    ]

    sns.histplot(
        latest_data["area"],
        bins=35,
        kde=True,
        ax=ax2
    )

    ax2.set_xlabel(
        "District Area (hectares)"
    )

    ax2.set_ylabel(
        "Number of Districts"
    )

    ax2.set_title(
        f"District Distribution in {latest_year}"
    )

    _finish(
        fig,
        f"district_distribution_{_slug(category)}.png"
    )


# Bar Chart: Year vs Year-over-Year Percentage Change
def plot_year_over_year_change(annual_data):
    """
    Shows the percentage change in a selected land-use metric from
    one year to the next.

    Why we are doing this:
    Long-term trend charts show the overall direction but may not
    clearly show sudden increases or decreases between consecutive years.

    What it compares/shows:
    It compares each year's value with the previous year's value
    and displays the percentage change.

    How it is useful:
    It helps identify years with unusually large increases or decreases
    in the selected land-use metric.
    """

    metrics = sorted(
        annual_data["land_use_metrics"]
        .dropna()
        .unique()
        .tolist()
    )

    metric = _pick_from_list(
        metrics,
        "SELECT LAND-USE METRIC"
    )

    if metric is None:
        return

    data = annual_data[
        annual_data["land_use_metrics"] == metric
    ].copy()

    data["start_year"] = data["start_year"].astype(int)

    data = data.sort_values(
        "start_year"
    )

    data["percentage_change"] = (
        data["area"]
        .pct_change()
        * 100
    )

    data = data.dropna(
        subset=["percentage_change"]
    )

    if data.empty:
        print(
            "Not enough data to calculate year-over-year change."
        )
        return

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        data["start_year"],
        data["percentage_change"]
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_xlabel(
        "Year (start of agricultural year)"
    )

    ax.set_ylabel(
        "Year-over-Year Change (%)"
    )

    ax.set_title(
        f"{metric}: Year-over-Year Percentage Change"
    )

    ax.tick_params(
        axis="x",
        rotation=60
    )

    _finish(
        fig,
        f"year_over_year_{_slug(metric)}.png"
    )


# Bar Chart: Year/Category vs Percentage of Missing Area Values
def plot_missing_values(cleaned_data):
    """
    Shows the percentage of missing area values across categories
    and years.

    Why we are doing this:
    To understand the completeness of the dataset before interpreting
    the main EDA results.

    What it compares/shows:
    The first chart compares missing area percentages across
    land-use categories. The second chart compares missing values
    across years.

    How it is useful:
    It helps identify categories or periods where missing data may
    affect the interpretation of the analysis.
    """

    data = cleaned_data[
        cleaned_data["start_year"].notna()
    ].copy()

    if data.empty:
        print("No data available.")
        return

    data["start_year"] = data[
        "start_year"
    ].astype(int)

    data["missing"] = data[
        "area"
    ].isna()

    by_category = (
        data
        .groupby("category")["missing"]
        .mean()
        .mul(100)
        .sort_values()
    )

    by_year = (
        data
        .groupby("start_year")["missing"]
        .mean()
        .mul(100)
    )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    ax1.barh(
        by_category.index,
        by_category.values
    )

    ax1.set_xlabel(
        "Missing Area Values (%)"
    )

    ax1.set_ylabel(
        "Land-Use Category"
    )

    ax1.set_title(
        "Missing Values by Land-Use Category"
    )

    ax2.bar(
        by_year.index,
        by_year.values
    )

    ax2.set_xlabel(
        "Year (start of agricultural year)"
    )

    ax2.set_ylabel(
        "Missing Area Values (%)"
    )

    ax2.set_title(
        "Missing Values by Year"
    )

    ax2.tick_params(
        axis="x",
        rotation=60
    )

    _finish(
        fig,
        "missing_values.png"
    )


def eda_visualization_menu(cleaned_data, annual_data):
    """
    Display the main EDA and visualization menu.

    The menu allows the user to select different exploratory
    visualizations without changing the main terminal application.
    """

    while True:

        _header(
            "EDA & VISUALIZATION"
        )

        print(
            "1. Overall Land-Use Trend"
        )

        print(
            "2. Compare Core Land-Use Metrics"
        )

        print(
            "3. Land-Use Composition"
        )

        print(
            "4. Cropping Intensity"
        )

        print(
            "5. State-wise Comparison"
        )

        print(
            "6. Top/Bottom States"
        )

        print(
            "7. State-wise Trend"
        )

        print(
            "8. State-Year Heatmap"
        )

        print(
            "9. District-wise Comparison"
        )

        print(
            "10. District Distribution"
        )

        print(
            "11. Year-over-Year Change"
        )

        print(
            "12. Missing Values Overview"
        )

        print(
            "0. Back"
        )

        choice = input(
            "\nEnter choice: "
        ).strip()

        if choice == "0":
            return

        try:

            if choice == "1":
                plot_category_trend(
                    cleaned_data
                )

            elif choice == "2":
                plot_core_metrics(
                    annual_data
                )

            elif choice == "3":
                plot_land_use_composition(
                    cleaned_data
                )

            elif choice == "4":
                plot_cropping_intensity(
                    cleaned_data
                )

            elif choice == "5":
                plot_state_comparison(
                    cleaned_data
                )

            elif choice == "6":
                plot_top_bottom_states(
                    cleaned_data
                )

            elif choice == "7":
                plot_state_trends(
                    cleaned_data
                )

            elif choice == "8":
                plot_state_heatmap(
                    cleaned_data
                )

            elif choice == "9":
                plot_district_comparison(
                    cleaned_data
                )

            elif choice == "10":
                plot_district_distribution(
                    cleaned_data
                )

            elif choice == "11":
                plot_year_over_year_change(
                    annual_data
                )

            elif choice == "12":
                plot_missing_values(
                    cleaned_data
                )

            else:
                print(
                    "Invalid choice. Please select a valid option."
                )

        except Exception as error:

            plt.close("all")

            print(
                f"\nCould not create the chart: {error}"
            )


if __name__ == "__main__":

    from data_preprocessor import (
        create_member2_dataset,
        create_member3_dataset
    )

    print(
        "Loading project data..."
    )

    cleaned_data = create_member2_dataset()

    annual_data = create_member3_dataset()

    eda_visualization_menu(
        cleaned_data,
        annual_data
    )