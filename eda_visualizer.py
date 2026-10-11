"""Exploratory charts for the Agricultural Land-Use Pattern Analysis Tool."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MILLION = 1_000_000

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
    "–": "-",
    "÷": "/",
})


def _print_line(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.translate(ASCII_UI))


def _header(title):
    width = 66
    _print_line("")
    _print_line("╭" + "─" * (width - 2) + "╮")
    _print_line("│ " + title.center(width - 4) + " │")
    _print_line("╰" + "─" * (width - 2) + "╯")


def _show(fig):
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    print("Close the chart window to return to the menu.")
    try:
        plt.show(block=True)
    finally:
        plt.close(fig)


def _category_rows(cleaned_data, category):
    rows = cleaned_data[
        (cleaned_data["category"] == category)
        & cleaned_data["area"].notna()
        & cleaned_data["start_year"].notna()
    ].copy()
    rows["start_year"] = rows["start_year"].astype(int)
    return rows


def _district_year_table(cleaned_data):
    rows = cleaned_data[
        cleaned_data["area"].notna() & cleaned_data["start_year"].notna()
    ].copy()
    rows["start_year"] = rows["start_year"].astype(int)
    table = (
        rows.groupby(
            ["start_year", "state_name", "district_name", "category"]
        )["area"]
        .sum()
        .unstack("category")
        .reset_index()
    )
    table.columns.name = None
    return table


def _label_last_value(axis, years, values, suffix=""):
    if len(years) == 0:
        return
    axis.annotate(
        f"{values[-1]:,.1f}{suffix}",
        (years[-1], values[-1]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )


def _label_bars(axis, bars, values, suffix=""):
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value:,.1f}{suffix}",
            va="center",
            fontsize=8,
        )
    if len(values):
        maximum = max(values)
        axis.set_xlim(right=maximum * 1.25 if maximum > 0 else 1)


def plot_national_trends(annual_data):
    core = annual_data[annual_data["land_use_metrics"].isin(CORE_METRICS)].copy()
    if core.empty:
        print("There is not enough data to draw the national trends.")
        return

    core["start_year"] = core["start_year"].astype(int)
    core["area_million"] = core["area"] / MILLION
    latest_year = int(core["start_year"].max())

    fig, (reporting_axis, crop_axis) = plt.subplots(1, 2, figsize=(14, 6))
    reporting = core[core["land_use_metrics"] == "Reporting Area"]
    sns.lineplot(
        data=reporting,
        x="start_year",
        y="area_million",
        marker="o",
        color="#1b7837",
        ax=reporting_axis,
    )
    reporting_axis.set_title("Total reporting area")
    reporting_axis.set_xlabel("Year (start of agricultural year)")
    reporting_axis.set_ylabel("Area (million hectares)")
    if not reporting.empty:
        reporting_axis.annotate(
            f"{reporting['area_million'].iloc[-1]:,.1f} million ha",
            (reporting["start_year"].iloc[-1], reporting["area_million"].iloc[-1]),
            xytext=(-95, 15),
            textcoords="offset points",
            fontsize=8,
        )

    crop_metrics = ["Net Area Sown", "Cropped Area", "Area Sown More Than Once"]
    crop_data = core[core["land_use_metrics"].isin(crop_metrics)]
    for metric in crop_metrics:
        metric_data = crop_data[crop_data["land_use_metrics"] == metric]
        if not metric_data.empty:
            sns.lineplot(
                data=metric_data,
                x="start_year",
                y="area_million",
                marker="o",
                label=metric,
                ax=crop_axis,
            )
    crop_axis.set_title("Sown and cropped area")
    crop_axis.set_xlabel("Year (start of agricultural year)")
    crop_axis.set_ylabel("Area (million hectares)")
    crop_axis.legend(title="Measure", fontsize=8)
    fig.suptitle(
        f"National land-use trends through {latest_year}",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.005,
        "The crop measures are shown separately from reporting area so their trends remain readable.",
        fontsize=8,
    )
    _show(fig)

def plot_land_use_patterns(cleaned_data):
    category_rows = cleaned_data[
        cleaned_data["area_classification"].isin(CLASSIFICATIONS)
        & cleaned_data["area"].notna()
        & cleaned_data["start_year"].notna()
    ].copy()
    if category_rows.empty:
        print("There is not enough classification data for this view.")
        return

    category_rows["start_year"] = category_rows["start_year"].astype(int)
    yearly = category_rows.pivot_table(
        index="start_year",
        columns="area_classification",
        values="area",
        aggfunc="sum",
        fill_value=0,
    )
    yearly = yearly[[name for name in CLASSIFICATIONS if name in yearly.columns]]
    yearly = yearly.loc[yearly.sum(axis=1) > 0]
    shares = yearly.div(yearly.sum(axis=1), axis=0) * 100

    latest_year = int(category_rows["start_year"].max())
    latest = category_rows[category_rows["start_year"] == latest_year]
    breakdown = (
        latest.groupby("area_classification")["area"]
        .sum()
        .sort_values(ascending=True)
    )

    table = _district_year_table(cleaned_data)
    needed = ["Net Area Sown", "Cropped Area", "Reporting Area"]
    if all(column in table.columns for column in needed):
        complete = table.dropna(subset=needed)
        complete = complete[complete["Net Area Sown"] > 0]
        yearly_metrics = complete.groupby("start_year")[needed].sum()
        if not yearly_metrics.empty:
            yearly_metrics["intensity"] = (
                yearly_metrics["Cropped Area"] / yearly_metrics["Net Area Sown"] * 100
            )
            yearly_metrics["sown_share"] = (
                yearly_metrics["Net Area Sown"]
                / yearly_metrics["Reporting Area"]
                * 100
            )
    else:
        yearly_metrics = pd.DataFrame()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    composition_axis, breakdown_axis, intensity_axis, sown_axis = axes.flatten()

    axes_colors = sns.color_palette("deep", len(shares.columns))
    composition_axis.stackplot(
        shares.index,
        shares.T.values,
        labels=shares.columns,
        colors=axes_colors,
    )
    composition_axis.set_ylim(0, 100)
    composition_axis.set_title("Share of classified land over time")
    composition_axis.set_xlabel("Year (start of agricultural year)")
    composition_axis.set_ylabel("Share of classified area (%)")
    composition_axis.legend(fontsize=7, loc="lower left")

    breakdown_values = breakdown.values / MILLION
    bars = breakdown_axis.barh(breakdown.index, breakdown_values, color="#66a061")
    breakdown_axis.set_title(f"Classified area by type in {latest_year}")
    breakdown_axis.set_xlabel("Area (million hectares)")
    breakdown_axis.set_ylabel("Classification")
    _label_bars(breakdown_axis, bars, breakdown_values)

    if yearly_metrics.empty:
        intensity_axis.text(
            0.5, 0.5, "Not enough complete records", ha="center", va="center"
        )
        sown_axis.text(
            0.5, 0.5, "Not enough complete records", ha="center", va="center"
        )
    else:
        intensity_axis.plot(
            yearly_metrics.index,
            yearly_metrics["intensity"],
            marker="o",
            color="#d95f02",
        )
        intensity_axis.set_title("Cropping intensity")
        intensity_axis.set_xlabel("Year (start of agricultural year)")
        intensity_axis.set_ylabel("Cropped area ÷ net sown area (%)")
        _label_last_value(
            intensity_axis,
            yearly_metrics.index.tolist(),
            yearly_metrics["intensity"].tolist(),
            "%",
        )

        sown_axis.plot(
            yearly_metrics.index,
            yearly_metrics["sown_share"],
            marker="o",
            color="#1b9e77",
        )
        sown_axis.set_title("Net sown area as a share of reporting area")
        sown_axis.set_xlabel("Year (start of agricultural year)")
        sown_axis.set_ylabel("Share (%)")
        _label_last_value(
            sown_axis,
            yearly_metrics.index.tolist(),
            yearly_metrics["sown_share"].tolist(),
            "%",
        )

    fig.suptitle("Land-use composition and cropping patterns", fontsize=15, fontweight="bold")
    fig.text(
        0.01,
        0.005,
        "Cropping measures include districts with all three reported metrics in a year.",
        fontsize=8,
    )
    _show(fig)

def plot_state_patterns(cleaned_data):
    rows = _category_rows(cleaned_data, "Reporting Area")
    if rows.empty:
        print("There is no state-level reporting-area data.")
        return

    year = int(rows["start_year"].max())
    ranking = (
        rows[rows["start_year"] == year]
        .groupby("state_name")["area"]
        .sum()
        .sort_values(ascending=False)
    )
    chosen_states = ranking.head(10).index.tolist()
    if not chosen_states:
        print(f"There is no state data for {year}.")
        return

    trend = rows[rows["state_name"].isin(chosen_states)]
    trend = (
        trend.groupby(["start_year", "state_name"])["area"]
        .sum()
        .reset_index()
    )
    trend["area_million"] = trend["area"] / MILLION
    heatmap_data = trend.pivot(
        index="state_name", columns="start_year", values="area"
    ).reindex(chosen_states)
    heatmap_data = heatmap_data / MILLION

    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(chosen_states) * 0.42)))
    bar_axis, trend_axis, heatmap_axis = axes
    bar_values = ranking.head(10).values / MILLION
    bars = bar_axis.barh(
        chosen_states[::-1],
        bar_values[::-1],
        color=sns.color_palette("viridis", len(chosen_states)),
    )
    bar_axis.set_title(f"Top states in {year}")
    bar_axis.set_xlabel("Area (million hectares)")
    bar_axis.set_ylabel("State")
    _label_bars(bar_axis, bars, bar_values[::-1])

    trend_states = chosen_states[:5]
    trend = trend[trend["state_name"].isin(trend_states)]
    sns.lineplot(
        data=trend,
        x="start_year",
        y="area_million",
        hue="state_name",
        hue_order=trend_states,
        marker="o",
        ax=trend_axis,
    )
    trend_axis.set_title("Trends for the five largest states")
    trend_axis.set_xlabel("Year (start of agricultural year)")
    trend_axis.set_ylabel("Area (million hectares)")
    trend_axis.legend(title="State", fontsize=7, loc="best")

    sns.heatmap(
        heatmap_data.loc[chosen_states],
        cmap="YlGn",
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Million hectares"},
        ax=heatmap_axis,
    )
    heatmap_axis.set_title("State area by year")
    heatmap_axis.set_xlabel("Year (start of agricultural year)")
    heatmap_axis.set_ylabel("State")
    heatmap_axis.tick_params(axis="x", rotation=60)

    fig.suptitle("State reporting-area comparison", fontsize=15, fontweight="bold")
    fig.text(
        0.01,
        0.005,
        f"Showing up to 10 states ranked in {year}; blank heatmap cells mean no data.",
        fontsize=8,
    )
    _show(fig)

def plot_district_patterns(cleaned_data):
    rows = _category_rows(cleaned_data, "Reporting Area")
    if rows.empty:
        print("There is no district-level reporting-area data.")
        return

    year = int(rows["start_year"].max())
    ranking = (
        rows[rows["start_year"] == year]
        .groupby(["state_name", "district_name"])["area"]
        .sum()
        .sort_values(ascending=False)
    )
    chosen_districts = ranking.head(10).index.tolist()
    if not chosen_districts:
        print(f"There is no district data for {year}.")
        return

    rows["place"] = rows["district_name"] + ", " + rows["state_name"]
    chosen_places = [f"{district}, {state}" for state, district in chosen_districts]
    trend = rows[rows["place"].isin(chosen_places)]
    trend = trend[trend["place"].isin(chosen_places[:5])]
    trend = (
        trend.groupby(["start_year", "place"])["area"]
        .sum()
        .reset_index()
    )
    trend["area_thousand"] = trend["area"] / 1000

    fig, (bar_axis, trend_axis) = plt.subplots(1, 2, figsize=(15, 7))
    values = ranking.head(10).values / 1000
    labels = chosen_places
    bars = bar_axis.barh(
        labels[::-1],
        values[::-1],
        color=sns.color_palette("mako", len(labels)),
    )
    bar_axis.set_title(f"Top districts across India in {year}")
    bar_axis.set_xlabel("Area (thousand hectares)")
    bar_axis.set_ylabel("District")
    _label_bars(bar_axis, bars, values[::-1])

    sns.lineplot(
        data=trend,
        x="start_year",
        y="area_thousand",
        hue="place",
        hue_order=chosen_places[:5],
        marker="o",
        ax=trend_axis,
    )
    trend_axis.set_title("Trends for the five largest districts")
    trend_axis.set_xlabel("Year (start of agricultural year)")
    trend_axis.set_ylabel("Area (thousand hectares)")
    trend_axis.legend(title="District", fontsize=7, loc="best")

    fig.suptitle(
        "District reporting-area comparison",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.005,
        f"Districts are ranked by reporting area in {year}; labels include the state.",
        fontsize=8,
    )
    _show(fig)

def plot_data_quality(cleaned_data):
    rows = cleaned_data[cleaned_data["start_year"].notna()].copy()
    rows["start_year"] = rows["start_year"].astype(int)
    rows["missing_area"] = rows["area"].isna()
    missing_by_category = (
        rows.groupby("category")["missing_area"].mean().mul(100).sort_values()
    )
    missing_by_year = rows.groupby("start_year")["missing_area"].mean().mul(100)

    usable_rows = cleaned_data[
        cleaned_data["area"].notna()
        & cleaned_data["start_year"].notna()
        & (cleaned_data["area"] > 0)
    ].copy()
    if usable_rows.empty:
        print("There are no positive area values to summarize.")
        return
    usable_rows["start_year"] = usable_rows["start_year"].astype(int)
    reporting_rows = usable_rows[
        usable_rows["land_use_metrics"] == "Reporting Area"
    ]
    if reporting_rows.empty:
        print("There are no positive reporting-area values to summarize.")
        return
    latest_year = int(reporting_rows["start_year"].max())
    latest_values = reporting_rows[reporting_rows["start_year"] == latest_year]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    distribution_axis, missing_category_axis = axes[0]
    missing_year_axis, summary_axis = axes[1]

    sns.histplot(
        latest_values["area"],
        bins=30,
        color="#1f78b4",
        ax=distribution_axis,
    )
    distribution_axis.set_xscale("log")
    distribution_axis.set_title(f"Reporting-area values in {latest_year}")
    distribution_axis.set_xlabel("Area (hectares, logarithmic scale)")
    distribution_axis.set_ylabel("Number of records")
    distribution_axis.text(
        0.98,
        0.95,
        f"Records: {len(latest_values):,}\n"
        f"Median: {latest_values['area'].median():,.0f} ha",
        transform=distribution_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )

    category_names = missing_by_category.index.tolist()
    display_names = [
        "Misc. tree crops and groves" if name.startswith("Land under Misc.")
        else "Other uncultivated land" if name.startswith("Other Uncultivated")
        else name
        for name in category_names
    ]
    missing_category_values = missing_by_category.values
    category_bars = missing_category_axis.barh(
        display_names,
        missing_category_values,
        color="#fc8d62",
    )
    missing_category_axis.set_title("Missing area records by category")
    missing_category_axis.set_xlabel("Records with missing area (%)")
    missing_category_axis.set_ylabel("Land-use category")
    _label_bars(
        missing_category_axis,
        category_bars,
        missing_category_values,
        "%",
    )
    missing_category_axis.tick_params(axis="y", labelsize=8)

    missing_year_axis.plot(
        missing_by_year.index,
        missing_by_year.values,
        marker="o",
        color="#756bb1",
    )
    missing_year_axis.set_title("Missing area records by year")
    missing_year_axis.set_xlabel("Year (start of agricultural year)")
    missing_year_axis.set_ylabel("Records with missing area (%)")
    _label_last_value(
        missing_year_axis,
        missing_by_year.index.tolist(),
        missing_by_year.values.tolist(),
        "%",
    )

    missing_year_axis.tick_params(axis="x", rotation=45)
    summary_axis.set_facecolor("#f2f6f2")
    summary_axis.set_title("Data at a glance")
    summary_axis.set_xticks([])
    summary_axis.set_yticks([])
    summary_axis.text(
        0.08,
        0.82,
        "RECORDS REVIEWED",
        transform=summary_axis.transAxes,
        fontsize=9,
        color="#526357",
        weight="bold",
    )
    summary_axis.text(
        0.08,
        0.70,
        f"{len(rows):,}",
        transform=summary_axis.transAxes,
        fontsize=20,
        color="#214e34",
    )
    summary_axis.text(
        0.08,
        0.52,
        f"Area reported: {rows['area'].notna().sum():,}\n"
        f"Area missing: {int(rows['missing_area'].sum()):,} "
        f"({rows['missing_area'].mean() * 100:.1f}%)\n"
        f"Latest reporting year: {latest_year}\n"
        f"Median reporting area: {latest_values['area'].median():,.0f} ha",
        transform=summary_axis.transAxes,
        fontsize=11,
        linespacing=1.8,
        va="top",
    )
    fig.suptitle(
        "Data quality and reporting-area distribution",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.005,
        "The distribution uses a logarithmic area axis because reported values vary widely.",
        fontsize=8,
    )
    _show(fig)

def eda_visualization_menu(cleaned_data, annual_data):
    while True:
        _header("EDA & VISUALIZATION")
        _print_line("  1  ▸ National trends")
        _print_line("  2  ▸ Land-use and cropping patterns")
        _print_line("  3  ▸ State comparisons")
        _print_line("  4  ▸ District comparisons")
        _print_line("  5  ▸ Data quality overview")
        _print_line("  0  ↩ Back")

        choice = input(
            "\nMain Menu > EDA & Visualization > Enter choice: "
        ).strip()
        if choice == "0":
            return
        if choice == "1":
            plot_national_trends(annual_data)
        elif choice == "2":
            plot_land_use_patterns(cleaned_data)
        elif choice == "3":
            plot_state_patterns(cleaned_data)
        elif choice == "4":
            plot_district_patterns(cleaned_data)
        elif choice == "5":
            plot_data_quality(cleaned_data)
        else:
            print("Please select one of the listed options.")


if __name__ == "__main__":
    from data_preprocessor import create_member2_dataset, create_member3_dataset

    print("Loading project data...")
    cleaned_data = create_member2_dataset()
    annual_data = create_member3_dataset()
    eda_visualization_menu(cleaned_data, annual_data)
