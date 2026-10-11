"""Exploratory charts for the Agricultural Land-Use Pattern Analysis Tool."""

import matplotlib.pyplot as plt
import os
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

THOUSAND = 1_000
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


OUTPUT_DIR = "eda_outputs"

def _show(fig, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    file_path = os.path.join(OUTPUT_DIR, filename + ".png")
    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    print(f"Chart saved: {file_path}")

    print("Close the chart window to return to the menu.")
    try:
        plt.show(block=True)
    finally:
        plt.close(fig)


def _interpret_change(label, years, values, unit=""):
    series = pd.Series(values, index=years).dropna().sort_index()
    if len(series) < 2:
        print(f"Interpretation — {label}: not enough observations to compare years.")
        return
    first_year, last_year = int(series.index[0]), int(series.index[-1])
    first_value, last_value = float(series.iloc[0]), float(series.iloc[-1])
    change = last_value - first_value
    direction = "increased" if change > 0 else "decreased" if change < 0 else "was unchanged"
    percent = f" ({change / abs(first_value) * 100:+.1f}%)" if first_value != 0 else " (percentage change unavailable: starting value is zero)"
    print(f"Interpretation — {label}: {direction} from {first_value:,.2f}{unit} in {first_year} to {last_value:,.2f}{unit} in {last_year}{percent}.")


def _interpret_rank(label, ranking, unit=""):
    ranking = ranking.dropna().sort_values(ascending=False)
    if ranking.empty:
        print(f"Interpretation — {label}: no values are available.")
        return
    print(f"Interpretation — {label}: {ranking.index[0]} has the largest recorded value ({ranking.iloc[0]:,.2f}{unit}).")
    if len(ranking) > 1:
        print(f"  {ranking.index[-1]} has the smallest value among the displayed groups ({ranking.iloc[-1]:,.2f}{unit}).")


def _category_rows(cleaned_data, category):
    rows = cleaned_data[(cleaned_data["category"] == category) & cleaned_data["area"].notna() & cleaned_data["start_year"].notna()].copy()
    rows["start_year"] = rows["start_year"].astype(int)
    return rows


def _district_rows(cleaned_data):
    rows = cleaned_data[cleaned_data["district_name"].notna()].copy()
    names = rows["district_name"].astype(str).str.strip().str.lower()
    return rows[names.ne("") & names.ne("unknown")]


def _label_last_value(axis, years, values, suffix=""):
    if len(years):
        axis.annotate(f"{values[-1]:,.1f}{suffix}", (years[-1], values[-1]), xytext=(5, 5), textcoords="offset points", fontsize=8)


def _label_bars(axis, bars, values, suffix=""):
    for bar, value in zip(bars, values):
        axis.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value:,.1f}{suffix}", va="center", fontsize=8)
    if len(values):
        axis.set_xlim(right=max(values) * 1.25 if max(values) > 0 else 1)


def plot_national_trends(annual_data):
    core = annual_data[annual_data["land_use_metrics"].isin(CORE_METRICS)].copy()
    if core.empty:
        print("There is not enough data to draw the national trends.")
        return
    core["start_year"] = core["start_year"].astype(int)
    core["area_million"] = core["area"] / MILLION
    latest_year = int(core["start_year"].max())
    fig, (reporting_axis, crop_axis) = plt.subplots(1, 2, figsize=(14, 6))
    reporting = core[core["land_use_metrics"] == "Reporting Area"].sort_values("start_year")
    sns.lineplot(data=reporting, x="start_year", y="area_million", marker="o", color="#1b7837", ax=reporting_axis)
    reporting_axis.set(title="Total reporting area", xlabel="Year (start of agricultural year)", ylabel="Area (million hectares)")
    if not reporting.empty:
        _interpret_change("Reporting Area", reporting["start_year"].tolist(), reporting["area_million"].tolist(), " million ha")
    crop_metrics = ["Net Area Sown", "Cropped Area", "Area Sown More Than Once"]
    for metric in crop_metrics:
        subset = core[core["land_use_metrics"] == metric].sort_values("start_year")
        if not subset.empty:
            sns.lineplot(data=subset, x="start_year", y="area_million", marker="o", label=metric, ax=crop_axis)
            _interpret_change(metric, subset["start_year"].tolist(), subset["area_million"].tolist(), " million ha")
    crop_axis.set(title="Sown and cropped area", xlabel="Year (start of agricultural year)", ylabel="Area (million hectares)")
    crop_axis.legend(title="Measure", fontsize=8)
    fig.suptitle(f"National land-use trends through {latest_year}", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.005, "Interpretations compare first and last available observations; trends do not establish causes.", fontsize=8)
    _show(fig, "national_trends")


def plot_land_use_patterns(cleaned_data):
    rows = cleaned_data[cleaned_data["area_classification"].isin(CLASSIFICATIONS) & cleaned_data["area"].notna() & cleaned_data["start_year"].notna()].copy()
    if rows.empty:
        print("There is not enough classification data for this view.")
        return
    rows["start_year"] = rows["start_year"].astype(int)
    yearly = rows.pivot_table(index="start_year", columns="area_classification", values="area", aggfunc="sum", fill_value=0)
    yearly = yearly[[name for name in CLASSIFICATIONS if name in yearly.columns]]
    yearly = yearly.loc[yearly.sum(axis=1) > 0]
    shares = yearly.div(yearly.sum(axis=1), axis=0) * 100
    latest_year = int(rows["start_year"].max())
    latest = rows[rows["start_year"] == latest_year]
    breakdown = latest.groupby("area_classification")["area"].sum().sort_values()

    district = _district_rows(cleaned_data)
    district = district[district["area"].notna() & district["start_year"].notna()].copy()
    district["start_year"] = district["start_year"].astype(int)
    district_table = district.groupby(["start_year", "state_name", "district_name", "category"])["area"].sum().unstack("category").reset_index()
    needed = ["Net Area Sown", "Cropped Area", "Reporting Area"]
    if all(name in district_table.columns for name in needed):
        complete = district_table.dropna(subset=needed)
        complete = complete[complete["Net Area Sown"] > 0]
        yearly_metrics = complete.groupby("start_year")[needed].sum()
        yearly_metrics["intensity"] = yearly_metrics["Cropped Area"] / yearly_metrics["Net Area Sown"] * 100
        yearly_metrics["sown_share"] = yearly_metrics["Net Area Sown"] / yearly_metrics["Reporting Area"] * 100
    else:
        yearly_metrics = pd.DataFrame()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    comp, breakdown_axis, intensity_axis, sown_axis = axes.flatten()
    comp.stackplot(shares.index, shares.T.values, labels=shares.columns, colors=sns.color_palette("deep", len(shares.columns)))
    comp.set_ylim(0, 100)
    comp.set(title="Share of selected classifications over time", xlabel="Year", ylabel="Share of selected classifications (%)")
    comp.legend(fontsize=7, loc="lower left")
    vals = breakdown.values / MILLION
    bars = breakdown_axis.barh(breakdown.index, vals, color="#66a061")
    breakdown_axis.set(title=f"Selected classified area in {latest_year}", xlabel="Area (million hectares)", ylabel="Classification")
    _label_bars(breakdown_axis, bars, vals)
    _interpret_rank(f"selected classifications in {latest_year}", breakdown / MILLION, " million ha")
    if yearly_metrics.empty:
        intensity_axis.text(.5, .5, "Not enough complete records", ha="center", va="center")
        sown_axis.text(.5, .5, "Not enough complete records", ha="center", va="center")
    else:
        yearly_metrics = yearly_metrics.sort_index()
        intensity_axis.plot(yearly_metrics.index, yearly_metrics["intensity"], marker="o", color="#d95f02")
        intensity_axis.set(title="Cropping intensity", xlabel="Year", ylabel="Cropped area ÷ net sown area (%)")
        _label_last_value(intensity_axis, yearly_metrics.index.tolist(), yearly_metrics["intensity"].tolist(), "%")
        _interpret_change("Cropping intensity", yearly_metrics.index.tolist(), yearly_metrics["intensity"].tolist(), "%")
        sown_axis.plot(yearly_metrics.index, yearly_metrics["sown_share"], marker="o", color="#1b9e77")
        sown_axis.set(title="Net sown area as share of reporting area", xlabel="Year", ylabel="Share (%)")
        _label_last_value(sown_axis, yearly_metrics.index.tolist(), yearly_metrics["sown_share"].tolist(), "%")
        _interpret_change("Net sown area share", yearly_metrics.index.tolist(), yearly_metrics["sown_share"].tolist(), "%")
    fig.suptitle("Land-use composition and cropping patterns", fontsize=15, fontweight="bold")
    fig.text(0.01, 0.005, "Composition uses only the four listed classifications. Cropping metrics use districts with all three metrics.", fontsize=8)
    _show(fig, "land_use_patterns")


def plot_state_patterns(cleaned_data):
    rows = _category_rows(cleaned_data, "Reporting Area")
    if rows.empty:
        print("There is no reporting-area data.")
        return
    year = int(rows["start_year"].max())
    ranking = rows[rows["start_year"] == year].groupby("state_name")["area"].sum().sort_values(ascending=False)
    chosen = ranking.head(10).index.tolist()
    trend = rows[rows["state_name"].isin(chosen)].groupby(["start_year", "state_name"])["area"].sum().reset_index()
    trend["area_million"] = trend["area"] / MILLION
    heat = trend.pivot(index="state_name", columns="start_year", values="area").reindex(chosen) / MILLION
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(chosen) * .42)))
    bars = axes[0].barh(chosen[::-1], ranking.head(10).values[::-1] / MILLION, color=sns.color_palette("viridis", len(chosen)))
    axes[0].set(title=f"Top states/UTs in {year}", xlabel="Area (million hectares)", ylabel="State/UT")
    _label_bars(axes[0], bars, ranking.head(10).values[::-1] / MILLION)
    _interpret_rank(f"reporting area by state/UT in {year}", ranking / MILLION, " million ha")
    top_five = chosen[:5]
    trend5 = trend[trend["state_name"].isin(top_five)]
    sns.lineplot(data=trend5, x="start_year", y="area_million", hue="state_name", hue_order=top_five, marker="o", ax=axes[1])
    axes[1].set(title="Trends for five largest states/UTs", xlabel="Year", ylabel="Area (million hectares)")
    axes[1].legend(title="State/UT", fontsize=7)
    for state in top_five:
        subset = trend5[trend5["state_name"] == state].sort_values("start_year")
        _interpret_change(f"{state} reporting area", subset["start_year"].tolist(), subset["area_million"].tolist(), " million ha")
    sns.heatmap(heat, cmap="YlGn", linewidths=.25, linecolor="white", cbar_kws={"label": "Million hectares"}, ax=axes[2])
    axes[2].set(title="State/UT area by year", xlabel="Year", ylabel="State/UT")
    axes[2].tick_params(axis="x", rotation=60)
    fig.suptitle("State reporting-area comparison", fontsize=15, fontweight="bold")
    fig.text(.01, .005, f"States/UTs are ranked in {year}; blank heatmap cells mean no data.", fontsize=8)
    _show(fig, "state_comparisons")


def plot_district_patterns(cleaned_data):
    rows = _category_rows(cleaned_data, "Reporting Area")
    rows = rows[rows["district_name"].notna() & rows["district_name"].astype(str).str.strip().ne("") & rows["district_name"].astype(str).str.lower().ne("unknown")].copy()
    if rows.empty:
        print("There is no district-level reporting-area data.")
        return
    year = int(rows["start_year"].max())
    ranking = rows[rows["start_year"] == year].groupby(["state_name", "district_name"])["area"].sum().sort_values(ascending=False)
    chosen = ranking.head(10).index.tolist()
    rows["place"] = rows["district_name"].astype(str) + ", " + rows["state_name"].astype(str)
    places = [f"{district}, {state}" for state, district in chosen]
    trend = rows[rows["place"].isin(places[:5])].groupby(["start_year", "place"])["area"].sum().reset_index()
    trend["area_thousand"] = trend["area"] / THOUSAND
    fig, (bar_axis, trend_axis) = plt.subplots(1, 2, figsize=(15, 7))
    vals = ranking.head(10).values / THOUSAND
    bars = bar_axis.barh(places[::-1], vals[::-1], color=sns.color_palette("mako", len(places)))
    bar_axis.set(title=f"Top districts in {year}", xlabel="Area (thousand hectares)", ylabel="District, state/UT")
    _label_bars(bar_axis, bars, vals[::-1])
    _interpret_rank(f"district reporting area in {year}", ranking / THOUSAND, " thousand ha")
    sns.lineplot(data=trend, x="start_year", y="area_thousand", hue="place", hue_order=places[:5], marker="o", ax=trend_axis)
    trend_axis.set(title="Trends for five largest districts", xlabel="Year", ylabel="Area (thousand hectares)")
    trend_axis.legend(title="District, state/UT", fontsize=7)
    for place in places[:5]:
        subset = trend[trend["place"] == place].sort_values("start_year")
        _interpret_change(f"{place} reporting area", subset["start_year"].tolist(), subset["area_thousand"].tolist(), " thousand ha")
    fig.suptitle("District reporting-area comparison", fontsize=15, fontweight="bold")
    fig.text(.01, .005, f"Districts are ranked by reporting area in {year}; labels include state/UT.", fontsize=8)
    _show(fig, "district_comparisons")


def plot_district_changes(cleaned_data):
    """Show districts with largest increases/decreases in Net Area Sown."""
    rows = _category_rows(cleaned_data, "Net Area Sown")
    rows = rows[rows["district_name"].notna() & rows["district_name"].astype(str).str.strip().ne("") & rows["district_name"].astype(str).str.lower().ne("unknown")].copy()
    if rows.empty:
        print("There is no district-level Net Area Sown data.")
        return
    rows["place"] = rows["district_name"].astype(str) + ", " + rows["state_name"].astype(str)
    table = rows.pivot_table(index="place", columns="start_year", values="area", aggfunc="sum")
    table = table.dropna(axis=0, how="any")
    if table.shape[1] < 2 or table.empty:
        print("Not enough comparable district-year data for this chart.")
        return
    first_year, last_year = int(table.columns.min()), int(table.columns.max())
    changes = (table[last_year] - table[first_year]).dropna()
    selected = pd.concat([changes.nsmallest(5).sort_values(), changes.nlargest(5).sort_values()])
    values = selected.values / THOUSAND
    fig, axis = plt.subplots(figsize=(12, 7))
    colors = ["#b2182b" if value < 0 else "#2166ac" for value in values]
    bars = axis.barh(selected.index, values, color=colors)
    axis.axvline(0, color="black", linewidth=.8)
    axis.set(title=f"Largest changes in Net Area Sown: {first_year} to {last_year}", xlabel="Change in area (thousand hectares)", ylabel="District, state/UT")
    for bar, value in zip(bars, values):
        axis.text(value + (.5 if value >= 0 else -.5), bar.get_y() + bar.get_height()/2, f"{value:+,.1f}", va="center", ha="left" if value >= 0 else "right", fontsize=8)
    print(f"Interpretation — district changes: {changes.idxmax()} had the largest increase ({changes.max()/THOUSAND:+,.1f} thousand ha); {changes.idxmin()} had the largest decrease ({changes.min()/THOUSAND:+,.1f} thousand ha.")
    fig.text(.01, .005, "Only districts with non-missing values in every year are included; changes are descriptive, not causal.", fontsize=8)
    _show(fig, "district_changes")

def plot_land_use_pie_chart(cleaned_data):
    category_rows = cleaned_data[
        cleaned_data["area_classification"].isin(CLASSIFICATIONS)
        & cleaned_data["area"].notna()
        & cleaned_data["start_year"].notna()
    ].copy()

    if category_rows.empty:
        print("There is not enough classification data for the pie chart.")
        return

    category_rows["start_year"] = category_rows["start_year"].astype(int)
    latest_year = category_rows["start_year"].max()

    latest_data = category_rows[
        category_rows["start_year"] == latest_year
    ]

    breakdown = (
        latest_data.groupby("area_classification")["area"]
        .sum()
        .sort_values(ascending=False)
    )

    breakdown = breakdown[breakdown > 0]

    if breakdown.empty:
        print("No positive area values are available for the pie chart.")
        return

    fig, axis = plt.subplots(figsize=(9, 7))

    axis.pie(
        breakdown.values,
        labels=breakdown.index,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.7,
    )

    axis.set_title(
        f"Land-use Classification Distribution in {latest_year}",
        fontsize=14,
        fontweight="bold",
    )

    fig.text(
        0.01,
        0.01,
        "Percentages represent only the four selected classifications.",
        fontsize=8,
    )

    print(f"\nInterpretation — Land-use distribution in {latest_year}:")
    total_area = breakdown.sum()

    for category, area in breakdown.items():
        percentage = area / total_area * 100
        print(f"{category}: {percentage:.1f}% of selected classified area.")

    largest = breakdown.idxmax()
    print(f"Largest share: {largest} ({breakdown[largest] / total_area * 100:.1f}%).")

    _show(fig, "land_use_pie_chart")


def plot_data_quality(cleaned_data):
    rows = cleaned_data[cleaned_data["start_year"].notna()].copy()
    rows["start_year"] = rows["start_year"].astype(int)
    rows["missing_area"] = rows["area"].isna()
    missing_by_category = rows.groupby("category")["missing_area"].mean().mul(100).sort_values()
    missing_by_year = rows.groupby("start_year")["missing_area"].mean().mul(100)
    usable = cleaned_data[cleaned_data["area"].notna() & cleaned_data["start_year"].notna() & (cleaned_data["area"] > 0)].copy()
    reporting = usable[usable["land_use_metrics"] == "Reporting Area"]
    if reporting.empty:
        print("There are no positive reporting-area values to summarize.")
        return
    latest_year = int(reporting["start_year"].max())
    latest_values = reporting[reporting["start_year"] == latest_year]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    sns.histplot(latest_values["area"], bins=30, color="#1f78b4", ax=axes[0, 0])
    axes[0, 0].set_xscale("log")
    axes[0, 0].set(title=f"Reporting-area values in {latest_year}", xlabel="Area (hectares, logarithmic scale)", ylabel="Number of records")
    axes[0, 0].text(.98, .95, f"Records: {len(latest_values):,}\nMedian: {latest_values['area'].median():,.0f} ha", transform=axes[0, 0].transAxes, ha="right", va="top", fontsize=8)
    names = ["Misc. tree crops and groves" if str(name).startswith("Land under Misc.") else "Other uncultivated land" if str(name).startswith("Other Uncultivated") else str(name) for name in missing_by_category.index]
    bars = axes[0, 1].barh(names, missing_by_category.values, color="#fc8d62")
    axes[0, 1].set(title="Missing area records by category", xlabel="Records with missing area (%)", ylabel="Land-use category")
    _label_bars(axes[0, 1], bars, missing_by_category.values, "%")
    axes[0, 1].tick_params(axis="y", labelsize=8)
    axes[1, 0].plot(missing_by_year.index, missing_by_year.values, marker="o", color="#756bb1")
    axes[1, 0].set(title="Missing area records by year", xlabel="Year", ylabel="Records with missing area (%)")
    axes[1, 0].tick_params(axis="x", rotation=45)
    _label_last_value(axes[1, 0], missing_by_year.index.tolist(), missing_by_year.values.tolist(), "%")
    axes[1, 1].set_facecolor("#f2f6f2")
    axes[1, 1].set(title="Data at a glance", xticks=[], yticks=[])
    axes[1, 1].text(.08, .82, "RECORDS REVIEWED", transform=axes[1, 1].transAxes, fontsize=9, color="#526357", weight="bold")
    axes[1, 1].text(.08, .70, f"{len(rows):,}", transform=axes[1, 1].transAxes, fontsize=20, color="#214e34")
    axes[1, 1].text(.08, .52, f"Area reported: {rows['area'].notna().sum():,}\nArea missing: {int(rows['missing_area'].sum()):,} ({rows['missing_area'].mean()*100:.1f}%)\nLatest reporting year: {latest_year}\nMedian reporting area: {latest_values['area'].median():,.0f} ha", transform=axes[1, 1].transAxes, fontsize=11, linespacing=1.8, va="top")
    _interpret_change("percentage of rows with missing area", missing_by_year.index.tolist(), missing_by_year.values.tolist(), "%")
    print(f"Interpretation — data completeness: {int(rows['missing_area'].sum()):,} of {len(rows):,} rows with a valid year have missing area values ({rows['missing_area'].mean()*100:.2f}%). Missing values are not assumed to mean zero.")
    fig.suptitle("Data quality and reporting-area distribution", fontsize=15, fontweight="bold")
    fig.text(.01, .005, "The distribution uses a logarithmic area axis because reported values vary widely.", fontsize=8)
    _show(fig, "data_quality")


def eda_visualization_menu(cleaned_data, annual_data):
    while True:
        _header("EDA & VISUALIZATION")
        _print_line("  1  ▸ Land-use classification pie chart")
        _print_line("  2  ▸ National trends")
        _print_line("  3  ▸ Land-use and cropping patterns")
        _print_line("  4  ▸ State comparisons")
        _print_line("  5  ▸ District comparisons")
        _print_line("  6  ▸ District changes in Net Area Sown")
        _print_line("  7  ▸ Data quality overview")
        _print_line("  0  ↩ Back")
        choice = input("\nMain Menu > EDA & Visualization > Enter choice: ").strip()
        if choice == "0":
            return
        if choice == "1":
            plot_land_use_pie_chart(cleaned_data)
        elif choice == "2":
            plot_national_trends(annual_data)
        elif choice == "3":
            plot_land_use_patterns(cleaned_data)
        elif choice == "4":
            plot_state_patterns(cleaned_data)
        elif choice == "5":
            plot_district_patterns(cleaned_data)
        elif choice == "6":
            plot_district_changes(cleaned_data)
        elif choice == "7":
            plot_data_quality(cleaned_data)
        else:
            print("Please select one of the listed options.")


if __name__ == "__main__":
    from data_preprocessor import create_member2_dataset, create_member3_dataset
    print("Loading project data...")
    cleaned_data = create_member2_dataset()
    annual_data = create_member3_dataset()
    eda_visualization_menu(cleaned_data, annual_data)

