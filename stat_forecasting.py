"""Statistical analysis and forecasting for the Agricultural Land-Use Pattern Analysis Tool.

Statistics (SciPy)
    1. Descriptive statistics and state ranking
    2. Trend analysis (linear regression on the yearly series)
    3. Early-vs-later period comparison (paired t-test / Wilcoxon)
    4. Year-to-year change analysis
    5. Correlation analysis (Pearson / Spearman)
    6. State comparison (t-test / Mann-Whitney, ANOVA / Kruskal-Wallis)

Forecasting (Statsmodels)
    1. Time series object and subset
    2. Stationarity test (ADF)
    3. Smoothing (moving average)
    4. Differencing
    5. ARIMA model, hold-out evaluation and forecast

Main series
    Cropping intensity = Cropped Area / Net Area Sown, calculated only for the
    states that report in every year. The number of reporting states changes a
    lot between years (35 states in 1998, 20 in 2021), so national totals mix
    real change with changes in reporting. A fixed set of states removes that
    problem.
"""

import os


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

OUTPUT_DIR = "stat_outputs"
ALPHA = 0.05
MILLION = 1_000_000
MIN_DISTRICTS_FOR_YEAR = 100
STABLE_RANGE = (0.80, 1.25)  # judgement-based coverage threshold around each state median

NET = "Net Area Sown"
CROPPED = "Cropped Area"
TWICE = "Area Sown More Than Once"
REPORTING = "Reporting Area"
INTENSITY = "cropping_intensity"

ASCII_UI = str.maketrans({
    "╭": "+", "╮": "+", "╰": "+", "╯": "+", "─": "-", "│": "|",
    "═": "=", "–": "-", "÷": "/", "²": "2",
})

# Display helpers
def _print_line(text=""):
    try:
        print(text)
    except UnicodeEncodeError:
        print(str(text).translate(ASCII_UI))


def _title(text):
    width = 70
    _print_line("")
    _print_line("╭" + "─" * (width - 2) + "╮")
    _print_line("│ " + text.center(width - 4) + " │")
    _print_line("╰" + "─" * (width - 2) + "╯")


def _table(frame, index=False, float_format="{:,.3f}".format):
    if frame is None or len(frame) == 0:
        _print_line("  (no records)")
        return
    _print_line(frame.to_string(index=index, float_format=float_format))


def _interpret(text):
    _print_line("")
    _print_line("Interpretation: " + text)


def _p_text(p_value):
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def _year_label(year):
    year = int(year)
    return f"{year}-{str(year + 1)[-2:]}"


def _show(fig, filename):
    """Save the chart and open it in a window (same behaviour as the EDA charts)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.tight_layout()
    file_path = os.path.join(OUTPUT_DIR, filename + ".png")
    fig.savefig(file_path, dpi=200, bbox_inches="tight")
    _print_line(f"Chart saved: {file_path}")
    _print_line("Close the chart window to return to the menu.")
    try:
        plt.show(block=True)
    finally:
        plt.close(fig)


# Data preparation
def district_metric_table(cleaned_data):
    """One row per state / district / year with the four land-use metrics as columns."""
    rows = cleaned_data[cleaned_data["land_use_metrics"].isin([REPORTING, NET, CROPPED, TWICE])]
    rows = rows.dropna(subset=["start_year", "area"])
    table = rows.pivot_table(
        index=["state_name", "district_name", "start_year"],
        columns="land_use_metrics",
        values="area",
        aggfunc="sum",
    ).reset_index()
    table.columns.name = None
    for column in [REPORTING, NET, CROPPED, TWICE]:
        if column not in table.columns:
            table[column] = np.nan
    table["start_year"] = table["start_year"].astype(int)
    # Cropping intensity needs both areas, and a Net Area Sown above zero.
    table = table[(table[NET] > 0) & table[CROPPED].notna()].copy()
    table[INTENSITY] = table[CROPPED] / table[NET]
    return table


def district_class_table(cleaned_data):
    """Fallow and forest area per state / district / year."""
    rows = cleaned_data[
        cleaned_data["land_use_metrics"].isna()
        & cleaned_data["area_classification"].isin(["Fallow Land", "Forests"])
    ].dropna(subset=["start_year", "area"])
    table = rows.pivot_table(
        index=["state_name", "district_name", "start_year"],
        columns="area_classification",
        values="area",
        aggfunc="sum",
    ).reset_index()
    table.columns.name = None
    for column in ["Fallow Land", "Forests"]:
        if column not in table.columns:
            table[column] = np.nan
    table["start_year"] = table["start_year"].astype(int)
    return table


def usable_years(table):
    """Years with enough valid districts for district-level comparisons."""
    if table.empty or INTENSITY not in table.columns:
        return []
    counts = table.groupby("start_year")[INTENSITY].count()
    return sorted(counts[counts >= MIN_DISTRICTS_FOR_YEAR].index.astype(int))


def stable_panel_states(table, stable_range=STABLE_RANGE):
    """Select states with records every year and stable Net Area Sown totals.

    The 80%-125% range is a judgement-based screening rule, not a statistical
    guarantee. It aims to flag abrupt coverage changes; genuine area changes
    can also cause a state to be excluded.
    """
    years = sorted(table["start_year"].dropna().astype(int).unique())
    if not years:
        return []

    state_years = table.groupby("state_name")["start_year"].agg(
        lambda values: set(values.astype(int))
    )
    required_years = set(years)
    complete_states = [
        state for state, reported_years in state_years.items()
        if required_years.issubset(reported_years)
    ]
    if not complete_states:
        return []

    net = (
        table[table["state_name"].isin(complete_states)]
        .groupby(["state_name", "start_year"])[NET]
        .sum(min_count=1)
        .unstack()
        .reindex(columns=years)
    )
    median = net.median(axis=1)
    share = net.div(median.replace(0, np.nan), axis=0)
    stable = (
        share.notna().all(axis=1)
        & (share.min(axis=1) >= stable_range[0])
        & (share.max(axis=1) <= stable_range[1])
    )
    return sorted(stable[stable].index.tolist())


def build_intensity_series(cleaned_data):
    """Build cropping intensity for states passing the fixed-panel screen.

    A state must be represented every year, and its yearly Net Area Sown total
    must stay within 80%-125% of its own median. This threshold is a judgement
    call intended to flag coverage breaks; it cannot prove consistent reporting.
    Returns (series, panel_states) with an annual DatetimeIndex.
    """
    table = district_metric_table(cleaned_data)
    years = sorted(table["start_year"].dropna().astype(int).unique())
    if not years:
        raise ValueError("No valid yearly cropping-intensity data was found.")

    panel_states = stable_panel_states(table)
    if len(panel_states) < 3:
        raise ValueError(
            f"Only {len(panel_states)} states pass the stable-reporting screen; "
            "at least 3 are needed for this time-series analysis."
        )

    panel = table[table["state_name"].isin(panel_states)]
    yearly = panel.groupby("start_year")[[NET, CROPPED]].sum(min_count=1)
    yearly = yearly.reindex(years)
    if yearly[[NET, CROPPED]].isna().any().any() or (yearly[NET] <= 0).any():
        raise ValueError("The stable-state series has missing or invalid yearly totals.")

    values = yearly[CROPPED] / yearly[NET]
    first, last = int(values.index.min()), int(values.index.max())
    values = values.reindex(range(first, last + 1))
    if values.isna().any():
        raise ValueError("The yearly series has missing years; do not interpolate silently for forecasting.")
    if len(values) < 8:
        raise ValueError("At least 8 consecutive yearly observations are needed for time-series analysis.")

    values.index = pd.date_range(start=f"{first}-01-01", periods=len(values), freq="YS")
    values.name = INTENSITY
    return values, panel_states



# PART A - STATISTICS (SciPy)

def descriptive_statistics(cleaned_data, annual_data):
    """Descriptive statistics of the yearly series and state ranking."""
    _title("DESCRIPTIVE STATISTICS & STATE RANKING")
    table = district_metric_table(cleaned_data)
    years = usable_years(table)
    if not years:
        _print_line("Not enough valid district data to calculate descriptive statistics.")
        return None
    latest = years[-1]

    # 1. Yearly national totals
    _print_line("\n1) Yearly national totals, 1998 onwards (million hectares)")
    rows = []
    for metric in [REPORTING, NET, CROPPED, TWICE]:
        values = annual_data[annual_data["land_use_metrics"] == metric]["area"].dropna() / MILLION
        if len(values) < 3:
            continue
        summary = stats.describe(values)
        rows.append({
            "Metric": metric, "Years": summary.nobs, "Mean": summary.mean,
            "Median": float(np.median(values)), "Std dev": float(np.sqrt(summary.variance)),
            "Min": summary.minmax[0], "Max": summary.minmax[1],
            "Skewness": summary.skewness, "Kurtosis": summary.kurtosis,
        })
    _table(pd.DataFrame(rows), float_format="{:,.2f}".format)
    _print_line("Note: totals depend on how many states reported that year (see option 4).")

    # 2. Fixed-panel cropping intensity series
    series, panel_states = build_intensity_series(cleaned_data)
    summary = stats.describe(series.values)
    _print_line(f"\n2) Cropping intensity, fixed panel of {len(panel_states)} states that report every year")
    _print_line(f"   Years: {len(series)}  Mean: {summary.mean:.3f}  Median: {np.median(series):.3f}  "
                f"Std dev: {np.sqrt(summary.variance):.3f}")
    _print_line(f"   Min: {summary.minmax[0]:.3f}  Max: {summary.minmax[1]:.3f}  "
                f"Skewness: {summary.skewness:.2f}  Kurtosis: {summary.kurtosis:.2f}")

    # 3. District distribution in the latest year
    latest_rows = table[table["start_year"] == latest]
    summary = stats.describe(latest_rows[INTENSITY].values)
    _print_line(f"\n3) District cropping intensity in {_year_label(latest)} ({summary.nobs} districts)")
    _print_line(f"   Mean: {summary.mean:.3f}  Median: {latest_rows[INTENSITY].median():.3f}  "
                f"Std dev: {np.sqrt(summary.variance):.3f}  Skewness: {summary.skewness:.2f}")
    _print_line(f"   Min: {summary.minmax[0]:.3f}  Max: {summary.minmax[1]:.3f}")

    # 4. State ranking
    states = latest_rows.groupby("state_name").agg(
        Districts=(INTENSITY, "count"),
        Net_Area_Sown_Mha=(NET, lambda s: s.sum() / MILLION),
        Mean_Intensity=(INTENSITY, "mean"),
    ).reset_index()
    states["Rank (Intensity)"] = stats.rankdata(-states["Mean_Intensity"], method="min").astype(int)
    states["Rank (Net Area Sown)"] = stats.rankdata(-states["Net_Area_Sown_Mha"], method="min").astype(int)
    states = states.sort_values("Rank (Intensity)")
    states = states.rename(columns={"state_name": "State", "Net_Area_Sown_Mha": "Net Sown (M ha)",
                                    "Mean_Intensity": "Mean intensity"})

    _print_line(f"\n4) State ranking by mean district cropping intensity, {_year_label(latest)}")
    _print_line("   Highest 10")
    _table(states.head(10))
    _print_line("   Lowest 5")
    _table(states.tail(5))

    _interpret(
        f"Skewness of the district intensity values is {summary.skewness:.2f}, so a few districts with very "
        "high intensity pull the mean above the median. Medians and rank-based tests are safer for districts. "
        f"{states.iloc[0]['State']} ranks first and {states.iloc[-1]['State']} last on intensity."
    )
    return states


def trend_analysis(cleaned_data):
    """Linear trend of the yearly cropping-intensity series."""
    _title("TREND ANALYSIS")
    series, panel_states = build_intensity_series(cleaned_data)
    years = series.index.year.values
    result = stats.linregress(years, series.values)
    fitted = result.intercept + result.slope * years
    change_total = fitted[-1] - fitted[0]

    _print_line(f"Series: cropping intensity of {len(panel_states)} states represented every year, "
                f"{years[0]} to {years[-1]}")
    _print_line(f"Slope           : {result.slope:+.5f} per year (standard error {result.stderr:.5f})")
    _print_line(f"R-squared       : {result.rvalue ** 2:.3f}")
    _print_line(f"p-value         : {_p_text(result.pvalue)}")
    _print_line(f"Fitted change   : {fitted[0]:.3f} -> {fitted[-1]:.3f} ({change_total:+.3f} over the period)")

    if result.pvalue < ALPHA:
        direction = "rising" if result.slope > 0 else "falling"
        _interpret(
            f"There is a statistically significant {direction} trend (p {_p_text(result.pvalue)}). "
            f"Cropping intensity changes by about {result.slope:+.4f} per year, and the straight line "
            f"explains {result.rvalue ** 2 * 100:.0f}% of the variation. "
            "The ADF stationarity test is checked separately before fitting ARIMA."
        )
    else:
        _interpret(f"No significant linear trend was found (p {_p_text(result.pvalue)}).")

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(years, series.values, marker="o", label="Cropping intensity")
    axis.plot(years, fitted, linestyle="--", color="crimson",
              label=f"Linear trend ({result.slope:+.4f} per year)")
    axis.set(title="Trend in cropping intensity (fixed panel of states)",
             xlabel="Year (start of agricultural year)", ylabel="Cropped Area / Net Area Sown")
    axis.grid(alpha=0.3)
    axis.legend()
    _show(fig, "trend_analysis")
    return result


def period_comparison(cleaned_data, early_year=None, late_year=None):
    """Compare district cropping intensity in two years using matched districts."""
    _title("EARLY vs LATER PERIOD COMPARISON")
    table = district_metric_table(cleaned_data)
    years = usable_years(table)
    if not years:
        _print_line("Not enough valid district data to compare years.")
        return None
    early_year = years[0] if early_year is None else int(early_year)
    late_year = years[-1] if late_year is None else int(late_year)
    if early_year not in years or late_year not in years or early_year >= late_year:
        _print_line("Choose two available years, with the later year after the early year.")
        return None

    early = table[table["start_year"] == early_year].set_index(["state_name", "district_name"])[INTENSITY]
    late = table[table["start_year"] == late_year].set_index(["state_name", "district_name"])[INTENSITY]
    early = early[~early.index.duplicated()]
    late = late[~late.index.duplicated()]
    pairs = pd.concat([early.rename("early"), late.rename("late")], axis=1, join="inner").dropna()
    _print_line(f"Districts available: {len(early)} in {_year_label(early_year)}; "
                f"{len(late)} in {_year_label(late_year)}.")
    _print_line(f"Matched districts: {len(pairs)}; unmatched early: {len(early) - len(pairs)}; "
                f"unmatched later: {len(late) - len(pairs)}.")
    if len(pairs) < 10:
        _print_line("Not enough matched districts to run a paired comparison (minimum 10).")
        return None

    differences = pairs["late"] - pairs["early"]
    t_stat, t_p = stats.ttest_rel(pairs["late"], pairs["early"])
    try:
        w_stat, w_p = stats.wilcoxon(differences)
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    shapiro_sample = differences.sample(min(len(differences), 5000), random_state=1)
    shapiro_p = stats.shapiro(shapiro_sample).pvalue
    skewness = float(stats.skew(differences, bias=False))
    up = int((differences > 0).sum())
    down = int((differences < 0).sum())
    sign_p = stats.binomtest(up, up + down, 0.5).pvalue if up + down else np.nan

    _print_line(f"Mean intensity: {pairs['early'].mean():.3f} -> {pairs['late'].mean():.3f} "
                f"(mean change {differences.mean():+.3f}).")
    _print_line(f"Districts increased: {up} | decreased: {down} | unchanged: {(differences == 0).sum()}")
    _print_line(f"Shapiro-Wilk on differences: p = {_p_text(shapiro_p)} "
                f"({'not rejected as non-normal' if shapiro_p >= ALPHA else 'evidence against normality'}).")
    _print_line(f"Paired t-test: t = {t_stat:.3f}, p = {_p_text(t_p)}")
    _print_line(f"Wilcoxon signed-rank: W = {w_stat:,.0f}, p = {_p_text(w_p)}")
    _print_line(f"Sign test: {up} increases vs {down} decreases, p = {_p_text(sign_p)}")
    _print_line(f"Skewness of paired differences: {skewness:.2f}")

    if shapiro_p >= ALPHA:
        chosen_name, chosen_p = "paired t-test", t_p
        reason = "the differences did not fail the Shapiro-Wilk normality check"
    elif abs(skewness) <= 1:
        chosen_name, chosen_p = "Wilcoxon signed-rank test", w_p
        reason = "the differences are non-normal but not strongly skewed"
    else:
        chosen_name, chosen_p = "sign test", sign_p
        reason = "the differences are strongly skewed, making Wilcoxon symmetry questionable"

    if chosen_p is not None and np.isfinite(chosen_p) and chosen_p < ALPHA:
        conclusion = f"A statistically significant change was detected (p {_p_text(chosen_p)})."
    elif chosen_p is not None and np.isfinite(chosen_p):
        conclusion = f"No statistically significant change was detected (p {_p_text(chosen_p)})."
    else:
        conclusion = "The selected test could not produce a valid p-value."
    _interpret(f"The {chosen_name} is used because {reason}. {conclusion}")
    _print_line("Note: changed or split district names may not match. Wilcoxon assumes roughly symmetric paired differences; the sign test does not require that symmetry.")

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.hist(differences.clip(-1.5, 1.5), bins=40, color="steelblue", edgecolor="white")
    axis.axvline(0, color="black", linewidth=1)
    axis.axvline(differences.mean(), color="crimson", linestyle="--", label=f"Mean {differences.mean():+.3f}")
    axis.set(title=f"Change in district cropping intensity, {_year_label(early_year)} to {_year_label(late_year)}",
             xlabel="Change in intensity (values beyond +-1.5 grouped at the edges)",
             ylabel="Number of districts")
    axis.legend()
    _show(fig, "period_comparison")
    return {"n": len(pairs), "paired_t_p": t_p, "wilcoxon_p": w_p,
            "sign_test_p": sign_p, "shapiro_p": shapiro_p, "difference_skewness": skewness,
            "chosen_test": chosen_name, "chosen_p": chosen_p}


def year_to_year_change(cleaned_data, annual_data):
    """Yearly change in the fixed-panel intensity and in the national totals."""
    _title("YEAR-TO-YEAR CHANGE ANALYSIS")
    series, _ = build_intensity_series(cleaned_data)
    years = series.index.year.values

    # Count states with any non-missing area record in the cleaned source data.
    # This is different from counting states that have enough data to calculate intensity.
    reporting_rows = cleaned_data.dropna(subset=["start_year", "area"])
    states_reporting = reporting_rows.groupby("start_year")["state_name"].nunique()
    national = annual_data[annual_data["land_use_metrics"] == NET].set_index("start_year")
    cropped = annual_data[annual_data["land_use_metrics"] == CROPPED].set_index("start_year")

    summary = pd.DataFrame({
        "Year": [_year_label(y) for y in years],
        "States with area records": [int(states_reporting.get(y, 0)) for y in years],
        "Intensity (panel)": series.values,
        "Intensity change": series.diff().values,
        "Net Sown % change": [national["pct_change"].get(y, np.nan) for y in years],
        "Cropped % change": [cropped["pct_change"].get(y, np.nan) for y in years],
    })
    _table(summary, float_format="{:,.3f}".format)

    change = series.diff().dropna()
    _print_line("")
    _print_line(f"Fixed-panel intensity: average yearly change {change.mean():+.4f}; "
                f"largest rise {change.max():+.3f} ({_year_label(change.idxmax().year)}), "
                f"largest fall {change.min():+.3f} ({_year_label(change.idxmin().year)}).")

    for label, frame in [(NET, national), (CROPPED, cropped)]:
        pct = frame["pct_change"].dropna()
        if pct.empty:
            continue
        z_scores = (pct - pct.mean()) / pct.std()
        unusual = [_year_label(y) for y in pct.index[np.abs(z_scores) > 2]]
        _print_line(f"National {label}: average change {pct.mean():+.2f}% a year; "
                    f"largest rise {pct.max():+.1f}% ({_year_label(pct.idxmax())}), "
                    f"largest fall {pct.min():+.1f}% ({_year_label(pct.idxmin())}); "
                    f"unusual years (|z| > 2): {', '.join(unusual) if unusual else 'none'}.")

    _interpret(
        "Large swings in national totals may reflect changes in reporting coverage as well as land-use change. "
        "The fixed-state intensity series reduces changes caused by states entering or leaving the dataset, "
        "but within-state district coverage and reporting methods may still vary. Treat the forecast as exploratory."
    )

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    bar_colors = ["seagreen" if v >= 0 else "indianred" for v in change.values]
    top.bar(change.index.year, change.values, color=bar_colors)
    top.axhline(0, color="black", linewidth=0.8)
    top.set(title="Yearly change in cropping intensity (fixed panel of states)", ylabel="Change in intensity")

    pct = national["pct_change"].dropna()
    bottom.bar(pct.index, pct.values, color="steelblue", label="National Net Area Sown, % change")
    bottom.axhline(0, color="black", linewidth=0.8)
    bottom.set(title="National Net Area Sown change and states with area records",
               xlabel="Year (start of agricultural year)", ylabel="% change")
    twin = bottom.twinx()
    twin.plot(states_reporting.index, states_reporting.values, color="darkorange", marker="o", label="States reporting")
    twin.set_ylabel("States with area records")
    handles = bottom.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = bottom.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    bottom.legend(handles, labels, loc="lower left")
    _show(fig, "year_to_year_change")
    return summary


def _correlation_table(cleaned_data, year):
    table = district_metric_table(cleaned_data)
    classes = district_class_table(cleaned_data)
    merged = table.merge(classes, on=["state_name", "district_name", "start_year"], how="left")
    merged = merged[merged["start_year"] == year].copy()
    base = merged[REPORTING].where(merged[REPORTING] > 0)
    merged["Fallow share"] = merged["Fallow Land"] / base
    merged["Forest share"] = merged["Forests"] / base
    merged["Net sown share"] = merged[NET] / base
    return merged


def correlation_analysis(cleaned_data, year=None):
    """Pearson and Spearman correlation between district-level variables."""
    _title("CORRELATION ANALYSIS")
    table = district_metric_table(cleaned_data)
    years = usable_years(table)
    if not years:
        _print_line("Not enough valid district data to calculate correlations.")
        return None
    year = years[-1] if year is None else int(year)
    data = _correlation_table(cleaned_data, year)
    # Shares above 1 mean the district's classification does not match its reporting area.
    for column in ["Fallow share", "Forest share", "Net sown share"]:
        data.loc[data[column] > 1, column] = np.nan

    pairs = [
        (NET, CROPPED),
        (NET, TWICE),
        ("Fallow share", INTENSITY),
        ("Forest share", "Net sown share"),
    ]
    rows = []
    for first, second in pairs:
        pair = data[[first, second]].dropna()
        if len(pair) < 10:
            continue
        r, r_p = stats.pearsonr(pair[first], pair[second])
        rho, rho_p = stats.spearmanr(pair[first], pair[second])
        strength = "strong" if abs(rho) >= 0.7 else "moderate" if abs(rho) >= 0.4 else "weak"
        rows.append({
            "Variable 1": first, "Variable 2": second, "n": len(pair),
            "Pearson r": r, "p (Pearson)": _p_text(r_p),
            "Spearman rho": rho, "p (Spearman)": _p_text(rho_p),
            "Strength": f"{strength} {'positive' if rho > 0 else 'negative'}",
        })
    results = pd.DataFrame(rows)
    _print_line(f"District-level correlations, {_year_label(year)}")
    _table(results, float_format="{:,.3f}".format)

    _interpret(
        "Net Area Sown and Cropped Area move together almost one-for-one because cropped area is built from "
        "net sown area. The more informative pairs are fallow share against intensity and forest share "
        "against net sown share. Spearman is preferred here because district values are skewed. "
        "Correlation does not show cause."
    )

    variables = [NET, CROPPED, TWICE, INTENSITY, "Fallow share", "Forest share"]
    matrix = data[variables].corr(method="spearman")
    fig, axis = plt.subplots(figsize=(8.5, 7))
    image = axis.imshow(matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(variables)))
    axis.set_yticks(range(len(variables)))
    axis.set_xticklabels(variables, rotation=40, ha="right")
    axis.set_yticklabels(variables)
    for i in range(len(variables)):
        for j in range(len(variables)):
            axis.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axis, label="Spearman correlation")
    axis.set_title(f"Spearman correlation of district variables, {_year_label(year)}")
    _show(fig, "correlation")
    return results


def eligible_states(cleaned_data, year=None, min_districts=10):
    """States with enough districts for a comparison, largest first."""
    table = district_metric_table(cleaned_data)
    years = usable_years(table)
    if not years:
        return pd.Series(dtype=int)
    year = years[-1] if year is None else int(year)
    counts = table[table["start_year"] == year].groupby("state_name")[INTENSITY].count()
    return counts[counts >= min_districts].sort_values(ascending=False)


def compare_states(cleaned_data, states, year=None):
    """Compare district cropping intensity across 2 states (t-test) or 3+ states (ANOVA)."""
    _title("STATE COMPARISON")
    table = district_metric_table(cleaned_data)
    years = usable_years(table)
    if not years:
        _print_line("Not enough valid district data to compare states.")
        return None
    year = years[-1] if year is None else int(year)
    data = table[table["start_year"] == year]
    groups = {s: data[data["state_name"] == s][INTENSITY].dropna().values for s in states}
    groups = {s: v for s, v in groups.items() if len(v) >= 3}
    if len(groups) < 2:
        _print_line("At least two states with three or more districts are needed.")
        return None

    _print_line(f"District cropping intensity, {_year_label(year)}")
    summary = pd.DataFrame({
        "State": list(groups),
        "Districts": [len(v) for v in groups.values()],
        "Mean": [v.mean() for v in groups.values()],
        "Median": [np.median(v) for v in groups.values()],
        "Std dev": [v.std(ddof=1) for v in groups.values()],
        "Shapiro p": [stats.shapiro(v).pvalue for v in groups.values()],
    })
    _table(summary.assign(**{"Shapiro p": summary["Shapiro p"].map(_p_text)}))

    all_normal = bool((summary["Shapiro p"] >= ALPHA).all())
    levene_p = stats.levene(*groups.values(), center="median").pvalue
    equal_var = levene_p >= ALPHA
    _print_line("")
    _print_line(f"Normality (Shapiro-Wilk)     : {'all groups normal' if all_normal else 'at least one group not normal'}")
    _print_line(f"Equal variances (Levene)     : p = {_p_text(levene_p)} "
                f"({'equal' if equal_var else 'not equal'})")

    values = list(groups.values())
    if len(groups) == 2:
        t_stat, t_p = stats.ttest_ind(values[0], values[1], equal_var=equal_var)
        u_stat, u_p = stats.mannwhitneyu(values[0], values[1], alternative="two-sided")
        test_name = "Student t-test" if equal_var else "Welch t-test"
        _print_line(f"{test_name:<29}: t = {t_stat:.3f}, p = {_p_text(t_p)}")
        _print_line(f"{'Mann-Whitney U test':<29}: U = {u_stat:,.0f}, p = {_p_text(u_p)}")
        chosen_name, chosen_p = (test_name, t_p) if all_normal else ("Mann-Whitney U test", u_p)
    else:
        f_stat, f_p = stats.f_oneway(*values)
        h_stat, h_p = stats.kruskal(*values)
        _print_line(f"{'One-way ANOVA':<29}: F = {f_stat:.3f}, p = {_p_text(f_p)}")
        _print_line(f"{'Kruskal-Wallis test':<29}: H = {h_stat:.3f}, p = {_p_text(h_p)}")
        chosen_name, chosen_p = ("One-way ANOVA", f_p) if (all_normal and equal_var) else ("Kruskal-Wallis test", h_p)

    names = list(groups)
    if len(groups) == 2:
        reason = (
            "the groups passed the normality checks"
            if all_normal else
            "at least one group did not pass the normality check"
        )
    elif all_normal and equal_var:
        reason = "the normality and equal-variance checks were satisfied"
    elif not all_normal and not equal_var:
        reason = "normality and equal-variance checks were not both satisfied"
    elif not all_normal:
        reason = "at least one group did not pass the normality check"
    else:
        reason = "the equal-variance check was not satisfied"
    _interpret(
        f"The {chosen_name} was selected because {reason}. "
        + (f"The test found a statistically significant difference between "
           f"{'the two states' if len(groups) == 2 else 'at least some of the states'} "
           f"(p {_p_text(chosen_p)}). Highest sample mean: {summary.loc[summary['Mean'].idxmax(), 'State']}; "
           f"lowest sample mean: {summary.loc[summary['Mean'].idxmin(), 'State']}."
           if chosen_p < ALPHA else
           f"The test did not find a statistically significant difference (p {_p_text(chosen_p)}).")
    )
    _print_line("Note: districts in one state share climate and policy, so they are not fully independent.")

    fig, axis = plt.subplots(figsize=(max(7, 1.8 * len(names)), 5.5))
    axis.boxplot(values, showmeans=True)
    axis.set_xticklabels(names, rotation=20, ha="right")
    axis.set(title=f"District cropping intensity by state, {_year_label(year)}", ylabel="Cropped Area / Net Area Sown")
    axis.grid(axis="y", alpha=0.3)
    _show(fig, "state_comparison")
    return summary



# PART B - FORECASTING (Statsmodels)
def _statsmodels_missing():
    _print_line("Statsmodels is not installed. Run: pip install statsmodels")


def time_series_overview(series, start_year=None):
    """Show the series as a time series object, and a subset of it."""
    _title("TIME SERIES OBJECT")
    _print_line(f"Type: {type(series).__name__} | index: {type(series.index).__name__} | "
                f"frequency: {series.index.freqstr} | observations: {len(series)}")
    _print_line(f"Period: {_year_label(series.index.year[0])} to {_year_label(series.index.year[-1])}")
    frame = pd.DataFrame({"Year": [_year_label(y) for y in series.index.year], "Cropping intensity": series.values})
    _table(frame, float_format="{:,.4f}".format)

    if start_year is not None:
        subset = series[series.index.year >= int(start_year)]
        _print_line(f"\nSubset from {start_year}: {len(subset)} observations")
        _print_line(f"Mean {subset.mean():.4f} | min {subset.min():.4f} | max {subset.max():.4f}")
        return subset
    return series


def _adf(values):
    from statsmodels.tsa.stattools import adfuller
    stat, p_value, lags, nobs, critical, _ = adfuller(np.asarray(values), autolag="AIC")
    return stat, p_value, lags, nobs, critical


def stationarity_test(series):
    """Augmented Dickey-Fuller test on the series."""
    _title("STATIONARITY TEST (ADF)")
    try:
        stat, p_value, lags, nobs, critical = _adf(series.values)
    except ImportError:
        _statsmodels_missing()
        return None
    _print_line("Null hypothesis: the series has a unit root (it is not stationary).")
    _print_line(f"ADF statistic : {stat:.4f}")
    _print_line(f"p-value       : {_p_text(p_value)}")
    _print_line(f"Lags used     : {lags} | observations: {nobs}")
    _print_line("Critical values: " + ", ".join(f"{k} = {v:.3f}" for k, v in critical.items()))
    if p_value < ALPHA:
        _interpret("The null hypothesis is rejected, so the series can be treated as stationary.")
    else:
        _interpret("The ADF null hypothesis is not rejected, so there is not enough evidence to call the series stationary. "
                   "Differencing may help before ARIMA. With only about 26 yearly values, the test has low power.")
    return p_value


def smoothing_chart(series, window=3):
    """Moving-average smoothing of the trend."""
    _title("SMOOTHING (MOVING AVERAGE)")
    smoothed = series.rolling(window=window).mean()
    frame = pd.DataFrame({
        "Year": [_year_label(y) for y in series.index.year],
        "Actual": series.values,
        f"{window}-year average": smoothed.values,
    })
    _table(frame.tail(10), float_format="{:,.4f}".format)
    _interpret(f"The {window}-year moving average removes year-to-year noise and shows the underlying "
               "direction of cropping intensity.")

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(series.index.year, series.values, marker="o", alpha=0.6, label="Actual")
    axis.plot(series.index.year, smoothed.values, linewidth=2.5, color="crimson", label=f"{window}-year moving average")
    axis.set(title="Cropping intensity with moving-average smoothing",
             xlabel="Year (start of agricultural year)", ylabel="Cropped Area / Net Area Sown")
    axis.grid(alpha=0.3)
    axis.legend()
    _show(fig, "smoothing")
    return smoothed


def choose_differencing(series, max_d=2, verbose=True):
    """Smallest number of differences (0 to max_d) after which the ADF test passes."""
    current = series.dropna().copy()
    if len(current) < 8:
        raise ValueError("At least 8 yearly observations are needed for the ADF test.")
    for d in range(max_d + 1):
        values = current.dropna().values
        try:
            _, p_value, _, _, _ = _adf(values)
        except ImportError:
            raise
        except Exception:
            p_value = np.nan
        if verbose:
            _print_line(f"  d = {d}: ADF p-value = {_p_text(p_value) if not np.isnan(p_value) else 'n/a'}")
        if not np.isnan(p_value) and p_value < ALPHA:
            return d
        current = current.diff()
    return max_d


def differencing_analysis(series):
    """Difference the series until it is stationary."""
    _title("DIFFERENCING")
    _print_line("ADF test after each round of differencing:")
    try:
        d = choose_differencing(series)
    except ImportError:
        _statsmodels_missing()
        return None
    _interpret(f"The series becomes stationary after {d} difference(s), so d = {d} is used in ARIMA."
               if d > 0 else "The series is already stationary, so d = 0.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(series.index.year, series.values, marker="o")
    axes[0].set(title="Original series", ylabel="Cropping intensity")
    first = series.diff()
    axes[1].plot(series.index.year, first.values, marker="o", color="darkorange")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(title="First difference", xlabel="Year (start of agricultural year)", ylabel="Change in intensity")
    for axis in axes:
        axis.grid(alpha=0.3)
    _show(fig, "differencing")
    return d


def _trend_for(d):
    # A constant cannot be used when d > 0. A linear trend ('t') acts as drift when d = 1.
    return "c" if d == 0 else "t" if d == 1 else "n"


def _fit_arima(values, order, trend):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(values, order=order, trend=trend).fit()


def arima_forecast(series, horizon=5, holdout=5, max_p=2, max_q=2):
    """Select an ARIMA order, evaluate it on a hold-out period and forecast."""
    _title("ARIMA MODEL & FORECAST")
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError:
        _statsmodels_missing()
        return None

    if len(series) < 12:
        _print_line("ARIMA forecasting needs at least 12 yearly observations.")
        return None
    holdout = min(holdout, max(2, len(series) // 4))
    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    # Select differencing from training data only, to avoid using hold-out years
    # when deciding the model specification.
    try:
        d = choose_differencing(train, verbose=False)
    except ImportError:
        _statsmodels_missing()
        return None
    trend = _trend_for(d)
    _print_line(f"Differencing order d = {d} | trend term: '{trend}' | training years: {len(train)} | "
                f"hold-out years: {holdout}")

    # ACF and PACF of the differenced series guide the choice of p and q.
    differenced = train.diff(d).dropna() if d > 0 else train
    lags = max(1, min(5, len(differenced) // 2 - 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_acf(differenced, lags=lags, ax=axes[0], title="ACF (autocorrelation)")
    plot_pacf(differenced, lags=lags, ax=axes[1], title="PACF (partial autocorrelation)", method="ywm")
    _show(fig, "acf_pacf")

    # Order selection on the training data only, by AIC.
    candidates = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                fit = _fit_arima(train.values, (p, d, q), trend)
                candidates.append({"Order (p,d,q)": f"({p},{d},{q})", "p": p, "q": q,
                                   "AIC": fit.aic, "BIC": fit.bic})
            except Exception:
                continue
    if not candidates:
        _print_line("No ARIMA model could be fitted to this series.")
        return None
    candidates = pd.DataFrame(candidates).sort_values("AIC").reset_index(drop=True)
    _print_line("\nCandidate models (lower AIC is better):")
    _table(candidates[["Order (p,d,q)", "AIC", "BIC"]], float_format="{:,.2f}".format)
    best_p, best_q = int(candidates.loc[0, "p"]), int(candidates.loc[0, "q"])
    order = (best_p, d, best_q)
    _print_line(f"\nSelected order: ARIMA{order}")

    # Hold-out evaluation.
    train_fit = _fit_arima(train.values, order, trend)
    predicted = np.asarray(train_fit.forecast(steps=holdout))
    actual = test.values
    errors = actual - predicted
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    nonzero_actual = actual != 0
    mape = (float(np.mean(np.abs(errors[nonzero_actual] / actual[nonzero_actual])) * 100)
            if nonzero_actual.any() else np.nan)
    _print_line("\nHold-out evaluation (model fitted on the training years only):")
    _table(pd.DataFrame({"Year": [_year_label(y) for y in test.index.year], "Actual": actual,
                         "Predicted": predicted, "Error": errors}), float_format="{:,.4f}".format)
    if np.isfinite(mape):
        mape_text = f"{mape:.2f}%"
    elif nonzero_actual.any():
        mape_text = "not available (no valid non-zero actual values)"
    else:
        mape_text = "not defined (all actual values are zero)"
    _print_line(f"RMSE = {rmse:.4f} | MAE = {mae:.4f} | MAPE = {mape_text}")

    # Final model on all data.
    final_fit = _fit_arima(series.values, order, trend)
    residuals = np.asarray(final_fit.resid)[d:]
    try:
        ljung_p = float(acorr_ljungbox(residuals, lags=[min(5, max(1, len(residuals) // 4))], return_df=True)["lb_pvalue"].iloc[0])
        _print_line(f"Residual check (Ljung-Box) p = {_p_text(ljung_p)}: "
                    + ("no leftover autocorrelation" if ljung_p >= ALPHA else "some autocorrelation remains"))
    except Exception:
        ljung_p = np.nan

    forecast = final_fit.get_forecast(steps=horizon)
    mean = np.asarray(forecast.predicted_mean)
    interval = np.asarray(forecast.conf_int(alpha=0.05))
    last_year = int(series.index.year[-1])
    forecast_years = [last_year + i for i in range(1, horizon + 1)]
    forecast_table = pd.DataFrame({
        "Year": [_year_label(y) for y in forecast_years],
        "Forecast": mean, "Lower 95%": interval[:, 0], "Upper 95%": interval[:, 1],
    })
    _print_line(f"\nForecast for the next {horizon} years (fitted on all {len(series)} years):")
    _table(forecast_table, float_format="{:,.4f}".format)

    _interpret(
        f"ARIMA{order} was selected by AIC. On the last {holdout} known years it was off by "
        f"{mape_text} on average (RMSE {rmse:.4f}). The forecast projects cropping intensity of "
        f"{mean[-1]:.3f} by {_year_label(forecast_years[-1])}. The interval widens each year because the "
        f"model is based on only {len(series)} yearly values, so treat distant years with caution. "
        "The forecast assumes the past pattern continues and does not model weather or policy."
    )

    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(series.index.year, series.values, marker="o", label="Actual")
    axis.plot(test.index.year, predicted, marker="s", linestyle="--", color="darkorange",
              label=f"Hold-out prediction (MAPE {mape_text})")
    axis.plot(forecast_years, mean, marker="o", color="crimson", label="Forecast")
    axis.fill_between(forecast_years, interval[:, 0], interval[:, 1], color="crimson", alpha=0.18, label="95% interval")
    axis.set(title=f"ARIMA{order} forecast of cropping intensity",
             xlabel="Year (start of agricultural year)", ylabel="Cropped Area / Net Area Sown")
    axis.grid(alpha=0.3)
    axis.legend()
    _show(fig, "arima_forecast")
    return {"order": order, "rmse": rmse, "mae": mae, "mape": mape, "forecast": forecast_table}