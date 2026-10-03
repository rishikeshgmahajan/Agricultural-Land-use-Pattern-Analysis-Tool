"""
Statistical analysis and forecasting module:

1. Descriptive statistics
2. Trend analysis using SciPy
3. Early-vs-later period comparison
4. Year-to-year change analysis
5. Correlation analysis using SciPy
6. Forecast model validation
7. Future forecasting using Statsmodels Holt model
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.holtwinters import Holt

# 1. PREPARE ANNUAL SERIES
def prepare_annual_series(data):
    """
    Prepare the data so it can be used for statistics and forecasting.
    The function:
    - converts year and value into numbers
    - removes rows with missing year or value
    - combines values if more than one row exists for a year
    - sorts the data from oldest year to newest year
    """

    df = data.copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["year", "value"])

    df["year"] = df["year"].astype(int)

    annual_data = (
        df.groupby("year", as_index=False)["value"]
        .sum()
        .sort_values("year")
    )

    return annual_data


# 2. DESCRIPTIVE STATISTICS
def descriptive_summary(data):
    """
    Calculate basic descriptive statistics for the selected annual agricultural metric.
    These values help us understand:
    - how many years are available
    - the average value
    - the middle value
    - the lowest and highest values
    - how much the values vary
    """

    df = prepare_annual_series(data)

    values = df["value"]

    result = {
        "observations": len(values),
        "mean": values.mean(),
        "median": values.median(),
        "minimum": values.min(),
        "maximum": values.max(),
        "standard_deviation": values.std(),
        "first_year": df["year"].min(),
        "last_year": df["year"].max()
    }

    return result


def interpret_descriptive_summary(result):
    return (
        f"Across {result['observations']} yearly observations, "
        f"the average value was {result['mean']:,.2f}. "
        f"The values ranged from {result['minimum']:,.2f} "
        f"to {result['maximum']:,.2f}. "
        f"The standard deviation was {result['standard_deviation']:,.2f}, "
        f"showing the amount of variation around the average."
    )


# 3. TREND ANALYSIS
def analyze_trend(data):
    """
    Perform linear trend analysis using scipy.stats.linregress.
    The year is treated as the independent variable and the
    agricultural metric as the dependent variable.
     
    Check the overall direction of the agricultural metric over time.

    A positive slope means the value generally increased.
    A negative slope means the value generally decreased.

    R-squared shows how well the straight-line trend represents
    the yearly data, while the p-value helps us judge whether
    the observed trend is statistically significant.
    """

    df = prepare_annual_series(data)

    x = df["year"]
    y = df["value"]

    result = stats.linregress(x, y)

    return {
        "slope": result.slope,
        "intercept": result.intercept,
        "r_squared": result.rvalue ** 2,
        "p_value": result.pvalue,
        "standard_error": result.stderr,
        "observations": len(df)
    }


def interpret_trend(result):
    """
    Interpret the direction, significance and strength of
    the observed linear trend.
    The explanation tells us:
    - whether the trend is increasing or decreasing
    - whether it is statistically significant
    - how well the trend line fits the data
    """

    slope = result["slope"]
    r_squared = result["r_squared"]
    p_value = result["p_value"]

    if slope > 0:
        direction = "increasing"
    elif slope < 0:
        direction = "decreasing"
    else:
        direction = "stable"

    if p_value < 0.05:
        significance = "statistically significant at the 5% level"
    else:
        significance = "not statistically significant at the 5% level"

    if r_squared >= 0.70:
        fit = "strong"
    elif r_squared >= 0.40:
        fit = "moderate"
    else:
        fit = "limited"

    return {
        "direction": direction,
        "significance": significance,
        "fit": fit,
        "summary": (
            f"The analysis shows an overall {direction} linear trend. "
            f"The estimated change is {slope:,.2f} units per year. "
            f"The trend is {significance}. "
            f"The model explains approximately {r_squared * 100:.2f}% "
            f"of the variation in the annual values, indicating a "
            f"{fit} linear fit."
        )
    }


# 4. EARLY PERIOD VS LATER PERIOD
def compare_period_trends(data):
    """
    Compare the average value in the first half and second half
    of the available years.
    This helps us see whether the typical value changed
    between the first half and second half of the dataset
    This is a descriptive comparison, not a hypothesis test.
    """

    df = prepare_annual_series(data)

    n = len(df)

    if n < 4:
        return None

    midpoint = n // 2

    first_period = df.iloc[:midpoint]
    second_period = df.iloc[midpoint:]

    first_mean = first_period["value"].mean()
    second_mean = second_period["value"].mean()

    difference = second_mean - first_mean

    if first_mean != 0:
        percentage_change = (difference / first_mean) * 100
    else:
        percentage_change = np.nan

    return {
        "first_period": (
            int(first_period["year"].min()),
            int(first_period["year"].max())
        ),
        "second_period": (
            int(second_period["year"].min()),
            int(second_period["year"].max())
        ),
        "first_mean": first_mean,
        "second_mean": second_mean,
        "difference": difference,
        "percentage_change": percentage_change
    }


def interpret_period_comparison(result):

    if result is None:
        return "There are not enough yearly observations for this comparison."

    if result["difference"] > 0:
        direction = "higher"
    elif result["difference"] < 0:
        direction = "lower"
    else:
        direction = "similar"

    return (
        f"The average value during "
        f"{result['second_period'][0]}–{result['second_period'][1]} "
        f"was {direction} than during "
        f"{result['first_period'][0]}–{result['first_period'][1]}. "
        f"The difference in averages was "
        f"{result['difference']:,.2f} units "
        f"({result['percentage_change']:.2f}%)."
    )


# 5. YEAR-OVER-YEAR CHANGE
def calculate_yearly_change(data):
    """
    Calculate absolute and percentage year-over-year changes.
    'change' shows the actual increase or decrease.
    'percentage_change' shows the same change as a percentage
    of the previous year's value.
    """

    df = prepare_annual_series(data)

    df["change"] = df["value"].diff()

    df["percentage_change"] = (
        df["value"].pct_change() * 100
    )

    return df


def interpret_yearly_changes(data):
    df = calculate_yearly_change(data)

    changes = df.dropna(subset=["change"])

    if changes.empty:
        return None

    largest_increase = changes.loc[changes["change"].idxmax()]
    largest_decrease = changes.loc[changes["change"].idxmin()]

    return {
        "largest_increase_year": int(largest_increase["year"]),
        "largest_increase": largest_increase["change"],
        "largest_increase_percentage": largest_increase["percentage_change"],

        "largest_decrease_year": int(largest_decrease["year"]),
        "largest_decrease": largest_decrease["change"],
        "largest_decrease_percentage": largest_decrease["percentage_change"],

        "summary": (
            f"The largest year-over-year increase occurred in "
            f"{int(largest_increase['year'])}, with an increase of "
            f"{largest_increase['change']:,.2f} "
            f"({largest_increase['percentage_change']:.2f}%). "
            f"The largest year-over-year decrease occurred in "
            f"{int(largest_decrease['year'])}, with a decrease of "
            f"{abs(largest_decrease['change']):,.2f} "
            f"({largest_decrease['percentage_change']:.2f}%)."
        )
    }


# 6. CORRELATION ANALYSIS
def analyze_correlation(data, column_x, column_y):
    """
    Calculate Pearson correlation between two numerical variables.
    Correlation indicates association and does not establish causation.
    
    A positive correlation means both variables generally
    increase or decrease together.

    A negative correlation means one generally increases
    when the other decreases.

    Correlation shows a relationship between variables.
    It does not prove that one variable causes the other
    """

    df = data[[column_x, column_y]].copy()

    df[column_x] = pd.to_numeric(df[column_x], errors="coerce")
    df[column_y] = pd.to_numeric(df[column_y], errors="coerce")

    df = df.dropna()

    if len(df) < 3:
        return None

    correlation, p_value = stats.pearsonr(
        df[column_x],
        df[column_y]
    )

    return {
        "correlation": correlation,
        "p_value": p_value,
        "observations": len(df)
    }


def interpret_correlation(result):
    """
    Interpret Pearson correlation.
    """

    if result is None:
        return "Not enough valid observations for correlation analysis."

    r = result["correlation"]
    p = result["p_value"]

    if abs(r) >= 0.70:
        strength = "strong"
    elif abs(r) >= 0.40:
        strength = "moderate"
    elif abs(r) >= 0.20:
        strength = "weak"
    else:
        strength = "very weak"

    if r > 0:
        direction = "positive"
    elif r < 0:
        direction = "negative"
    else:
        direction = "no"

    if p < 0.05:
        significance = "statistically significant"
    else:
        significance = "not statistically significant"

    return (
        f"The analysis indicates a {strength} {direction} association "
        f"(r = {r:.3f}). The relationship is {significance} "
        f"at the 5% level. Correlation describes association and "
        f"does not by itself establish causation."
    )

# 7. FORECAST ERROR CALCULATION
def calculate_errors(actual, predicted):
    """
    Measure how close the forecast values were to the actual values.

    MAE  = average size of the prediction error.
    RMSE = gives more weight to larger prediction errors.
    MAPE = average percentage error.

    Lower error values generally mean the forecasts were
    closer to the actual values.
    """

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = actual - predicted

    mae = np.mean(np.abs(error))

    rmse = np.sqrt(np.mean(error ** 2))

    # MAPE is only calculated for non-zero actual values.
    non_zero = actual != 0

    if np.any(non_zero):
        mape = (
            np.mean(
                np.abs(
                    (actual[non_zero] - predicted[non_zero])
                    / actual[non_zero]
                )
            ) * 100
        )
    else:
        mape = np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }

# 8. HOLT FORECASTING MODEL
def fit_holt_model(data, periods=5):
    """
    Use Holt's linear trend method to forecast future yearly values.

    Holt's method is useful when the data has a trend over time.

    We do not include seasonality because our dataset contains
    yearly observations rather than monthly or quarterly data.
    """

    df = prepare_annual_series(data)

    series = df.set_index("year")["value"]

    model = Holt(
        series,
        initialization_method="estimated"
    )

    fitted_model = model.fit(optimized=True)

    forecast = fitted_model.forecast(periods)

    return {
        "model": fitted_model,
        "historical": df,
        "forecast": forecast
    }


# 9. CHRONOLOGICAL VALIDATION
def rolling_origin_validation(data, test_size=5):
    """
    Test how well the forecasting methods perform on recent
    historical data.

    The older years are used to train the models.
    The last few years are kept aside for testing.

    This keeps the data in time order, which is important
    for time-series forecasting.
    """

    df = prepare_annual_series(data)

    if len(df) <= test_size + 2:
        return None

    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]

    train_series = train.set_index("year")["value"]

    actual = test["value"].to_numpy()


    # Baseline: last observed value
    baseline_value = train_series.iloc[-1]

    baseline_predictions = np.repeat(
        baseline_value,
        len(test)
    )

    baseline_errors = calculate_errors(
        actual,
        baseline_predictions
    )

    # Holt linear trend
    model = Holt(
        train_series,
        initialization_method="estimated"
    )

    fitted_model = model.fit(optimized=True)

    holt_predictions = np.asarray(
        fitted_model.forecast(len(test))
    )

    holt_errors = calculate_errors(
        actual,
        holt_predictions
    )

    validation = {
        "train_start": int(train["year"].min()),
        "train_end": int(train["year"].max()),

        "test_start": int(test["year"].min()),
        "test_end": int(test["year"].max()),

        "actual": actual,

        "baseline_predictions": baseline_predictions,
        "holt_predictions": holt_predictions,

        "baseline_errors": baseline_errors,
        "holt_errors": holt_errors
    }

    return validation


def interpret_model_evaluation(validation):
    """
    Compare the baseline and Holt model using validation errors.
    """

    if validation is None:
        return "Not enough historical observations for model validation."

    baseline_mae = validation["baseline_errors"]["mae"]
    holt_mae = validation["holt_errors"]["mae"]

    baseline_rmse = validation["baseline_errors"]["rmse"]
    holt_rmse = validation["holt_errors"]["rmse"]

    if holt_mae < baseline_mae:
        mae_result = "Holt produced a lower MAE"
    elif holt_mae > baseline_mae:
        mae_result = "the baseline produced a lower MAE"
    else:
        mae_result = "both models produced the same MAE"

    if holt_rmse < baseline_rmse:
        rmse_result = "Holt produced a lower RMSE"
    elif holt_rmse > baseline_rmse:
        rmse_result = "the baseline produced a lower RMSE"
    else:
        rmse_result = "both models produced the same RMSE"

    return (
        f"For the validation period "
        f"{validation['test_start']}–{validation['test_end']}, "
        f"{mae_result}. "
        f"{rmse_result}. "
        f"These results describe performance on the historical "
        f"validation period and do not guarantee future accuracy."
    )


# 10. FINAL FORECAST
def forecast_annual(data, periods=5, validation_size=5):
    """
    Select the better-performing forecasting method based on
    historical validation, then use that method to forecast
    future years.

    The forecast is based on historical patterns and should
    be treated as an estimate, not a guaranteed future value.
    """

    validation = rolling_origin_validation(
        data,
        test_size=validation_size
    )

    if validation is None:
        return None

    baseline_mae = validation["baseline_errors"]["mae"]
    holt_mae = validation["holt_errors"]["mae"]

    # Select the model with the lower validation MAE.
    if holt_mae < baseline_mae:
        selected_model = "Holt Linear Trend"
    else:
        selected_model = "Last Value Baseline"

    df = prepare_annual_series(data)

    series = df.set_index("year")["value"]

    if selected_model == "Holt Linear Trend":

        model = Holt(
            series,
            initialization_method="estimated"
        )

        fitted_model = model.fit(optimized=True)

        forecast_values = np.asarray(
            fitted_model.forecast(periods)
        )

    else:

        forecast_values = np.repeat(
            series.iloc[-1],
            periods
        )

    last_year = int(df["year"].max())

    forecast_years = list(
        range(
            last_year + 1,
            last_year + periods + 1
        )
    )

    forecast_table = pd.DataFrame({
        "year": forecast_years,
        "forecast": forecast_values
    })

    result = {
        "selected_model": selected_model,
        "forecast": forecast_table,
        "validation": validation,
        "interpretation": interpret_forecast(
            selected_model,
            forecast_table,
            validation
        )
    }

    return result


# 11. FORECAST INTERPRETATION
def interpret_forecast(model_name, forecast_table, validation):
    """
    Interpret the selected forecasting model and projected values.
    """

    first_forecast = forecast_table["forecast"].iloc[0]
    last_forecast = forecast_table["forecast"].iloc[-1]

    if last_forecast > first_forecast:
        forecast_direction = "increase"
    elif last_forecast < first_forecast:
        forecast_direction = "decrease"
    else:
        forecast_direction = "remain approximately stable"

    baseline_mae = validation["baseline_errors"]["mae"]
    holt_mae = validation["holt_errors"]["mae"]

    if model_name == "Holt Linear Trend":
        model_statement = (
            "Holt's linear trend model was selected because it produced "
            "a lower MAE than the last-value baseline on the historical "
            "validation period."
        )
    else:
        model_statement = (
            "The last-value baseline was selected because it produced "
            "a lower or equal MAE compared with Holt's model on the "
            "historical validation period."
        )

    return (
        f"{model_statement} "
        f"The forecast indicates an overall {forecast_direction} "
        f"from approximately {first_forecast:,.2f} in "
        f"{int(forecast_table['year'].iloc[0])} to "
        f"{last_forecast:,.2f} in "
        f"{int(forecast_table['year'].iloc[-1])}. "
        f"The validation MAE was {min(baseline_mae, holt_mae):,.2f} "
        f"for the selected model. "
        f"These forecasts are projections based on historical patterns "
        f"and should not be interpreted as guaranteed future values."
    )