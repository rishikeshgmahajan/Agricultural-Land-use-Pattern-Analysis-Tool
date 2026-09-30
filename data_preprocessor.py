from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).with_name("classification-of-area.csv")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data.columns = data.columns.str.strip().str.lower()

    required_cols = {"year", "state_name", "land_use_metrics", "area"}
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_cols))}")

    text_columns = [
        "state_name",
        "district_name",
        "land_use_metrics",
        "area_classification",
        "area_sub_classification",
    ]

    for col in text_columns:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()
            data[col] = data[col].replace({"nan": None, "NaN": None, "": None})

    data["state_name"] = data["state_name"].fillna("Unknown")
    if "district_name" in data.columns:
        data["district_name"] = data["district_name"].fillna("Unknown")

    metric_col = data["land_use_metrics"]
    sub_col = (
        data["area_sub_classification"]
        if "area_sub_classification" in data.columns
        else pd.Series(None, index=data.index)
    )
    class_col = (
        data["area_classification"]
        if "area_classification" in data.columns
        else pd.Series(None, index=data.index)
    )

    data["category"] = (
        metric_col.fillna(sub_col).fillna(class_col).fillna("Unspecified")
    )

    data["start_year"] = (
        data["year"].astype(str).str.split("-").str[0].str.extract(r"(\d{4})")[0]
    )
    data["start_year"] = pd.to_numeric(data["start_year"], errors="coerce").astype(
        "Int64"
    )

    for code_col in ["state_code", "district_code"]:
        if code_col in data.columns:
            data[code_col] = pd.to_numeric(data[code_col], errors="coerce").astype(
                "Int64"
            )

    data["area"] = pd.to_numeric(data["area"], errors="coerce")
    data.loc[data["area"] < 0, "area"] = None

    return data


def create_member2_dataset(data: pd.DataFrame | None = None) -> pd.DataFrame:
    if data is None:
        data = pd.read_csv(DATA_PATH)
    return preprocess_data(data)


def create_member3_dataset(data: pd.DataFrame | None = None) -> pd.DataFrame:
    cleaned_df = create_member2_dataset(data)

    time_series = (
        cleaned_df.dropna(subset=["start_year", "land_use_metrics", "area"])
        .groupby(["land_use_metrics", "start_year"], as_index=False)["area"]
        .sum()
    )

    time_series = time_series.sort_values(
        ["land_use_metrics", "start_year"]
    ).reset_index(drop=True)

    time_series["abs_change"] = time_series.groupby("land_use_metrics")["area"].diff()
    time_series["pct_change"] = (
        time_series.groupby("land_use_metrics")["area"].pct_change() * 100
    )

    return time_series