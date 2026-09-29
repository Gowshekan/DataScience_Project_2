"""Analyze hourly restaurant demand and staffing using the bundled sample data."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "restaurant_staffing.csv"
SYNTHETIC_DATASET_SHA256 = "6253645F7C2EE7EE8BC24727FC03176AE6C51B43865D29B5EB96EE46482EF31A"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models"
REQUIRED_COLUMNS = [
    "Date", "Outlet_ID", "Hour", "Order_Volume", "Average_Preparation_Time",
    "Queue_Length", "Staff_Count", "Day_Type", "Holiday",
]
NUMERIC_COLUMNS = ["Order_Volume", "Average_Preparation_Time", "Queue_Length", "Staff_Count"]
FEATURE_COLUMNS = [
    "Hour", "Outlet_ID", "Order_Volume", "Average_Preparation_Time", "Staff_Count", "Day_Type", "Holiday",
]


def ensure_directories() -> None:
    for directory in (DATA_PATH.parent, OUTPUT_DIR, FIGURE_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def generate_synthetic_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Create a fixed-seed teaching dataset; it is not observed restaurant data."""
    rng = np.random.default_rng(2026)
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    outlets = {"OUTLET_A": 1.18, "OUTLET_B": 1.00, "OUTLET_C": 0.84}
    hourly_demand = {
        7: 7, 8: 10, 9: 9, 10: 9, 11: 14, 12: 23, 13: 27, 14: 22,
        15: 16, 16: 14, 17: 18, 18: 26, 19: 30, 20: 27, 21: 21, 22: 14,
    }
    synthetic_holidays = {
        pd.Timestamp(day)
        for day in ("2025-01-15", "2025-02-14", "2025-03-17", "2025-04-15", "2025-05-15", "2025-06-15")
    }
    rows = []
    for date in dates:
        day_type = "Weekend" if date.dayofweek >= 5 else "Weekday"
        holiday = date in synthetic_holidays
        calendar_multiplier = (1.16 if day_type == "Weekend" else 1.0) * (1.18 if holiday else 1.0)
        for outlet, outlet_multiplier in outlets.items():
            for hour, baseline in hourly_demand.items():
                order_volume = int(rng.poisson(baseline * calendar_multiplier * outlet_multiplier))
                planned_staff = max(2, round(2 + order_volume / 9.0 + (0.5 if day_type == "Weekend" else 0)))
                staff_count = max(1, planned_staff + int(rng.choice([-2, -1, 0, 0, 0, 1, 2])))
                orders_per_employee = order_volume / staff_count
                preparation_time = max(1.0, 4.0 + 0.25 * orders_per_employee + rng.normal(0, 0.75))
                queue_length = max(
                    0,
                    int(round(0.62 * orders_per_employee + 0.13 * preparation_time - 3.1 + rng.normal(0, 1.25))),
                )
                rows.append({
                    "Date": date.strftime("%Y-%m-%d"), "Outlet_ID": outlet, "Hour": hour,
                    "Order_Volume": order_volume, "Average_Preparation_Time": round(preparation_time, 2),
                    "Queue_Length": queue_length, "Staff_Count": staff_count,
                    "Day_Type": day_type, "Holiday": holiday,
                })
    data = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return data


def load_source_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, bool]:
    """Load the supplied CSV, or create the documented sample if it is absent."""
    if path.exists():
        data = pd.read_csv(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        return data, digest == SYNTHETIC_DATASET_SHA256
    warnings.warn(
        "No input CSV was found. Creating the bundled fixed-seed synthetic teaching dataset.",
        stacklevel=2,
    )
    return generate_synthetic_dataset(path), True


def inspect_data(data: pd.DataFrame) -> dict[str, object]:
    return {
        "shape": data.shape,
        "columns": data.columns.tolist(),
        "first_five_rows": data.head(),
        "data_types": data.dtypes.astype(str),
        "missing_values": data.isna().sum(),
        "duplicate_rows": int(data.duplicated().sum()),
    }


def clean_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    cleaned = data[REQUIRED_COLUMNS].copy().drop_duplicates()
    report = {"input_rows": len(data), "duplicates_removed": len(data) - len(cleaned)}
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")
    cleaned["Outlet_ID"] = cleaned["Outlet_ID"].astype("string").str.strip()
    cleaned.loc[cleaned["Outlet_ID"].eq(""), "Outlet_ID"] = pd.NA
    for column in ["Hour", *NUMERIC_COLUMNS]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned.loc[~cleaned["Hour"].between(0, 23), "Hour"] = np.nan
    for column in NUMERIC_COLUMNS:
        invalid = cleaned[column] < 0
        if column == "Average_Preparation_Time":
            invalid |= cleaned[column].eq(0)
        cleaned.loc[invalid, column] = np.nan
    cleaned["Day_Type"] = cleaned["Day_Type"].astype("string").str.strip().str.lower()
    cleaned["Day_Type"] = cleaned["Day_Type"].map(
        {"weekday": "Weekday", "week day": "Weekday", "weekend": "Weekend", "week end": "Weekend"}
    )
    derived_day_type = np.where(cleaned["Date"].dt.dayofweek >= 5, "Weekend", "Weekday")
    cleaned["Day_Type"] = cleaned["Day_Type"].fillna(pd.Series(derived_day_type, index=cleaned.index))
    holiday_values = cleaned["Holiday"].astype("string").str.strip().str.lower()
    cleaned["Holiday"] = holiday_values.map(
        {"true": True, "1": True, "yes": True, "y": True, "false": False, "0": False, "no": False, "n": False}
    ).fillna(False).astype(bool)
    invalid_key = cleaned["Date"].isna() | cleaned["Outlet_ID"].isna() | cleaned["Hour"].isna()
    report["rows_missing_valid_date_outlet_or_hour"] = int(invalid_key.sum())
    cleaned = cleaned.loc[~invalid_key].copy()
    cleaned["Hour"] = cleaned["Hour"].astype(int)
    for column in NUMERIC_COLUMNS:
        grouped_median = cleaned.groupby(["Outlet_ID", "Hour"], observed=True)[column].transform("median")
        fallback = cleaned[column].median()
        if pd.isna(fallback):
            raise ValueError(f"Column {column} has no valid numeric values to impute missing entries.")
        cleaned[column] = cleaned[column].fillna(grouped_median).fillna(fallback)
        if column in {"Order_Volume", "Queue_Length", "Staff_Count"}:
            cleaned[column] = cleaned[column].round().astype(int)
    cleaned["Date"] = cleaned["Date"].dt.strftime("%Y-%m-%d")
    cleaned["Day_Type"] = cleaned["Day_Type"].astype(str)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    report["output_rows"] = len(cleaned)
    report["missing_values_after_cleaning"] = int(cleaned.isna().sum().sum())
    return cleaned, report


def add_orders_per_employee(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    staff = result["Staff_Count"].to_numpy(dtype=float)
    result["Orders_Per_Employee"] = np.divide(
        result["Order_Volume"].to_numpy(dtype=float), staff,
        out=np.full(len(result), np.nan), where=staff > 0,
    )
    return result


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    return data[NUMERIC_COLUMNS].agg(["count", "mean", "median", "std", "min", "max"]).T.rename_axis("Variable").reset_index().round(3)


def hourly_summary(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby("Hour", as_index=False).agg(
        Order_Volume=("Order_Volume", "mean"),
        Average_Preparation_Time=("Average_Preparation_Time", "mean"),
        Queue_Length=("Queue_Length", "mean"),
        Staff_Count=("Staff_Count", "mean"),
        Orders_Per_Employee=("Orders_Per_Employee", "mean"),
    ).sort_values("Hour")


def create_visualizations(data: pd.DataFrame, actual: pd.Series | None = None, predicted: np.ndarray | None = None) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    saved = []

    def save_figure(name: str) -> None:
        destination = FIGURE_DIR / name
        plt.tight_layout()
        plt.savefig(destination, dpi=160, bbox_inches="tight")
        plt.close()
        saved.append(destination)

    hourly = hourly_summary(data)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=data, x="Hour", y="Order_Volume", hue="Day_Type", estimator="mean", errorbar=None, marker="o")
    plt.title("Average Order Volume by Hour and Day Type")
    plt.xlabel("Hour of day")
    plt.ylabel("Mean orders per outlet-hour")
    save_figure("order_volume_by_hour.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x="Staff_Count", y="Queue_Length", hue="Day_Type", alpha=0.45, s=28)
    plt.title("Staff Count vs Queue Length")
    plt.xlabel("Staff count")
    plt.ylabel("Queue length (queue-pressure indicator)")
    save_figure("staff_count_vs_queue_length.png")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=hourly, x="Hour", y="Orders_Per_Employee", color="#347c72")
    plt.title("Mean Orders per Employee by Hour")
    plt.xlabel("Hour of day")
    plt.ylabel("Orders per employee")
    save_figure("orders_per_employee.png")

    comparison = data.groupby("Outlet_ID", as_index=False).agg(
        Order_Volume=("Order_Volume", "mean"), Staff_Count=("Staff_Count", "mean"),
        Queue_Length=("Queue_Length", "mean"),
        Average_Preparation_Time=("Average_Preparation_Time", "mean"),
        Orders_Per_Employee=("Orders_Per_Employee", "mean"),
    )
    metrics = [
        ("Order_Volume", "Mean orders per outlet-hour", "Orders"),
        ("Staff_Count", "Mean staff count", "Staff"),
        ("Queue_Length", "Mean queue length", "Queue"),
        ("Average_Preparation_Time", "Mean preparation time", "Minutes"),
    ]
    _, axes = plt.subplots(2, 3, figsize=(14, 8))
    metrics.append(("Orders_Per_Employee", "Mean orders per employee", "Orders per employee"))
    for axis, (column, title, ylabel) in zip(axes.flat, metrics):
        sns.barplot(data=comparison, x="Outlet_ID", y=column, color="#d07c42", ax=axis)
        axis.set_title(title)
        axis.set_xlabel("Outlet")
        axis.set_ylabel(ylabel)
    axes.flat[-1].set_visible(False)
    save_figure("outlet_comparison.png")

    for column, title, ylabel, filename in [
        ("Average_Preparation_Time", "Average Preparation Time by Hour", "Mean preparation time", "preparation_time_by_hour.png"),
        ("Queue_Length", "Average Queue Length by Hour", "Mean queue length", "queue_length_by_hour.png"),
        ("Staff_Count", "Average Staff Count by Hour", "Mean staff count", "staff_count_by_hour.png"),
    ]:
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=data, x="Hour", y=column, hue="Day_Type", estimator="mean", errorbar=None, marker="o")
        plt.title(title)
        plt.xlabel("Hour of day")
        plt.ylabel(ylabel)
        save_figure(filename)

    if actual is not None and predicted is not None:
        plt.figure(figsize=(7, 6))
        sns.scatterplot(x=actual, y=predicted, alpha=0.55, color="#347c72")
        lower = min(float(actual.min()), float(np.min(predicted)))
        upper = max(float(actual.max()), float(np.max(predicted)))
        plt.plot([lower, upper], [lower, upper], linestyle="--", color="#c34f42", label="Perfect prediction")
        plt.title("Random Forest: Actual vs Predicted Queue Length")
        plt.xlabel("Actual queue length")
        plt.ylabel("Predicted queue length")
        plt.legend()
        save_figure("actual_vs_predicted_queue.png")
    return saved


def peak_demand_periods(data: pd.DataFrame) -> pd.DataFrame:
    dimensions = {
        "Hour": ["Hour"], "Outlet and hour": ["Outlet_ID", "Hour"],
        "Day type and hour": ["Day_Type", "Hour"], "Holiday and hour": ["Holiday", "Hour"],
    }
    peak_tables = []
    for level, keys in dimensions.items():
        grouped = data.groupby(keys, as_index=False, observed=True).agg(
            Order_Volume=("Order_Volume", "mean"), Staff_Count=("Staff_Count", "mean"),
            Queue_Length=("Queue_Length", "mean"),
            Average_Preparation_Time=("Average_Preparation_Time", "mean"),
            Orders_Per_Employee=("Orders_Per_Employee", "mean"),
            Observation_Count=("Order_Volume", "size"),
        )
        grouped = grouped.loc[grouped["Order_Volume"] >= grouped["Order_Volume"].quantile(0.90)].copy()
        for key in ["Hour", "Outlet_ID", "Day_Type", "Holiday"]:
            if key not in grouped:
                grouped[key] = pd.NA
        grouped["Analysis_Level"] = level
        peak_tables.append(grouped[[
            "Analysis_Level", "Hour", "Outlet_ID", "Day_Type", "Holiday", "Order_Volume",
            "Staff_Count", "Queue_Length", "Average_Preparation_Time", "Orders_Per_Employee", "Observation_Count",
        ]])
    return pd.concat(peak_tables, ignore_index=True).sort_values(
        ["Analysis_Level", "Order_Volume"], ascending=[True, False]
    ).reset_index(drop=True)


def classify_staffing(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    thresholds = {
        "orders_per_employee_q25": float(data["Orders_Per_Employee"].quantile(0.25)),
        "orders_per_employee_q75": float(data["Orders_Per_Employee"].quantile(0.75)),
        "queue_length_median": float(data["Queue_Length"].median()),
        "queue_length_q75": float(data["Queue_Length"].quantile(0.75)),
        "preparation_time_median": float(data["Average_Preparation_Time"].median()),
        "preparation_time_q75": float(data["Average_Preparation_Time"].quantile(0.75)),
    }

    def classify(row: pd.Series) -> str:
        workload = row["Orders_Per_Employee"]
        high_load = pd.notna(workload) and workload > thresholds["orders_per_employee_q75"]
        elevated_pressure = (
            row["Queue_Length"] > thresholds["queue_length_q75"]
            and row["Average_Preparation_Time"] > thresholds["preparation_time_median"]
        )
        low_load = pd.notna(workload) and workload < thresholds["orders_per_employee_q25"]
        low_pressure = (
            row["Queue_Length"] <= thresholds["queue_length_median"]
            and row["Average_Preparation_Time"] < thresholds["preparation_time_median"]
        )
        if high_load or elevated_pressure:
            return "Understaffed"
        if low_load and low_pressure:
            return "Overstaffed"
        return "Balanced"

    grouped = data.groupby(["Outlet_ID", "Hour", "Day_Type", "Holiday"], as_index=False, observed=True).agg(
        Order_Volume=("Order_Volume", "mean"), Staff_Count=("Staff_Count", "mean"),
        Queue_Length=("Queue_Length", "mean"),
        Average_Preparation_Time=("Average_Preparation_Time", "mean"),
        Orders_Per_Employee=("Orders_Per_Employee", "mean"),
        Observation_Count=("Order_Volume", "size"),
    )
    grouped["Staffing_Status"] = grouped.apply(classify, axis=1)
    return grouped, thresholds


def outlet_comparison(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby("Outlet_ID", as_index=False, observed=True).agg(
        Order_Volume=("Order_Volume", "mean"), Staff_Count=("Staff_Count", "mean"),
        Queue_Length=("Queue_Length", "mean"),
        Average_Preparation_Time=("Average_Preparation_Time", "mean"),
        Orders_Per_Employee=("Orders_Per_Employee", "mean"),
        Observation_Count=("Order_Volume", "size"),
    ).round(3)


def staffing_queue_relationship(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = ["Staff_Count", "Queue_Length", "Order_Volume", "Average_Preparation_Time"]
    correlations = data[variables].corr(method="pearson").round(3)
    grouped = data.groupby("Staff_Count", as_index=False, observed=True).agg(
        Queue_Length=("Queue_Length", "mean"), Order_Volume=("Order_Volume", "mean"),
        Average_Preparation_Time=("Average_Preparation_Time", "mean"),
        Observation_Count=("Queue_Length", "size"),
    )
    return correlations, grouped


def train_random_forest(data: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, np.ndarray]:
    model_data = data.copy()
    model_data["Date"] = pd.to_datetime(model_data["Date"])
    dates = np.sort(model_data["Date"].unique())
    if len(dates) < 2:
        raise ValueError("At least two distinct dates are required for a chronological train/test split.")
    cutoff_index = min(max(1, int(len(dates) * 0.8)), len(dates) - 1)
    split_date = pd.Timestamp(dates[cutoff_index])
    train = model_data[model_data["Date"] < split_date]
    test = model_data[model_data["Date"] >= split_date]
    categorical_features = ["Outlet_ID", "Day_Type", "Holiday"]
    numeric_features = [column for column in FEATURE_COLUMNS if column not in categorical_features]
    preprocessor = ColumnTransformer(transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", "passthrough", numeric_features),
    ])
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ])
    model.fit(train[FEATURE_COLUMNS], train["Queue_Length"])
    predictions = model.predict(test[FEATURE_COLUMNS])
    actual = test["Queue_Length"].astype(float).reset_index(drop=True)
    mse = mean_squared_error(actual, predictions)
    metrics = pd.DataFrame([{
        "Target": "Queue_Length", "Split": "Chronological 80/20 by unique date",
        "Train_Rows": len(train), "Test_Rows": len(test),
        "MAE": mean_absolute_error(actual, predictions), "MSE": mse,
        "RMSE": np.sqrt(mse), "R2": r2_score(actual, predictions),
    }]).round(4)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["regressor"].feature_importances_
    importance_table = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_table["Feature"] = importance_table["Feature"].str.replace(r"^(categorical|numeric)__", "", regex=True)
    importance_table = importance_table.sort_values("Importance", ascending=False).reset_index(drop=True).round(5)
    return model, metrics, importance_table, actual, predictions


def generate_insights(data: pd.DataFrame, peaks: pd.DataFrame, staffing: pd.DataFrame, outlets: pd.DataFrame, correlations: pd.DataFrame) -> list[str]:
    hourly = data.groupby("Hour")["Order_Volume"].mean()
    peak_hour = int(hourly.idxmax())
    peak_orders = float(hourly.max())
    day_demand = data.groupby("Day_Type")["Order_Volume"].mean()
    holiday_demand = data.groupby("Holiday")["Order_Volume"].mean()
    busiest_day_type = str(day_demand.idxmax())
    busiest_holiday_state = "holiday" if bool(holiday_demand.idxmax()) else "non-holiday"
    holiday_mean = float(holiday_demand.get(True, np.nan))
    nonholiday_mean = float(holiday_demand.get(False, np.nan))
    holiday_ratio = holiday_mean / nonholiday_mean if nonholiday_mean else np.nan
    queue_outlet = outlets.loc[outlets["Queue_Length"].idxmax()]
    workload_periods = data.groupby(["Outlet_ID", "Hour"], as_index=False)["Orders_Per_Employee"].mean()
    workload_peak = workload_periods.loc[workload_periods["Orders_Per_Employee"].idxmax()]
    under_share = float(staffing["Staffing_Status"].value_counts(normalize=True).get("Understaffed", 0.0) * 100)
    staff_queue_corr = float(correlations.loc["Staff_Count", "Queue_Length"])
    outlet_workload = outlets.loc[outlets["Orders_Per_Employee"].idxmax()]
    status_shares = staffing["Staffing_Status"].value_counts(normalize=True)
    over_share = float(status_shares.get("Overstaffed", 0.0) * 100)
    return [
        f"Mean demand peaks at {peak_hour:02d}:00, averaging {peak_orders:.2f} orders per outlet-hour.",
        f"{busiest_day_type} observations have the higher mean demand ({day_demand.max():.2f} orders per outlet-hour).",
        f"The {busiest_holiday_state} group has the higher mean demand ({holiday_demand.max():.2f} orders per outlet-hour; holiday/non-holiday ratio {holiday_ratio:.2f}).",
        f"{queue_outlet['Outlet_ID']} has the highest outlet-average queue length ({queue_outlet['Queue_Length']:.2f}) and averages {queue_outlet['Staff_Count']:.2f} staff.",
        f"The highest mean workload is at {workload_peak['Outlet_ID']} at {int(workload_peak['Hour']):02d}:00 ({workload_peak['Orders_Per_Employee']:.2f} orders per employee); {outlet_workload['Outlet_ID']} has the highest overall outlet-average workload ({outlet_workload['Orders_Per_Employee']:.2f}).",
        f"The data-derived classification marks {under_share:.1f}% of outlet-hour-day-type-holiday periods understaffed and {over_share:.1f}% overstaffed. The raw staff-count/queue-length Pearson correlation is {staff_queue_corr:.3f}; this is observational and does not establish causation.",
    ]


def create_action_plan(staffing: pd.DataFrame) -> pd.DataFrame:
    action_map = {
        "Understaffed": "Increase staffing during comparable periods and monitor queue pressure.",
        "Overstaffed": "Consider reallocating staff to higher-pressure periods; monitor service measures.",
        "Balanced": "Maintain current staffing and continue monitoring queue pressure.",
    }
    plan = staffing.copy()
    plan["Time_Period"] = plan["Hour"].map(lambda hour: f"{int(hour):02d}:00")
    demand_low, demand_high = plan["Order_Volume"].quantile([0.25, 0.75])
    queue_low, queue_high = plan["Queue_Length"].quantile([0.25, 0.75])
    plan["Demand_Situation"] = np.select(
        [plan["Order_Volume"] >= demand_high, plan["Order_Volume"] <= demand_low],
        ["High relative to observed periods", "Low relative to observed periods"],
        default="Middle range of observed periods",
    )
    plan["Queue_Situation"] = np.select(
        [plan["Queue_Length"] >= queue_high, plan["Queue_Length"] <= queue_low],
        ["High relative to observed periods", "Low relative to observed periods"],
        default="Middle range of observed periods",
    )
    plan["Recommended_Action"] = plan["Staffing_Status"].map(action_map)
    return plan[[
        "Time_Period", "Outlet_ID", "Day_Type", "Holiday", "Demand_Situation", "Order_Volume", "Staff_Count",
        "Orders_Per_Employee", "Queue_Length", "Queue_Situation", "Staffing_Status", "Recommended_Action", "Observation_Count",
    ]].sort_values(["Outlet_ID", "Day_Type", "Holiday", "Time_Period"]).reset_index(drop=True).round(3)


def run_analysis() -> dict[str, object]:
    ensure_directories()
    source, synthetic = load_source_data()
    inspection = inspect_data(source)
    clean, cleaning_report = clean_dataset(source)
    data = add_orders_per_employee(clean)
    data.to_csv(OUTPUT_DIR / "cleaned_dataset.csv", index=False)
    statistics = descriptive_statistics(data)
    statistics.to_csv(OUTPUT_DIR / "descriptive_statistics.csv", index=False)
    peaks = peak_demand_periods(data)
    peaks.to_csv(OUTPUT_DIR / "peak_demand_periods.csv", index=False)
    staffing, thresholds = classify_staffing(data)
    staffing.to_csv(OUTPUT_DIR / "staffing_analysis.csv", index=False)
    outlets = outlet_comparison(data)
    outlets.to_csv(OUTPUT_DIR / "outlet_comparison.csv", index=False)
    correlations, grouped_relationship = staffing_queue_relationship(data)
    correlations.to_csv(OUTPUT_DIR / "staff_queue_correlations.csv")
    grouped_relationship.to_csv(OUTPUT_DIR / "staff_queue_grouped_statistics.csv", index=False)
    model, metrics, importances, actual, predictions = train_random_forest(data)
    joblib.dump(model, MODEL_DIR / "random_forest_model.pkl")
    evaluation = metrics.copy()
    evaluation["Top_Features"] = "; ".join(
        f"{row.Feature} ({row.Importance:.3f})" for row in importances.head(5).itertuples()
    )
    evaluation.to_csv(OUTPUT_DIR / "model_evaluation.csv", index=False)
    importances.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    figures = create_visualizations(data, actual, predictions)
    insights = generate_insights(data, peaks, staffing, outlets, correlations)
    with (OUTPUT_DIR / "key_insights.txt").open("w", encoding="utf-8") as file:
        provenance = "Fixed-seed synthetic teaching data; not observed restaurant data." if synthetic else "Dataset read from the project CSV."
        file.write(f"DATA PROVENANCE: {provenance}\n\n")
        for number, insight in enumerate(insights, start=1):
            file.write(f"{number}. {insight}\n")
    action_plan = create_action_plan(staffing)
    action_plan.to_csv(OUTPUT_DIR / "practical_action_plan.csv", index=False)
    return {
        "source": source, "synthetic": synthetic, "inspection": inspection, "cleaned": data,
        "cleaning_report": cleaning_report, "statistics": statistics, "hourly": hourly_summary(data),
        "peaks": peaks, "staffing": staffing, "thresholds": thresholds, "outlets": outlets,
        "correlations": correlations, "grouped_relationship": grouped_relationship,
        "model": model, "metrics": metrics, "importances": importances, "actual": actual,
        "predictions": predictions, "figures": figures, "insights": insights, "action_plan": action_plan,
    }


if __name__ == "__main__":
    results = run_analysis()
    print("Synthetic sample generated:" if results["synthetic"] else "Source dataset loaded:", DATA_PATH)
    print("Rows analyzed:", len(results["cleaned"]))
    print("Cleaning report:", results["cleaning_report"])
    print("\nModel evaluation:\n", results["metrics"].to_string(index=False))
    print("\nKey insights:")
    for insight in results["insights"]:
        print("-", insight)
    print("\nGenerated figures:", len(results["figures"]))
