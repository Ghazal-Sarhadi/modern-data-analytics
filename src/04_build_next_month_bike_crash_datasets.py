from pathlib import Path
import pandas as pd

transformed_data_dir = Path("transformed_data")


def build_next_month_dataset(radius: str) -> None:
    input_path = transformed_data_dir / "model_outputs" / f"station_month_bike_crash_dataset_{radius}.csv"
    output_path = transformed_data_dir / "model_outputs" / f"station_month_bike_crash_dataset_{radius}_next_month.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crash_count_col = f"bike_crash_count_{radius}"
    crash_happened_col = f"bike_crash_happened_{radius}"

    df = pd.read_csv(input_path)
    df = df.sort_values(["site_id", "year", "month"]).copy()
    df["date"] = pd.to_datetime(
        df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str).str.zfill(2) + "-01"
    )

    lag_feature_columns = [
        "total_count_in",
        "total_count_out",
        "mean_interval_count_in",
        "mean_interval_count_out",
        "zero_count_intervals_in",
        "zero_count_intervals_out",
        "avg_temperature_2m",
        "total_precipitation",
        "avg_wind_speed_10m",
        "avg_cloud_cover",
        "rainy_days",
        "heavy_rain_days",
        "frost_days",
        "storm_days",
        crash_count_col,
    ]

    for column in lag_feature_columns:
        df[f"{column}_lag1"] = df.groupby("site_id")[column].shift(1)
        df[f"{column}_lag2"] = df.groupby("site_id")[column].shift(2)
        df[f"{column}_roll3_mean"] = (
            df.groupby("site_id")[column]
            .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
        )
        df[f"{column}_roll6_mean"] = (
            df.groupby("site_id")[column]
            .transform(lambda s: s.shift(1).rolling(window=6, min_periods=1).mean())
        )
        df[f"{column}_roll3_std"] = (
            df.groupby("site_id")[column]
            .transform(lambda s: s.shift(1).rolling(window=3, min_periods=2).std())
        )

    df[f"{crash_happened_col}_lag1"] = df.groupby("site_id")[crash_happened_col].shift(1)
    df[f"{crash_happened_col}_lag2"] = df.groupby("site_id")[crash_happened_col].shift(2)
    df[f"{crash_count_col}_roll3_sum"] = (
        df.groupby("site_id")[crash_count_col]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).sum())
    )
    df[f"{crash_count_col}_roll6_sum"] = (
        df.groupby("site_id")[crash_count_col]
        .transform(lambda s: s.shift(1).rolling(window=6, min_periods=1).sum())
    )

    df["total_count_in_change_lag1"] = df["total_count_in"] - df["total_count_in_lag1"]
    df["total_count_out_change_lag1"] = df["total_count_out"] - df["total_count_out_lag1"]
    df["avg_temperature_2m_change_lag1"] = df["avg_temperature_2m"] - df["avg_temperature_2m_lag1"]
    df["total_precipitation_change_lag1"] = df["total_precipitation"] - df["total_precipitation_lag1"]
    df["in_out_ratio"] = df["total_count_in"] / df["total_count_out"].replace(0, pd.NA)
    df["in_out_ratio_lag1"] = df["total_count_in_lag1"] / df["total_count_out_lag1"].replace(0, pd.NA)
    df["total_count_sum"] = df["total_count_in"] + df["total_count_out"]
    df["total_count_sum_lag1"] = df["total_count_in_lag1"] + df["total_count_out_lag1"]
    df["bike_crashes_per_1000_cyclists_lag1"] = (
        1000 * df[f"{crash_count_col}_lag1"] / df["total_count_sum_lag1"].replace(0, pd.NA)
    )

    df["site_avg_bike_crash_count_history"] = (
        df.groupby("site_id")[crash_count_col]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df["site_avg_total_count_in_history"] = (
        df.groupby("site_id")["total_count_in"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df["site_avg_total_count_out_history"] = (
        df.groupby("site_id")["total_count_out"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )

    df[f"target_{crash_count_col}_next_month"] = df.groupby("site_id")[crash_count_col].shift(-1)
    df[f"target_{crash_happened_col}_next_month"] = df.groupby("site_id")[crash_happened_col].shift(-1)
    df["target_date_next_month"] = df.groupby("site_id")["date"].shift(-1)

    df["expected_next_month"] = df["date"] + pd.offsets.MonthBegin(1)
    forecast_df = df[df["target_date_next_month"] == df["expected_next_month"]].copy()
    forecast_df = forecast_df.drop(columns=["date", "expected_next_month", "target_date_next_month"])
    forecast_df.to_csv(output_path, index=False)
    print(f"Built next-month dataset for {radius}: {output_path}")


for radius in ["500m", "1km", "2_5km"]:
    build_next_month_dataset(radius)