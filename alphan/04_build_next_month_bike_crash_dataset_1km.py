from pathlib import Path

import pandas as pd


transformed_data_dir = Path("transformed_data")
input_path = transformed_data_dir / "model_outputs" / "station_month_bike_crash_dataset_1km.csv"
output_path = transformed_data_dir / "model_outputs" / "station_month_bike_crash_dataset_1km_next_month.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)


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
    "bike_crash_count_1km",
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

df["bike_crash_happened_1km_lag1"] = df.groupby("site_id")["bike_crash_happened_1km"].shift(1)
df["bike_crash_happened_1km_lag2"] = df.groupby("site_id")["bike_crash_happened_1km"].shift(2)
df["bike_crash_count_1km_roll3_sum"] = (
    df.groupby("site_id")["bike_crash_count_1km"]
    .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).sum())
)
df["bike_crash_count_1km_roll6_sum"] = (
    df.groupby("site_id")["bike_crash_count_1km"]
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
    1000 * df["bike_crash_count_1km_lag1"] / df["total_count_sum_lag1"].replace(0, pd.NA)
)

df["site_avg_bike_crash_count_history"] = (
    df.groupby("site_id")["bike_crash_count_1km"]
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

df["target_bike_crash_count_next_month_1km"] = df.groupby("site_id")["bike_crash_count_1km"].shift(-1)
df["target_bike_crash_happened_next_month_1km"] = df.groupby("site_id")["bike_crash_happened_1km"].shift(-1)
df["target_date_next_month"] = df.groupby("site_id")["date"].shift(-1)

df["expected_next_month"] = df["date"] + pd.offsets.MonthBegin(1)
forecast_df = df[df["target_date_next_month"] == df["expected_next_month"]].copy()

forecast_df = forecast_df.drop(columns=["date", "expected_next_month", "target_date_next_month"])
forecast_df.to_csv(output_path, index=False)

print(f"Built next-month bike-only 1 km forecasting dataset: {output_path}")
