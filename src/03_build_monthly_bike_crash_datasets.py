# %%
#Libraries
from pathlib import Path
import math

import pandas as pd


# %%
# Paths
base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
transformed_data_dir = base_dir / "transformed_data"
weather_dir = transformed_data_dir / "weather_outputs"
crash_dir = transformed_data_dir / "crash_outputs"
output_dir = transformed_data_dir / "model_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

weather_csv = weather_dir / "station_weather_hourly.csv"

matched_bike_500m_csv = crash_dir / "crashes_bike_only_matched_to_sites_500m.csv"
matched_bike_1km_csv = crash_dir / "crashes_bike_only_matched_to_sites_1km.csv"
matched_bike_2_5km_csv = crash_dir / "crashes_bike_only_matched_to_sites_2_5km.csv"

station_month_bike_500m_csv = output_dir / "station_month_bike_crash_dataset_500m.csv"
station_month_bike_1km_csv = output_dir / "station_month_bike_crash_dataset_1km.csv"
station_month_bike_2_5km_csv = output_dir / "station_month_bike_crash_dataset_2_5km.csv"


def build_traffic_monthly() -> pd.DataFrame:
    data_columns = ["site_id", "richting", "type", "van", "tot", "aantal"]
    traffic_monthly_parts = []

    for data_file in sorted(data_dir.glob("data-*.csv")):
        for chunk in pd.read_csv(data_file, names=data_columns, chunksize=200_000):
            fietsers_chunk = chunk[
                (chunk["type"] == "FIETSERS") & 
                (~chunk["site_id"].isin([123, 144]))
            ].copy()

            if fietsers_chunk.empty:
                continue

            fietsers_chunk["van"] = pd.to_datetime(fietsers_chunk["van"], errors="coerce")
            fietsers_chunk = fietsers_chunk.dropna(subset=["van"])
            fietsers_chunk["year"] = fietsers_chunk["van"].dt.year
            fietsers_chunk["month"] = fietsers_chunk["van"].dt.month

            grouped_chunk = (
                fietsers_chunk.groupby(["site_id", "year", "month", "richting"])["aantal"]
                .agg(
                    total_count="sum",
                    mean_interval_count="mean",
                    max_interval_count="max",
                    min_interval_count="min",
                    zero_count_intervals=lambda s: (s == 0).sum(),
                    interval_count="count",
                )
                .reset_index()
            )
            traffic_monthly_parts.append(grouped_chunk)

    traffic_monthly_long_df = (
        pd.concat(traffic_monthly_parts, ignore_index=True)
        .groupby(["site_id", "year", "month", "richting"], as_index=False)
        .agg(
            total_count=("total_count", "sum"),
            mean_interval_count=("mean_interval_count", "mean"),
            max_interval_count=("max_interval_count", "max"),
            min_interval_count=("min_interval_count", "min"),
            zero_count_intervals=("zero_count_intervals", "sum"),
            interval_count=("interval_count", "sum"),
        )
    )

    traffic_monthly_df = (
        traffic_monthly_long_df.pivot_table(
            index=["site_id", "year", "month"],
            columns="richting",
            values=[
                "total_count",
                "mean_interval_count",
                "max_interval_count",
                "min_interval_count",
                "zero_count_intervals",
                "interval_count",
            ],
            aggfunc="first",
        )
        .sort_index(axis=1)
    )
    traffic_monthly_df.columns = [
        f"{metric.lower()}_{direction.lower().replace('/', '_')}"
        for metric, direction in traffic_monthly_df.columns
    ]
    return traffic_monthly_df.reset_index()


def build_weather_monthly() -> pd.DataFrame:
    weather_use_columns = [
        "site_id",
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "snowfall",
        "cloud_cover",
        "wind_speed_10m",
        "weather_code",
        "station_to_weather_grid_km",
    ]

    weather_df = pd.read_csv(weather_csv, usecols=weather_use_columns)
    weather_df["time"] = pd.to_datetime(weather_df["time"], errors="coerce")
    weather_df = weather_df.dropna(subset=["time"])
    weather_df["year"] = weather_df["time"].dt.year
    weather_df["month"] = weather_df["time"].dt.month

    return (
        weather_df.groupby(["site_id", "year", "month"], as_index=False)
        .agg(
            weather_hours_available=("time", "count"),
            avg_temperature_2m=("temperature_2m", "mean"),
            min_temperature_2m=("temperature_2m", "min"),
            max_temperature_2m=("temperature_2m", "max"),
            avg_relative_humidity_2m=("relative_humidity_2m", "mean"),
            total_precipitation=("precipitation", "sum"),
            total_rain=("rain", "sum"),
            total_snowfall=("snowfall", "sum"),
            avg_cloud_cover=("cloud_cover", "mean"),
            avg_wind_speed_10m=("wind_speed_10m", "mean"),
            most_common_weather_code=("weather_code", lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA),
            station_to_weather_grid_km=("station_to_weather_grid_km", "first"),
            rainy_days=("precipitation", lambda x: (x > 0).sum()),
            heavy_rain_days=("precipitation", lambda x: (x > 10).sum()),
            frost_days=("temperature_2m", lambda x: (x < 0).sum()),
            storm_days=("wind_speed_10m", lambda x: (x > 50).sum()),
        )
    )


def build_bike_crash_dataset(
    matched_crashes_csv: Path,
    count_column: str,
    happened_column: str,
    output_csv: Path,
    traffic_monthly_df: pd.DataFrame,
    weather_monthly_df: pd.DataFrame,
) -> pd.DataFrame:
    matched_crashes_df = pd.read_csv(matched_crashes_csv)

    crash_periods_df = (
        matched_crashes_df[["DT_YEAR_COLLISION", "DT_MONTH_COLLISION"]]
        .drop_duplicates()
        .rename(columns={"DT_YEAR_COLLISION": "year", "DT_MONTH_COLLISION": "month"})
    )

    crash_monthly_df = (
        matched_crashes_df.groupby(["site_id", "DT_YEAR_COLLISION", "DT_MONTH_COLLISION"], as_index=False)
        .agg(**{count_column: ("crash_row_id", "count")})
        .rename(columns={"DT_YEAR_COLLISION": "year", "DT_MONTH_COLLISION": "month"})
    )
    crash_monthly_df[happened_column] = (crash_monthly_df[count_column] > 0).astype(int)

    dataset_df = (
        traffic_monthly_df.merge(weather_monthly_df, on=["site_id", "year", "month"], how="inner")
        .merge(crash_periods_df, on=["year", "month"], how="inner")
        .merge(crash_monthly_df, on=["site_id", "year", "month"], how="left")
    )

    dataset_df[count_column] = dataset_df[count_column].fillna(0)
    dataset_df[happened_column] = dataset_df[happened_column].fillna(0).astype(int)

    dataset_df["month_sin"] = dataset_df["month"].apply(lambda m: math.sin(2 * math.pi * m / 12))
    dataset_df["month_cos"] = dataset_df["month"].apply(lambda m: math.cos(2 * math.pi * m / 12))

    dataset_df.to_csv(output_csv, index=False)
    return dataset_df


# %%
# Shared monthly inputs
traffic_monthly_df = build_traffic_monthly()
weather_monthly_df = build_weather_monthly()


# %%
# Build bike-only monthly datasets for both radii.
bike_500m_df = build_bike_crash_dataset(
    matched_crashes_csv=matched_bike_500m_csv,
    count_column="bike_crash_count_500m",
    happened_column="bike_crash_happened_500m",
    output_csv=station_month_bike_500m_csv,
    traffic_monthly_df=traffic_monthly_df,
    weather_monthly_df=weather_monthly_df,
)

bike_1km_df = build_bike_crash_dataset(
    matched_crashes_csv=matched_bike_1km_csv,
    count_column="bike_crash_count_1km",
    happened_column="bike_crash_happened_1km",
    output_csv=station_month_bike_1km_csv,
    traffic_monthly_df=traffic_monthly_df,
    weather_monthly_df=weather_monthly_df,
)

bike_2_5km_df = build_bike_crash_dataset(
    matched_crashes_csv=matched_bike_2_5km_csv,
    count_column="bike_crash_count_2_5km",
    happened_column="bike_crash_happened_2_5km",
    output_csv=station_month_bike_2_5km_csv,
    traffic_monthly_df=traffic_monthly_df,
    weather_monthly_df=weather_monthly_df,
)


# %%
print(
    "Built bike-only monthly datasets. "
    f"500 m rows={len(bike_500m_df)}, "
    f"1.0 km rows={len(bike_1km_df)}, "
    f"2.5 km rows={len(bike_2_5km_df)}"
)

# %%
