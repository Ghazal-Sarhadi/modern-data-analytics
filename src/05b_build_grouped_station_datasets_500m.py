from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRANSFORMED_DATA_DIR = BASE_DIR / "transformed_data"
OUTPUT_DIR = TRANSFORMED_DATA_DIR / "combined_station_groups_500m" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SITES_CSV = DATA_DIR / "sites.csv"
MONTHLY_500M_CSV = TRANSFORMED_DATA_DIR / "model_outputs" / "station_month_bike_crash_dataset_500m.csv"
MATCHED_500M_CSV = TRANSFORMED_DATA_DIR / "crash_outputs" / "crashes_bike_only_matched_to_sites_500m.csv"

GROUP_MAPPING_CSV = OUTPUT_DIR / "group_mapping.csv"
GROUPED_MONTHLY_CSV = OUTPUT_DIR / "station_group_month_bike_crash_dataset_500m.csv"
GROUPED_NEXT_MONTH_CSV = OUTPUT_DIR / "station_group_month_bike_crash_dataset_500m_next_month.csv"
GROUP_SUMMARY_TXT = OUTPUT_DIR / "group_dataset_summary.txt"

SITE_COLUMNS = [
    "site_id", "site_nr", "long", "lat", "naam", "domein",
    "wegnr", "district", "gemeente", "interval", "datum_van",
]

TRAFFIC_SUM_COLUMNS = [
    "interval_count_in", "interval_count_out",
    "total_count_in", "total_count_out",
    "zero_count_intervals_in", "zero_count_intervals_out",
]

TRAFFIC_MAX_COLUMNS = ["max_interval_count_in", "max_interval_count_out"]
TRAFFIC_MIN_COLUMNS = ["min_interval_count_in", "min_interval_count_out"]
DISTANCE_THRESHOLD_KM = 0.20


def normalize_name(name: str) -> str:
    text = str(name).lower().strip()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\b(debug|test|validatie)\b", "", text)
    text = re.sub(r"\b(teller|totem)\b", "", text)
    text = re.sub(r"\b[123]\b", "", text)
    text = re.sub(r"\b[a-z]*\d+[a-z0-9]*\b", "", text)

    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or str(name).lower().strip()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def connected_components(indices: list[int], coords: dict[int, tuple[float, float]]) -> list[list[int]]:
    remaining = set(indices)
    components: list[list[int]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            current = stack.pop()
            current_lat, current_lon = coords[current]
            neighbors = []
            for candidate in list(remaining):
                cand_lat, cand_lon = coords[candidate]
                if haversine_km(current_lat, current_lon, cand_lat, cand_lon) <= DISTANCE_THRESHOLD_KM:
                    neighbors.append(candidate)
            for neighbor in neighbors:
                remaining.remove(neighbor)
                stack.append(neighbor)
                component.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda comp: comp[0])


def build_group_mapping() -> pd.DataFrame:
    sites_df = pd.read_csv(SITES_CSV, names=SITE_COLUMNS)
    sites_df = sites_df[~sites_df["site_id"].isin([123, 144])]
    sites_df = sites_df[["site_id", "naam", "gemeente", "lat", "long", "datum_van"]].copy()
    sites_df["base_name"] = sites_df["naam"].map(normalize_name)
    coords = {
        int(row.site_id): (float(row.lat), float(row.long))
        for row in sites_df.itertuples(index=False)
    }
    mapping_rows = []
    next_group_id = 1
    for (_, base_name), group_df in sites_df.groupby(["gemeente", "base_name"], dropna=False):
        site_ids = sorted(group_df["site_id"].astype(int).tolist())
        components = connected_components(site_ids, coords)
        for component in components:
            component_df = group_df[group_df["site_id"].isin(component)].copy()
            representative_name = (
                component_df.assign(name_len=component_df["naam"].astype(str).str.len())
                .sort_values(["name_len", "site_id"])
                .iloc[0]["naam"]
            )
            for row in component_df.itertuples(index=False):
                mapping_rows.append({
                    "site_id": int(row.site_id),
                    "group_id": next_group_id,
                    "group_name": representative_name,
                    "group_base_name": base_name,
                    "member_site_count": len(component),
                    "group_lat": float(component_df["lat"].mean()),
                    "group_long": float(component_df["long"].mean()),
                    "gemeente": row.gemeente,
                    "datum_van": row.datum_van,
                })
            next_group_id += 1
    mapping_df = pd.DataFrame(mapping_rows).sort_values(["group_id", "site_id"]).reset_index(drop=True)
    mapping_df.to_csv(GROUP_MAPPING_CSV, index=False)
    return mapping_df


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    if not valid.any():
        return float("nan")
    weights_valid = weights[valid]
    if float(weights_valid.sum()) == 0:
        return float(values[valid].mean())
    return float((values[valid] * weights_valid).sum() / weights_valid.sum())


def first_mode(series: pd.Series):
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else pd.NA


def build_grouped_monthly(mapping_df: pd.DataFrame) -> pd.DataFrame:
    monthly_df = pd.read_csv(MONTHLY_500M_CSV)
    monthly_df = monthly_df.merge(mapping_df, on="site_id", how="inner")
    grouped_rows = []
    for (group_id, year, month), part in monthly_df.groupby(["group_id", "year", "month"], sort=True):
        row = {
            "group_id": int(group_id),
            "year": int(year),
            "month": int(month),
            "group_name": part["group_name"].iloc[0],
            "gemeente": part["gemeente"].iloc[0],
            "member_site_count": int(part["member_site_count"].max()),
            "group_lat": float(part["group_lat"].iloc[0]),
            "group_long": float(part["group_long"].iloc[0]),
        }
        for column in TRAFFIC_SUM_COLUMNS:
            row[column] = float(part[column].fillna(0).sum())
        for column in TRAFFIC_MAX_COLUMNS:
            row[column] = float(part[column].max())
        for column in TRAFFIC_MIN_COLUMNS:
            row[column] = float(part[column].min())
        row["mean_interval_count_in"] = weighted_mean(part["mean_interval_count_in"], part["interval_count_in"])
        row["mean_interval_count_out"] = weighted_mean(part["mean_interval_count_out"], part["interval_count_out"])
        row["weather_hours_available"] = float(part["weather_hours_available"].max())
        row["avg_temperature_2m"] = weighted_mean(part["avg_temperature_2m"], part["weather_hours_available"])
        row["min_temperature_2m"] = float(part["min_temperature_2m"].min())
        row["max_temperature_2m"] = float(part["max_temperature_2m"].max())
        row["avg_relative_humidity_2m"] = weighted_mean(part["avg_relative_humidity_2m"], part["weather_hours_available"])
        row["total_precipitation"] = weighted_mean(part["total_precipitation"], part["weather_hours_available"])
        row["total_rain"] = weighted_mean(part["total_rain"], part["weather_hours_available"])
        row["total_snowfall"] = weighted_mean(part["total_snowfall"], part["weather_hours_available"])
        row["avg_cloud_cover"] = weighted_mean(part["avg_cloud_cover"], part["weather_hours_available"])
        row["rainy_days"] = weighted_mean(part["rainy_days"], part["weather_hours_available"])
        row["heavy_rain_days"] = weighted_mean(part["heavy_rain_days"], part["weather_hours_available"])
        row["frost_days"] = weighted_mean(part["frost_days"], part["weather_hours_available"])
        row["storm_days"] = weighted_mean(part["storm_days"], part["weather_hours_available"])
        row["avg_wind_speed_10m"] = weighted_mean(part["avg_wind_speed_10m"], part["weather_hours_available"])
        row["most_common_weather_code"] = first_mode(part["most_common_weather_code"])
        row["station_to_weather_grid_km"] = weighted_mean(part["station_to_weather_grid_km"], part["weather_hours_available"])
        grouped_rows.append(row)

    grouped_monthly_df = pd.DataFrame(grouped_rows)
    matched_df = pd.read_csv(MATCHED_500M_CSV)
    matched_df = matched_df.merge(mapping_df[["site_id", "group_id"]], on="site_id", how="inner")
    crash_periods_df = (
        matched_df[["DT_YEAR_COLLISION", "DT_MONTH_COLLISION"]]
        .drop_duplicates()
        .rename(columns={"DT_YEAR_COLLISION": "year", "DT_MONTH_COLLISION": "month"})
    )
    crash_monthly_df = (
        matched_df.groupby(["group_id", "DT_YEAR_COLLISION", "DT_MONTH_COLLISION"], as_index=False)["crash_row_id"]
        .nunique()
        .rename(columns={
            "DT_YEAR_COLLISION": "year",
            "DT_MONTH_COLLISION": "month",
            "crash_row_id": "bike_crash_count_500m_grouped",
        })
    )
    crash_monthly_df["bike_crash_happened_500m_grouped"] = (
        crash_monthly_df["bike_crash_count_500m_grouped"] > 0
    ).astype(int)
    dataset_df = (
        grouped_monthly_df.merge(crash_periods_df, on=["year", "month"], how="inner")
        .merge(crash_monthly_df, on=["group_id", "year", "month"], how="left")
        .sort_values(["group_id", "year", "month"])
        .reset_index(drop=True)
    )
    dataset_df["bike_crash_count_500m_grouped"] = dataset_df["bike_crash_count_500m_grouped"].fillna(0)
    dataset_df["bike_crash_happened_500m_grouped"] = (
        dataset_df["bike_crash_happened_500m_grouped"].fillna(0).astype(int)
    )
    dataset_df["month_sin"] = dataset_df["month"].apply(lambda m: math.sin(2 * math.pi * m / 12))
    dataset_df["month_cos"] = dataset_df["month"].apply(lambda m: math.cos(2 * math.pi * m / 12))
    dataset_df.to_csv(GROUPED_MONTHLY_CSV, index=False)
    return dataset_df


def build_next_month_dataset(grouped_monthly_df: pd.DataFrame) -> pd.DataFrame:
    df = grouped_monthly_df.copy()
    df = df.sort_values(["group_id", "year", "month"]).copy()
    df["date"] = pd.to_datetime(
        df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    lag_feature_columns = [
        "total_count_in", "total_count_out",
        "mean_interval_count_in", "mean_interval_count_out",
        "zero_count_intervals_in", "zero_count_intervals_out",
        "avg_temperature_2m", "total_precipitation",
        "avg_wind_speed_10m", "avg_cloud_cover",
        "rainy_days", "heavy_rain_days", "frost_days", "storm_days",
        "bike_crash_count_500m_grouped",
    ]
    for column in lag_feature_columns:
        df[f"{column}_lag1"] = df.groupby("group_id")[column].shift(1)
        df[f"{column}_lag2"] = df.groupby("group_id")[column].shift(2)
        df[f"{column}_roll3_mean"] = (
            df.groupby("group_id")[column].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
        )
        df[f"{column}_roll6_mean"] = (
            df.groupby("group_id")[column].transform(lambda s: s.shift(1).rolling(window=6, min_periods=1).mean())
        )
        df[f"{column}_roll3_std"] = (
            df.groupby("group_id")[column].transform(lambda s: s.shift(1).rolling(window=3, min_periods=2).std())
        )
    df["bike_crash_happened_500m_grouped_lag1"] = df.groupby("group_id")["bike_crash_happened_500m_grouped"].shift(1)
    df["bike_crash_happened_500m_grouped_lag2"] = df.groupby("group_id")["bike_crash_happened_500m_grouped"].shift(2)
    df["bike_crash_count_500m_grouped_roll3_sum"] = (
        df.groupby("group_id")["bike_crash_count_500m_grouped"]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).sum())
    )
    df["bike_crash_count_500m_grouped_roll6_sum"] = (
        df.groupby("group_id")["bike_crash_count_500m_grouped"]
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
        1000 * df["bike_crash_count_500m_grouped_lag1"] / df["total_count_sum_lag1"].replace(0, pd.NA)
    )
    df["group_avg_bike_crash_count_history"] = (
        df.groupby("group_id")["bike_crash_count_500m_grouped"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df["group_avg_total_count_in_history"] = (
        df.groupby("group_id")["total_count_in"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df["group_avg_total_count_out_history"] = (
        df.groupby("group_id")["total_count_out"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df["target_bike_crash_count_next_month_500m_grouped"] = df.groupby("group_id")["bike_crash_count_500m_grouped"].shift(-1)
    df["target_bike_crash_happened_next_month_500m_grouped"] = (
        df.groupby("group_id")["bike_crash_happened_500m_grouped"].shift(-1)
    )
    df["target_date_next_month"] = df.groupby("group_id")["date"].shift(-1)
    df["expected_next_month"] = df["date"] + pd.offsets.MonthBegin(1)
    forecast_df = df[df["target_date_next_month"] == df["expected_next_month"]].copy()
    forecast_df = forecast_df.drop(columns=["date", "expected_next_month", "target_date_next_month"])
    forecast_df.to_csv(GROUPED_NEXT_MONTH_CSV, index=False)
    return forecast_df


def write_summary(mapping_df: pd.DataFrame, grouped_monthly_df: pd.DataFrame, grouped_next_df: pd.DataFrame) -> None:
    singleton_groups = int((mapping_df.groupby("group_id")["site_id"].count() == 1).sum())
    combined_groups = int((mapping_df.groupby("group_id")["site_id"].count() > 1).sum())
    positive_share = float(grouped_next_df["target_bike_crash_happened_next_month_500m_grouped"].mean())
    mean_count = float(grouped_monthly_df["bike_crash_count_500m_grouped"].mean())
    summary = [
        "Combined station group experiment — 500m",
        f"Original site count: {mapping_df['site_id'].nunique()}",
        f"Grouped counter count: {mapping_df['group_id'].nunique()}",
        f"Singleton groups: {singleton_groups}",
        f"Combined multi-site groups: {combined_groups}",
        f"Monthly grouped rows: {len(grouped_monthly_df)}",
        f"Next-month grouped rows: {len(grouped_next_df)}",
        f"Mean monthly grouped bike crash count: {mean_count:.4f}",
        f"Next-month positive target share: {positive_share:.4f}",
    ]
    GROUP_SUMMARY_TXT.write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    mapping_df = build_group_mapping()
    grouped_monthly_df = build_grouped_monthly(mapping_df)
    grouped_next_df = build_next_month_dataset(grouped_monthly_df)
    write_summary(mapping_df, grouped_monthly_df, grouped_next_df)
    print(f"Saved group mapping to: {GROUP_MAPPING_CSV}")
    print(f"Saved grouped monthly dataset to: {GROUPED_MONTHLY_CSV}")
    print(f"Saved grouped next-month dataset to: {GROUPED_NEXT_MONTH_CSV}")


if __name__ == "__main__":
    main()