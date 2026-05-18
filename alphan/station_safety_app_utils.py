from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
APP_ASSETS_DIR = BASE_DIR / "app_assets"

SITE_COLUMNS = [
    "site_id",
    "site_nr",
    "long",
    "lat",
    "naam",
    "domein",
    "wegnr",
    "district",
    "gemeente",
    "interval",
    "datum_van",
]

RADIUS_CONFIGS = {
    "1km_bike": {
        "label": "1.0 km",
        "meters": 1000,
        "monthly_dataset_path": BASE_DIR / "model_outputs/station_month_bike_crash_dataset_1km.csv",
        "next_month_dataset_path": BASE_DIR / "model_outputs/station_month_bike_crash_dataset_1km_next_month.csv",
        "crash_count_column": "bike_crash_count_1km",
        "crash_happened_column": "bike_crash_happened_1km",
        "target_class_column": "target_bike_crash_happened_next_month_1km",
        "target_count_column": "target_bike_crash_count_next_month_1km",
        "historical_rank_column": "historical_risk_rank",
        "rate_column": "smoothed_crashes_per_10000_cyclists",
        "model_filename": "next_month_catboost_bike_1km.cbm",
        "metadata_filename": "next_month_catboost_bike_1km_metadata.json",
        "importance_filename": "next_month_catboost_bike_1km_feature_importance.csv",
    },
}

EDITABLE_FEATURE_LABELS = {
    "site_avg_bike_crash_count_history": "Historical average monthly bike crashes",
    "bike_crash_count_1km_lag1": "Previous month bike crashes (1.0 km)",
    "bike_crash_count_1km_roll3_sum": "Previous 3-month bike crashes (1.0 km)",
    "bike_crash_count_1km_roll6_sum": "Previous 6-month bike crashes (1.0 km)",
    "bike_crashes_per_1000_cyclists_lag1": "Previous month bike crashes per 1,000 cyclists",
    "total_count_sum_lag1": "Previous month total cyclists",
    "total_count_in_lag1": "Previous month inbound cyclists",
    "total_count_out_lag1": "Previous month outbound cyclists",
    "avg_temperature_2m_lag1": "Previous month average temperature",
    "total_precipitation_lag1": "Previous month total precipitation",
    "avg_wind_speed_10m_lag1": "Previous month average wind speed",
    "avg_cloud_cover_lag1": "Previous month average cloud cover",
    "total_count_in_roll3_mean": "Previous 3-month avg inbound cyclists",
    "total_count_out_roll3_mean": "Previous 3-month avg outbound cyclists",
    "avg_temperature_2m_roll3_mean": "Previous 3-month avg temperature",
    "total_precipitation_roll3_mean": "Previous 3-month avg precipitation",
    "avg_wind_speed_10m_roll3_mean": "Previous 3-month avg wind speed",
    "avg_cloud_cover_roll3_mean": "Previous 3-month avg cloud cover",
}

EDITABLE_FEATURE_PRIORITY = [
    "site_avg_bike_crash_count_history",
    "bike_crashes_per_1000_cyclists_lag1",
    "total_count_sum_lag1",
    "total_count_in_lag1",
    "total_count_out_lag1",
    "avg_temperature_2m_lag1",
    "total_precipitation_lag1",
    "avg_wind_speed_10m_lag1",
    "avg_cloud_cover_lag1",
    "bike_crash_count_1km_lag1",
    "bike_crash_count_1km_roll3_sum",
    "bike_crash_count_1km_roll6_sum",
    "total_count_in_roll3_mean",
    "total_count_out_roll3_mean",
    "avg_temperature_2m_roll3_mean",
    "total_precipitation_roll3_mean",
    "avg_wind_speed_10m_roll3_mean",
    "avg_cloud_cover_roll3_mean",
]

NON_EDITABLE_FEATURES = {
    "site_id",
    "year",
    "month",
    "month_sin",
    "month_cos",
    "station_to_weather_grid_km",
    "weather_hours_available",
    "most_common_weather_code",
}


def safe_step(min_value: float, max_value: float) -> float:
    span = float(max_value) - float(min_value)
    if span <= 0:
        return 1.0
    return max(round(span / 100, 4), 0.01)


def humanize_radius_key(radius_key: str) -> str:
    return RADIUS_CONFIGS[radius_key]["label"]
