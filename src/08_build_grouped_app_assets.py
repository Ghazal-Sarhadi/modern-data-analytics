from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
TRANSFORMED_DATA_DIR = BASE_DIR / "transformed_data"
GROUPED_DIR = TRANSFORMED_DATA_DIR / "combined_station_groups_1km"
OUTPUT_DIR = GROUPED_DIR / "outputs"
MODELS_DIR = GROUPED_DIR / "models"
APP_ASSETS_DIR = GROUPED_DIR / "app_assets"
APP_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

MONTHLY_CSV = OUTPUT_DIR / "station_group_month_bike_crash_dataset_1km.csv"
NEXT_MONTH_CSV = OUTPUT_DIR / "station_group_month_bike_crash_dataset_1km_next_month.csv"
MAPPING_CSV = OUTPUT_DIR / "group_mapping.csv"
MODEL_PATH = MODELS_DIR / "next_month_catboost_grouped_bike_1km.cbm"
METADATA_PATH = MODELS_DIR / "next_month_catboost_grouped_bike_1km_metadata.json"
IMPORTANCE_PATH = MODELS_DIR / "next_month_catboost_grouped_bike_1km_feature_importance.csv"


EDITABLE_FEATURE_LABELS = {
    "group_avg_bike_crash_count_history": "Historical average monthly bike crashes",
    "bike_crashes_per_1000_cyclists_lag1": "Previous month bike crashes per 1,000 cyclists",
    "total_count_sum_lag1": "Previous month total cyclists",
    "total_count_in_lag1": "Previous month inbound cyclists",
    "total_count_out_lag1": "Previous month outbound cyclists",
    "avg_temperature_2m_lag1": "Previous month average temperature",
    "total_precipitation_lag1": "Previous month total precipitation",
    "avg_wind_speed_10m_lag1": "Previous month average wind speed",
    "avg_cloud_cover_lag1": "Previous month average cloud cover",
    "bike_crash_count_1km_grouped_lag1": "Previous month bike crashes (1.0 km grouped)",
    "bike_crash_count_1km_grouped_roll3_sum": "Previous 3-month bike crashes (1.0 km grouped)",
    "bike_crash_count_1km_grouped_roll6_sum": "Previous 6-month bike crashes (1.0 km grouped)",
}

EDITABLE_FEATURE_PRIORITY = list(EDITABLE_FEATURE_LABELS)

NON_EDITABLE_FEATURES = {
    "group_id",
    "year",
    "month",
    "month_sin",
    "month_cos",
    "station_to_weather_grid_km",
    "weather_hours_available",
    "most_common_weather_code",
    "group_lat",
    "group_long",
}


def safe_step(min_value: float, max_value: float) -> float:
    span = float(max_value) - float(min_value)
    if span <= 0:
        return 1.0
    return max(round(span / 100, 4), 0.01)


def load_group_metadata() -> pd.DataFrame:
    mapping_df = pd.read_csv(MAPPING_CSV)
    return (
        mapping_df.groupby("group_id", as_index=False)
        .agg(
            group_name=("group_name", "first"),
            gemeente=("gemeente", "first"),
            lat=("group_lat", "first"),
            long=("group_long", "first"),
            member_site_count=("member_site_count", "max"),
            member_site_ids=("site_id", lambda s: ",".join(str(int(x)) for x in sorted(s))),
        )
    )


def build_historical_ranking(monthly_df: pd.DataFrame, group_meta_df: pd.DataFrame) -> pd.DataFrame:
    work_df = monthly_df.copy()
    work_df["total_cyclists_month"] = work_df["total_count_in"].fillna(0) + work_df["total_count_out"].fillna(0)
    global_rate = work_df["bike_crash_count_1km_grouped"].sum() / work_df["total_cyclists_month"].replace(0, np.nan).sum()
    stabilizer = work_df["total_cyclists_month"].median()

    ranking_df = (
        work_df.groupby("group_id", as_index=False)
        .agg(
            months_observed=("group_id", "count"),
            total_crashes=("bike_crash_count_1km_grouped", "sum"),
            average_monthly_crashes=("bike_crash_count_1km_grouped", "mean"),
            total_cyclists=("total_cyclists_month", "sum"),
            average_monthly_cyclists=("total_cyclists_month", "mean"),
            crash_month_share=("bike_crash_happened_1km_grouped", "mean"),
        )
    )
    ranking_df["crashes_per_10000_cyclists"] = (
        10000 * ranking_df["total_crashes"] / ranking_df["total_cyclists"].replace(0, np.nan)
    )
    ranking_df["smoothed_crashes_per_10000_cyclists"] = (
        10000 * (ranking_df["total_crashes"] + global_rate * stabilizer) / (ranking_df["total_cyclists"] + stabilizer)
    )
    ranking_df = ranking_df.merge(group_meta_df, on="group_id", how="left")
    ranking_df = ranking_df.sort_values(
        ["smoothed_crashes_per_10000_cyclists", "crashes_per_10000_cyclists"],
        ascending=False,
    ).reset_index(drop=True)
    ranking_df["historical_risk_rank"] = ranking_df.index + 1
    return ranking_df


def prepare_model_frame(df: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    frame = df[list(metadata["feature_columns"])].copy()
    categorical_features = set(metadata["categorical_features"])
    fill_values = metadata["numeric_fill_values"]
    for column in frame.columns:
        if column in categorical_features:
            frame[column] = frame[column].fillna("missing").astype(str)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(fill_values.get(column, 0.0))
    return frame


def build_forecast_ranking(next_month_df: pd.DataFrame, group_meta_df: pd.DataFrame, model, metadata):
    baseline_df = next_month_df.sort_values(["group_id", "year", "month"]).groupby("group_id", as_index=False).tail(1).copy()
    model_frame = prepare_model_frame(baseline_df, metadata)
    baseline_df["predicted_probability"] = model.predict_proba(model_frame)[:, 1]
    threshold = float(metadata["threshold"])
    baseline_df["predicted_label"] = (baseline_df["predicted_probability"] >= threshold).astype(int)
    baseline_df = baseline_df.drop(
        columns=["group_name", "gemeente", "member_site_count", "group_lat", "group_long"],
        errors="ignore",
    )
    baseline_df = baseline_df.merge(group_meta_df, on="group_id", how="left")
    baseline_df = baseline_df.rename(columns={"lat": "group_lat", "long": "group_long"})

    forecast_df = baseline_df[
        [
            "group_id",
            "group_name",
            "gemeente",
            "group_lat",
            "group_long",
            "member_site_count",
            "member_site_ids",
            "year",
            "month",
            "predicted_probability",
            "predicted_label",
        ]
    ].copy()
    forecast_df = forecast_df.rename(columns={"group_lat": "lat", "group_long": "long"})
    forecast_df = forecast_df.sort_values(["predicted_probability", "group_id"], ascending=[False, True]).reset_index(drop=True)
    forecast_df["forecast_risk_rank"] = forecast_df.index + 1
    forecast_df["risk_label"] = np.where(forecast_df["predicted_label"] == 1, "Higher risk", "Lower risk")
    return forecast_df, baseline_df


def choose_editable_features(next_month_df: pd.DataFrame) -> list[str]:
    importance_df = pd.read_csv(IMPORTANCE_PATH)
    numeric_columns = set(next_month_df.select_dtypes(include=["number"]).columns)
    chosen: list[str] = []
    for feature in importance_df["feature"]:
        if feature not in numeric_columns or feature in NON_EDITABLE_FEATURES or feature.startswith("target_"):
            continue
        if feature in EDITABLE_FEATURE_LABELS or feature in EDITABLE_FEATURE_PRIORITY:
            chosen.append(feature)
        if len(chosen) >= 10:
            break
    for fallback in EDITABLE_FEATURE_PRIORITY:
        if fallback in next_month_df.columns and fallback not in chosen:
            chosen.append(fallback)
        if len(chosen) >= 10:
            break
    return chosen[:10]


def build_control_ranges(next_month_df: pd.DataFrame, editable_features: list[str]) -> dict[str, dict[str, float | str]]:
    ranges = {}
    for feature in editable_features:
        series = pd.to_numeric(next_month_df[feature], errors="coerce").dropna()
        if series.empty:
            continue
        min_value = float(series.quantile(0.05))
        max_value = float(series.quantile(0.95))
        if min_value == max_value:
            min_value = float(series.min())
            max_value = float(series.max())
        ranges[feature] = {
            "label": EDITABLE_FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
            "min": min_value,
            "max": max_value,
            "mean": float(series.mean()),
            "step": safe_step(min_value, max_value),
        }
    return ranges


def main() -> None:
    monthly_df = pd.read_csv(MONTHLY_CSV)
    next_month_df = pd.read_csv(NEXT_MONTH_CSV)
    group_meta_df = load_group_metadata()

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    historical_df = build_historical_ranking(monthly_df, group_meta_df)
    forecast_df, baseline_df = build_forecast_ranking(next_month_df, group_meta_df, model, metadata)
    editable_features = choose_editable_features(next_month_df)
    control_ranges = build_control_ranges(next_month_df, editable_features)

    historical_df.to_csv(APP_ASSETS_DIR / "historical_ranking.csv", index=False)
    forecast_df.to_csv(APP_ASSETS_DIR / "forecast_ranking.csv", index=False)
    baseline_df.to_csv(APP_ASSETS_DIR / "group_baselines.csv", index=False)
    group_meta_df.to_csv(APP_ASSETS_DIR / "group_metadata.csv", index=False)
    (APP_ASSETS_DIR / "control_ranges.json").write_text(json.dumps(control_ranges, indent=2), encoding="utf-8")

    bootstrap = {
        "summary": {
            "label": "1.0 km grouped bike-only",
            "meters": 1000,
            "group_count": int(historical_df["group_id"].nunique()),
            "total_observed_crashes": float(historical_df["total_crashes"].sum()),
            "average_monthly_crashes": float(historical_df["average_monthly_crashes"].mean()),
            "average_monthly_cyclists": float(historical_df["average_monthly_cyclists"].mean()),
            "median_smoothed_rate": float(historical_df["smoothed_crashes_per_10000_cyclists"].median()),
            "highest_risk_group": str(historical_df.sort_values("historical_risk_rank").iloc[0]["group_name"]),
            "mean_forecast_probability": float(forecast_df["predicted_probability"].mean()),
            "positive_forecast_share": float(forecast_df["predicted_label"].mean()),
            "deployed_model": {
                "name": metadata["model_type"],
                "threshold": float(metadata["threshold"]),
                "accuracy": float(metadata["metrics"]["accuracy"]),
                "f1": float(metadata["metrics"]["f1"]),
                "roc_auc": float(metadata["metrics"]["roc_auc"]),
            },
        },
        "municipalities": sorted(group_meta_df["gemeente"].dropna().unique().tolist()),
        "method_evidence": [
            "This grouped-counter app merges nearby paired counters into grouped station entities before modeling.",
            "Bike-only crashes are deduplicated within each grouped counter so one crash is not counted twice across paired members.",
            "The grouped forecasting model keeps the same 1.0 km bike-only target and the same time-based validation logic as the current site-level app.",
            "This app is separate from the current deployed site-level app and is intended only for comparison.",
        ],
        "groups": {
            "historical": historical_df[["group_id", "group_name", "gemeente", "lat", "long", "historical_risk_rank", "smoothed_crashes_per_10000_cyclists"]]
            .rename(columns={"historical_risk_rank": "rank", "smoothed_crashes_per_10000_cyclists": "score"})
            .to_dict(orient="records"),
            "forecast": forecast_df[["group_id", "group_name", "gemeente", "lat", "long", "forecast_risk_rank", "predicted_probability"]]
            .rename(columns={"forecast_risk_rank": "rank", "predicted_probability": "score"})
            .to_dict(orient="records"),
        },
    }
    (APP_ASSETS_DIR / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    print(f"Saved grouped app assets to: {APP_ASSETS_DIR}")


if __name__ == "__main__":
    main()
