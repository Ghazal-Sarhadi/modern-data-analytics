from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from station_safety_app_utils import (
    APP_ASSETS_DIR,
    EDITABLE_FEATURE_LABELS,
    EDITABLE_FEATURE_PRIORITY,
    NON_EDITABLE_FEATURES,
    RADIUS_CONFIGS,
    SITE_COLUMNS,
    humanize_radius_key,
    safe_step,
)


def load_sites() -> pd.DataFrame:
    return pd.read_csv(Path("data/sites.csv"), names=SITE_COLUMNS)[
        ["site_id", "site_nr", "long", "lat", "naam", "gemeente", "datum_van"]
    ]


def build_historical_ranking(radius_key: str, monthly_df: pd.DataFrame, sites_df: pd.DataFrame) -> pd.DataFrame:
    config = RADIUS_CONFIGS[radius_key]
    crash_count_column = str(config["crash_count_column"])
    crash_happened_column = str(config["crash_happened_column"])

    work_df = monthly_df.copy()
    work_df["total_cyclists_month"] = work_df["total_count_in"].fillna(0) + work_df["total_count_out"].fillna(0)
    global_crash_rate = work_df[crash_count_column].sum() / work_df["total_cyclists_month"].replace(0, np.nan).sum()
    stabilizer = work_df["total_cyclists_month"].median()

    ranking_df = (
        work_df.groupby("site_id", as_index=False)
        .agg(
            months_observed=("site_id", "count"),
            total_crashes=(crash_count_column, "sum"),
            average_monthly_crashes=(crash_count_column, "mean"),
            total_cyclists=("total_cyclists_month", "sum"),
            average_monthly_cyclists=("total_cyclists_month", "mean"),
            crash_month_share=(crash_happened_column, "mean"),
        )
    )

    ranking_df["crashes_per_10000_cyclists"] = (
        10000 * ranking_df["total_crashes"] / ranking_df["total_cyclists"].replace(0, np.nan)
    )
    ranking_df["smoothed_crashes_per_10000_cyclists"] = (
        10000 * (ranking_df["total_crashes"] + global_crash_rate * stabilizer) / (ranking_df["total_cyclists"] + stabilizer)
    )
    ranking_df["radius_key"] = radius_key
    ranking_df["radius_label"] = config["label"]
    ranking_df = ranking_df.merge(sites_df, on="site_id", how="left")
    ranking_df = ranking_df.sort_values(
        ["smoothed_crashes_per_10000_cyclists", "crashes_per_10000_cyclists"],
        ascending=False,
    ).reset_index(drop=True)
    ranking_df["historical_risk_rank"] = ranking_df.index + 1
    return ranking_df


def load_model_bundle(radius_key: str) -> tuple[CatBoostClassifier, dict[str, object]]:
    config = RADIUS_CONFIGS[radius_key]
    metadata_path = Path("models") / str(config["metadata_filename"])
    model_path = Path("models") / str(config["model_filename"])
    if not metadata_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing deployable model artifacts for {radius_key}. Run export_app_models.py first.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model, metadata


def prepare_model_frame(df: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    frame = df[list(metadata["feature_columns"])].copy()

    categorical_features = set(metadata["categorical_features"])
    numeric_fill_values = metadata["numeric_fill_values"]
    for column in frame.columns:
        if column in categorical_features:
            frame[column] = frame[column].fillna("missing").astype(str)
        else:
            fill_value = numeric_fill_values.get(column, 0.0)
            frame[column] = frame[column].fillna(fill_value)

    return frame


def build_forecast_ranking(
    radius_key: str,
    next_month_df: pd.DataFrame,
    sites_df: pd.DataFrame,
    model: CatBoostClassifier,
    metadata: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_df = next_month_df.sort_values(["site_id", "year", "month"]).groupby("site_id", as_index=False).tail(1).copy()
    model_frame = prepare_model_frame(baseline_df, metadata)
    probabilities = model.predict_proba(model_frame)[:, 1]
    threshold = float(metadata["threshold"])

    baseline_df["predicted_probability"] = probabilities
    baseline_df["predicted_label"] = (baseline_df["predicted_probability"] >= threshold).astype(int)
    baseline_df["radius_key"] = radius_key
    baseline_df["radius_label"] = humanize_radius_key(radius_key)
    baseline_df = baseline_df.merge(sites_df, on="site_id", how="left")

    forecast_df = baseline_df[
        [
            "site_id",
            "year",
            "month",
            "predicted_probability",
            "predicted_label",
            "radius_key",
            "radius_label",
            "naam",
            "gemeente",
            "datum_van",
            "long",
            "lat",
        ]
    ].copy()
    forecast_df = forecast_df.sort_values(["predicted_probability", "site_id"], ascending=[False, True]).reset_index(drop=True)
    forecast_df["forecast_risk_rank"] = forecast_df.index + 1
    forecast_df["risk_label"] = np.where(forecast_df["predicted_label"] == 1, "Higher risk", "Lower risk")

    return forecast_df, baseline_df


def choose_editable_features(radius_key: str, next_month_df: pd.DataFrame) -> list[str]:
    importance_path = Path("models") / str(RADIUS_CONFIGS[radius_key]["importance_filename"])
    importance_df = pd.read_csv(importance_path)
    numeric_columns = set(next_month_df.select_dtypes(include=["number"]).columns)

    chosen = []
    for feature in importance_df["feature"]:
        if feature not in numeric_columns:
            continue
        if feature in NON_EDITABLE_FEATURES:
            continue
        if feature in {RADIUS_CONFIGS[radius_key]["target_class_column"], RADIUS_CONFIGS[radius_key]["target_count_column"]}:
            continue
        if feature.startswith("target_"):
            continue
        if feature in EDITABLE_FEATURE_LABELS:
            chosen.append(feature)
        elif feature in EDITABLE_FEATURE_PRIORITY:
            chosen.append(feature)
        if len(chosen) >= 12:
            break

    for fallback in EDITABLE_FEATURE_PRIORITY:
        if fallback in next_month_df.columns and fallback not in chosen and fallback not in NON_EDITABLE_FEATURES:
            chosen.append(fallback)
        if len(chosen) >= 10:
            break

    return chosen[:10]


def build_control_ranges(radius_key: str, next_month_df: pd.DataFrame, editable_features: list[str]) -> dict[str, dict[str, float | str]]:
    ranges = {}
    for feature in editable_features:
        series = pd.to_numeric(next_month_df[feature], errors="coerce").dropna()
        if series.empty:
            continue
        min_value = float(series.quantile(0.05))
        max_value = float(series.quantile(0.95))
        mean_value = float(series.mean())
        if min_value == max_value:
            min_value = float(series.min())
            max_value = float(series.max())
        ranges[feature] = {
            "label": EDITABLE_FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
            "step": safe_step(min_value, max_value),
        }
    return ranges


def build_radius_summary(radius_key: str, historical_df: pd.DataFrame, forecast_df: pd.DataFrame, model_metadata: dict[str, object]) -> dict[str, object]:
    top_station = historical_df.sort_values("historical_risk_rank").iloc[0]
    return {
        "radius_key": radius_key,
        "radius_label": humanize_radius_key(radius_key),
        "meters": int(RADIUS_CONFIGS[radius_key]["meters"]),
        "station_count": int(historical_df["site_id"].nunique()),
        "total_observed_crashes": float(historical_df["total_crashes"].sum()),
        "average_monthly_crashes": float(historical_df["average_monthly_crashes"].mean()),
        "average_monthly_cyclists": float(historical_df["average_monthly_cyclists"].mean()),
        "median_smoothed_rate": float(historical_df["smoothed_crashes_per_10000_cyclists"].median()),
        "highest_risk_station": str(top_station["naam"]),
        "mean_forecast_probability": float(forecast_df["predicted_probability"].mean()),
        "positive_forecast_share": float(forecast_df["predicted_label"].mean()),
        "deployed_model": {
            "name": model_metadata["model_type"],
            "threshold": float(model_metadata["threshold"]),
            "accuracy": float(model_metadata["metrics"]["accuracy"]),
            "f1": float(model_metadata["metrics"]["f1"]),
            "roc_auc": float(model_metadata["metrics"]["roc_auc"]),
        },
    }


def main() -> None:
    APP_ASSETS_DIR.mkdir(exist_ok=True)
    sites_df = load_sites()
    sites_df.to_csv(APP_ASSETS_DIR / "station_metadata.csv", index=False)

    bootstrap = {
        "radius_summaries": {},
        "municipalities": sorted(sites_df["gemeente"].dropna().unique().tolist()),
        "method_evidence": [
            "Monthly and next-month station datasets were built from bike counts, weather summaries, and crash matches.",
            "Leakage fields tied to same-month crash distance were removed before the final models were used.",
            "The deployed app uses bike-only crashes within 1.0 km because that is the strongest compromise between local interpretability and non-sparse modeling.",
            "The app uses saved deployable next-month classifiers and does not retrain models during interaction.",
        ],
        "stations": {},
    }

    for radius_key, config in RADIUS_CONFIGS.items():
        monthly_df = pd.read_csv(Path(config["monthly_dataset_path"]))
        next_month_df = pd.read_csv(Path(config["next_month_dataset_path"]))
        model, metadata = load_model_bundle(radius_key)

        historical_df = build_historical_ranking(radius_key, monthly_df, sites_df)
        forecast_df, baseline_df = build_forecast_ranking(radius_key, next_month_df, sites_df, model, metadata)

        editable_features = choose_editable_features(radius_key, next_month_df)
        control_ranges = build_control_ranges(radius_key, next_month_df, editable_features)

        historical_path = APP_ASSETS_DIR / f"historical_ranking_{radius_key}.csv"
        forecast_path = APP_ASSETS_DIR / f"forecast_ranking_{radius_key}.csv"
        baseline_path = APP_ASSETS_DIR / f"station_baselines_{radius_key}.csv"
        controls_path = APP_ASSETS_DIR / f"control_ranges_{radius_key}.json"

        historical_df.to_csv(historical_path, index=False)
        forecast_df.to_csv(forecast_path, index=False)
        baseline_df.to_csv(baseline_path, index=False)
        controls_path.write_text(json.dumps(control_ranges, indent=2), encoding="utf-8")

        radius_summary = build_radius_summary(radius_key, historical_df, forecast_df, metadata)
        bootstrap["radius_summaries"][radius_key] = radius_summary

        stations_payload = {}
        for mode_name, ranking_df in {"historical": historical_df, "forecast": forecast_df}.items():
            rank_column = "historical_risk_rank" if mode_name == "historical" else "forecast_risk_rank"
            metric_column = (
                "smoothed_crashes_per_10000_cyclists" if mode_name == "historical" else "predicted_probability"
            )
            stations_payload[mode_name] = (
                ranking_df[
                    ["site_id", "naam", "gemeente", "lat", "long", rank_column, metric_column]
                ]
                .rename(columns={rank_column: "rank", metric_column: "score", "naam": "name", "gemeente": "municipality"})
                .sort_values("rank")
                .to_dict(orient="records")
            )
        bootstrap["stations"][radius_key] = stations_payload

    (APP_ASSETS_DIR / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    print(f"Saved app assets to: {APP_ASSETS_DIR}")


if __name__ == "__main__":
    main()
