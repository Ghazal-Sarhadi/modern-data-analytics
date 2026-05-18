from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from flask import Flask, jsonify, render_template, request

from station_safety_app_utils import APP_ASSETS_DIR, MODEL_DIR, RADIUS_CONFIGS


class AppState:
    def __init__(self) -> None:
        bootstrap_path = APP_ASSETS_DIR / "bootstrap.json"
        metadata_path = APP_ASSETS_DIR / "station_metadata.csv"
        if not bootstrap_path.exists() or not metadata_path.exists():
            raise RuntimeError("Missing app assets. Run export_app_models.py and build_app_assets.py first.")

        self.bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        self.station_metadata_df = pd.read_csv(metadata_path).set_index("site_id")

        self.historical_rankings = {}
        self.forecast_rankings = {}
        self.station_baselines = {}
        self.control_ranges = {}
        self.models = {}
        self.model_metadata = {}

        for radius_key, config in RADIUS_CONFIGS.items():
            historical_path = APP_ASSETS_DIR / f"historical_ranking_{radius_key}.csv"
            forecast_path = APP_ASSETS_DIR / f"forecast_ranking_{radius_key}.csv"
            baseline_path = APP_ASSETS_DIR / f"station_baselines_{radius_key}.csv"
            controls_path = APP_ASSETS_DIR / f"control_ranges_{radius_key}.json"
            model_path = MODEL_DIR / str(config["model_filename"])
            model_metadata_path = MODEL_DIR / str(config["metadata_filename"])

            required_paths = [
                historical_path,
                forecast_path,
                baseline_path,
                controls_path,
                model_path,
                model_metadata_path,
            ]
            missing_paths = [str(path) for path in required_paths if not path.exists()]
            if missing_paths:
                raise RuntimeError(
                    f"Missing assets for {radius_key}: {', '.join(missing_paths)}. "
                    "Run export_app_models.py and build_app_assets.py first."
                )

            self.historical_rankings[radius_key] = pd.read_csv(historical_path).set_index("site_id")
            self.forecast_rankings[radius_key] = pd.read_csv(forecast_path).set_index("site_id")
            self.station_baselines[radius_key] = pd.read_csv(baseline_path).set_index("site_id")
            self.control_ranges[radius_key] = json.loads(controls_path.read_text(encoding="utf-8"))
            self.model_metadata[radius_key] = json.loads(model_metadata_path.read_text(encoding="utf-8"))

            model = CatBoostClassifier()
            model.load_model(model_path)
            self.models[radius_key] = model

    def validate_radius(self, radius_key: str) -> None:
        if radius_key not in RADIUS_CONFIGS:
            raise ValueError(f"Unsupported radius: {radius_key}")

    def station_detail(self, site_id: int, radius_key: str) -> dict[str, object]:
        self.validate_radius(radius_key)

        if site_id not in self.station_metadata_df.index or site_id not in self.station_baselines[radius_key].index:
            raise KeyError(f"Unknown station id: {site_id}")

        metadata_row = self.station_metadata_df.loc[site_id]
        historical_row = self.historical_rankings[radius_key].loc[site_id]
        forecast_row = self.forecast_rankings[radius_key].loc[site_id]
        baseline_row = self.station_baselines[radius_key].loc[site_id]

        controls = []
        for feature, control_meta in self.control_ranges[radius_key].items():
            default_value = baseline_row.get(feature, control_meta["mean"])
            if pd.isna(default_value):
                default_value = control_meta["mean"]
            controls.append(
                {
                    "feature": feature,
                    "label": control_meta["label"],
                    "min": float(control_meta["min"]),
                    "max": float(control_meta["max"]),
                    "step": float(control_meta["step"]),
                    "default": float(default_value),
                    "mean": float(control_meta["mean"]),
                }
            )

        return {
            "site_id": int(site_id),
            "radius_key": radius_key,
            "radius_label": RADIUS_CONFIGS[radius_key]["label"],
            "radius_meters": int(RADIUS_CONFIGS[radius_key]["meters"]),
            "metadata": {
                "name": str(metadata_row["naam"]),
                "municipality": str(metadata_row["gemeente"]),
                "opening_date": str(metadata_row["datum_van"]),
                "lat": float(metadata_row["lat"]),
                "long": float(metadata_row["long"]),
            },
            "historical": {
                "rank": int(historical_row["historical_risk_rank"]),
                "total_crashes": float(historical_row["total_crashes"]),
                "average_monthly_crashes": float(historical_row["average_monthly_crashes"]),
                "total_cyclists": float(historical_row["total_cyclists"]),
                "average_monthly_cyclists": float(historical_row["average_monthly_cyclists"]),
                "crash_month_share": float(historical_row["crash_month_share"]),
                "crashes_per_10000_cyclists": float(historical_row["crashes_per_10000_cyclists"]),
                "smoothed_crashes_per_10000_cyclists": float(historical_row["smoothed_crashes_per_10000_cyclists"]),
            },
            "forecast": {
                "rank": int(forecast_row["forecast_risk_rank"]),
                "probability": float(forecast_row["predicted_probability"]),
                "predicted_label": int(forecast_row["predicted_label"]),
                "risk_label": str(forecast_row["risk_label"]),
                "forecast_year": int(forecast_row["year"]),
                "forecast_month": int(forecast_row["month"]),
            },
            "baseline_context": {
                "feature_year": int(baseline_row["year"]),
                "feature_month": int(baseline_row["month"]),
            },
            "controls": controls,
        }

    def predict(self, site_id: int, radius_key: str, overrides: dict[str, float]) -> dict[str, object]:
        self.validate_radius(radius_key)
        if site_id not in self.station_baselines[radius_key].index:
            raise KeyError(f"Unknown station id: {site_id}")

        metadata = self.model_metadata[radius_key]
        baseline_series = self.station_baselines[radius_key].loc[site_id].copy()
        baseline_series["site_id"] = site_id
        for feature, value in overrides.items():
            baseline_series[feature] = value

        model_frame = pd.DataFrame([baseline_series])[list(metadata["feature_columns"])].copy()
        categorical_features = set(metadata["categorical_features"])
        numeric_fill_values = metadata["numeric_fill_values"]
        for column in model_frame.columns:
            if column in categorical_features:
                model_frame[column] = model_frame[column].fillna("missing").astype(str)
            else:
                model_frame[column] = pd.to_numeric(model_frame[column], errors="coerce").fillna(
                    numeric_fill_values.get(column, 0.0)
                )

        probability = float(self.models[radius_key].predict_proba(model_frame)[:, 1][0])
        threshold = float(metadata["threshold"])
        predicted_label = int(probability >= threshold)
        return {
            "site_id": int(site_id),
            "radius_key": radius_key,
            "radius_label": RADIUS_CONFIGS[radius_key]["label"],
            "predicted_probability": probability,
            "threshold": threshold,
            "predicted_label": predicted_label,
            "risk_label": "Higher risk" if predicted_label == 1 else "Lower risk",
        }


STATE = AppState()
def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            bootstrap=STATE.bootstrap,
        )

    @app.get("/api/station/<int:site_id>")
    def station_detail(site_id: int):
        radius_key = request.args.get("radius", "1km_bike")
        try:
            detail = STATE.station_detail(site_id, radius_key)
        except (KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(detail)

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(force=True) or {}
        try:
            site_id = int(payload["site_id"])
            radius_key = str(payload["radius"])
            overrides = {str(key): float(value) for key, value in (payload.get("overrides") or {}).items()}
            result = STATE.predict(site_id=site_id, radius_key=radius_key, overrides=overrides)
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
