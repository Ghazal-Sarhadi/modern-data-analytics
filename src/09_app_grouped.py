from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent.parent
TRANSFORMED_DATA_DIR = BASE_DIR / "transformed_data"
GROUPED_DIR = TRANSFORMED_DATA_DIR / "combined_station_groups_1km"
APP_ASSETS_DIR = GROUPED_DIR / "app_assets"
MODELS_DIR = GROUPED_DIR / "models"
TEMPLATE_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

class GroupedAppState:
    def __init__(self) -> None:
        bootstrap_path = APP_ASSETS_DIR / "bootstrap.json"
        metadata_path = APP_ASSETS_DIR / "group_metadata.csv"
        historical_path = APP_ASSETS_DIR / "historical_ranking.csv"
        forecast_path = APP_ASSETS_DIR / "forecast_ranking.csv"
        baseline_path = APP_ASSETS_DIR / "group_baselines.csv"
        controls_path = APP_ASSETS_DIR / "control_ranges.json"
        model_path = MODELS_DIR / "next_month_catboost_grouped_bike_1km.cbm"
        model_metadata_path = MODELS_DIR / "next_month_catboost_grouped_bike_1km_metadata.json"

        required = [
            bootstrap_path,
            metadata_path,
            historical_path,
            forecast_path,
            baseline_path,
            controls_path,
            model_path,
            model_metadata_path,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(
                "Missing grouped app assets. Run 07_export_grouped_app_model.py and 08_build_grouped_app_assets.py first. "
                + ", ".join(missing)
            )

        self.bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        self.group_metadata_df = pd.read_csv(metadata_path).set_index("group_id")
        self.historical_df = pd.read_csv(historical_path).set_index("group_id")
        self.forecast_df = pd.read_csv(forecast_path).set_index("group_id")
        self.baseline_df = pd.read_csv(baseline_path).set_index("group_id")
        self.controls = json.loads(controls_path.read_text(encoding="utf-8"))
        self.model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))

        self.model = CatBoostClassifier()
        self.model.load_model(model_path)

    def group_detail(self, group_id: int) -> dict[str, object]:
        if group_id not in self.group_metadata_df.index or group_id not in self.baseline_df.index:
            raise KeyError(f"Unknown group id: {group_id}")

        meta = self.group_metadata_df.loc[group_id]
        hist = self.historical_df.loc[group_id]
        fc = self.forecast_df.loc[group_id]
        baseline = self.baseline_df.loc[group_id]

        controls = []
        for feature, control_meta in self.controls.items():
            default_value = baseline.get(feature, control_meta["mean"])
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
            "group_id": int(group_id),
            "label": "1.0 km grouped bike-only",
            "radius_meters": 1000,
            "metadata": {
                "name": str(meta["group_name"]),
                "municipality": str(meta["gemeente"]),
                "lat": float(meta["lat"]),
                "long": float(meta["long"]),
                "member_site_count": int(meta["member_site_count"]),
                "member_site_ids": str(meta["member_site_ids"]),
            },
            "historical": {
                "rank": int(hist["historical_risk_rank"]),
                "total_crashes": float(hist["total_crashes"]),
                "average_monthly_crashes": float(hist["average_monthly_crashes"]),
                "total_cyclists": float(hist["total_cyclists"]),
                "average_monthly_cyclists": float(hist["average_monthly_cyclists"]),
                "crash_month_share": float(hist["crash_month_share"]),
                "crashes_per_10000_cyclists": float(hist["crashes_per_10000_cyclists"]),
                "smoothed_crashes_per_10000_cyclists": float(hist["smoothed_crashes_per_10000_cyclists"]),
            },
            "forecast": {
                "rank": int(fc["forecast_risk_rank"]),
                "probability": float(fc["predicted_probability"]),
                "predicted_label": int(fc["predicted_label"]),
                "risk_label": str(fc["risk_label"]),
                "hidden_risk_spot": bool(fc.get("hidden_risk_spot", False)),
                "forecast_year": int(fc["year"]),
                "forecast_month": int(fc["month"]),
            },
            "baseline_context": {
                "feature_year": int(baseline["year"]),
                "feature_month": int(baseline["month"]),
            },
            "controls": controls,
        }

    def predict(self, group_id: int, overrides: dict[str, float]) -> dict[str, object]:
        if group_id not in self.baseline_df.index:
            raise KeyError(f"Unknown group id: {group_id}")

        baseline = self.baseline_df.loc[group_id].copy()
        baseline["group_id"] = group_id
        for feature, value in overrides.items():
            baseline[feature] = value

        feature_columns = list(self.model_metadata["feature_columns"])
        frame = pd.DataFrame([baseline])[feature_columns].copy()
        categorical = set(self.model_metadata["categorical_features"])
        fill_values = self.model_metadata["numeric_fill_values"]
        for column in frame.columns:
            if column in categorical:
                frame[column] = frame[column].fillna("missing").astype(str)
            else:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(fill_values.get(column, 0.0))

        probability = float(self.model.predict_proba(frame)[:, 1][0])
        threshold = float(self.model_metadata["threshold"])
        predicted_label = int(probability >= threshold)
        return {
            "group_id": int(group_id),
            "predicted_probability": probability,
            "threshold": threshold,
            "predicted_label": predicted_label,
            "risk_label": "Higher risk" if predicted_label == 1 else "Lower risk",
        }


STATE = GroupedAppState()


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))

    @app.route("/")
    def index():
        return render_template("index_grouped.html", bootstrap=STATE.bootstrap)

    @app.get("/api/group/<int:group_id>")
    def group_detail(group_id: int):
        try:
            detail = STATE.group_detail(group_id)
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(detail)

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(force=True) or {}
        try:
            group_id = int(payload["group_id"])
            overrides = {str(k): float(v) for k, v in (payload.get("overrides") or {}).items()}
            result = STATE.predict(group_id, overrides)
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
