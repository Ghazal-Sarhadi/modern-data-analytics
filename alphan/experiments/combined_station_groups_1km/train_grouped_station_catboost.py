from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

GROUPED_NEXT_MONTH_CSV = OUTPUT_DIR / "station_group_month_bike_crash_dataset_1km_next_month.csv"
CURRENT_METADATA_JSON = BASE_DIR / "models/next_month_catboost_bike_1km_metadata.json"
OPTUNA_PARAMS_JSON = BASE_DIR / "model_outputs/optuna_catboost_1km_best_params.json"

METRICS_JSON = OUTPUT_DIR / "grouped_catboost_metrics.json"
PREDICTIONS_CSV = OUTPUT_DIR / "grouped_catboost_predictions.csv"
SUMMARY_TXT = OUTPUT_DIR / "grouped_catboost_summary.txt"


def split_timewise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_df = df.copy()
    split_df["year_month_index"] = split_df["year"] * 100 + split_df["month"]
    unique_periods = sorted(split_df["year_month_index"].unique())
    split_point = int(len(unique_periods) * 0.8)
    train_periods = set(unique_periods[:split_point])
    test_periods = set(unique_periods[split_point:])
    return (
        split_df[split_df["year_month_index"].isin(train_periods)].copy(),
        split_df[split_df["year_month_index"].isin(test_periods)].copy(),
    )


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    target_columns = [
        "target_bike_crash_happened_next_month_1km_grouped",
        "target_bike_crash_count_next_month_1km_grouped",
    ]
    feature_df = df.drop(columns=target_columns + ["year_month_index"], errors="ignore").copy()
    categorical_features = feature_df.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_features = feature_df.select_dtypes(include=["number"]).columns.tolist()
    fill_values: dict[str, float] = {}

    for column in categorical_features:
        feature_df[column] = feature_df[column].fillna("missing").astype(str)

    for column in numeric_features:
        median = feature_df[column].median()
        fill_value = float(median) if not pd.isna(median) else 0.0
        fill_values[column] = fill_value
        feature_df[column] = feature_df[column].fillna(fill_value)

    return feature_df, categorical_features, fill_values


def choose_threshold(y_true: pd.Series, y_score: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(round(threshold, 2))
    return best_threshold


def evaluate(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, object]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def load_training_params() -> dict[str, object]:
    if OPTUNA_PARAMS_JSON.exists():
        params = json.loads(OPTUNA_PARAMS_JSON.read_text(encoding="utf-8"))
    else:
        params = {"iterations": 400, "depth": 6, "learning_rate": 0.05}
    params.update(
        {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": False,
            "random_seed": 42,
        }
    )
    return params


def main() -> None:
    df = pd.read_csv(GROUPED_NEXT_MONTH_CSV)
    train_df, test_df = split_timewise(df)

    X_train, categorical_features, fill_values = prepare_features(train_df)
    X_test, _, _ = prepare_features(test_df)
    y_train = train_df["target_bike_crash_happened_next_month_1km_grouped"]
    y_test = test_df["target_bike_crash_happened_next_month_1km_grouped"]

    cat_indices = [X_train.columns.get_loc(name) for name in categorical_features]
    params = load_training_params()

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, cat_features=cat_indices)
    y_score = model.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(y_test, y_score)
    metrics = evaluate(y_test, y_score, threshold)

    predictions_df = test_df[["group_id", "group_name", "gemeente", "year", "month"]].copy()
    predictions_df["actual"] = y_test.values
    predictions_df["predicted_probability"] = y_score
    predictions_df["predicted_label"] = (y_score >= threshold).astype(int)
    predictions_df.to_csv(PREDICTIONS_CSV, index=False)

    baseline_metrics = {}
    if CURRENT_METADATA_JSON.exists():
        baseline_metrics = json.loads(CURRENT_METADATA_JSON.read_text(encoding="utf-8")).get("metrics", {})

    payload = {
        "grouped_dataset_rows": int(len(df)),
        "group_count": int(df["group_id"].nunique()),
        "positive_share": float(df["target_bike_crash_happened_next_month_1km_grouped"].mean()),
        "training_params": params,
        "grouped_metrics": metrics,
        "current_site_level_metrics": baseline_metrics,
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_lines = [
        "Combined station group experiment vs current site-level 1 km model",
        f"Grouped rows: {payload['grouped_dataset_rows']}",
        f"Grouped counter count: {payload['group_count']}",
        f"Grouped positive share: {payload['positive_share']:.4f}",
        "",
        "Grouped CatBoost metrics:",
        f"  Threshold: {metrics['threshold']:.2f}",
        f"  Accuracy: {metrics['accuracy']:.4f}",
        f"  Precision: {metrics['precision']:.4f}",
        f"  Recall: {metrics['recall']:.4f}",
        f"  F1: {metrics['f1']:.4f}",
        f"  ROC AUC: {metrics['roc_auc']:.4f}",
    ]
    if baseline_metrics:
        summary_lines.extend(
            [
                "",
                "Current site-level deployed metrics:",
                f"  Threshold: {float(baseline_metrics['threshold']):.2f}",
                f"  Accuracy: {float(baseline_metrics['accuracy']):.4f}",
                f"  Precision: {float(baseline_metrics.get('precision', 0.0)):.4f}",
                f"  Recall: {float(baseline_metrics.get('recall', 0.0)):.4f}",
                f"  F1: {float(baseline_metrics['f1']):.4f}",
                f"  ROC AUC: {float(baseline_metrics['roc_auc']):.4f}",
            ]
        )

    SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved grouped experiment metrics to: {METRICS_JSON}")
    print(f"Saved grouped experiment summary to: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
