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
MODELS_DIR = EXPERIMENT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CSV = OUTPUT_DIR / "station_group_month_bike_crash_dataset_1km_next_month.csv"
OPTUNA_PARAMS_JSON = BASE_DIR / "model_outputs/optuna_catboost_1km_best_params.json"

MODEL_PATH = MODELS_DIR / "next_month_catboost_grouped_bike_1km.cbm"
METADATA_PATH = MODELS_DIR / "next_month_catboost_grouped_bike_1km_metadata.json"
IMPORTANCE_PATH = MODELS_DIR / "next_month_catboost_grouped_bike_1km_feature_importance.csv"
SUMMARY_PATH = MODELS_DIR / "deployed_grouped_model_summary.json"

TARGET_CLASS_COLUMN = "target_bike_crash_happened_next_month_1km_grouped"
TARGET_COUNT_COLUMN = "target_bike_crash_count_next_month_1km_grouped"


def split_timewise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_df = df.copy()
    split_df["year_month_index"] = split_df["year"] * 100 + split_df["month"]
    unique_periods = sorted(split_df["year_month_index"].unique())
    if len(unique_periods) < 6:
        raise ValueError("Not enough monthly periods for time-based validation.")
    split_point = int(len(unique_periods) * 0.8)
    train_periods = set(unique_periods[:split_point])
    test_periods = set(unique_periods[split_point:])
    return (
        split_df[split_df["year_month_index"].isin(train_periods)].copy(),
        split_df[split_df["year_month_index"].isin(test_periods)].copy(),
    )


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    feature_df = df.drop(columns=[TARGET_CLASS_COLUMN, TARGET_COUNT_COLUMN, "year_month_index"], errors="ignore").copy()
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


def get_params() -> dict[str, object]:
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
    df = pd.read_csv(DATASET_CSV)
    train_df, test_df = split_timewise(df)

    X_train, categorical_features, _ = prepare_features(train_df)
    X_test, _, _ = prepare_features(test_df)
    y_train = train_df[TARGET_CLASS_COLUMN]
    y_test = test_df[TARGET_CLASS_COLUMN]

    cat_indices = [X_train.columns.get_loc(name) for name in categorical_features]
    params = get_params()

    validation_model = CatBoostClassifier(**params)
    validation_model.fit(X_train, y_train, cat_features=cat_indices)
    y_score = validation_model.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(y_test, y_score)
    metrics = evaluate(y_test, y_score, threshold)

    full_X, _, full_fill_values = prepare_features(df)
    full_y = df[TARGET_CLASS_COLUMN]
    full_cat_indices = [full_X.columns.get_loc(name) for name in categorical_features]

    final_model = CatBoostClassifier(**params)
    final_model.fit(full_X, full_y, cat_features=full_cat_indices)
    final_model.save_model(MODEL_PATH)

    metadata = {
        "radius_key": "1km_grouped_bike",
        "radius_label": "1.0 km grouped bike-only",
        "model_type": "catboost_classifier",
        "dataset_path": str(DATASET_CSV),
        "target_column": TARGET_CLASS_COLUMN,
        "sibling_target_column": TARGET_COUNT_COLUMN,
        "feature_columns": full_X.columns.tolist(),
        "categorical_features": categorical_features,
        "numeric_fill_values": full_fill_values,
        "threshold": threshold,
        "metrics": metrics,
        "training_params": params,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    importance_df = pd.DataFrame(
        {"feature": full_X.columns, "importance": final_model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    summary = {
        "model_path": str(MODEL_PATH),
        "metadata_path": str(METADATA_PATH),
        "importance_path": str(IMPORTANCE_PATH),
        "metrics": metrics,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved grouped deployable model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
