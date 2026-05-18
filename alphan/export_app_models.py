from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from station_safety_app_utils import MODEL_DIR, RADIUS_CONFIGS


VALIDATION_THRESHOLD_PATH = Path("model_outputs/next_month_500m_threshold_best.csv")
OPTUNA_1KM_PARAMS_PATH = Path("model_outputs/optuna_catboost_1km_best_params.json")
OPTUNA_1KM_METRICS_PATH = Path("model_outputs/optuna_catboost_1km_best_metrics.json")


def split_timewise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_df = df.copy()
    split_df["year_month_index"] = split_df["year"] * 100 + split_df["month"]
    unique_periods = sorted(split_df["year_month_index"].unique())
    if len(unique_periods) < 6:
        raise ValueError("Not enough monthly periods for a time-based split.")

    split_point = int(len(unique_periods) * 0.8)
    train_periods = set(unique_periods[:split_point])
    test_periods = set(unique_periods[split_point:])
    train_df = split_df[split_df["year_month_index"].isin(train_periods)].copy()
    test_df = split_df[split_df["year_month_index"].isin(test_periods)].copy()
    return train_df, test_df


def prepare_features(df: pd.DataFrame, target_column: str, sibling_target_column: str) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    feature_df = df.drop(columns=[target_column, sibling_target_column]).copy()
    categorical_features = feature_df.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_features = feature_df.select_dtypes(include=["number"]).columns.tolist()
    numeric_fill_values = {}

    for column in categorical_features:
        feature_df[column] = feature_df[column].fillna("missing").astype(str)

    for column in numeric_features:
        median_value = float(feature_df[column].median()) if not pd.isna(feature_df[column].median()) else 0.0
        numeric_fill_values[column] = median_value
        feature_df[column] = feature_df[column].fillna(median_value)

    return feature_df, categorical_features, numeric_fill_values


def choose_threshold(radius_key: str, y_true: pd.Series, y_score: np.ndarray) -> float:
    if radius_key == "500m" and VALIDATION_THRESHOLD_PATH.exists():
        threshold_df = pd.read_csv(VALIDATION_THRESHOLD_PATH)
        match_df = threshold_df[
            (threshold_df["model"] == "catboost_classifier") & (threshold_df["objective"] == "f1")
        ]
        if not match_df.empty:
            return float(match_df.iloc[0]["threshold"])

    if radius_key == "1km_bike" and OPTUNA_1KM_METRICS_PATH.exists():
        tuned_metrics = json.loads(OPTUNA_1KM_METRICS_PATH.read_text(encoding="utf-8"))
        threshold = tuned_metrics.get("threshold")
        if threshold is not None:
            return float(threshold)

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(round(threshold, 2))
    return best_threshold


def evaluate_model(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, object]:
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


def get_catboost_params(radius_key: str) -> dict[str, object]:
    default_params: dict[str, object] = {
        "iterations": 400,
        "depth": 6,
        "learning_rate": 0.05,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": False,
        "random_seed": 42,
    }

    if radius_key == "1km_bike" and OPTUNA_1KM_PARAMS_PATH.exists():
        tuned_params = json.loads(OPTUNA_1KM_PARAMS_PATH.read_text(encoding="utf-8"))
        tuned_params.update(
            {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "verbose": False,
                "random_seed": 42,
            }
        )
        return tuned_params

    return default_params


def train_radius_model(radius_key: str, config: dict[str, object]) -> dict[str, object]:
    dataset_path = Path(config["next_month_dataset_path"])
    target_column = str(config["target_class_column"])
    sibling_target_column = str(config["target_count_column"])

    df = pd.read_csv(dataset_path)
    train_df, test_df = split_timewise(df)

    X_train, categorical_features, numeric_fill_values = prepare_features(
        train_df,
        target_column=target_column,
        sibling_target_column=sibling_target_column,
    )
    X_test, _, _ = prepare_features(
        test_df,
        target_column=target_column,
        sibling_target_column=sibling_target_column,
    )
    y_train = train_df[target_column]
    y_test = test_df[target_column]

    cat_feature_indices = [X_train.columns.get_loc(name) for name in categorical_features]
    model_params = get_catboost_params(radius_key)

    validation_model = CatBoostClassifier(**model_params)
    validation_model.fit(X_train, y_train, cat_features=cat_feature_indices)
    validation_scores = validation_model.predict_proba(X_test)[:, 1]
    threshold = choose_threshold(radius_key, y_test, validation_scores)
    metrics = evaluate_model(y_test, validation_scores, threshold)

    full_X, _, full_numeric_fill_values = prepare_features(
        df,
        target_column=target_column,
        sibling_target_column=sibling_target_column,
    )
    full_y = df[target_column]
    full_cat_indices = [full_X.columns.get_loc(name) for name in categorical_features]

    final_model = CatBoostClassifier(**model_params)
    final_model.fit(full_X, full_y, cat_features=full_cat_indices)

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / str(config["model_filename"])
    metadata_path = MODEL_DIR / str(config["metadata_filename"])
    importance_path = MODEL_DIR / str(config["importance_filename"])

    final_model.save_model(model_path)

    metadata = {
        "radius_key": radius_key,
        "radius_label": config["label"],
        "model_type": "catboost_classifier",
        "dataset_path": str(dataset_path),
        "target_column": target_column,
        "sibling_target_column": sibling_target_column,
        "feature_columns": full_X.columns.tolist(),
        "categorical_features": categorical_features,
        "numeric_fill_values": full_numeric_fill_values,
        "threshold": threshold,
        "metrics": metrics,
        "training_params": model_params,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    importance_df = pd.DataFrame(
        {
            "radius_key": radius_key,
            "feature": full_X.columns,
            "importance": final_model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(importance_path, index=False)

    return {
        "radius_key": radius_key,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "importance_path": str(importance_path),
        "metrics": metrics,
    }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    summary = {"models": []}
    for radius_key, config in RADIUS_CONFIGS.items():
        result = train_radius_model(radius_key, config)
        summary["models"].append(result)

    summary_path = MODEL_DIR / "deployed_model_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved deployed model summary to: {summary_path}")


if __name__ == "__main__":
    main()
