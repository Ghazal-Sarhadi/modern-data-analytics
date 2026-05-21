from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


TRANSFORMED_DATA_DIR = Path("transformed_data")
MODEL_OUTPUTS_DIR = TRANSFORMED_DATA_DIR / "model_outputs"
MODEL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = MODEL_OUTPUTS_DIR / "station_month_bike_crash_dataset_1km_next_month.csv"
STUDY_CSV = MODEL_OUTPUTS_DIR / "optuna_catboost_1km_study.csv"
BEST_PARAMS_JSON = MODEL_OUTPUTS_DIR / "optuna_catboost_1km_best_params.json"
BEST_METRICS_JSON = MODEL_OUTPUTS_DIR / "optuna_catboost_1km_best_metrics.json"
SUMMARY_TXT = MODEL_OUTPUTS_DIR / "optuna_catboost_1km_summary.txt"

TARGET_COLUMN = "target_bike_crash_happened_next_month_1km"
SIBLING_TARGET_COLUMN = "target_bike_crash_count_next_month_1km"


def split_timewise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_df = df.copy()
    split_df["year_month_index"] = split_df["year"] * 100 + split_df["month"]
    unique_periods = sorted(split_df["year_month_index"].unique())
    if len(unique_periods) < 6:
        raise ValueError("Not enough monthly periods for a time-based split.")
    split_point = int(len(unique_periods) * 0.8)
    train_periods = set(unique_periods[:split_point])
    test_periods = set(unique_periods[split_point:])
    return (
        split_df[split_df["year_month_index"].isin(train_periods)].copy(),
        split_df[split_df["year_month_index"].isin(test_periods)].copy(),
    )


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    feature_df = df.drop(columns=[TARGET_COLUMN, SIBLING_TARGET_COLUMN, "year_month_index"], errors="ignore").copy()
    categorical_features = feature_df.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_features = feature_df.select_dtypes(include=["number"]).columns.tolist()
    numeric_fill_values: dict[str, float] = {}

    for column in categorical_features:
        feature_df[column] = feature_df[column].fillna("missing").astype(str)

    for column in numeric_features:
        median_value = float(feature_df[column].median()) if not pd.isna(feature_df[column].median()) else 0.0
        numeric_fill_values[column] = median_value
        feature_df[column] = feature_df[column].fillna(median_value)

    return feature_df, categorical_features, numeric_fill_values


def evaluate_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, object]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def find_best_f1_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(round(threshold, 2))
    return best_threshold, float(best_f1)


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    train_df, test_df = split_timewise(df)

    X_train, categorical_features, _ = prepare_features(train_df)
    X_test, _, _ = prepare_features(test_df)
    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]
    cat_feature_indices = [X_train.columns.get_loc(name) for name in categorical_features]

    trial_rows: list[dict[str, object]] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1200),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": False,
            "random_seed": 42,
        }

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, cat_features=cat_feature_indices)
        y_score = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_score))
        threshold, best_f1 = find_best_f1_threshold(y_test, y_score)

        trial_rows.append(
            {
                "trial": trial.number,
                **params,
                "validation_roc_auc": auc,
                "best_f1_threshold": threshold,
                "best_f1": best_f1,
            }
        )
        return auc

    study = optuna.create_study(direction="maximize", study_name="catboost_1km_next_month_auc")
    study.optimize(objective, n_trials=40)

    best_params = study.best_trial.params
    full_params = {
        **best_params,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": False,
        "random_seed": 42,
    }

    best_model = CatBoostClassifier(**full_params)
    best_model.fit(X_train, y_train, cat_features=cat_feature_indices)
    y_score = best_model.predict_proba(X_test)[:, 1]
    threshold, best_f1 = find_best_f1_threshold(y_test, y_score)
    metrics = evaluate_metrics(y_test, y_score, threshold)
    metrics["threshold_selection_f1"] = best_f1

    pd.DataFrame(trial_rows).sort_values("validation_roc_auc", ascending=False).to_csv(STUDY_CSV, index=False)
    BEST_PARAMS_JSON.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    BEST_METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary_lines = [
        f"Trials: {len(trial_rows)}",
        f"Best validation ROC AUC: {study.best_value:.4f}",
        f"Best threshold by F1: {threshold:.2f}",
        "",
        "Metrics at that threshold:",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1: {metrics['f1']:.4f}",
        f"ROC AUC: {metrics['roc_auc']:.4f}",
        "",
        "Confusion matrix:",
        str(metrics["confusion_matrix"]),
        "",
        "Best parameters:",
        json.dumps(best_params, indent=2),
    ]
    SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved Optuna study to: {STUDY_CSV}")
    print(f"Saved best params to: {BEST_PARAMS_JSON}")
    print(f"Saved best metrics to: {BEST_METRICS_JSON}")


if __name__ == "__main__":
    main()
