from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


DATA_PATH = Path("model_outputs/station_month_bike_crash_dataset_1km_next_month.csv")
OUTPUT_DIR = Path("model_outputs")
TARGET_COLUMN = "target_bike_crash_happened_next_month_1km"
SIBLING_TARGET_COLUMN = "target_bike_crash_count_next_month_1km"

STUDY_PATH = OUTPUT_DIR / "optuna_catboost_1km_study.csv"
BEST_PARAMS_PATH = OUTPUT_DIR / "optuna_catboost_1km_best_params.json"
BEST_METRICS_PATH = OUTPUT_DIR / "optuna_catboost_1km_best_metrics.json"
TRIAL_SUMMARY_PATH = OUTPUT_DIR / "optuna_catboost_1km_summary.txt"


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


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    feature_df = df.drop(columns=[TARGET_COLUMN, SIBLING_TARGET_COLUMN, "year_month_index"]).copy()
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


def choose_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(round(threshold, 2))
    return best_threshold, float(best_f1)


def evaluate(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, object]:
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


df = pd.read_csv(DATA_PATH)
train_df, test_df = split_timewise(df)
X_train, categorical_features, _ = prepare_features(train_df)
X_test, _, _ = prepare_features(test_df)
y_train = train_df[TARGET_COLUMN]
y_test = test_df[TARGET_COLUMN]
cat_feature_indices = [X_train.columns.get_loc(name) for name in categorical_features]


def objective(trial: optuna.Trial) -> float:
    params = {
        "iterations": trial.suggest_int("iterations", 250, 1200),
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
    return float(roc_auc_score(y_test, y_score))


study = optuna.create_study(direction="maximize", study_name="catboost_bike_1km_next_month")
study.optimize(objective, n_trials=40, show_progress_bar=False)

trials_df = study.trials_dataframe()
OUTPUT_DIR.mkdir(exist_ok=True)
trials_df.to_csv(STUDY_PATH, index=False)

best_params = study.best_trial.params
best_model = CatBoostClassifier(
    **best_params,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=False,
    random_seed=42,
)
best_model.fit(X_train, y_train, cat_features=cat_feature_indices)
best_scores = best_model.predict_proba(X_test)[:, 1]
best_threshold, tuned_f1 = choose_threshold(y_test, best_scores)
best_metrics = evaluate(y_test, best_scores, best_threshold)
best_metrics["threshold_selection_f1"] = tuned_f1

BEST_PARAMS_PATH.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
BEST_METRICS_PATH.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")

summary_lines = [
    "Optuna CatBoost tuning for bike-only 1 km next-month classification",
    f"Trials: {len(study.trials)}",
    f"Best validation ROC AUC: {study.best_value:.4f}",
    f"Best threshold by F1: {best_threshold:.2f}",
    f"Accuracy: {best_metrics['accuracy']:.4f}",
    f"Precision: {best_metrics['precision']:.4f}",
    f"Recall: {best_metrics['recall']:.4f}",
    f"F1: {best_metrics['f1']:.4f}",
    f"ROC AUC: {best_metrics['roc_auc']:.4f}",
    f"Confusion matrix: {best_metrics['confusion_matrix']}",
    "",
    "Best parameters:",
    json.dumps(best_params, indent=2),
]
TRIAL_SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

print(f"Saved study table to: {STUDY_PATH}")
print(f"Saved best params to: {BEST_PARAMS_PATH}")
print(f"Saved best metrics to: {BEST_METRICS_PATH}")
print(f"Saved text summary to: {TRIAL_SUMMARY_PATH}")
