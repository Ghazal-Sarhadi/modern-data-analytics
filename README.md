# Grouped Bike Crash App

This project contains the grouped bike crash forecasting pipeline and the Flask map app.

The active setup consists of:

- Dataset Creation
- EDA in notebooks
- Model Selection and Hyperparameter Tuning
- Map App


Generated outputs are written under:

- `transformed_data/`

Raw source datas stay under:

- `data/`

## Structure

- `src/`
  - pipeline and app scripts
- `app/`
  - Flask templates and static frontend files
- `data/`
  - raw bike files, station metadata, directions, crash workbook, official dangerous points workbook
- `notebooks/`
  - exploratory analysis and presentation notebooks
- `transformed_data/`
  - generated weather, crash, model, grouped model, and app assets

## Run Order

If weather has not been fetched yet:

```cmd
python src/01_weather_fetch.py
python src/02_crash_preprocess_match_bike_radii.py
python src/03_build_monthly_bike_crash_datasets.py
python src/04_build_next_month_bike_crash_datasets.py
python src/05a_build_grouped_station_datasets_1km.py
python src/06_train_optuna_catboost_1km.py
python src/07_export_grouped_app_model.py
python src/08_build_grouped_app_assets.py
python src/09_app_grouped.py
```

Taking the weather data from the OpenMeteo API can take significant time due to API request limits. If weather is already present at `transformed_data/weather_outputs/station_weather_hourly.csv`, start from script `02`. Since this file is bigger than the size GitHub accepts it, it can be downloaded in this link https://drive.google.com/file/d/1nG5Do5cDaKh88VBImOh3seyomLJKNNBy/view?usp=sharing

Open:

```text
http://127.0.0.1:5000
```

## `src/` Scripts

### `src/01_weather_fetch.py`

Fetches hourly weather per station from Open-Meteo, starting from each station's opening date and ending at the configured latest available date. This produces the raw hourly weather table used later for monthly station weather summaries.

Inputs:
- `data/sites.csv`
- Open-Meteo API

Outputs:
- `transformed_data/weather_outputs/station_weather_hourly.csv`
- weather checkpoint and failure files in `transformed_data/weather_outputs/`

### `src/02_crash_preprocess_match_bike_radii.py`

Loads the crash workbook, filters to crashes involving a bicycle road user, converts crash coordinates, and matches crashes to stations at several radii. The active grouped pipeline uses the bike-only `1 km` matching.

Inputs:
- `data/OPENDATA_MAP_2017-2024.xlsx`
- `data/sites.csv`

Outputs:
- `transformed_data/crash_outputs/crashes_bike_only.csv`
- `transformed_data/crash_outputs/crashes_bike_only_matched_to_sites_500m.csv`
- `transformed_data/crash_outputs/crashes_bike_only_matched_to_sites_1km.csv`
- `transformed_data/crash_outputs/crashes_bike_only_matched_to_sites_2_5km.csv`
- corresponding summary CSVs in `transformed_data/crash_outputs/`

### `src/03_build_monthly_bike_crash_datasets.py`

Aggregates raw bike counts to station's month level, aggregates hourly weather to monthly weather summaries, and merges in bike only crash counts. This creates the base monthly station datasets used for later forecasting features.

Inputs:
- raw monthly bike files in `data/`
- `transformed_data/weather_outputs/station_weather_hourly.csv`
- bike-only matched crash files in `transformed_data/crash_outputs/`

Outputs:
- `transformed_data/model_outputs/station_month_bike_crash_dataset_500m.csv`
- `transformed_data/model_outputs/station_month_bike_crash_dataset_1km.csv`
- `transformed_data/model_outputs/station_month_bike_crash_dataset_2_5km.csv`

### `src/04_build_next_month_bike_crash_datasets.py`

Takes each monthly station dataset and builds a next month forecasting table with lagged, rolling, change, and historical features. It keeps only rows where a true consecutive next month exists, so the result is a forecasting dataset.

Inputs:
- `transformed_data/model_outputs/station_month_bike_crash_dataset_500m.csv`
- `transformed_data/model_outputs/station_month_bike_crash_dataset_1km.csv`
- `transformed_data/model_outputs/station_month_bike_crash_dataset_2_5km.csv`

Outputs:
- `transformed_data/model_outputs/station_month_bike_crash_dataset_500m_next_month.csv`
- `transformed_data/model_outputs/station_month_bike_crash_dataset_1km_next_month.csv`
- `transformed_data/model_outputs/station_month_bike_crash_dataset_2_5km_next_month.csv`

### `src/05a_build_grouped_station_datasets_1km.py`

Groups nearby related counters into grouped entities for the `1 km` setup using names, municipality, and proximity. It then rebuilds the monthly and next month datasets at the grouped level, so the modeling unit becomes grouped counters instead of individual stations.

Inputs:
- `data/sites.csv`
- `transformed_data/model_outputs/station_month_bike_crash_dataset_1km.csv`
- `transformed_data/crash_outputs/crashes_bike_only_matched_to_sites_1km.csv`

Outputs:
- `transformed_data/combined_station_groups_1km/outputs/group_mapping.csv`
- `transformed_data/combined_station_groups_1km/outputs/station_group_month_bike_crash_dataset_1km.csv`
- `transformed_data/combined_station_groups_1km/outputs/station_group_month_bike_crash_dataset_1km_next_month.csv`
- `transformed_data/combined_station_groups_1km/outputs/group_dataset_summary.txt`

### `src/05b_build_grouped_station_datasets_500m.py`

Builds the grouped-counter datasets for the `500 m radius` version of the dataset included in the report. This is only a  comparison path, results from this script is not used in the app.

Inputs:
- `data/sites.csv`
- `transformed_data/model_outputs/station_month_bike_crash_dataset_500m.csv`
- `transformed_data/crash_outputs/crashes_bike_only_matched_to_sites_500m.csv`

Outputs:
- grouped `500 m` outputs under `transformed_data/combined_station_groups_500m/`

### `src/06_train_optuna_catboost_1km.py`

Tunes the grouped `1 km` CatBoost classifier's hyperparameters with Optuna. It uses a time-based train/validation/test split, optimizes validation ROC AUC, then chooses the classification threshold on validation F1 and reports final performance on untouched test months.

Inputs:
- `transformed_data/combined_station_groups_1km/outputs/station_group_month_bike_crash_dataset_1km_next_month.csv`

Outputs:
- `transformed_data/model_outputs/optuna_catboost_1km_study.csv`
- `transformed_data/model_outputs/optuna_catboost_1km_best_params.json`
- `transformed_data/model_outputs/optuna_catboost_1km_best_metrics.json`
- `transformed_data/model_outputs/optuna_catboost_1km_summary.txt`

### `src/07_export_grouped_app_model.py`

Trains and exports the final grouped CatBoost model used by the app. It reads the grouped next month dataset, applies the tuned parameter set, selects the deployed threshold, and writes model metadata and feature importance for the UI layer.

Inputs:
- `transformed_data/combined_station_groups_1km/outputs/station_group_month_bike_crash_dataset_1km_next_month.csv`
- `transformed_data/model_outputs/optuna_catboost_1km_best_params.json`

Outputs:
- `transformed_data/combined_station_groups_1km/models/next_month_catboost_grouped_bike_1km.cbm`
- `transformed_data/combined_station_groups_1km/models/next_month_catboost_grouped_bike_1km_metadata.json`
- `transformed_data/combined_station_groups_1km/models/next_month_catboost_grouped_bike_1km_feature_importance.csv`
- `transformed_data/combined_station_groups_1km/models/deployed_grouped_model_summary.json`

### `src/08_build_grouped_app_assets.py`

Builds the grouped app assets from the grouped datasets and grouped model. This includes historical ranking, forecast ranking, baseline prediction rows, editable control ranges, bootstrap summary data, and the hidden risk flags used in the UI.

Inputs:
- `transformed_data/combined_station_groups_1km/outputs/station_group_month_bike_crash_dataset_1km.csv`
- `transformed_data/combined_station_groups_1km/outputs/station_group_month_bike_crash_dataset_1km_next_month.csv`
- `transformed_data/combined_station_groups_1km/outputs/group_mapping.csv`
- grouped model files in `transformed_data/combined_station_groups_1km/models/`
- `data/Dynamische lijst 2025.xlsx`

Outputs:
- `transformed_data/combined_station_groups_1km/app_assets/bootstrap.json`
- `transformed_data/combined_station_groups_1km/app_assets/historical_ranking.csv`
- `transformed_data/combined_station_groups_1km/app_assets/forecast_ranking.csv`
- `transformed_data/combined_station_groups_1km/app_assets/group_baselines.csv`
- `transformed_data/combined_station_groups_1km/app_assets/group_metadata.csv`
- `transformed_data/combined_station_groups_1km/app_assets/control_ranges.json`

### `src/09_app_grouped.py`

Runs the grouped Flask app. It loads grouped assets and the grouped CatBoost model into memory, serves the map UI, returns group details, and performs live prediction when the user adjusts the prediction engine controls.

Inputs:
- grouped app assets from `transformed_data/combined_station_groups_1km/app_assets/`
- grouped model files from `transformed_data/combined_station_groups_1km/models/`
- `app/templates/index_grouped.html`
- `app/static/grouped_app.js`
- `app/static/grouped_styles.css`

Outputs:
- local Flask app at `http://127.0.0.1:5000`

## Notebooks

### `notebooks/01_eda.ipynb`

Exploratory data analysis notebook for the raw bike count files, station metadata, directions table, and crash workbook. It focuses on sensor coverage, null patterns, descriptive distributions, and early data-quality checks.

### `notebooks/02_model_comparison.ipynb`

Exploratory model-comparison notebook for the grouped forecasting dataset. It compares several model families and also contains radius-comparison work.

### `notebooks/03_model_results.ipynb`

Presentation-oriented notebook for reading and summarizing saved grouped model results. It is used to inspect exported metrics, rankings, and model outputs.

### `notebooks/04_risk_analyisis.ipynb`

Risk interpretation notebook that compares grouped forecast outputs against the official dangerous-points workbook. It identifies model predicted high risk groups, including the “hidden risk spots” whose municipalities are not present in the official list.

## Notes

- The active grouped app depends on `src/`, `app/`, `data/`, and `transformed_data/`.
- The weather CSV is intentionally ignored by git.
