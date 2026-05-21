# Grouped 1 km Bike Crash App

This folder contains the current reduced pipeline for:

- bike-only crash matching around stations
- monthly and next-month `1 km` station datasets
- Optuna tuning for CatBoost
- grouped-counter dataset creation
- grouped Flask app generation and serving

The active project definition in this folder is:

- bike-only crashes
- within `1.0 km` of stations
- grouped nearby related counters
- next-month crash occurrence prediction

All generated outputs are written under:

- `transformed_data/`

Raw source inputs stay under:

- `data/`

## Directory Structure

- `data/`
  - raw bike counter files
  - `sites.csv`
  - `richtingen.csv`
  - crash workbook
- `templates/`
  - grouped Flask HTML template
- `static/`
  - grouped Flask JavaScript and CSS
- `transformed_data/`
  - generated weather, crash, model, grouped model, and app assets

## Run Order

If weather has not been fetched yet:

```cmd
.\.venv\Scripts\python.exe 01_weather_fetch.py
.\.venv\Scripts\python.exe 02_crash_preprocess_match_bike_radii.py
.\.venv\Scripts\python.exe 03_build_monthly_bike_crash_datasets.py
.\.venv\Scripts\python.exe 04_build_next_month_bike_crash_dataset_1km.py
.\.venv\Scripts\python.exe 05_train_optuna_catboost_1km.py
.\.venv\Scripts\python.exe 06_build_grouped_station_datasets.py
.\.venv\Scripts\python.exe 07_export_grouped_app_model.py
.\.venv\Scripts\python.exe 08_build_grouped_app_assets.py
.\.venv\Scripts\python.exe 09_app_grouped.py
```

If weather is already present at `transformed_data/weather_outputs/station_weather_hourly.csv`, start from script `02`.

Open:

```text
http://127.0.0.1:5000
```

## Script Overview

### `01_weather_fetch.py`

Fetches hourly weather history per station from Open-Meteo, starting at the station opening date. It stores the raw hourly weather table that later gets aggregated into monthly station weather features.

Inputs:
- `data/sites.csv`
- Open-Meteo API

Outputs:
- `transformed_data/weather_outputs/station_weather_hourly.csv`
- checkpoint and failure files in `transformed_data/weather_outputs/`

### `02_crash_preprocess_match_bike_radii.py`

Loads the crash workbook, filters to crashes involving a cyclist, converts the coordinates, and spatially matches crashes to stations at several radii. For the active pipeline, the relevant result is the `1 km` bike-only station-crash matching.

Inputs:
- `data/OPENDATA_MAP_2017-2024.xlsx`
- `data/sites.csv`

Outputs:
- `transformed_data/crash_outputs/crashes_bike_only.csv`
- `transformed_data/crash_outputs/crashes_bike_only_matched_to_sites_1km.csv`
- `transformed_data/crash_outputs/site_bike_crash_summary_1km.csv`

### `03_build_monthly_bike_crash_datasets.py`

Reads the raw bike count files and aggregates them to station-month level. It also aggregates hourly weather to monthly weather summaries and merges those with matched crash counts to create the main monthly modeling dataset.

Inputs:
- raw monthly bike files in `data/`
- `transformed_data/weather_outputs/station_weather_hourly.csv`
- `transformed_data/crash_outputs/crashes_bike_only_matched_to_sites_1km.csv`

Outputs:
- `transformed_data/model_outputs/station_month_bike_crash_dataset_1km.csv`

### `04_build_next_month_bike_crash_dataset_1km.py`

Takes the monthly station dataset and adds lagged, rolling, and historical features. It also creates the next-month target so each row is a forecasting row rather than only a descriptive monthly observation.

Inputs:
- `transformed_data/model_outputs/station_month_bike_crash_dataset_1km.csv`

Outputs:
- `transformed_data/model_outputs/station_month_bike_crash_dataset_1km_next_month.csv`

### `05_train_optuna_catboost_1km.py`

Tunes CatBoost with Optuna on the site-level `1 km` next-month dataset using a time-based validation split. It saves the best parameter set and validation metrics so the grouped final model can reuse that tuned configuration.

Inputs:
- `transformed_data/model_outputs/station_month_bike_crash_dataset_1km_next_month.csv`

Outputs:
- `transformed_data/model_outputs/optuna_catboost_1km_study.csv`
- `transformed_data/model_outputs/optuna_catboost_1km_best_params.json`
- `transformed_data/model_outputs/optuna_catboost_1km_best_metrics.json`
- `transformed_data/model_outputs/optuna_catboost_1km_summary.txt`

### `06_build_grouped_station_datasets.py`

Groups nearby related counters into combined entities using station names, municipality, and proximity. It then rebuilds the monthly and next-month datasets at the grouped level so the modeling unit becomes grouped counters rather than individual stations.

Inputs:
- `data/sites.csv`
- `transformed_data/model_outputs/station_month_bike_crash_dataset_1km.csv`
- `transformed_data/crash_outputs/crashes_bike_only_matched_to_sites_1km.csv`

Outputs:
- `transformed_data/combined_station_groups_1km/outputs/group_mapping.csv`
- `transformed_data/combined_station_groups_1km/outputs/station_group_month_bike_crash_dataset_1km.csv`
- `transformed_data/combined_station_groups_1km/outputs/station_group_month_bike_crash_dataset_1km_next_month.csv`
- `transformed_data/combined_station_groups_1km/outputs/group_dataset_summary.txt`

### `07_export_grouped_app_model.py`

Trains the final grouped CatBoost classifier that the app uses. It evaluates the grouped model on held-out later months, selects the prediction threshold, and exports the deployable model plus metadata and feature importance.

Inputs:
- `transformed_data/combined_station_groups_1km/outputs/station_group_month_bike_crash_dataset_1km_next_month.csv`
- `transformed_data/model_outputs/optuna_catboost_1km_best_params.json`

Outputs:
- `transformed_data/combined_station_groups_1km/models/next_month_catboost_grouped_bike_1km.cbm`
- `transformed_data/combined_station_groups_1km/models/next_month_catboost_grouped_bike_1km_metadata.json`
- `transformed_data/combined_station_groups_1km/models/next_month_catboost_grouped_bike_1km_feature_importance.csv`
- `transformed_data/combined_station_groups_1km/models/deployed_grouped_model_summary.json`

### `08_build_grouped_app_assets.py`

Builds the grouped app’s ready-to-serve assets from the grouped datasets and grouped model. This includes historical ranking, forecast ranking, baseline prediction rows, editable control ranges, and the bootstrap summary file used by the UI.

Inputs:
- `transformed_data/combined_station_groups_1km/outputs/station_group_month_bike_crash_dataset_1km.csv`
- `transformed_data/combined_station_groups_1km/outputs/station_group_month_bike_crash_dataset_1km_next_month.csv`
- `transformed_data/combined_station_groups_1km/outputs/group_mapping.csv`
- grouped model files in `transformed_data/combined_station_groups_1km/models/`

Outputs:
- `transformed_data/combined_station_groups_1km/app_assets/bootstrap.json`
- `transformed_data/combined_station_groups_1km/app_assets/historical_ranking.csv`
- `transformed_data/combined_station_groups_1km/app_assets/forecast_ranking.csv`
- `transformed_data/combined_station_groups_1km/app_assets/group_baselines.csv`
- `transformed_data/combined_station_groups_1km/app_assets/group_metadata.csv`
- `transformed_data/combined_station_groups_1km/app_assets/control_ranges.json`

### `09_app_grouped.py`

Runs the grouped Flask app. It loads the grouped app assets and grouped model into memory, serves the map interface, returns grouped station details, and performs live prediction when the user changes inputs.

Inputs:
- grouped app assets from `transformed_data/combined_station_groups_1km/app_assets/`
- grouped model files from `transformed_data/combined_station_groups_1km/models/`
- `templates/index_grouped.html`
- `static/grouped_app.js`
- `static/grouped_styles.css`

Outputs:
- local Flask app at `http://127.0.0.1:5000`

## Notes

- The pipeline now depends only on the numbered root scripts plus `data/`, `templates/`, `static/`, and `transformed_data/`.
- The active grouped app no longer depends on the old `experiments/` folder.
- The weather CSV is intentionally ignored by git.
