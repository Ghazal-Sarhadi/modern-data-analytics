# Bike Counts, Weather, and Crash Risk Project

This folder is a self-contained subset of the larger project. It keeps only the files needed for the current final pipeline:

- build bike-only crash matches around stations
- build monthly and next-month `1.0 km` bike-crash datasets
- train and export the deployed next-month classifier
- build app assets
- run the Flask dashboard

The current deployed lens in this folder is:
- **bike-only crashes**
- within **`1.0 km`** of stations
- with **next-month classification** in the app

## Run Order For This Folder

From this directory:

```cmd
python build_next_month_bike_crash_dataset_1km.py
python export_app_models.py
python build_app_assets.py
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

If you want to rebuild more of the pipeline first, use:

```cmd
python crash_preprocess_match_bike_radii.py
python build_monthly_bike_crash_datasets.py
python build_next_month_bike_crash_dataset_1km.py
python export_app_models.py
python build_app_assets.py
python app.py
```

Note:
- this folder does **not** include a virtual environment
- create your own environment here, or run it with the parent environment if you are still inside the original workspace

## Data Sources

### Bike counter data

The original AWV EcoCounter data is stored in:
- `data/sites.csv`
- `data/richtingen.csv`
- `data/data-<year>-<month>.csv`

The bike count files contain 15-minute interval counts. We focused on `FIETSERS`.

### Weather data

Weather was fetched from the Open-Meteo archive API per station location, starting from each station's opening date in `sites.csv`.

Generated files:
- `weather_outputs/station_weather_hourly.csv`
- `weather_outputs/weather_fetch_checkpoint.json`
- `weather_outputs/weather_fetch_failures.csv`

### Crash data

Crash data was added through:
- `data/OPENDATA_MAP_2017-2024.xlsx`

This file was cleaned, converted to WGS84 coordinates, filtered to the bike-station geographic extent, and matched to stations by radius.

## Original Raw Data Structure

### `data/sites.csv`

| Kolom | Type | Beschrijving |
|---|---|---|
| site ID | int | ID (primary key) |
| site nr | int | Ecocounter ID van site |
| long | float | WGS84 longitude |
| lat | float | WGS84 latitude |
| naam | text | naam site |
| domein | text | domein naam van site |
| wegnr | text | wegnummer van site |
| district | text | nummer wegendistrict |
| gemeente | text | gemeente van site |
| interval | int | meetinterval in minuten |
| datum_van | datum | datum van installatie |

### `data/richtingen.csv`

| Kolom | Type | Beschrijving |
|---|---|---|
| site ID | int | ID (primary key) |
| richting | text | IN, OUT of IN/OUT |
| naam | text | omschrijving richting |

### `data/data-<year>-<month>.csv`

| Kolom | Type | Beschrijving |
|---|---|---|
| site ID | int | FKey naar site |
| richting | text | IN, OUT of IN/OUT |
| type | text | type telling |
| van | datetime | start van interval |
| tot | datetime | einde van interval |
| aantal | int | aantal geteld in interval |

Type telling can include `FIETSERS`, `VOETGANGERS`, `PAARDEN`, `AUTOS`, `BUSSEN`, `MINIBUSSEN`, `NIET GEDEFINIEERD`, `MOTORFIETSEN`, and `KAYAKS`.

## Main Scripts

### Exploration and maps
- `read.py`: notebook-style exploration of the bike counter data
- `map_with_weather.py`: Folium map of stations with traffic and weather summaries
- `map_crash_radius_review.py`: station circles and matched crashes at `2.5 km`
- `map_crash_radius_review_500m.py`: station circles and matched crashes at `500 m`
- `map_site_confusion_500m.py`: station map colored by per-site classification performance
- `map_station_safety_historical_500m.py`: historical safety ranking map

### Weather pipeline
- `weather_fetch.py`: fetch hourly weather from Open-Meteo with checkpointing, retry, and rate-limit handling

### Crash preprocessing and matching
- `crash_preprocess_match.py`: preprocess crashes and match to stations at `2.5 km`
- `crash_preprocess_match_500m.py`: preprocess crashes and match to stations at `500 m`

### Dataset builders
- `build_monthly_crash_dataset.py`
- `build_monthly_crash_dataset_500m.py`
- `build_next_month_crash_dataset_500m.py`
- `build_next_month_crash_dataset_2_5km.py`
- `build_next_quarter_crash_dataset_500m.py`
- `build_station_safety_rankings_500m.py`

### Modeling scripts
- `train_monthly_crash_model.py`
- `train_monthly_crash_model_500m.py`
- `train_monthly_crash_count_model.py`
- `train_monthly_crash_count_model_500m.py`
- `train_next_month_crash_models_500m.py`
- `train_next_month_rnn_models_500m.py`
- `train_next_month_radius_siteid_compare.py`
- `train_next_quarter_crash_models_500m.py`
- `train_next_quarter_boosted_models_500m.py`
- `train_next_month_catboost_classifier_500m.py`
- `tune_next_month_500m_thresholds.py`

## Generated Datasets

### Crash preprocessing outputs
- `crash_outputs/crashes_filtered_to_site_bounds.csv`
- `crash_outputs/crashes_matched_to_sites_2_5km.csv`
- `crash_outputs/site_crash_summary_2_5km.csv`
- `crash_outputs/crashes_matched_to_sites_500m.csv`
- `crash_outputs/site_crash_summary_500m.csv`

### Modeling datasets
- `model_outputs/station_month_crash_dataset.csv`
- `model_outputs/station_month_crash_dataset_500m.csv`
- `model_outputs/station_month_crash_labels.csv`
- `model_outputs/station_day_features.csv`
- `model_outputs/station_day_with_monthly_crash_context.csv`

### Ranking outputs
- `model_outputs/station_safety_ranking_historical_500m.csv`
- `model_outputs/station_safety_ranking_forecast_500m.csv`

## Crash Matching Results

Crash preprocessing summary:
- original crash workbook rows: `289,532`
- after filtering to the bike-station bounding box: `180,981`

Crash matching summary:
- total bike stations: `151`
- stations used at `2.5 km`: `151`
- stations used at `500 m`: `151`

Matched crashes:
- `2.5 km`
  - unique crash records matched to at least one station: `41,196`
  - station-crash match rows: `86,811`
- `500 m`
  - unique crash records matched to at least one station: `3,691`
  - station-crash match rows: `5,948`

The `500 m` radius is much stricter and more local. The `2.5 km` radius produces a broader area-risk definition.

## Important Modeling Decision: Leakage Removal

The first monthly modeling datasets accidentally contained same-month crash-distance information:
- `nearest_crash_distance_km`
- `average_crash_distance_km`

These leak the target into the features. They were removed from the final modeling datasets and the models were retrained. All headline results below refer to the cleaned versions.

## Why Monthly and Quarterly Targets

The crash file provides year and month, but not day-level timestamps. Because of that:
- daily crash prediction was not defensible
- monthly station-level modeling became the main setup
- later, quarter-ahead forecasting was added because it is more stable than month-ahead forecasting

## Monthly Modeling Results

Average target size:
- `2.5 km` mean monthly crash count per station-month: `6.54`
- `500 m` mean monthly crash count per station-month: `0.445`

### Monthly classification

Target:
- `crash_happened = 1` if at least one crash occurred near the station in that month

`2.5 km` results:
- logistic regression: `accuracy 0.5978`, `F1 0.7286`, `ROC AUC 0.6975`
- random forest: `accuracy 0.9220`, `F1 0.9589`, `ROC AUC 0.7794`

`500 m` results:
- logistic regression: `accuracy 0.6267`, `F1 0.4932`, `ROC AUC 0.6612`
- random forest: `accuracy 0.7231`, `F1 0.4384`, `ROC AUC 0.7096`

### Monthly count regression

Target:
- monthly crash count near each station

`2.5 km` results:
- Poisson regression: `MAE 3.9720`, `RMSE 5.2218`, `R2 0.2535`
- random forest regressor: `MAE 2.5223`, `RMSE 3.4017`, `R2 0.6832`

`500 m` results:
- Poisson regression: `MAE 0.5881`, `RMSE 0.7623`, `R2 0.0780`
- random forest regressor: `MAE 0.5495`, `RMSE 0.7219`, `R2 0.1731`

Interpretation:
- `2.5 km` is easier to model, but less local
- `500 m` is more station-specific, but much sparser and harder

## Next-Month Forecasting Results

We built next-month forecasting datasets using lagged crash, traffic, and weather features.

### 500 m next-month classification

Baseline tabular models:
- logistic regression: `accuracy 0.6531`, `F1 0.5229`, `ROC AUC 0.7083`
- random forest: `accuracy 0.7409`, `F1 0.4413`, `ROC AUC 0.7112`

### 500 m next-month count regression

- Poisson regression: `MAE 0.5503`, `RMSE 0.7273`, `R2 0.1279`
- random forest regressor: `MAE 0.6083`, `RMSE 0.7674`, `R2 0.0290`

### 500 m next-month recurrent models

Classification:
- LSTM: `accuracy 0.6633`, `F1 0.4646`, `ROC AUC 0.6793`
- GRU: `accuracy 0.7173`, `F1 0.3624`, `ROC AUC 0.6358`

Regression:
- LSTM regressor: `MAE 0.5428`, `RMSE 0.7698`, `R2 0.0092`
- GRU regressor: `MAE 0.5008`, `RMSE 0.7362`, `R2 0.0937`

Interpretation:
- RNNs did not beat the best tabular baselines
- the current data is better handled by tabular models than by sequence models

## Site-ID Ablation and Radius Comparison

We also tested next-month forecasting with and without `site_id`.

Main conclusion:
- removing `site_id` barely changed performance
- the models are using real crash-history, traffic, weather, and seasonality signals rather than only memorizing station identity

Next-month summary:
- `500 m` remains difficult with or without `site_id`
- `2.5 km` remains substantially easier with or without `site_id`

## Next-Quarter Forecasting Results

Quarter-ahead prediction is more stable than month-ahead prediction.

### 500 m next-quarter baseline models

Classification:
- logistic regression: `accuracy 0.6494`, `F1 0.7026`, `ROC AUC 0.7114`
- random forest: `accuracy 0.6653`, `F1 0.7406`, `ROC AUC 0.7464`

Regression:
- Poisson regression: `MAE 1.0679`, `RMSE 1.4441`, `R2 0.2610`
- random forest regressor: `MAE 1.0301`, `RMSE 1.3713`, `R2 0.3336`

### 500 m next-quarter boosted models

Classification:
- CatBoost: `accuracy 0.7018`, `F1 0.7455`, `ROC AUC 0.7570`
- XGBoost: `accuracy 0.6706`, `F1 0.7256`, `ROC AUC 0.7369`
- HistGradientBoosting: `accuracy 0.6899`, `F1 0.7347`, `ROC AUC 0.7500`

Regression:
- CatBoost regressor: `MAE 0.9800`, `RMSE 1.3521`, `R2 0.3521`
- XGBoost regressor: `MAE 1.0443`, `RMSE 1.3836`, `R2 0.3216`
- HistGradientBoosting regressor: `MAE 1.0305`, `RMSE 1.4060`, `R2 0.2994`

Main conclusion:
- the best current `500 m` quarter-ahead models are CatBoost for both classification and regression
- quarter-ahead prediction is clearly easier than month-ahead prediction

## Threshold Tuning for Next-Month 500m CatBoost

At the default threshold `0.50`, the next-month `500 m` CatBoost classifier gave:
- accuracy `0.7397`
- F1 `0.4401`
- ROC AUC `0.7152`

After threshold tuning, the best F1 was at threshold `0.23`:
- accuracy `0.6628`
- precision `0.4532`
- recall `0.6728`
- F1 `0.5416`

Interpretation:
- default threshold is conservative
- lower threshold improves recall and F1 substantially
- CatBoost with threshold tuning is the best current next-month `500 m` classifier by F1

## Safety Ranking Outputs

We built two station-level safety ranking tables for `500 m`:

### Historical ranking

Built from:
- crashes within `500 m`
- cyclist exposure
- smoothed crashes per `10,000` cyclists

Top historically riskiest stations:
1. `Maasmechelen teller 2`
2. `Maasmechelen teller 1`
3. `St. pieters leeuw teller 2`
4. `St. pieters leeuw teller 1`
5. `Mechelen teller 1`
6. `Gent teller 2`
7. `test leuven`
8. `Diest teller 2`
9. `Deinze teller 1`
10. `Genk`

### Forecast ranking

Built from:
- predicted next-month crash probability
- predicted next-month crash count

Top forecast-risk stations:
1. `Leuven teller 1`
2. `TEST Validatie BRUGGE Y2H22022134`
3. `leuven totem`
4. `Kortrijk 1`
5. `Kortrijk 2`
6. `Hasselt-Kempische brug`
7. `Eco Display Classic Budastraat`
8. `Leuven teller 2`
9. `Gent teller 1`
10. `Evergem 1`

## Map Outputs

Generated Folium maps:
- `belgium_sites_map_with_weather.html`
- `crash_radius_review_map.html`
- `crash_radius_review_map_500m.html`
- `site_confusion_500m_map.html`
- `station_safety_historical_500m_map.html`

These maps were used to:
- validate station coordinates
- compare `500 m` and `2.5 km` radius assumptions
- inspect per-site model behavior
- present the historical safety ranking geographically

## Environment Setup

The repository now includes:
- `requirements.txt`
- `pyproject.toml`

Recommended clean setup with `uv`:

```cmd
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

Or:

```cmd
uv venv
.venv\Scripts\activate
uv pip install .
```

## Run The App

Build the deployable model artifacts and app assets:

```cmd
python export_app_models.py
python build_app_assets.py
```

Start the local app:

```cmd
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The app is now the main project entrypoint. It provides:
- ranked station markers for `500 m` and `2.5 km`
- historical and forecast safety modes
- dense station summaries
- a live next-month prediction engine per selected radius

## Practical Conclusions

Main project conclusions:
- `500 m` is the better local station-safety definition, but it is harder to predict
- `2.5 km` is easier to predict, but it represents broader area risk rather than strict station-local risk
- monthly modeling works, but quarter-ahead forecasting is stronger and more stable than month-ahead forecasting
- leakage had to be removed before trusting results
- CatBoost is the strongest current `500 m` quarter-ahead model
- tuned CatBoost is the strongest current `500 m` next-month classifier by F1
- historical and forecast safety rankings can now be used to order stations by observed and predicted risk
