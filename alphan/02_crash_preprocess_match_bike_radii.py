from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.neighbors import BallTree


# %%
# Paths and configuration
data_dir = Path("data")
transformed_data_dir = Path("transformed_data")
output_dir = transformed_data_dir / "crash_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

sites_csv = data_dir / "sites.csv"
crash_xlsx = data_dir / "OPENDATA_MAP_2017-2024.xlsx"

bike_only_crashes_csv = output_dir / "crashes_bike_only.csv"
bike_only_filtered_csv = output_dir / "crashes_bike_only_filtered_to_site_bounds.csv"

radius_configs = {
    "2_5km": {
        "radius_km": 2.5,
        "matched_csv": output_dir / "crashes_bike_only_matched_to_sites_2_5km.csv",
        "summary_csv": output_dir / "site_bike_crash_summary_2_5km.csv",
        "summary_count_column": "bike_crash_count_2_5km",
    },
    "1km": {
        "radius_km": 1.0,
        "matched_csv": output_dir / "crashes_bike_only_matched_to_sites_1km.csv",
        "summary_csv": output_dir / "site_bike_crash_summary_1km.csv",
        "summary_count_column": "bike_crash_count_1km",
    },
    "500m": {
        "radius_km": 0.5,
        "matched_csv": output_dir / "crashes_bike_only_matched_to_sites_500m.csv",
        "summary_csv": output_dir / "site_bike_crash_summary_500m.csv",
        "summary_count_column": "bike_crash_count_500m",
    },
    "250m": {
        "radius_km": 0.25,
        "matched_csv": output_dir / "crashes_bike_only_matched_to_sites_250m.csv",
        "summary_csv": output_dir / "site_bike_crash_summary_250m.csv",
        "summary_count_column": "bike_crash_count_250m",
    },
}

earth_radius_km = 6371.0
bicycle_code = 3


# %%
# Load site metadata and coordinate envelope.
site_columns = [
    "site_id",
    "site_nr",
    "long",
    "lat",
    "naam",
    "domein",
    "wegnr",
    "district",
    "gemeente",
    "interval",
    "datum_van",
]
sites_df = pd.read_csv(sites_csv, names=site_columns)
site_lat_min = sites_df["lat"].min()
site_lat_max = sites_df["lat"].max()
site_long_min = sites_df["long"].min()
site_long_max = sites_df["long"].max()


# %%
# Load and normalize the crash workbook.
crashes_df = pd.read_excel(crash_xlsx)

for column in crashes_df.columns:
    if crashes_df[column].dtype != "object":
        continue

    cleaned = crashes_df[column].astype(str).str.strip().str.replace(",", ".", regex=False)
    numeric_values = pd.to_numeric(cleaned, errors="coerce")
    non_null_count = crashes_df[column].notna().sum()

    if non_null_count > 0 and numeric_values.notna().sum() >= 0.95 * non_null_count:
        crashes_df[column] = numeric_values


# %%
# Keep only crashes where one of the two road users is a bicycle.
bike_mask = (
    (crashes_df["CD_ROAD_USR_TYPE1"] == bicycle_code)
    | (crashes_df["CD_ROAD_USR_TYPE2"] == bicycle_code)
)
bike_crashes_df = crashes_df[bike_mask].copy()
bike_crashes_df.to_csv(bike_only_crashes_csv, index=False)


# %%
# Convert Lambert 72 coordinates to latitude/longitude.
transformer = Transformer.from_crs("EPSG:31370", "EPSG:4326", always_xy=True)
converted_longitudes, converted_latitudes = transformer.transform(
    bike_crashes_df["MS_X_COORD"].tolist(),
    bike_crashes_df["MS_Y_COORD"].tolist(),
)
bike_crashes_df["crash_long"] = converted_longitudes
bike_crashes_df["crash_lat"] = converted_latitudes


# %%
# Filter the bike-only crashes to the station coordinate envelope.
bike_crashes_filtered_df = bike_crashes_df[
    bike_crashes_df["crash_lat"].between(site_lat_min, site_lat_max)
    & bike_crashes_df["crash_long"].between(site_long_min, site_long_max)
].copy()
bike_crashes_filtered_df.to_csv(bike_only_filtered_csv, index=False)


# %%
# Prepare the station BallTree once and reuse it for both radii.
site_radians = np.radians(sites_df[["lat", "long"]].astype(float).to_numpy())
crash_radians = np.radians(bike_crashes_filtered_df[["crash_lat", "crash_long"]].astype(float).to_numpy())
tree = BallTree(site_radians, metric="haversine")


def build_radius_matches(radius_label: str, radius_km: float, matched_csv: Path, summary_csv: Path, summary_count_column: str):
    radius_radians = radius_km / earth_radius_km
    match_indices, match_distances = tree.query_radius(
        crash_radians,
        r=radius_radians,
        return_distance=True,
        sort_results=True,
    )

    match_rows = []

    for crash_row_position, (site_positions, distances) in enumerate(zip(match_indices, match_distances)):
        if len(site_positions) == 0:
            continue

        crash_row = bike_crashes_filtered_df.iloc[crash_row_position]
        for site_position, distance_radians in zip(site_positions, distances):
            site_row = sites_df.iloc[site_position]
            match_rows.append(
                {
                    "crash_row_id": int(crash_row_position),
                    "site_id": int(site_row["site_id"]),
                    "site_name": site_row["naam"],
                    "site_gemeente": site_row["gemeente"],
                    "site_lat": site_row["lat"],
                    "site_long": site_row["long"],
                    "distance_km": distance_radians * earth_radius_km,
                    "crash_lat": crash_row["crash_lat"],
                    "crash_long": crash_row["crash_long"],
                    "DT_YEAR_COLLISION": crash_row["DT_YEAR_COLLISION"],
                    "DT_MONTH_COLLISION": crash_row["DT_MONTH_COLLISION"],
                    "DT_TIME": crash_row["DT_TIME"],
                    "CD_NIS": crash_row["CD_NIS"],
                    "TX_MUNTY_COLLISION_NL": crash_row.get("TX_MUNTY_COLLISION_NL"),
                    "TX_WEATHER_NL": crash_row.get("TX_WEATHER_NL"),
                    "TX_ROAD_CONDITION_NL": crash_row.get("TX_ROAD_CONDITION_NL"),
                    "TX_LIGHT_CONDITION_NL": crash_row.get("TX_LIGHT_CONDITION_NL"),
                    "TX_CLASS_ACCIDENTS_NL": crash_row.get("TX_CLASS_ACCIDENTS_NL"),
                    "CD_ROAD_USR_TYPE1": crash_row.get("CD_ROAD_USR_TYPE1"),
                    "TX_ROAD_USR_TYPE1_NL": crash_row.get("TX_ROAD_USR_TYPE1_NL"),
                    "CD_ROAD_USR_TYPE2": crash_row.get("CD_ROAD_USR_TYPE2"),
                    "TX_ROAD_USR_TYPE2_NL": crash_row.get("TX_ROAD_USR_TYPE2_NL"),
                }
            )

    matched_df = pd.DataFrame(match_rows)
    matched_df.to_csv(matched_csv, index=False)

    summary_df = (
        matched_df.groupby(["site_id", "site_name", "site_gemeente"], as_index=False)
        .agg(
            **{
                summary_count_column: ("crash_row_id", "count"),
                "nearest_crash_distance_km": ("distance_km", "min"),
                "average_crash_distance_km": ("distance_km", "mean"),
            }
        )
        .sort_values(summary_count_column, ascending=False)
    )
    summary_df.to_csv(summary_csv, index=False)

    unique_matched_crashes = matched_df["crash_row_id"].nunique()
    used_stations = matched_df["site_id"].nunique()

    print(
        f"{radius_label}: matched rows={len(matched_df)}, "
        f"unique bike crashes matched={unique_matched_crashes}, "
        f"stations used={used_stations}"
    )


# %%
# Build radius-specific bike-only crash matches and summaries.
for radius_label, radius_config in radius_configs.items():
    build_radius_matches(
        radius_label=radius_label,
        radius_km=radius_config["radius_km"],
        matched_csv=radius_config["matched_csv"],
        summary_csv=radius_config["summary_csv"],
        summary_count_column=radius_config["summary_count_column"],
    )


# %%
print(
    "Bike-only crash preprocessing complete. "
    f"Original crashes={len(crashes_df)}, "
    f"bike-only crashes={len(bike_crashes_df)}, "
    f"bike-only crashes in site bounds={len(bike_crashes_filtered_df)}"
)
