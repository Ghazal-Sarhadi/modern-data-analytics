#%%
from __future__ import annotations

import json
import math
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


# %%
# Install requirements once before running this script:
# pip install openmeteo-requests requests-cache retry-requests numpy pandas


# %%
# Configuration
DATA_DIR = Path("data")
OUTPUT_DIR = Path("weather_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SITES_CSV = DATA_DIR / "sites.csv"
WEATHER_CSV = OUTPUT_DIR / "station_weather_hourly.csv"
CHECKPOINT_JSON = OUTPUT_DIR / "weather_fetch_checkpoint.json"
FAILED_CSV = OUTPUT_DIR / "weather_fetch_failures.csv"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo archive data is typically available with a delay.
END_DATE = date.today() - timedelta(days=5)

# Add an explicit pause between stations to reduce 429 Too Many Requests responses.
REQUEST_DELAY_SECONDS = 5.0
RATE_LIMIT_WAIT_SECONDS = 65
MAX_RATE_LIMIT_RETRIES = 5

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "weather_code",
]


# %%
# Setup the Open-Meteo API client with cache and retry on error.
cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.5)
openmeteo = openmeteo_requests.Client(session=retry_session)


# %%
# Load the sites metadata.
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

sites_df = pd.read_csv(SITES_CSV, names=site_columns)
sites_df["datum_van"] = pd.to_datetime(sites_df["datum_van"], errors="coerce").dt.date
sites_df = sites_df.dropna(subset=["lat", "long", "datum_van"]).copy()

sites_df.head()


# %%
def load_checkpoint() -> Dict[str, object]:
    if CHECKPOINT_JSON.exists():
        return json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))
    return {
        "completed_site_ids": [],
        "last_completed_site_id": None,
        "updated_at": None,
    }


def save_checkpoint(completed_site_ids: List[int], last_completed_site_id: int | None) -> None:
    payload = {
        "completed_site_ids": sorted(set(completed_site_ids)),
        "last_completed_site_id": last_completed_site_id,
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    CHECKPOINT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_failure(site_row: pd.Series, error_message: str) -> None:
    failure_row = pd.DataFrame(
        [
            {
                "site_id": site_row["site_id"],
                "naam": site_row["naam"],
                "gemeente": site_row["gemeente"],
                "lat": site_row["lat"],
                "long": site_row["long"],
                "datum_van": site_row["datum_van"],
                "error": error_message,
                "logged_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
        ]
    )
    failure_row.to_csv(
        FAILED_CSV,
        mode="a",
        index=False,
        header=not FAILED_CSV.exists(),
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


def fetch_weather_response(site_row: pd.Series):
    params = {
        "latitude": float(site_row["lat"]),
        "longitude": float(site_row["long"]),
        "start_date": site_row["datum_van"].isoformat(),
        "end_date": END_DATE.isoformat(),
        "hourly": HOURLY_VARIABLES,
        "timezone": "Europe/Brussels",
    }
    responses = openmeteo.weather_api(OPEN_METEO_ARCHIVE_URL, params=params)
    return responses[0]


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "request limit exceeded" in message or "try again in one minute" in message


def weather_response_to_frame(response, site_row: pd.Series) -> pd.DataFrame:
    hourly = response.Hourly()

    hourly_data = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }

    for index, variable in enumerate(HOURLY_VARIABLES):
        hourly_data[variable] = hourly.Variables(index).ValuesAsNumpy()

    weather_df = pd.DataFrame(data=hourly_data)

    weather_grid_latitude = response.Latitude()
    weather_grid_longitude = response.Longitude()
    weather_grid_elevation = response.Elevation()
    timezone_name = response.Timezone()
    timezone_abbreviation = response.TimezoneAbbreviation()
    utc_offset_seconds = response.UtcOffsetSeconds()

    distance_km = haversine_km(
        float(site_row["lat"]),
        float(site_row["long"]),
        float(weather_grid_latitude),
        float(weather_grid_longitude),
    )

    weather_df.insert(0, "site_id", site_row["site_id"])
    weather_df.insert(1, "site_name", site_row["naam"])
    weather_df.insert(2, "gemeente", site_row["gemeente"])
    weather_df.insert(3, "latitude", site_row["lat"])
    weather_df.insert(4, "longitude", site_row["long"])
    weather_df.insert(5, "opening_date", site_row["datum_van"].isoformat())
    weather_df.insert(6, "weather_grid_latitude", weather_grid_latitude)
    weather_df.insert(7, "weather_grid_longitude", weather_grid_longitude)
    weather_df.insert(8, "weather_grid_elevation", weather_grid_elevation)
    weather_df.insert(9, "station_to_weather_grid_km", distance_km)
    weather_df.insert(10, "timezone", timezone_name)
    weather_df.insert(11, "timezone_abbreviation", timezone_abbreviation)
    weather_df.insert(12, "utc_offset_seconds", utc_offset_seconds)

    return weather_df


# %%
# Determine which sites still need to be fetched.
checkpoint = load_checkpoint()
completed_site_ids = {int(site_id) for site_id in checkpoint["completed_site_ids"]}
sites_to_fetch_df = sites_df[~sites_df["site_id"].isin(completed_site_ids)].copy()

len(sites_to_fetch_df), END_DATE.isoformat()


# %%
# Print a short resume summary before the fetch starts.
total_sites = len(sites_df)
completed_sites = len(completed_site_ids)
remaining_sites = len(sites_to_fetch_df)
last_completed_site_id = checkpoint.get("last_completed_site_id")

print(f"Total sites: {total_sites}")
print(f"Completed sites: {completed_sites}")
print(f"Remaining sites: {remaining_sites}")
print(f"Last completed site ID: {last_completed_site_id}")

if remaining_sites > 0:
    next_site = sites_to_fetch_df.iloc[0]
    print(
        "Next site to fetch: "
        f"{int(next_site['site_id'])} - {next_site['naam']} ({next_site['gemeente']})"
    )
else:
    print("No remaining sites to fetch.")


# %%
# Fetch hourly weather per site from its opening date until END_DATE.
for site_row in sites_to_fetch_df.itertuples(index=False):
    if site_row.datum_van > END_DATE:
        save_checkpoint(list(completed_site_ids), checkpoint.get("last_completed_site_id"))
        continue

    site_series = pd.Series(site_row._asdict())

    try:
        response = None
        for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = fetch_weather_response(site_series)
                break
            except Exception as exc:
                append_failure(
                    site_series,
                    f"attempt={attempt}; retryable={is_rate_limit_error(exc)}; error={exc}",
                )

                if not is_rate_limit_error(exc) or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise

                wait_seconds = RATE_LIMIT_WAIT_SECONDS * attempt
                print(
                    f"Rate limit hit for site {site_row.site_id}. "
                    f"Waiting {wait_seconds} seconds before retry {attempt + 1}."
                )
                time.sleep(wait_seconds)

        weather_df = weather_response_to_frame(response, site_series)

        weather_df.to_csv(
            WEATHER_CSV,
            mode="a",
            index=False,
            header=not WEATHER_CSV.exists(),
        )

        completed_site_ids.add(int(site_row.site_id))
        save_checkpoint(list(completed_site_ids), int(site_row.site_id))
        time.sleep(REQUEST_DELAY_SECONDS)

    except Exception as exc:
        append_failure(site_series, f"final_failure={exc}")
        save_checkpoint(list(completed_site_ids), checkpoint.get("last_completed_site_id"))
        raise


# %%
# Quick preview of the created weather file.
if WEATHER_CSV.exists():
    weather_preview_df = pd.read_csv(WEATHER_CSV, nrows=5)
    weather_preview_df


# %%
# Restart snippet:
# If a request fails mid-run, keep the existing output files and run only these lines.
#
# import runpy
# from pathlib import Path
#
# checkpoint_path = Path("weather_outputs/weather_fetch_checkpoint.json")
# print(checkpoint_path.read_text(encoding="utf-8"))
# runpy.run_path("weather_fetch.py")
