from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st

from app import STATE


RADIUS_KEY = "1km_bike"


st.set_page_config(
    page_title="Station Safety Dashboard",
    page_icon="🚲",
    layout="wide",
)


def format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{digits}f}"


def get_ranked_stations(mode: str) -> pd.DataFrame:
    stations = pd.DataFrame(STATE.bootstrap["stations"][RADIUS_KEY][mode]).copy()
    valid_site_ids = set(STATE.station_baselines[RADIUS_KEY].index.tolist())
    stations = stations[stations["site_id"].isin(valid_site_ids)].copy()
    stations["rank_label"] = stations["rank"].astype(str)
    return stations


def filter_stations(stations: pd.DataFrame, municipality: str, search: str, view: str) -> pd.DataFrame:
    filtered = stations.copy()
    if municipality:
        filtered = filtered[filtered["municipality"] == municipality]
    if search:
        filtered = filtered[filtered["name"].str.contains(search, case=False, na=False)]
    filtered = filtered.sort_values("rank")
    if view == "Top 50 riskiest":
        filtered = filtered.head(50)
    elif view == "Top 50 safest":
        filtered = filtered.tail(50)
    return filtered


def build_map(stations: pd.DataFrame, selected_site_id: int | None):
    if stations.empty:
        return None

    plot_df = stations.copy()
    plot_df["radius"] = plot_df["score"].rank(pct=True, ascending=False).fillna(0.5) * 1200 + 250
    plot_df["fill_color"] = plot_df["rank"].rank(pct=True).apply(
        lambda x: [int(210 - 150 * x), int(120 + 80 * x), int(70 + 60 * x), 190]
    )

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=plot_df,
            get_position="[long, lat]",
            get_fill_color="fill_color",
            get_radius="radius",
            pickable=True,
            radius_min_pixels=5,
            radius_max_pixels=25,
        ),
        pdk.Layer(
            "TextLayer",
            data=plot_df,
            get_position="[long, lat]",
            get_text="rank_label",
            get_size=14,
            get_color=[255, 255, 255, 240],
            get_alignment_baseline="'center'",
            get_text_anchor="'middle'",
            pickable=False,
        ),
    ]

    if selected_site_id is not None and selected_site_id in plot_df["site_id"].values:
        selected = plot_df[plot_df["site_id"] == selected_site_id].iloc[0]
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame([selected]),
                get_position="[long, lat]",
                get_fill_color=[170, 63, 55, 40],
                get_line_color=[170, 63, 55, 220],
                get_line_width=4,
                stroked=True,
                filled=True,
                get_radius=1000,
                radius_min_pixels=12,
                radius_max_pixels=200,
                pickable=False,
            )
        )

    center_lat = float(plot_df["lat"].mean())
    center_long = float(plot_df["long"].mean())
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_long, zoom=7.5, pitch=0),
        tooltip={"text": "#{rank} {name}\n{municipality}\nScore: {score}"},
        map_style="light",
    )


st.title("Station Safety Dashboard")
st.caption("Bike-only crashes within 1.0 km, with historical ranking and next-month prediction.")

summary = STATE.bootstrap["radius_summaries"][RADIUS_KEY]

with st.sidebar:
    st.header("Controls")
    st.text_input("Radius", value="1.0 km bike-only", disabled=True)
    mode = st.selectbox(
        "Ranking mode",
        ["historical", "forecast"],
        format_func=lambda v: "Historical risk" if v == "historical" else "Forecast risk",
    )
    municipality = st.selectbox(
        "Municipality",
        [""] + STATE.bootstrap["municipalities"],
        format_func=lambda v: "All municipalities" if not v else v,
    )
    search = st.text_input("Station search")
    view = st.selectbox("View", ["All stations", "Top 50 riskiest", "Top 50 safest"])

stations = get_ranked_stations(mode)
filtered_stations = filter_stations(stations, municipality, search, view)

if filtered_stations.empty:
    st.warning("No stations match the current filters.")
    st.stop()

station_options = filtered_stations[["site_id", "rank", "name", "municipality"]].copy()
station_options["label"] = station_options.apply(
    lambda row: f"#{int(row['rank'])} {row['name']} ({row['municipality']})", axis=1
)

selected_label = st.selectbox("Select station", station_options["label"].tolist(), index=0)
selected_site_id = int(station_options.loc[station_options["label"] == selected_label, "site_id"].iloc[0])
station = STATE.station_detail(selected_site_id, RADIUS_KEY)

metric_cols = st.columns(5)
metric_cols[0].metric("Stations", summary["station_count"])
metric_cols[1].metric("Avg monthly crashes", format_number(summary["average_monthly_crashes"], 2))
metric_cols[2].metric("Avg monthly cyclists", format_number(summary["average_monthly_cyclists"], 0))
metric_cols[3].metric("Mean forecast probability", format_number(summary["mean_forecast_probability"], 3))
metric_cols[4].metric("Model ROC AUC", format_number(summary["deployed_model"]["roc_auc"], 3))

left_col, right_col = st.columns([1.25, 1])

with left_col:
    st.subheader("Ranked station map")
    deck = build_map(filtered_stations, selected_site_id)
    if deck is not None:
        st.pydeck_chart(deck, width="stretch")
    st.subheader("Visible ranking table")
    st.dataframe(
        filtered_stations[["rank", "name", "municipality", "score"]].rename(
            columns={"rank": "Rank", "name": "Station", "municipality": "Municipality", "score": "Score"}
        ),
        width="stretch",
        hide_index=True,
    )

with right_col:
    st.subheader(station["metadata"]["name"])
    st.caption(
        f"{station['metadata']['municipality']} · opened {station['metadata']['opening_date']} · radius {station['radius_label']}"
    )

    stat_cols = st.columns(2)
    stat_cols[0].metric("Historical rank", f"#{station['historical']['rank']}")
    stat_cols[1].metric("Forecast rank", f"#{station['forecast']['rank']}")
    stat_cols[0].metric("Crash rate / 10k cyclists", format_number(station["historical"]["crashes_per_10000_cyclists"], 2))
    stat_cols[1].metric("Smoothed risk score", format_number(station["historical"]["smoothed_crashes_per_10000_cyclists"], 2))
    stat_cols[0].metric("Forecast probability", format_number(station["forecast"]["probability"], 3))
    stat_cols[1].metric("Forecast label", station["forecast"]["risk_label"])

    st.markdown("**Historical context**")
    st.write(
        {
            "Total crashes": int(station["historical"]["total_crashes"]),
            "Avg monthly crashes": round(station["historical"]["average_monthly_crashes"], 3),
            "Total cyclists": int(station["historical"]["total_cyclists"]),
            "Avg monthly cyclists": round(station["historical"]["average_monthly_cyclists"], 1),
            "Crash month share": round(station["historical"]["crash_month_share"], 3),
            "Baseline month": f"{station['baseline_context']['feature_year']}-{station['baseline_context']['feature_month']:02d}",
        }
    )

    st.markdown("**Prediction engine**")
    st.caption("Adjust the top numeric features and run the deployed next-month classifier.")

    overrides: dict[str, float] = {}
    for control in station["controls"]:
        overrides[control["feature"]] = st.slider(
            control["label"],
            min_value=float(control["min"]),
            max_value=float(control["max"]),
            value=float(control["default"]),
            step=float(control["step"]),
            help=f"Mean observed value: {format_number(control['mean'], 2)}",
        )

    if st.button("Predict next-month risk", type="primary", width="stretch"):
        prediction = STATE.predict(selected_site_id, RADIUS_KEY, overrides)
        st.success(
            f"{prediction['risk_label']} · probability {format_number(prediction['predicted_probability'], 3)} "
            f"(threshold {format_number(prediction['threshold'], 2)})"
        )

st.markdown("---")
st.subheader("Method and evidence")
for line in STATE.bootstrap["method_evidence"]:
    st.write(f"- {line}")
