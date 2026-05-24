const bootstrap = window.APP_BOOTSTRAP;
const map = L.map("map", {
  zoomControl: true,
  preferCanvas: true,
}).setView([50.85, 4.35], 8);

L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
  attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
  subdomains: "abcd",
  maxZoom: 20,
}).addTo(map);

const state = {
  radius: "1km_bike",
  mode: "historical",
  municipality: "",
  search: "",
  view: "all",
  selectedSiteId: null,
  selectedStation: null,
};

const markerLayer = L.layerGroup().addTo(map);
const circleLayer = L.layerGroup().addTo(map);

function byId(id) {
  return document.getElementById(id);
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits === 0 ? 0 : 0,
  });
}

function scoreColor(rank, count) {
  const ratio = count <= 1 ? 0 : (rank - 1) / (count - 1);
  const hue = 2 + ratio * 112;
  return `hsl(${hue}, 62%, 42%)`;
}

function markerIcon(rank, color) {
  return L.divIcon({
    className: "",
    html: `<div class="marker-badge" style="background:${color}">${rank}</div>`,
    iconSize: [34, 34],
    iconAnchor: [17, 17],
  });
}

function getCurrentStations() {
  let stations = bootstrap.stations[state.radius][state.mode].slice();

  if (state.search) {
    const query = state.search.toLowerCase();
    stations = stations.filter((station) => station.name.toLowerCase().includes(query));
  }

  if (state.municipality) {
    stations = stations.filter((station) => station.municipality === state.municipality);
  }

  if (state.view === "top50") {
    stations = stations.slice(0, 50);
  } else if (state.view === "bottom50") {
    stations = stations.slice(-50);
  }

  return stations;
}

function renderMarkers() {
  markerLayer.clearLayers();
  const stations = getCurrentStations();
  const totalCount = bootstrap.stations[state.radius][state.mode].length;

  stations.forEach((station) => {
    const color = scoreColor(station.rank, totalCount);
    const marker = L.marker([station.lat, station.long], {
      icon: markerIcon(station.rank, color),
      title: `${station.rank}. ${station.name}`,
    });
    marker.bindTooltip(
      `<strong>#${station.rank} ${station.name}</strong><br>${station.municipality}<br>` +
      `${state.mode === "historical" ? "Historical score" : "Forecast probability"}: ${formatNumber(station.score, 3)}`,
      { direction: "top" }
    );
    marker.on("click", () => selectStation(station.site_id));
    marker.addTo(markerLayer);
  });
}

function renderRadiusSummary() {
  const summary = bootstrap.radius_summaries[state.radius];
  byId("radius-summary").innerHTML = `
    <div class="detail-title">
      <h2>${summary.radius_label} summary</h2>
      <span class="rank-chip">${state.mode === "historical" ? "Observed" : "Forecast"}</span>
    </div>
    <div class="summary-grid">
      <div class="summary-card">
        <div class="label">Stations</div>
        <div class="value">${summary.station_count}</div>
      </div>
      <div class="summary-card">
        <div class="label">Top station</div>
        <div class="value">${summary.highest_risk_station}</div>
      </div>
      <div class="summary-card">
        <div class="label">Avg monthly crashes</div>
        <div class="value">${formatNumber(summary.average_monthly_crashes, 2)}</div>
      </div>
      <div class="summary-card">
        <div class="label">Avg monthly cyclists</div>
        <div class="value">${formatNumber(summary.average_monthly_cyclists, 0)}</div>
      </div>
      <div class="summary-card">
        <div class="label">Median smoothed rate</div>
        <div class="value">${formatNumber(summary.median_smoothed_rate, 2)}</div>
      </div>
      <div class="summary-card">
        <div class="label">Mean forecast probability</div>
        <div class="value">${formatNumber(summary.mean_forecast_probability, 3)}</div>
      </div>
    </div>
    <p class="secondary-note">
      Deployed model: ${summary.deployed_model.name} · threshold ${formatNumber(summary.deployed_model.threshold, 2)}
      · F1 ${formatNumber(summary.deployed_model.f1, 3)} · ROC AUC ${formatNumber(summary.deployed_model.roc_auc, 3)}
    </p>
    <p class="secondary-note">
      This app is intentionally locked to <strong>1.0 km bike-only</strong> because that is the final compromise between local interpretability and usable signal.
    </p>
  `;
}

function renderEvidence() {
  byId("method-evidence").innerHTML = bootstrap.method_evidence
    .map((line) => `<li>${line}</li>`)
    .join("");
}

function clearSelectedCircle() {
  circleLayer.clearLayers();
}

function drawSelectedCircle(station) {
  clearSelectedCircle();
  if (!station) {
    return;
  }
  L.circle([station.metadata.lat, station.metadata.long], {
    radius: station.radius_meters,
    color: "#aa3f37",
    weight: 2,
    fillOpacity: 0.08,
  }).addTo(circleLayer);
}

function renderStationDetail() {
  const container = byId("station-detail");
  if (!state.selectedStation) {
    container.innerHTML = `
      <div class="detail-title">
        <h2>Select a station</h2>
      </div>
      <p class="secondary-note">
        Click a ranked station marker to inspect the local crash ratio, the 1.0 km bike-only summary, and the
        next-month prediction engine for that station.
      </p>
    `;
    return;
  }

  const station = state.selectedStation;
  const controlsMarkup = station.controls.map((control) => `
    <div class="predict-control">
      <div class="control-top">
        <strong>${control.label}</strong>
        <span id="value-${control.feature}">${formatNumber(control.default, 2)}</span>
      </div>
      <input
        type="range"
        id="slider-${control.feature}"
        min="${control.min}"
        max="${control.max}"
        step="${control.step}"
        value="${control.default}"
      />
      <div class="secondary-note">Range ${formatNumber(control.min, 2)} to ${formatNumber(control.max, 2)} · mean ${formatNumber(control.mean, 2)}</div>
    </div>
  `).join("");

  container.innerHTML = `
    <div class="detail-title">
      <h2>${station.metadata.name}</h2>
      <span class="rank-chip">#${station.historical.rank} historical</span>
    </div>
    <p class="secondary-note">
      ${station.metadata.municipality} · opened ${station.metadata.opening_date} · current radius ${station.radius_label}
    </p>

    <div class="dense-list">
      <div class="dense-row"><strong>Historical rank</strong><span>#${station.historical.rank}</span></div>
      <div class="dense-row"><strong>Forecast rank</strong><span>#${station.forecast.rank}</span></div>
      <div class="dense-row"><strong>Total crashes</strong><span>${formatNumber(station.historical.total_crashes, 0)}</span></div>
      <div class="dense-row"><strong>Total cyclists</strong><span>${formatNumber(station.historical.total_cyclists, 0)}</span></div>
      <div class="dense-row"><strong>Crash rate / 10k cyclists</strong><span>${formatNumber(station.historical.crashes_per_10000_cyclists, 2)}</span></div>
      <div class="dense-row"><strong>Smoothed risk score</strong><span>${formatNumber(station.historical.smoothed_crashes_per_10000_cyclists, 2)}</span></div>
      <div class="dense-row"><strong>Forecast probability</strong><span id="forecast-probability">${formatNumber(station.forecast.probability, 3)}</span></div>
      <div class="dense-row"><strong>Forecast label</strong><span id="forecast-label">${station.forecast.risk_label}</span></div>
      <div class="dense-row"><strong>Avg monthly crashes</strong><span>${formatNumber(station.historical.average_monthly_crashes, 2)}</span></div>
      <div class="dense-row"><strong>Avg monthly cyclists</strong><span>${formatNumber(station.historical.average_monthly_cyclists, 0)}</span></div>
      <div class="dense-row"><strong>Crash month share</strong><span>${formatNumber(station.historical.crash_month_share, 3)}</span></div>
      <div class="dense-row"><strong>Baseline month</strong><span>${station.baseline_context.feature_year}-${String(station.baseline_context.feature_month).padStart(2, "0")}</span></div>
    </div>

    <h3>Prediction engine</h3>
    <p class="secondary-note">
      Adjust a small set of high-impact numeric inputs for the selected station, then run the deployed
      next-month classifier for ${station.radius_label}.
    </p>
    <div class="controls-stack">${controlsMarkup}</div>
    <div class="predict-actions">
      <button class="primary-btn" id="predict-btn">Predict next-month risk</button>
      <span class="secondary-note">Uses the saved tuned classifier for the 1.0 km bike-only view.</span>
    </div>
    <div class="prediction-result" id="prediction-result">
      <strong>Prediction</strong>
      <div class="result-value">Waiting for prediction</div>
    </div>
  `;

  station.controls.forEach((control) => {
    const slider = byId(`slider-${control.feature}`);
    const valueLabel = byId(`value-${control.feature}`);
    slider.addEventListener("input", () => {
      valueLabel.textContent = formatNumber(slider.value, 2);
    });
  });
  byId("predict-btn").addEventListener("click", runPrediction);
}

async function selectStation(siteId) {
  const response = await fetch(`/api/station/${siteId}?radius=${state.radius}`);
  const data = await response.json();
  if (data.error) {
    return;
  }
  state.selectedSiteId = siteId;
  state.selectedStation = data;
  drawSelectedCircle(data);
  renderStationDetail();
}

async function runPrediction() {
  if (!state.selectedStation) {
    return;
  }
  const overrides = {};
  state.selectedStation.controls.forEach((control) => {
    overrides[control.feature] = Number(byId(`slider-${control.feature}`).value);
  });
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      site_id: state.selectedStation.site_id,
      radius: state.radius,
      overrides,
    }),
  });
  const data = await response.json();
  const resultNode = byId("prediction-result");
  if (data.error) {
    resultNode.innerHTML = `<strong>Prediction</strong><div class="result-value">${data.error}</div>`;
    return;
  }
  resultNode.innerHTML = `
    <strong>Prediction</strong>
    <div class="result-value">${data.risk_label} · probability ${formatNumber(data.predicted_probability, 3)}</div>
    <div class="secondary-note">Threshold ${formatNumber(data.threshold, 2)} on the ${data.radius_label} classifier.</div>
  `;
  const forecastProbabilityNode = byId("forecast-probability");
  const forecastLabelNode = byId("forecast-label");
  if (forecastProbabilityNode) {
    forecastProbabilityNode.textContent = formatNumber(data.predicted_probability, 3);
  }
  if (forecastLabelNode) {
    forecastLabelNode.textContent = `${data.risk_label} (live)`;
  }
  state.selectedStation.forecast.probability = data.predicted_probability;
  state.selectedStation.forecast.risk_label = `${data.risk_label} (live)`;
}

function populateMunicipalities() {
  const select = byId("municipality-select");
  bootstrap.municipalities.forEach((municipality) => {
    const option = document.createElement("option");
    option.value = municipality;
    option.textContent = municipality;
    select.appendChild(option);
  });
}

function bindControls() {
  byId("mode-select").addEventListener("change", (event) => {
    state.mode = event.target.value;
    renderMarkers();
    renderRadiusSummary();
  });

  byId("search-input").addEventListener("input", (event) => {
    state.search = event.target.value.trim();
    renderMarkers();
  });

  byId("municipality-select").addEventListener("change", (event) => {
    state.municipality = event.target.value;
    renderMarkers();
  });

  byId("view-select").addEventListener("change", (event) => {
    state.view = event.target.value;
    renderMarkers();
  });
}

function init() {
  populateMunicipalities();
  bindControls();
  renderEvidence();
  renderRadiusSummary();
  renderMarkers();
  renderStationDetail();
}

init();
