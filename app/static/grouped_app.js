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
  mode: "historical",
  municipality: "",
  search: "",
  view: "all",
  selectedGroupId: null,
  selectedGroup: null,
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

function rankLabel(group) {
  return group.hidden_risk_spot ? `${group.rank}!` : `${group.rank}`;
}

function getCurrentGroups() {
  let groups = bootstrap.groups[state.mode].slice();
  if (state.search) {
    const query = state.search.toLowerCase();
    groups = groups.filter((group) => group.group_name.toLowerCase().includes(query));
  }
  if (state.municipality) {
    groups = groups.filter((group) => group.gemeente === state.municipality);
  }
  if (state.view === "top50") {
    groups = groups.slice(0, 50);
  } else if (state.view === "bottom50") {
    groups = groups.slice(-50);
  }
  return groups;
}

function renderMarkers() {
  markerLayer.clearLayers();
  const groups = getCurrentGroups();
  const totalCount = bootstrap.groups[state.mode].length;
  groups.forEach((group) => {
    const color = scoreColor(group.rank, totalCount);
    const marker = L.marker([group.lat, group.long], {
      icon: markerIcon(rankLabel(group), color),
      title: `${rankLabel(group)}. ${group.group_name}`,
    });
    marker.bindTooltip(
      `<strong>#${rankLabel(group)} ${group.group_name}</strong><br>${group.gemeente}<br>` +
      `${state.mode === "historical" ? "Historical score" : "Forecast probability"}: ${formatNumber(group.score, 3)}` +
      (group.hidden_risk_spot ? `<br><em>Hidden risk spot</em>` : ``),
      { direction: "top" }
    );
    marker.on("click", () => selectGroup(group.group_id));
    marker.addTo(markerLayer);
  });
}

function renderSummary() {
  const summary = bootstrap.summary;
  byId("radius-summary").innerHTML = `
    <div class="detail-title">
      <h2>${summary.label} summary</h2>
      <span class="rank-chip">${state.mode === "historical" ? "Observed" : "Forecast"}</span>
    </div>
    <div class="summary-grid">
      <div class="summary-card"><div class="label">Groups</div><div class="value">${summary.group_count}</div></div>
      <div class="summary-card"><div class="label">Top group</div><div class="value">${summary.highest_risk_group}</div></div>
      <div class="summary-card"><div class="label">Avg monthly crashes</div><div class="value">${formatNumber(summary.average_monthly_crashes, 2)}</div></div>
      <div class="summary-card"><div class="label">Avg monthly cyclists</div><div class="value">${formatNumber(summary.average_monthly_cyclists, 0)}</div></div>
      <div class="summary-card"><div class="label">Median smoothed rate</div><div class="value">${formatNumber(summary.median_smoothed_rate, 2)}</div></div>
      <div class="summary-card"><div class="label">Mean forecast probability</div><div class="value">${formatNumber(summary.mean_forecast_probability, 3)}</div></div>
    </div>
    <p class="secondary-note">
      Deployed model: ${summary.deployed_model.name} · threshold ${formatNumber(summary.deployed_model.threshold, 2)}
      · F1 ${formatNumber(summary.deployed_model.f1, 3)} · ROC AUC ${formatNumber(summary.deployed_model.roc_auc, 3)}
    </p>
  `;
}

function renderEvidence() {
  byId("method-evidence").innerHTML = bootstrap.method_evidence.map((line) => `<li>${line}</li>`).join("");
}

function drawSelectedCircle(group) {
  circleLayer.clearLayers();
  if (!group) return;
  L.circle([group.metadata.lat, group.metadata.long], {
    radius: group.radius_meters,
    color: "#aa3f37",
    weight: 2,
    fillOpacity: 0.08,
  }).addTo(circleLayer);
}

function renderDetail() {
  const container = byId("station-detail");
  if (!state.selectedGroup) {
    container.innerHTML = `
      <div class="detail-title"><h2>Select a grouped counter</h2></div>
      <p class="secondary-note">
        Click a ranked grouped marker to inspect the merged historical ratio and the grouped next-month prediction engine.
      </p>
    `;
    return;
  }

  const group = state.selectedGroup;
  const forecastRankLabel = group.forecast.hidden_risk_spot ? `#${group.forecast.rank}!` : `#${group.forecast.rank}`;
  const hiddenRiskNote = group.forecast.hidden_risk_spot
    ? `<p class="secondary-note"><strong>Hidden Risk Spot:</strong> This group is predicted as higher risk but its municipality is not present in the official dangerous-points list used in the analysis notebook.</p>`
    : "";
  const controlsMarkup = group.controls.map((control) => `
    <div class="predict-control">
      <div class="control-top">
        <strong>${control.label}</strong>
        <span id="value-${control.feature}">${formatNumber(control.default, 2)}</span>
      </div>
      <input type="range" id="slider-${control.feature}" min="${control.min}" max="${control.max}" step="${control.step}" value="${control.default}" />
      <div class="secondary-note">Range ${formatNumber(control.min, 2)} to ${formatNumber(control.max, 2)} · mean ${formatNumber(control.mean, 2)}</div>
    </div>
  `).join("");

  container.innerHTML = `
    <div class="detail-title">
      <h2>${group.metadata.name}</h2>
      <span class="rank-chip">#${group.historical.rank} historical</span>
    </div>
    <p class="secondary-note">
      ${group.metadata.municipality} · members ${group.metadata.member_site_count} · site ids ${group.metadata.member_site_ids}
    </p>
    ${hiddenRiskNote}
    <div class="dense-list">
      <div class="dense-row"><strong>Historical rank</strong><span>#${group.historical.rank}</span></div>
      <div class="dense-row"><strong>Forecast rank</strong><span>${forecastRankLabel}</span></div>
      <div class="dense-row"><strong>Total crashes</strong><span>${formatNumber(group.historical.total_crashes, 0)}</span></div>
      <div class="dense-row"><strong>Total cyclists</strong><span>${formatNumber(group.historical.total_cyclists, 0)}</span></div>
      <div class="dense-row"><strong>Crash rate / 10k cyclists</strong><span>${formatNumber(group.historical.crashes_per_10000_cyclists, 2)}</span></div>
      <div class="dense-row"><strong>Smoothed risk score</strong><span>${formatNumber(group.historical.smoothed_crashes_per_10000_cyclists, 2)}</span></div>
      <div class="dense-row"><strong>Forecast probability</strong><span id="forecast-probability">${formatNumber(group.forecast.probability, 3)}</span></div>
      <div class="dense-row"><strong>Forecast label</strong><span id="forecast-label">${group.forecast.risk_label}</span></div>
      <div class="dense-row"><strong>Avg monthly crashes</strong><span>${formatNumber(group.historical.average_monthly_crashes, 2)}</span></div>
      <div class="dense-row"><strong>Avg monthly cyclists</strong><span>${formatNumber(group.historical.average_monthly_cyclists, 0)}</span></div>
      <div class="dense-row"><strong>Crash month share</strong><span>${formatNumber(group.historical.crash_month_share, 3)}</span></div>
      <div class="dense-row"><strong>Baseline month</strong><span>${group.baseline_context.feature_year}-${String(group.baseline_context.feature_month).padStart(2, "0")}</span></div>
    </div>

    <h3>Prediction engine</h3>
    <p class="secondary-note">
      Adjust a small set of high-impact numeric inputs for the selected grouped counter, then run the grouped next-month classifier.
    </p>
    <div class="controls-stack">${controlsMarkup}</div>
    <div class="predict-actions">
      <button class="primary-btn" id="predict-btn">Predict next-month risk</button>
    </div>
    <div class="prediction-result" id="prediction-result">
      <strong>Prediction</strong>
      <div class="result-value">Waiting for prediction</div>
    </div>
  `;

  group.controls.forEach((control) => {
    const slider = byId(`slider-${control.feature}`);
    const valueLabel = byId(`value-${control.feature}`);
    slider.addEventListener("input", () => {
      valueLabel.textContent = formatNumber(slider.value, 2);
    });
  });
  byId("predict-btn").addEventListener("click", runPrediction);
}

async function selectGroup(groupId) {
  const response = await fetch(`/api/group/${groupId}`);
  const data = await response.json();
  if (data.error) return;
  state.selectedGroupId = groupId;
  state.selectedGroup = data;
  drawSelectedCircle(data);
  renderDetail();
}

async function runPrediction() {
  if (!state.selectedGroup) return;
  const overrides = {};
  state.selectedGroup.controls.forEach((control) => {
    overrides[control.feature] = Number(byId(`slider-${control.feature}`).value);
  });
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({group_id: state.selectedGroup.group_id, overrides}),
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
    <div class="secondary-note">Threshold ${formatNumber(data.threshold, 2)} on the grouped classifier.</div>
  `;
  byId("forecast-probability").textContent = formatNumber(data.predicted_probability, 3);
  byId("forecast-label").textContent = `${data.risk_label} (live)`;
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
    renderSummary();
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

populateMunicipalities();
bindControls();
renderSummary();
renderEvidence();
renderMarkers();
renderDetail();
