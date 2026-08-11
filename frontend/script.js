const API_URL = window.location.protocol.startsWith("http") ? window.location.origin : "http://127.0.0.1:8000";

// DOM Elements
const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const fileChip = document.getElementById("fileChip");
const fileNameEl = document.getElementById("fileName");
const clearFileBtn = document.getElementById("clearFile");
const runBtn = document.getElementById("runBtn");
const errorMsg = document.getElementById("errorMsg");
const resultsContainer = document.getElementById("resultsContainer");
const apiStatus = document.getElementById("apiStatus");
const previewPanel = document.getElementById("previewPanel");
const previewStatus = document.getElementById("previewStatus");
const previewChartCanvas = document.getElementById("previewChart");
const channelLegendSwatches = document.getElementById("channelLegendSwatches");
const resultsSidebar = document.getElementById("resultsSidebar");
const sidebarToggleBtn = document.getElementById("sidebarToggleBtn");
const apiDocsLink = document.getElementById("apiDocsLink");

if (apiDocsLink) {
  apiDocsLink.href = `${API_URL}/docs`;
}

// Section Availability State Tracking
const sectionState = {
  preview: false,
  predictedActivity: false,
  probBreakdown: false,
  aiInsight: false,
};

// Routing Elements
const navLinks = {
  home: document.getElementById("navHome"),
  classify: document.getElementById("navClassify"),
  about: document.getElementById("navAbout"),
  "model-info": document.getElementById("navModelInfo"),
};

const pageViews = {
  home: document.getElementById("viewHome"),
  classify: document.getElementById("viewClassify"),
  about: document.getElementById("viewAbout"),
  "model-info": document.getElementById("viewModelInfo"),
};

let selectedFile = null;
let previewChartInstance = null;
let probChartInstance = null;
let modelInfoFetched = false;
let scrollSpyObserver = null;

const channelColors = [
  "#3EE3C0", // body_acc_x
  "#72B4FF", // body_acc_y
  "#FFD166", // body_acc_z
  "#FF6B6B", // body_gyro_x
  "#A68EFF", // body_gyro_y
  "#6EE7B7", // body_gyro_z
  "#F472B6", // total_acc_x
  "#38BDF8", // total_acc_y
  "#FBBF24", // total_acc_z
];

// Sidebar Toggle Listener (Expand / Collapse Rail)
if (sidebarToggleBtn && resultsSidebar) {
  sidebarToggleBtn.addEventListener("click", () => {
    resultsSidebar.classList.toggle("collapsed");
    const classifyView = document.getElementById("viewClassify");
    if (classifyView) classifyView.classList.toggle("sidebar-collapsed");
  });
}

// ================= CLIENT-SIDE ROUTER =================
function handleRoute() {
  const hash = (window.location.hash || "#home").replace("#", "").toLowerCase();
  const activeRoute = pageViews[hash] ? hash : "home";

  // Update Nav Links
  Object.keys(navLinks).forEach((key) => {
    if (navLinks[key]) {
      if (key === activeRoute) {
        navLinks[key].classList.add("active");
      } else {
        navLinks[key].classList.remove("active");
      }
    }
  });

  // Update Active View
  Object.keys(pageViews).forEach((key) => {
    if (pageViews[key]) {
      if (key === activeRoute) {
        pageViews[key].classList.add("active");
      } else {
        pageViews[key].classList.remove("active");
      }
    }
  });

  // Update sidebar visibility based on route
  if (activeRoute === "classify") {
    updateSidebarNavState();
  } else if (resultsSidebar) {
    resultsSidebar.style.display = "none";
  }

  // Lazy-load Model Info if navigating to model-info view
  if (activeRoute === "model-info" && !modelInfoFetched) {
    fetchModelInfo();
  }
}

window.addEventListener("hashchange", handleRoute);
window.addEventListener("DOMContentLoaded", handleRoute);

// ================= API CHECK & MODEL INFO =================
async function checkApi() {
  try {
    const response = await fetch(`${API_URL}/status`);
    if (!response.ok) throw new Error();
    apiStatus.classList.add("online");
    apiStatus.classList.remove("offline");
    apiStatus.innerHTML = '<span class="dot"></span> backend online';
  } catch (error) {
    apiStatus.classList.add("offline");
    apiStatus.classList.remove("online");
    apiStatus.innerHTML = '<span class="dot"></span> backend unreachable';
  }
}

async function fetchModelInfo() {
  const totalParamsEl = document.getElementById("modelTotalParams");

  try {
    const res = await fetch(`${API_URL}/model-info`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    modelInfoFetched = true;

    if (totalParamsEl) {
      totalParamsEl.textContent = data.total_params
        ? data.total_params.toLocaleString()
        : "157,798";
    }
  } catch (e) {
    if (totalParamsEl) totalParamsEl.textContent = "157,798";
  }
}

// ================= FILE SELECTION & PREVIEW =================
function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = false;
}

function clearError() {
  errorMsg.hidden = true;
  errorMsg.textContent = "";
}

function setFile(file) {
  selectedFile = file;
  clearError();
  
  if (resultsContainer) resultsContainer.innerHTML = "";
  if (probChartInstance) {
    probChartInstance.destroy();
    probChartInstance = null;
  }

  // Reset section state
  sectionState.preview = false;
  sectionState.predictedActivity = false;
  sectionState.probBreakdown = false;
  sectionState.aiInsight = false;

  const previewMeta = document.getElementById("previewMeta");
  if (file) {
    fileChip.hidden = false;
    fileNameEl.textContent = file.name;
    runBtn.disabled = true;
    previewPanel.hidden = true;
    if (previewMeta) previewMeta.hidden = true;
    if (previewChartInstance) {
      previewChartInstance.destroy();
      previewChartInstance = null;
    }
    previewFile(file);
  } else {
    fileChip.hidden = true;
    fileNameEl.textContent = "";
    fileInput.value = "";
    runBtn.disabled = true;
    previewPanel.hidden = true;
    if (previewMeta) previewMeta.hidden = true;
    if (previewChartInstance) {
      previewChartInstance.destroy();
      previewChartInstance = null;
    }
    updateSidebarNavState();
  }
}

async function previewFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_URL}/preview`, { method: "POST", body: formData });
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.detail || "Preview failed");

    renderPreviewMeta(payload);
    previewStatus.textContent = payload.warning ? "Warning detected during preview" : "Preview loaded successfully";
    renderPreviewChart(payload.preview_series, payload.resolved_channel_labels);
    previewPanel.hidden = false;
    runBtn.disabled = false;

    // Enable Preview section item
    sectionState.preview = true;
    updateSidebarNavState();
  } catch (error) {
    showError(error.message || "Unable to preview the uploaded file.");
    runBtn.disabled = true;
    sectionState.preview = false;
    updateSidebarNavState();
  }
}

function renderSwatches(labels) {
  if (!channelLegendSwatches) return;
  channelLegendSwatches.innerHTML = labels
    .map(
      (label, idx) => `
      <div class="swatch-item">
        <span class="swatch-color" style="background-color: ${channelColors[idx % channelColors.length]}"></span>
        <span>${label}</span>
      </div>
    `
    )
    .join("");
}

function renderPreviewChart(dataRows, channelLabels = null) {
  const defaultLabels = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
  ];
  const activeLabels = channelLabels || defaultLabels;
  renderSwatches(activeLabels);

  const labels = Array.from({ length: dataRows.length }, (_, i) => i + 1);
  const datasets = dataRows[0].map((_, channelIndex) => ({
    label: activeLabels[channelIndex] ?? `Channel ${channelIndex + 1}`,
    data: dataRows.map((row) => row[channelIndex]),
    borderColor: channelColors[channelIndex % channelColors.length],
    borderWidth: 1.2,
    pointRadius: 0,
    tension: 0.2,
  }));

  if (previewChartInstance) {
    previewChartInstance.destroy();
  }

  previewChartInstance = new Chart(previewChartCanvas, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: "#22303C" }, ticks: { color: "#8E9EAF" } },
        y: { grid: { color: "#22303C" }, ticks: { color: "#8E9EAF" } },
      },
    },
  });
}

function renderPreviewMeta(payload) {
  const previewMeta = document.getElementById("previewMeta");
  const previewCopy = document.getElementById("previewCopy");
  const mappingList = document.getElementById("mappingList");

  if (!previewMeta || !previewCopy || !mappingList) return;

  const sourceLabels = payload.source_labels;
  const mappingItems = payload.mapping || [];

  if (sourceLabels?.length) {
    previewCopy.textContent = `Header detected: ${sourceLabels.join(", ")}. Ready to predict with channel mapping.`;
  } else {
    previewCopy.textContent = "No header row detected. Interpreted using default expected 9-channel sensor order.";
  }

  if (mappingItems.length) {
    mappingList.innerHTML = mappingItems
      .map((item, index) => {
        const source = item.source || `col_${index + 1}`;
        const resolved = item.resolved || "unrecognized";
        return `<li><strong>${source}</strong> ➔ ${resolved}</li>`;
      })
      .join("");
  } else {
    mappingList.innerHTML = "";
  }

  previewMeta.hidden = false;
}

// ================= RUN CLASSIFICATION =================
async function runClassification() {
  if (!selectedFile) return;
  runBtn.disabled = true;
  runBtn.textContent = "Predicting…";
  clearError();

  // Reset prediction section states
  sectionState.predictedActivity = false;
  sectionState.probBreakdown = false;
  sectionState.aiInsight = false;
  updateSidebarNavState();

  if (resultsContainer) {
    resultsContainer.innerHTML = `
      <div class="panel loading-panel">
        <div class="spinner"></div>
        <p class="loading-text">Running model prediction…</p>
      </div>
    `;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("sampling_rate", "50");
  formData.append(
    "channel_labels",
    JSON.stringify([
      "body_acc_x",
      "body_acc_y",
      "body_acc_z",
      "body_gyro_x",
      "body_gyro_y",
      "body_gyro_z",
      "total_acc_x",
      "total_acc_y",
      "total_acc_z",
    ])
  );

  try {
    const res = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
    renderResult(data);
  } catch (error) {
    if (resultsContainer) resultsContainer.innerHTML = "";
    showError(error.message || "Prediction failed.");
    sectionState.predictedActivity = false;
    sectionState.probBreakdown = false;
    sectionState.aiInsight = false;
    updateSidebarNavState();
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Get Prediction";
  }
}

function renderResult(data) {
  if (!resultsContainer) return;

  const activityName = data.activity || "Unknown";
  const confidencePct = ((data.confidence || 0) * 100).toFixed(1);

  resultsContainer.innerHTML = `
    <div class="results">
      <div class="panel result-card primary-result" id="secPredictedActivity">
        <div class="result-header">
          <p class="result-label">PREDICTED ACTIVITY</p>
          <span class="live-badge">Sampled at 50 Hz</span>
        </div>
        <h2>${activityName}</h2>
        <p>Confidence: ${confidencePct}%</p>
      </div>

      <div class="panel chart-panel prominent-chart" id="secProbBreakdown">
        <div class="chart-header">
          <h3>Full probability breakdown</h3>
          <span class="chart-tag">Softmax Probabilities</span>
        </div>
        <div class="prob-chart-wrap">
          <canvas id="probChart"></canvas>
        </div>
      </div>

      <!-- AI Motion Insight Card -->
      <div class="panel ai-summary-panel" id="secAiInsight">
        <div class="ai-summary-header">
          <div class="ai-title-wrap">
            <span class="ai-badge">✦ AI Motion Insight</span>
            <h3>AI Motion Insight</h3>
          </div>
        </div>
        <div class="ai-summary-content" id="aiSummaryContent">
          <div class="ai-loading">
            <div class="spinner-sm"></div>
            <span>Generating AI motion insight…</span>
          </div>
        </div>
      </div>
    </div>
  `;

  const labels = Object.keys(data.probabilities || {});
  const values = labels.map((label) => (data.probabilities[label] || 0) * 100);

  const ctx = document.getElementById("probChart");
  if (probChartInstance) {
    probChartInstance.destroy();
  }

  probChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Confidence (%)",
          data: values,
          backgroundColor: labels.map((label) =>
            label === data.activity ? "#3EE3C0" : "rgba(62, 227, 192, 0.25)"
          ),
          borderColor: labels.map((label) =>
            label === data.activity ? "#3EE3C0" : "rgba(62, 227, 192, 0.4)"
          ),
          borderWidth: 1,
          borderRadius: 8,
          maxBarThickness: 48,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => ` ${context.raw.toFixed(1)}%`,
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: {
            color: "#8E9EAF",
            callback: (val) => `${val}%`,
          },
          grid: { color: "#22303C" },
        },
        x: {
          ticks: { color: "#F0F4F8", font: { family: "Inter", size: 12, weight: "500" } },
          grid: { display: false },
        },
      },
    },
  });

  // Mark prediction & probability breakdown active
  sectionState.predictedActivity = true;
  sectionState.probBreakdown = true;
  sectionState.aiInsight = false;
  updateSidebarNavState();

  fetchAiExplanation(data);
}

// ================= SIDEBAR NAV & SCROLL SPY =================
function updateSidebarNavState() {
  if (!resultsSidebar) return;

  const route = (window.location.hash || "#home").replace("#", "").toLowerCase();
  if (route !== "classify" && route !== "") {
    resultsSidebar.style.display = "none";
    return;
  }
  resultsSidebar.style.display = "flex";

  const itemMap = [
    { btnId: "sideLinkPreview", state: sectionState.preview, targetId: "previewPanel" },
    { btnId: "sideLinkPredictedActivity", state: sectionState.predictedActivity, targetId: "secPredictedActivity" },
    { btnId: "sideLinkProbBreakdown", state: sectionState.probBreakdown, targetId: "secProbBreakdown" },
    { btnId: "sideLinkAiInsight", state: sectionState.aiInsight, targetId: "secAiInsight" },
  ];

  const enabledTargets = [];

  itemMap.forEach(({ btnId, state, targetId }) => {
    const btn = document.getElementById(btnId);
    if (!btn) return;

    // Clone button to strip existing listeners
    const newBtn = btn.cloneNode(true);
    btn.parentNode.replaceChild(newBtn, btn);

    if (state) {
      newBtn.classList.remove("disabled");
      newBtn.disabled = false;
      const targetEl = document.getElementById(targetId);
      if (targetEl && !targetEl.hidden && targetEl.offsetParent !== null) {
        enabledTargets.push({ btn: newBtn, targetEl });
        newBtn.addEventListener("click", (e) => {
          e.preventDefault();
          targetEl.scrollIntoView({ behavior: "smooth", block: "start" });
        });
      }
    } else {
      newBtn.classList.add("disabled");
      newBtn.classList.remove("active");
      newBtn.disabled = true;
    }
  });

  setupScrollSpy(enabledTargets);
}

function setupScrollSpy(enabledTargets) {
  if (scrollSpyObserver) scrollSpyObserver.disconnect();

  if (enabledTargets.length === 0) return;

  scrollSpyObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const targetId = entry.target.id;
          enabledTargets.forEach(({ btn, targetEl }) => {
            if (targetEl.id === targetId) {
              btn.classList.add("active");
            } else {
              btn.classList.remove("active");
            }
          });
        }
      });
    },
    { threshold: 0.25, rootMargin: "-70px 0px -40% 0px" }
  );

  enabledTargets.forEach(({ targetEl }) => {
    scrollSpyObserver.observe(targetEl);
  });
}

// ================= LLM EXPLANATION =================
async function fetchAiExplanation(data) {
  const contentEl = document.getElementById("aiSummaryContent");
  if (!contentEl) return;

  try {
    const res = await fetch(`${API_URL}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        activity: data.activity,
        confidence: data.confidence,
        probabilities: data.probabilities,
      }),
    });

    if (res.ok) {
      const payload = await res.json();
      contentEl.innerHTML = `<p class="ai-explanation-text">${payload.explanation}</p>`;
      // AI Insight response succeeded: enable item in sidebar
      sectionState.aiInsight = true;
      updateSidebarNavState();
    } else if (res.status === 429) {
      contentEl.innerHTML = `<p class="ai-error-text">⚠️ AI motion insight temporarily unavailable — usage limit reached. Try again in a moment or check your Groq API key quota.</p>`;
      sectionState.aiInsight = false;
      updateSidebarNavState();
    } else {
      contentEl.innerHTML = `<p class="ai-error-text">⚠️ AI motion insight temporarily unavailable. Try again shortly.</p>`;
      sectionState.aiInsight = false;
      updateSidebarNavState();
    }
  } catch (err) {
    contentEl.innerHTML = `<p class="ai-error-text">⚠️ AI motion insight temporarily unavailable. Try again shortly.</p>`;
    sectionState.aiInsight = false;
    updateSidebarNavState();
  }
}

// ================= EVENT LISTENERS =================
fileInput.addEventListener("change", (event) => {
  if (event.target.files.length) setFile(event.target.files[0]);
});

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  if (event.dataTransfer.files.length) setFile(event.dataTransfer.files[0]);
});

clearFileBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  setFile(null);
});

runBtn.addEventListener("click", runClassification);

// Initial Execution
checkApi();
handleRoute();
