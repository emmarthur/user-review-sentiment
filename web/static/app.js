const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const statusMeta = document.getElementById("status-meta");
const reviewInput = document.getElementById("review-text");
const analyzeBtn = document.getElementById("btn-analyze");
const errorEl = document.getElementById("error-text");
const resultCard = document.getElementById("result-card");
const resultSentiment = document.getElementById("result-sentiment");
const barNeg = document.getElementById("bar-neg");
const barPos = document.getElementById("bar-pos");
const pctNeg = document.getElementById("pct-neg");
const pctPos = document.getElementById("pct-pos");

let modelReady = false;
let pollTimer = null;

function setError(message) {
  if (!message) {
    errorEl.hidden = true;
    errorEl.textContent = "";
    return;
  }
  errorEl.hidden = false;
  errorEl.textContent = message;
}

function applyStatus(data) {
  statusDot.classList.remove("ready", "training", "error");
  if (data.error) {
    statusDot.classList.add("error");
    statusText.textContent = "Model failed to load.";
    statusMeta.hidden = false;
    statusMeta.textContent = data.error;
    modelReady = false;
    analyzeBtn.disabled = true;
    return;
  }
  if (data.ready) {
    statusDot.classList.add("ready");
    statusText.textContent = "Model ready";
    statusMeta.hidden = !data.cv_mean_accuracy;
    statusMeta.textContent = data.cv_mean_accuracy
      ? `5-fold cross-validation accuracy on training data: ${data.cv_mean_accuracy}%`
      : "";
    modelReady = true;
    analyzeBtn.disabled = false;
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    return;
  }
  statusDot.classList.add("training");
  statusText.textContent = data.training
    ? "Training model (first run may take about a minute)…"
    : "Starting model load…";
  statusMeta.hidden = true;
  modelReady = false;
  analyzeBtn.disabled = true;
}

async function pollStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    applyStatus(data);
  } catch {
    statusDot.classList.add("error");
    statusText.textContent = "Could not reach the server.";
  }
}

async function analyze() {
  setError(null);
  const text = reviewInput.value.trim();
  if (!text) {
    setError("Enter some review text first.");
    return;
  }
  analyzeBtn.disabled = true;
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(
        typeof data.detail === "string" ? data.detail : "Analysis failed.",
      );
    }
    resultCard.hidden = false;
    const isPos = data.sentiment === "positive";
    resultSentiment.textContent = isPos ? "Positive" : "Negative";
    resultSentiment.className = `result-sentiment ${isPos ? "is-pos" : "is-neg"}`;
    barNeg.style.width = `${data.probabilities.negative}%`;
    barPos.style.width = `${data.probabilities.positive}%`;
    pctNeg.textContent = `${data.probabilities.negative}%`;
    pctPos.textContent = `${data.probabilities.positive}%`;
  } catch (err) {
    setError(err.message);
  } finally {
    if (modelReady) analyzeBtn.disabled = false;
  }
}

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    reviewInput.value = chip.dataset.text || "";
    reviewInput.focus();
  });
});

analyzeBtn.addEventListener("click", () => {
  void analyze();
});

pollStatus();
pollTimer = setInterval(pollStatus, 2500);

document.querySelectorAll("[data-help-jump]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.getElementById("help-heading")?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  });
});
