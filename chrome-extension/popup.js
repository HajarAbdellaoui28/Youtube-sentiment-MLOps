
const API_URL = "http://127.0.0.1:8000";

let allResults = [];

const analyzeBtn = document.getElementById("analyzeBtn");
const statusEl = document.getElementById("status");
const resultsSection = document.getElementById("results");

analyzeBtn.addEventListener("click", async () => {
  analyzeBtn.disabled = true;
  statusEl.textContent = "Extraction en cours...";

  try {
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true
    });

    if (!tab?.url?.startsWith(
      "https://www.youtube.com/watch"
    )) {
      throw new Error("Ouvre une vidéo YouTube.");
    }

    const response = await chrome.tabs.sendMessage(
      tab.id,
      { action: "GET_COMMENTS" }
    );

    const comments = response?.comments || [];

    if (comments.length === 0) {
      throw new Error(
        "Aucun commentaire chargé. Fais défiler la page."
      );
    }

    statusEl.textContent =
      `${comments.length} commentaires extraits. Analyse...`;

    // Limite compatible avec notre API (500 maximum)
    const batchSize = 50;
    const predictions = [];

    for (let i = 0; i < comments.length; i += batchSize) {
      const batch = comments.slice(i, i + batchSize);

      const apiResponse = await fetch(
        `${API_URL}/predict_batch`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ comments: batch })
        }
      );

      if (!apiResponse.ok) {
        throw new Error(
          `Erreur API : ${apiResponse.status}`
        );
      }

      const data = await apiResponse.json();
      predictions.push(...data.results);
    }

    allResults = predictions;
    displayResults();
    statusEl.textContent = "Analyse terminée.";

  } catch (error) {
    statusEl.textContent = error.message;
  } finally {
    analyzeBtn.disabled = false;
  }
});

function displayResults() {
  const total = allResults.length;

  const count = sentiment =>
    allResults.filter(
      item => item.sentiment === sentiment
    ).length;

  const positive = count("positive");
  const neutral = count("neutral");
  const negative = count("negative");

  const percent = value =>
    total ? (value / total * 100).toFixed(1) : 0;

  document.getElementById("total").textContent = total;
  document.getElementById("positive").textContent =
    percent(positive);
  document.getElementById("neutral").textContent =
    percent(neutral);
  document.getElementById("negative").textContent =
    percent(negative);

  const chart = document.getElementById("chart");
  chart.replaceChildren();

  [
    [positive, "#22c55e"],
    [neutral, "#94a3b8"],
    [negative, "#ef4444"]
  ].forEach(([value, color]) => {
    const segment = document.createElement("div");
    segment.style.width = `${percent(value)}%`;
    segment.style.background = color;
    chart.appendChild(segment);
  });

  resultsSection.hidden = false;
  renderComments();
}

function renderComments() {
  const filter = document.getElementById("filter").value;
  const container = document.getElementById("commentList");

  container.replaceChildren();

  const filtered = allResults.filter(item =>
    filter === "all" || item.sentiment === filter
  );

  filtered.forEach(item => {
    const div = document.createElement("div");
    div.className = "comment";

    const label = document.createElement("strong");
    label.textContent = item.sentiment.toUpperCase();

    const paragraph = document.createElement("p");
    paragraph.textContent = item.text;

    div.append(label, paragraph);
    container.appendChild(div);
  });
}

document.getElementById("filter")
  .addEventListener("change", renderComments);

document.getElementById("themeBtn")
  .addEventListener("click", () => {
    document.body.classList.toggle("dark");
    const dark = document.body.classList.contains("dark");
    chrome.storage.local.set({ darkMode: dark });
  });

chrome.storage.local.get(["darkMode"], data => {
  if (data.darkMode) {
    document.body.classList.add("dark");
  }
});

document.getElementById("copyBtn")
  .addEventListener("click", async () => {
    const text = allResults.map(item =>
      `${item.sentiment}: ${item.text}`
    ).join("\n");

    await navigator.clipboard.writeText(text);
    statusEl.textContent = "Résultats copiés.";
  });
