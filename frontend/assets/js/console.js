let currentPredictions = null;
let currentDayIndex = 0;

window.addEventListener("load", () => {
  if (localStorage.getItem("systemInitialized") !== "true") {
    window.location.href = "index.html";
    return;
  }

  initializeConsole();
  setupEventListeners();
});

async function initializeConsole() {
  const loader = document.querySelector(".console-loader");
  const startupTexts = document.querySelectorAll(".startup-text span");

  for (let text of startupTexts) {
    text.style.display = "block";
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  setTimeout(() => {
    loader.classList.add("fade-out");
    setTimeout(() => {
      loader.style.display = "none";
    }, 1500);
  }, 1500);
}

function setupEventListeners() {
  const predictionDate = document.getElementById("predictionDate");
  const predictBtn = document.getElementById("predictBtn");
  const loadingIndicator = document.getElementById("loadingIndicator");
  const prevDayBtn = document.getElementById("prevDay");
  const nextDayBtn = document.getElementById("nextDay");

  const today = new Date();
  predictionDate.value = today.toISOString().split("T")[0];
  prevDayBtn?.addEventListener("click", () => {
    if (currentDayIndex > 0) {
      currentDayIndex--;
      updatePredictionDisplay();
    }
  });

  nextDayBtn?.addEventListener("click", () => {
    if (currentDayIndex < 4) {
      currentDayIndex++;
      updatePredictionDisplay();
    }
  });

  predictBtn.addEventListener("click", async () => {
    const selectedDate = predictionDate.value;

    if (!selectedDate) {
      alert("Please select a date");
      return;
    }

    const dateParts = selectedDate.split("-");
    const year = parseInt(dateParts[0], 10);
    const month = parseInt(dateParts[1], 10) - 1;
    const day = parseInt(dateParts[2], 10);

    const parsedDate = new Date(year, month, day);

    if (
      parsedDate.getFullYear() >= 9999 ||
      parsedDate.getMonth() >= 13 ||
      parsedDate.getDate() >= 32 ||
      parsedDate.getFullYear() !== year ||
      parsedDate.getMonth() !== month ||
      parsedDate.getDate() !== day
    ) {
      alert("Please select a valid date");
      return;
    }

    loadingIndicator.style.display = "block";
    clearResults();

    try {
      const response = await fetchPredictions(selectedDate);
      currentPredictions = response;
      currentDayIndex = 0;
      console.log(response);
      updateUI(response);
    } catch (error) {
      console.error("Error:", error);
      alert("Error getting predictions. Please try again.");
    } finally {
      loadingIndicator.style.display = "none";
    }
  });
}

function clearResults() {
  document.getElementById("highestPrice").textContent = "---.--";
  document.getElementById("lowestPrice").textContent = "---.--";
  document.getElementById("avgClosingPrice").textContent = "---.--";
  document.getElementById("nvdqHighestPrice").textContent = "---.--";
  document.getElementById("nvdqLowestPrice").textContent = "---.--";
  document.getElementById("nvdqAvgClosingPrice").textContent = "---.--";
  document.getElementById("strategyTableBody").innerHTML = "";

  const selectedDateEl = document.getElementById("selectedDate");
  if (selectedDateEl) {
    selectedDateEl.textContent = "Trading Date: --/--/----";
  }

  const currentDayEl = document.getElementById("currentDay");
  if (currentDayEl) {
    currentDayEl.textContent = "DAY 1 of 5";
  }
}

function updatePredictionDisplay() {
  if (!currentPredictions || !currentPredictions.success) return;

  const predictions = currentPredictions.predictions[currentDayIndex];
  if (!predictions) return;

  // Update navigation elements
  const prevDayBtn = document.getElementById("prevDay");
  const nextDayBtn = document.getElementById("nextDay");
  const currentDayEl = document.getElementById("currentDay");
  const selectedDateEl = document.getElementById("selectedDate");

  if (prevDayBtn) {
    prevDayBtn.disabled = currentDayIndex === 0;
  }
  if (nextDayBtn) {
    nextDayBtn.disabled = currentDayIndex === 4;
  }
  if (currentDayEl) {
    currentDayEl.textContent = `DAY ${currentDayIndex + 1} of 5`;
  }
  if (selectedDateEl) {
    selectedDateEl.textContent = `Trading Date: ${formatDate(predictions.date)}`;
  }

  // Update prediction values
  document.getElementById("highestPrice").textContent =
    `$${predictions.nvda.max_price.toFixed(2)}`;
  document.getElementById("lowestPrice").textContent =
    `$${predictions.nvda.min_price.toFixed(2)}`;
  document.getElementById("avgClosingPrice").textContent =
    `$${predictions.nvda.avg_price.toFixed(2)}`;
  document.getElementById("nvdqHighestPrice").textContent =
    `$${predictions.nvdq.max_price.toFixed(2)}`;
  document.getElementById("nvdqLowestPrice").textContent =
    `$${predictions.nvdq.min_price.toFixed(2)}`;
  document.getElementById("nvdqAvgClosingPrice").textContent =
    `$${predictions.nvdq.avg_price.toFixed(2)}`;

  updateStrategyTable(currentPredictions.trading_strategy);
}

function updateStrategyTable(strategies) {
  const strategyTableBody = document.getElementById("strategyTableBody");
  strategyTableBody.innerHTML = "";

  strategies.forEach((strategyData, index) => {
    const row = document.createElement("tr");
    row.classList.toggle("active-day", index === currentDayIndex);

    row.innerHTML = `
      <td>${formatDate(strategyData.date)}</td>
      <td>${strategyData.strategy}</td>
    `;
    strategyTableBody.appendChild(row);
  });
}

function updateUI(data) {
  if (!data.success) {
    console.error("Error in response:", data.error);
    alert("Error getting predictions");
    return;
  }

  // Just call updatePredictionDisplay which will handle both predictions and strategy table
  updatePredictionDisplay();
}

function formatDate(dateString) {
  const date = new Date(dateString + "T00:00:00");
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    timeZone: "UTC",
  });
}

function resetSystem() {
  localStorage.removeItem("systemInitialized");
  window.location.href = "index.html";
}
