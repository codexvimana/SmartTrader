const API_URL =
  window.location.hostname === "localhost"
    ? "http://localhost:8989"
    : "https://smarttrader.manju59.net";

async function fetchPredictions(selectedDate) {
  try {
    const response = await fetch(`https://smarttrader.manju59.net/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        selected_date: selectedDate,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    if (data.success && data.predictions) {
      const dailyPredictions = data.predictions.map((dayPred, index) => ({
        date: dayPred.date,
        nvda: {
          max_price: dayPred.nvda.max_price,
          min_price: dayPred.nvda.min_price,
          avg_price: dayPred.nvda.avg_price,
        },
        nvdq: {
          max_price: dayPred.nvdq.max_price,
          min_price: dayPred.nvdq.min_price,
          avg_price: dayPred.nvdq.avg_price,
        },
      }));

      return {
        success: true,
        predictions: dailyPredictions,
        trading_strategy: data.trading_strategy,
      };
    }

    return data;
  } catch (error) {
    console.log(error);
    return {
      success: false,
      error: error.message,
    };
  }
}
