# CMPE 257 - Term Project
#SmartTrader
## Team Members:
Manju Shettar (@manjunath.shettar@sjsu.edu) 17886440
Rajorshi Sarkar (rajorshi.sarkar@sjsu.edu) 14547260

## link: smarttrader.manju59.net

Pipeline for SmartTrader Model
1. Data Collection: Download historical stock data using yfinance for specified tickers and date ranges.
2. Feature Engineering: Calculate derived features:
    * Moving averages (5-day, 10-day).
    * Daily returns (percentage change).
    * Volatility.
    * Lagged closing prices.
3. Target Generation: Create prediction targets:
    * Maximum, minimum, and average prices over a rolling window.
4. Preprocessing:
    * Save processed data to CSV files.
    * Remove metadata rows (if present).
    * Ensure all numerical columns are properly typed.
5. Training:
    * Train XGBoost regression models on specified features for each target.
6. Evaluation:
    * Compute Mean Squared Error (MSE) and R² scores for model predictions.
7. Visualization:
    * Plot feature importance for model interpretability.
    * Compare actual vs. predicted results using line charts.
    * Display stock price trends (Open, High, Low, Close) in OHLC plots.
8. Model Saving:
    * Save trained models as .pkl files for future use.
