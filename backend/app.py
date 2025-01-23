import os
import time
import joblib
from re import I
import numpy as np
import pandas as pd
from flask_cors import CORS
from tkinter.constants import W
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from typing import Any, Callable, Iterable


app = Flask(__name__)
CORS(app)

nvda_model_max = None
nvda_model_min = None
nvda_model_avg = None

nvdq_model_max = None
nvdq_model_min = None
nvdq_model_avg = None

def load_models():
    global nvda_model_max, nvda_model_min, nvda_model_avg, nvdq_model_max, nvdq_model_min, nvdq_model_avg

    if nvda_model_max is None or nvda_model_min is None or nvda_model_avg is None:
        try:
           # current_dir = os.path.dirname(os.path.abspath(__file__))
            current_dir = '/var/www/app/'
            models_dir = os.path.join(current_dir, 'models/')


            nvda_model_max = joblib.load(os.path.join(models_dir, 'model_max_nvda.pkl'))
            nvda_model_min = joblib.load(os.path.join(models_dir, 'model_min_nvda.pkl'))
            nvda_model_avg = joblib.load(os.path.join(models_dir, 'model_avg_nvda.pkl'))

            nvdq_model_max = joblib.load(os.path.join(models_dir, 'model_max_nvdq.pkl'))
            nvdq_model_min = joblib.load(os.path.join(models_dir, 'model_min_nvdq.pkl'))
            nvdq_model_avg = joblib.load(os.path.join(models_dir, 'model_avg_nvdq.pkl'))

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

def prepare_features(target_date: str, lookback_days=10) -> Any:
    try:
        #current_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = '/var/www/app/'
        data_path = os.path.join(os.path.dirname(current_dir), 'data', 'nvda_data.csv')
        nvdq_path = os.path.join(os.path.dirname(current_dir), 'data', 'nvdq_data.csv')

        data = pd.read_csv(data_path)
        nvdq_data = pd.read_csv(nvdq_path)

        recent_data = data.tail(lookback_days).copy()
        latest_data = recent_data.copy()

        nvdq_recent_data = nvdq_data.tail(lookback_days).copy()
        nvdq_latest_data = nvdq_recent_data.copy()

        processed_data = pd.DataFrame({
            'Open': [latest_data['Open'].iloc[-1]],
            'Close': [latest_data['Close'].iloc[-1]],
            '5-Day MA': [latest_data['5-Day MA'].iloc[-1]],
            '10-Day MA': [latest_data['10-Day MA'].iloc[-1]],
            'Daily Return (%)': [latest_data['Daily Return (%)'].iloc[-1]],
            'Volatility': [(latest_data['High'].iloc[-1] - latest_data['Low'].iloc[-1]) / latest_data['Low'].iloc[-1] * 100],
            'Day_Of_Week': [pd.to_datetime(target_date).weekday()],
            'Price_Change_Period': [(latest_data['Close'].iloc[-1] - recent_data.iloc[0]['Close']) / recent_data.iloc[0]['Close'] * 100],
            'Avg_Volume_Period': [recent_data['Volume'].mean()],
            'Volatility_Period': [recent_data['High'].max() - recent_data['Low'].min()],
            'Return_Std_Period': [recent_data['Daily Return (%)'].std()]
        })

        nvdq_processed_data = pd.DataFrame({
            'Open': [nvdq_latest_data['Open'].iloc[-1]],
            'Close': [nvdq_latest_data['Close'].iloc[-1]],
            '5-Day MA': [nvdq_latest_data['5-Day MA'].iloc[-1]],
            '10-Day MA': [nvdq_latest_data['10-Day MA'].iloc[-1]],
            'Daily Return (%)': [nvdq_latest_data['Daily Return (%)'].iloc[-1]],
            'Volatility': [(nvdq_latest_data['High'].iloc[-1] - nvdq_latest_data['Low'].iloc[-1]) / nvdq_latest_data['Low'].iloc[-1] * 100],
            'Day_Of_Week': [pd.to_datetime(target_date).weekday()],
            'Price_Change_Period': [(nvdq_latest_data['Close'].iloc[-1] - nvdq_recent_data.iloc[0]['Close']) / nvdq_recent_data.iloc[0]['Close'] * 100],
            'Avg_Volume_Period': [nvdq_recent_data['Volume'].mean()],
            'Volatility_Period': [nvdq_recent_data['High'].max() - nvdq_recent_data['Low'].min()],
            'Return_Std_Period': [nvdq_recent_data['Daily Return (%)'].std()]
        })

        if processed_data.isna().any().any() or nvdq_processed_data.isna().any().any():
            print("Warning: NaN values in features:", processed_data.isna().sum(), nvdq_processed_data.isna().sum())
            raise ValueError("NaN values present in features")

        return (processed_data, nvdq_processed_data)

    except Exception as e:
        print(f"Error in prepare_features: {str(e)}")
        raise

def get_predictions(target_date: str,
                    model_max_nvda: Any,
                    model_min_nvda: Any,
                    model_avg_nvda: Any,
                    model_max_nvdq: Any,
                    model_min_nvdq: Any,
                    model_avg_nvdq: Any
    ) -> Iterable:

    try:
        nyse = mcal.get_calendar('NYSE')
        start_date = pd.to_datetime(target_date)
        end_date = (start_date + pd.DateOffset(days=10)).strftime('%Y-%m-%d')
        next_five = nyse.schedule(start_date=start_date, end_date=end_date).index[:5]

        predictions_nvda_max = []
        predictions_nvda_min = []
        predictions_nvda_avg = []
        predictions_nvdq_max = []
        predictions_nvdq_min = []
        predictions_nvdq_avg = []

        nvda_frame, nvdq_frame = prepare_features(target_date)
        for i in range(5):
            # Create features for current prediction
            nvda_features = nvda_frame.to_numpy().reshape(1, -1)
            nvdq_features = nvdq_frame.to_numpy().reshape(1, -1)

            # Get predictions for current day
            max_price = model_max_nvda.predict(nvda_features)[0]
            min_price = model_min_nvda.predict(nvda_features)[0]
            avg_price = model_avg_nvda.predict(nvda_features)[0]

            nvdq_max_price = model_max_nvdq.predict(nvdq_features)[0]
            nvdq_min_price = model_min_nvdq.predict(nvdq_features)[0]
            nvdq_avg_price = model_avg_nvdq.predict(nvdq_features)[0]

            # Store predictions
            predictions_nvda_max.append(max_price)
            predictions_nvda_min.append(min_price)
            predictions_nvda_avg.append(avg_price)
            predictions_nvdq_max.append(nvdq_max_price - 2)
            predictions_nvdq_min.append(nvdq_min_price - 2)
            predictions_nvdq_avg.append(nvdq_avg_price - 2)

            # Update features for next prediction
            nvda_frame.loc[0, 'Open'] = avg_price
            nvda_frame.loc[0, 'Close'] = avg_price
            nvda_frame.loc[0, '5-Day MA'] = (nvda_frame.loc[0, '5-Day MA'] * 4 + avg_price) / 5
            nvda_frame.loc[0, '10-Day MA'] = (nvda_frame.loc[0, '10-Day MA'] * 9 + avg_price) / 10

            nvdq_frame.loc[0, 'Open'] = nvdq_avg_price
            nvdq_frame.loc[0, 'Close'] = nvdq_avg_price
            nvdq_frame.loc[0, '5-Day MA'] = (nvdq_frame.loc[0, '5-Day MA'] * 4 + nvdq_avg_price) / 5
            nvdq_frame.loc[0, '10-Day MA'] = (nvdq_frame.loc[0, '10-Day MA'] * 9 + nvdq_avg_price) / 10

        # Create arrays for trading strategy using all predictions
        nvda_open = [float(nvda_frame.iloc[0]['Open'])] * 5
        nvdq_open = [float(nvdq_frame.iloc[0]['Open'])] * 5
        nvda_close = predictions_nvda_avg
        nvdq_close = predictions_nvdq_avg

        strategy = generate_trading_strategy(
            predictions_nvda_max,
            predictions_nvda_min,
            predictions_nvdq_max,
            predictions_nvdq_min,
            nvda_open,
            nvdq_open,
            nvda_close,
            nvdq_close,
        )

        final_value, day_values, actions = strategy
    except Exception as e:
                print(f"Error in get_predictions: {str(e)}")
                raise

    return {
        'days': [d.strftime('%Y-%m-%d') for d in next_five],
        'max_price': predictions_nvda_max,
        'min_price': predictions_nvda_min,
        'avg_price': predictions_nvda_avg,
        'nvdq_max_price': predictions_nvdq_max,
        'nvdq_min_price': predictions_nvdq_min,
        'nvdq_avg_price': predictions_nvdq_avg,
        'trading_strategy': actions,
    }
def generate_trading_strategy(y_pred_max_nvda, y_pred_min_nvda, y_pred_max_nvdq, y_pred_min_nvdq,
                                            open_prices_nvda, open_prices_nvdq, close_prices_nvda, close_prices_nvdq,
                                            initial_nvda_shares=10000, initial_nvdq_shares=100000,
                                            threshold=0.05, hold_threshold=0.02):
    """
    Backtesting function with HOLD strategy to simulate trading and calculate final portfolio value.
    """
    # Initialize portfolio
    try:
        '''
        print(f"y_pred_max_nvda: {y_pred_max_nvda}")
        print(f"y_pred_min_nvda: {y_pred_min_nvda}")
        print(f"y_pred_max_nvdq: {y_pred_max_nvdq}")
        print(f"y_pred_min_nvdq: {y_pred_min_nvdq}")
        print(f"Open Prices NVDA: {open_prices_nvda}")
        print(f"Open Prices NVDQ: {open_prices_nvdq}")
        print(f"Close Prices NVDA: {close_prices_nvda}")
        print(f"Close Prices NVDQ: {close_prices_nvdq}")
        '''
        nvda_shares = initial_nvda_shares
        nvdq_shares = initial_nvdq_shares
        cash = 0  # Track cash flow for flexibility
        actions = []
        consolidated_actions = []  # Store the simplified BULLISH/BEARISH/IDLE actions
        portfolio_values = []

        for i in range(len(open_prices_nvda)):
            max_pred_nvda = y_pred_max_nvda[i]
            min_pred_nvda = y_pred_min_nvda[i]
            max_pred_nvdq = y_pred_max_nvdq[i]
            min_pred_nvdq = y_pred_min_nvdq[i]
            open_nvda = open_prices_nvda[i]
            open_nvdq = open_prices_nvdq[i]
            close_nvda = close_prices_nvda[i]
            close_nvdq = close_prices_nvdq[i]

            daily_actions = []

            # Trading Logic for NVDA
            if max_pred_nvda > open_nvda * (1 + threshold):
                nvda_action = "NVDA - BUY"
                if cash > 0:
                    shares_to_buy = cash / open_nvda
                    nvda_shares += shares_to_buy
                    cash = 0
            elif min_pred_nvda < open_nvda * (1 - threshold):
                nvda_action = "NVDA - SELL"
                cash += nvda_shares * open_nvda
                nvda_shares = 0
            elif abs(max_pred_nvda - open_nvda) < open_nvda * hold_threshold:
                nvda_action = "NVDA - HOLD"
            else:
                nvda_action = "NVDA - IDLE"

            # Trading Logic for NVDQ
            if max_pred_nvdq > open_nvdq * (1 + threshold):
                nvdq_action = "NVDQ - BUY"
                if cash > 0:
                    shares_to_buy = cash / open_nvdq
                    nvdq_shares += shares_to_buy
                    cash = 0
            elif min_pred_nvdq < open_nvdq * (1 - threshold):
                nvdq_action = "NVDQ - SELL"
                cash += nvdq_shares * open_nvdq
                nvdq_shares = 0
            elif abs(max_pred_nvdq - open_nvdq) < open_nvdq * hold_threshold:
                nvdq_action = "NVDQ - HOLD"
            else:
                nvdq_action = "NVDQ - IDLE"

            # Determine consolidated action based on the pair of actions
            if nvda_action == "NVDA - BUY" and nvdq_action == "NVDQ - SELL":
                consolidated_action = "BULLISH"
            elif nvda_action == "NVDA - SELL" and nvdq_action == "NVDQ - BUY":
                consolidated_action = "BEARISH"
            else:
                consolidated_action = "IDLE"

            daily_actions = [consolidated_action]
            actions.append(daily_actions)
            consolidated_actions.append(consolidated_action)

            # Calculate portfolio value
            portfolio_value = (nvda_shares * close_nvda) + (nvdq_shares * close_nvdq) + cash
            portfolio_values.append(portfolio_value)

        # Final portfolio value
        final_portfolio_value = portfolio_values[-1]

    except Exception as e:
        print(f"Error in generate_trading_strategy: {str(e)}")
        raise

    return final_portfolio_value, portfolio_values, consolidated_actions


'''
def generate_trading_strategy(max_price,
    min_price,
    avg_price,
    nvdq_max_price,
    nvdq_min_price,
    nvdq_avg_price,
    next_five
):

        strategy:
            curr_max = base_avg + (base_max - base_avg) * spread_multiplier
            curr_min = base_avg + (base_min - base_avg) * spread_multiplier

            upside ptl = calculate max ratio * 100 * decay ^ (day_idx)
            downside risk = calculate min ratio * 100 * decay ^ (day_idx)



    strategies = []



    base_max = max_price
    base_min = min_price
    base_avg = avg_price

    spread_multiplier = 1.0
    confidence_decay = 0.8

    for i in range(len(next_five)):
        current_max = base_avg + (base_max - base_avg) * spread_multiplier
        current_min = base_avg + (base_min - base_avg) * spread_multiplier

        upside_potential = ((current_max - base_avg) / base_avg) * 100 * (confidence_decay ** i)
        downside_risk = ((current_min - base_avg) / base_avg) * 100 * (confidence_decay ** i)

        base_threshold = 2.5
        day_threshold = base_threshold * (1 + i * 0.2)

        if i == len(next_five) - 1:
            if abs(downside_risk) > day_threshold:
                strategy = 'BEARISH'
            else:
                strategy = 'IDLE'
        else:
            if upside_potential > day_threshold and upside_potential > abs(downside_risk) * 1.2:
                strategy = 'BULLISH'
                spread_multiplier *= 1.005
            elif abs(downside_risk) > day_threshold and abs(downside_risk) > upside_potential * 1.2:
                strategy = 'BEARISH'
                spread_multiplier *= 0.995
            else:
                strategy = 'IDLE'

        strategies.append(strategy)

    return strategies
'''
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        selected_date = None
        if data:
            try:
                selected_date = data.get('selected_date')
                #print(f"Processing for date: {selected_date}")
            except:
                print("No date given!")
                return
        else:
            print("No date provided in body.")
            return

        if nvda_model_max is None:
            load_models()

        res: Iterable = get_predictions(selected_date,
            nvda_model_max,
            nvda_model_min,
            nvda_model_avg,
            nvdq_model_max,
            nvdq_model_min,
            nvdq_model_avg
        )
        daily_predictions = []
        for i, day in enumerate(res['days']):
            daily_predictions.append({
                'date': day,
                'nvda': {
                    'max_price': float(res['max_price'][i]),
                    'min_price': float(res['min_price'][i]),
                    'avg_price': float(res['avg_price'][i])
                },
                'nvdq': {
                    'max_price': float(res['nvdq_max_price'][i]),
                    'min_price': float(res['nvdq_min_price'][i]),
                    'avg_price': float(res['nvdq_avg_price'][i])
                }
            })
        trading_strat = []
        for day, strategy in zip(res['days'], res['trading_strategy']):
            tmp = {
                'date': day,
                'strategy': strategy
            }
            trading_strat.append(tmp)
        return jsonify({
                    'success': True,
                    'predictions': daily_predictions,
                    'trading_strategy': trading_strat,
                })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    load_models()
    app.run(port=8989)
