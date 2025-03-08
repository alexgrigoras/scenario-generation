from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import base64
import logging
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
from io import BytesIO
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from flask_cors import CORS

from source.scenario_generation import ScenarioGeneration

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- API Authentication (Replace with secure method in production) ---
API_KEYS = {"secure_api_key_123"}  # Set of allowed API keyss

# Forecasting
forecast_model = "transformer"

# Tranformer                  # initial              # best results
# ---------------------------------------------------------------------- 
simulation_size = 1           # 10                   # 10
num_layers = 8                # 1                    # 8
size_layer = 128              # 128                  # 128
epoch = 100                   # 300                  # 500
dropout_rate = 0.8            # 0.8                  # 0.7
learning_rate = 0.001         # 0.001                # 0.001
batch_size = 5                # 5                    # 5

def authenticate():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split(" ")[1]
    return token in API_KEYS

# --- Optimal Profit Window with Machine Learning ---
def determine_optimal_window(forecasted_revenue, max_window=6):
    optimal_windows = {}

    for scenario, df in forecasted_revenue.items():
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['revenue'].values
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)

        max_profit_idx = np.argmax(y_pred)
        start_idx = max(0, max_profit_idx - max_window // 2)
        end_idx = min(len(df) - 1, start_idx + max_window)

        optimal_windows[scenario] = {
            "scenario": scenario,
            "start_index": start_idx,
            "end_index": end_idx,
            "max_profit": y_pred[max_profit_idx]
        }

    return optimal_windows


# --- API Endpoint ---
@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Authentication Check
        if not authenticate():
            logging.warning("Unauthorized access attempt")
            return jsonify({"error": "Unauthorized"}), 403

        logging.info("Received a forecast request")

        # Receive JSON input
        data = request.json
        time_series = pd.DataFrame(data['time_series'])
        price_changes = data['price_changes']
        forecast_length = data.get('forecast_length', 12)

        from source.scenario_generation import ScenarioGeneration
        scenario_generator = ScenarioGeneration(forecast_model, num_layers, size_layer, batch_size, epoch, dropout_rate, forecast_length, learning_rate, simulation_size)

        # Generate scenarios
        scenarios, adjusted_input = scenario_generator.generate_scenarios(time_series, price_changes)

        # Apply forecasting
        forecasted_results = scenario_generator.apply_scenarios(time_series, scenarios, "price", model=forecast_model, forecast_horizon=forecast_length)

        # Determine optimal profit window
        optimal_windows = scenario_generator.determine_optimal_window(forecasted_results)

        logging.info("Forecast successfully processed")

        return jsonify({
            "forecasted_scenarios": {k: v.astype(float).to_dict() for k, v in forecasted_results.items()},
            "optimal_profit_window": {k: {key: (value.item() if isinstance(value, np.generic) else value) for key, value in v.items()} for k, v in optimal_windows.items()}
        })

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/forecast', methods=['OPTIONS'])
def options_request():
    response = jsonify()
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)


## Example call to the API
"""
curl -X POST http://127.0.0.1:5000/forecast \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer secure_api_key_123" \
     -d '{
         "time_series": {
             "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
             "price": [100, 105, 110],
             "demand": [500, 480, 460]
         },
         "price_changes": [5, 10],
         "forecast_length": 12
     }'
"""