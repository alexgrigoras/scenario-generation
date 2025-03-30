from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import base64
import logging
import os
import pickle
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from flask_cors import CORS

from source.transformer_forecasting import TransformerForecasting
from source.scenario_generation import ScenarioGeneration

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- API Authentication (Replace with secure method in production) ---
API_KEYS = {"secure_api_key_123"}  # Set of allowed API keyss


# --- Model Parameters ---
dataset_name = 'awsce'                              # Dataset name
input_col = 'price'                                 # Dataset column of the independent variable
output_col = 'demand'                               # Dataset column of the dependent variable
num_layers = 4                                      # range: 2-6
size_layer = 128                                    # range: 64-256
embedded_size = 128                                 # range: 64-256
epochs = 500                                        # range: 200-500 w/ early stopping
dropout_rate = 0.2                                  # range: 0.1-0.3
learning_rate = 0.0001                              # range: 0.0001-0.001
batch_size = 64                                     # 

def get_model(column, dataset_name="dataset"):
    """Training the transformer model on the training set"""

    model_file_path = f"trained-models/transformer-{dataset_name}-{column}.pkl"

    if os.path.exists(model_file_path):
        print(f"Trained model already exists -> loading model {model_file_path}")
        with open(model_file_path, 'rb') as file:
            model = pickle.load(file)
    else:
        print(f"Trained model not found -> model needs to be trained first {model_file_path}")
        
    return model

def authenticate():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return False
    token = auth_header.split(" ")[1]
    return token in API_KEYS

# --- Optimal Profit Window with Machine Learning ---
def determine_optimal_window(forecasted_revenue, col, max_window=6):
    optimal_windows = {}

    for scenario, df in forecasted_revenue.items():
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[col].values
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
        #time_series['timestamp'] = pd.to_datetime(time_series['timestamp'])
        time_series.set_index('timestamp', inplace=True)
        time_series.index = pd.to_datetime(time_series.index)
        price_changes = data['price_changes']
        forecast_length = data.get('forecast_length', 12)

        #logging.info(time_series)

        model = get_model(input_col, dataset_name)

        scenario_generator = ScenarioGeneration()

        # Generate scenarios
        #scenarios, adjusted_input = scenario_generator.generate_scenarios(time_series, price_changes)
        scenarios_input, scenarios_output = scenario_generator.generate_scenarios(time_series, input_col, output_col, price_changes)

        # Apply forecasting
        #forecasted_results = scenario_generator.apply_scenarios(time_series, scenarios, "price", model=forecast_model, forecast_horizon=forecast_length)
        forecasted_results = scenario_generator.apply_scenarios(time_series, scenarios_input, output_col, model=model, forecast_horizon=forecast_length)

        #for k, v in forecasted_results.items():
            #logging.info(k)
            #logging.info(v)

        # Determine optimal profit window
        optimal_windows = determine_optimal_window(forecasted_results, output_col)

        logging.info("Forecast successfully processed")

        return jsonify({
            "forecasted_scenarios": {k: v.rename_axis('timestamp').reset_index().assign(timestamp=lambda df: df['timestamp'].astype(str)).set_index('timestamp').to_dict() for k, v in forecasted_results.items()},
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