import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from autots import AutoTS
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
from math import sqrt

from source.transformer import TransformerForecasting

class ScenarioGeneration:
    def __init__(self, forecast_model="transformer", num_layers=1, size_layer=128, batch_size=32, epoch=100, dropout_rate=0.1, forecast_length=30, learning_rate=0.0001, simulation_size=10):
        print("Init transformer")

        self.forecast_model = forecast_model
        self.num_layers = num_layers
        self.size_layer = size_layer
        self.batch_size = batch_size
        self.epoch = epoch
        self.dropout_rate = dropout_rate
        self.forecast_length = forecast_length
        self.learning_rate = learning_rate
        self.simulation_size = simulation_size

        if forecast_model == "timegpt":
            nixtla_client = NixtlaClient(api_key = 'nixtla-tok-jNt1fIlgsGeTI3nto5VlwxGMPezKso6HBT2JGFsX9uQFP9SSaEbjXlNOrvRkSu9knzRanQYCIdebAZrE')
            nixtla_client.validate_api_key()

    ## Scenario generation

    # Calculate price elasticity of demand using a linear regression model
    def calculate_elasticity(self, demand, price):
        regression = LinearRegression()
        log_price = np.log(price).values.reshape(-1, 1)
        log_demand = np.log(demand).values.reshape(-1, 1)
        regression.fit(log_price, log_demand)
        elasticity = regression.coef_[0][0]
        return elasticity

    # Non-linear elasticity function
    def non_linear_elasticity(self, price_change, base_elasticity):
        # Quadratic function, with larger price changes having a stronger effect
        return base_elasticity * (1 + 0.5 * price_change**2)

    # Apply stochastic adjustment
    def apply_randomness(self, demand, scenario_name, randomness_factor=0.01, cap=0.05):
        # Introduce randomness based on scenario (e.g., larger changes have more variability)
        if 'Price Change' in scenario_name:
            change = int(scenario_name.split()[2].replace('%', ''))
        else:
            change = 0
        randomness = np.random.normal(loc=0, scale=randomness_factor * abs(change), size=len(demand))
        
        # Cap the randomness to prevent extreme deviations
        capped_randomness = np.clip(randomness, -cap, cap)
        
        stochastic_demand = demand * (1 + capped_randomness)
        return stochastic_demand

    # Forecasting function using SARIMAX
    def forecast_with_sarimax(self, series, steps=30):
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=steps)
        return forecast

    # Forecasting function using Transformer
    def forecast_with_transformer(self, series, steps=30):
        transformer_forecasting = TransformerForecasting()

        series = pd.DataFrame(series)
        minmax = MinMaxScaler().fit(series.astype('float32'))
        df_log = minmax.transform(series.astype('float32'))
        df_train = pd.DataFrame(df_log)

        results = []
        for i in range(self.simulation_size):
            print('simulation %d'%(i + 1))
            results.append(np.abs(transformer_forecasting.forecast(df_train, minmax, self.num_layers, self.size_layer, self.batch_size, self.epoch, self.dropout_rate, self.forecast_length, self.learning_rate)))

        results_df = pd.DataFrame(series.iloc[-self.forecast_length:])
            
        # Adding the number of months to the timestamp index
        results_df.index = results_df.index + pd.DateOffset(months=self.forecast_length)

        for idx, result in enumerate(results):
            results_df[idx] = result
        results_df['value'] = results_df.mean(axis=1)

        return results_df[["value"]]

    # Forecasting function using AutoTS
    def forecast_with_autots(self, series, steps=30):
        #model = AutoTS(forecast_length=steps, frequency='infer', ensemble='simple')
        model = AutoTS(
            forecast_length=steps,
            frequency='infer',
            #prediction_interval=0.95,
            ensemble='simple',
            models_mode='default',              # or 'deep', 'regressor'
            model_list = 'fast_parallel',       #['ARIMA','ETS'],       # or ['ARIMA','ETS'] or "fast", "superfast", "default", "fast_parallel"
            max_generations=4,
            num_validations=2,
            no_negatives=True,
            n_jobs='auto',
            holiday_country='US',
            verbose=0,
        )
        model = model.fit(series)
        prediction = model.predict()
        forecast = prediction.forecast
        return forecast

    # Forecasting function using timegpt (nixtla)
    def forecast_with_timegpt(self, series, steps=30):
        df = series.reset_index()

        target_col = df.columns[1]

        forecast = nixtla_client.forecast(df=df, h=steps, time_col='timestamp', target_col=target_col, freq='MS', model='timegpt-1-long-horizon')

        forecast.set_index('timestamp', inplace=True)
        forecast.rename(columns={'TimeGPT': target_col}, inplace=True)

        return forecast

    # Wrapper for selecting the forecasting model
    def forecast(self, series, model="", steps=30):
        if model == "":
            model = self.forecast_model
        if model == "sarimax":
            return self.forecast_with_sarimax(series, steps)
        elif model == "transformer":
            return self.forecast_with_transformer(series, steps)
        elif model == "autots":
            return self.forecast_with_autots(series, steps)
        elif model == "timegpt":
            return self.forecast_with_timegpt(series, steps)

    # Generate scenarios based on price change
    def generate_scenarios(self, data, price_change_percentages):
        base_elasticity = self.calculate_elasticity(data['demand'], data['price'])
        adjusted_input = {}
        scenarios = {}
        for change in price_change_percentages:
            adjusted_price = data['price'] * (1 + change / 100)
            adjusted_elasticity = self.non_linear_elasticity(change / 100, base_elasticity)
            adjusted_demand = data['demand'] * (adjusted_price / data['price']) ** adjusted_elasticity
            adjusted_demand = self.apply_randomness(adjusted_demand, f'Price Change {change}%', randomness_factor=0.005, cap=0.005)
            scenarios[f'Price Change {change}%'] = adjusted_demand
            adjusted_input[f'Price Change {change}%'] = adjusted_price
        return scenarios, adjusted_input

    # Apply the what-if scenarios and forecast
    def apply_scenarios(self, data, scenarios, column_name, model="autots", forecast_horizon=30):
        results = {}
        for scenario, value in scenarios.items():
            print("Scenario " + scenario)
            forecasted_data = self.forecast(value, model=model, steps=forecast_horizon)
            forecasted_data.loc[data.index[-1]] = data[column_name][-1]
            forecasted_data.sort_index(inplace=True)
            results[scenario] = forecasted_data
        return results

    # Visualize the results
    def visualize_results(self, original_series, forecasted_results, column_name):
        plt.figure(figsize=(14, 7))
        plt.plot(original_series.index, original_series, label='Original ' + str(column_name), color='black')

        for scenario, forecast in forecasted_results.items():
            if scenario == "Baseline":
                plt.plot(forecast.index, forecast, label=scenario, color='black')
            else:
                plt.plot(forecast.index, forecast, label=scenario)
        
        plt.title('What-If Scenario Analysis of ' + str(column_name) + ' with Non-Linear Elasticity')
        plt.xlabel('Date')
        plt.ylabel(column_name)
        plt.legend()
        plt.show()

    # Function to prepare the dataset for training the model with a quadratic penalty for large windows
    def prepare_training_data_with_quadratic_penalty(self, forecasted_revenue, max_window_size, penalty_factor=0.1):
        X = []
        y = []
        for scenario, revenue_series in forecasted_revenue.items():
            revenue_values = revenue_series.values
            
            for window_size in range(1, max_window_size + 1):
                for start in range(len(revenue_values) - window_size + 1):
                    window_revenue = revenue_values[start:start + window_size]
                    total_profit = np.sum(window_revenue)
                    rolling_mean = np.mean(window_revenue)
                    rolling_std = np.std(window_revenue)

                    # Apply quadratic penalty based on window size
                    window_size_penalty = penalty_factor * (window_size ** 2)

                    X.append([
                        start,  # Start index
                        window_size,  # Window size
                        rolling_mean,  # Mean revenue
                        rolling_std,  # Std deviation in window
                        window_size_penalty  # Quadratic penalty term for large windows
                    ])
                    y.append(total_profit)

        return np.array(X), np.array(y)

    # Function to determine the optimal profit window using the model with a quadratic penalty for large windows
    def find_optimal_window_with_quadratic_penalty(self, forecasted_revenue, model, max_window_size, penalty_factor=0.1):
        optimal_result = {
            "scenario": None,
            "start_index": None,
            "end_index": None,
            "profit": float('-inf')
        }

        for scenario, revenue_series in forecasted_revenue.items():
            revenue_values = revenue_series.values
            scenario_X = []
            scenario_indices = []

            for window_size in range(1, max_window_size + 1):
                for start in range(len(revenue_values) - window_size + 1):
                    window_revenue = revenue_values[start:start + window_size]
                    rolling_mean = np.mean(window_revenue)
                    rolling_std = np.std(window_revenue)

                    # Apply quadratic penalty based on window size
                    window_size_penalty = penalty_factor * (window_size ** 2)

                    scenario_X.append([
                        start,
                        window_size,
                        rolling_mean,
                        rolling_std,
                        window_size_penalty  # Quadratic penalty term
                    ])
                    scenario_indices.append((start, start + window_size - 1))
            
            scenario_X = np.array(scenario_X)
            predicted_profits = model.predict(scenario_X)
            best_window_idx = np.argmax(predicted_profits)

            if predicted_profits[best_window_idx] > optimal_result["profit"]:
                optimal_result.update({
                    "scenario": scenario,
                    "start_index": scenario_indices[best_window_idx][0],
                    "end_index": scenario_indices[best_window_idx][1],
                    "profit": predicted_profits[best_window_idx]
                })

        return optimal_result

    # Visualization of the results with quadratic penalty
    def visualize_optimal_window_with_history(self, original_series, forecasted_revenue, optimal_window_ml):
        """
        Visualizes the forecasted results with the optimal profit window highlighted, along with the historical series.

        Args:
        - original_series: The original time series data (historical values).
        - forecasted_revenue: Dictionary of forecasted revenue scenarios.
        - optimal_window_ml: Dictionary containing details about the optimal profit window.
        """
        plt.figure(figsize=(14, 7))
        
        # Plot the original (historical) revenue series in black
        plt.plot(original_series.index, original_series.values, label='Original Series', color='black', linewidth=2)
        
        # Initialize variable to track the global minimum y-value for all plots
        global_min_value = original_series.min()  # Start with the min value from historical data
        
        # Plot the forecasted revenue for each scenario and update the global minimum
        for scenario, forecast in forecasted_revenue.items():
            # Ensure forecast is a Series and use the first column if it's a DataFrame
            if isinstance(forecast, pd.DataFrame):
                forecast = forecast.iloc[:, 0]
            
            plt.plot(forecast.index, forecast.values, label=f'Forecasted: {scenario}')
            global_min_value = min(global_min_value, forecast.min())  # Safely update the global minimum

        # Extract optimal scenario, start index, and end index
        optimal_scenario = optimal_window_ml['scenario']
        start_idx = optimal_window_ml['start_index']
        end_idx = optimal_window_ml['end_index']
        
        # Extract the specific scenario's forecasted revenue series
        optimal_forecast = forecasted_revenue[optimal_scenario]
        if isinstance(optimal_forecast, pd.DataFrame):
            optimal_forecast = optimal_forecast.iloc[:, 0]  # Take the first column if it's a DataFrame
        
        # Get the optimal window based on start and end indices
        optimal_window = optimal_forecast.iloc[start_idx:end_idx + 1]
        
        # Highlight the optimal window on the plot
        plt.fill_between(
            optimal_window.index,  # Use the index for x-axis (time)
            global_min_value,  # Lower bound (global minimum of all scenarios)
            optimal_window.values,  # Upper bound (actual values for the window)
            color='yellow',  # Color for the optimal window
            alpha=0.3,  # Transparency
            label=f'Optimal Window: {optimal_scenario} ({start_idx}-{end_idx})'
        )

        # Add plot labels, title, and legend
        plt.title('Forecasted Revenue with Optimal Profit Window Highlighted')
        plt.xlabel('Time')
        plt.ylabel('Revenue')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
