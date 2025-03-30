import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class ScenarioGeneration:
    def calculate_structural_elasticity(self, input, output):
        """Structural causal elasticity calculation"""
        regression = LinearRegression()
        log_input = np.log(input).values.reshape(-1, 1)
        log_output = np.log(output).values.reshape(-1, 1)
        regression.fit(log_input, log_output)
        elasticity = regression.coef_[0][0]
        intercept = regression.intercept_[0]
        residuals = log_output - regression.predict(log_input)
        return elasticity, intercept, residuals.flatten()

    def non_linear_elasticity(self, change, base_elasticity):
        """Non-linear elasticity adjustment"""
        return base_elasticity * (1 + 0.5 * change**2)

    def apply_randomness(self, demand, change, randomness_factor=0.01, cap=0.05, seed=42):
        np.random.seed(seed)
        randomness = np.random.normal(loc=0, scale=randomness_factor * abs(change), size=len(demand))
        capped_randomness = np.clip(randomness, -cap, cap)
        stochastic_demand = demand * (1 + capped_randomness)
        return stochastic_demand

    # Scenario generation using structural causal inference
    def generate_scenarios(self, data, input_col, output_col, price_change_percentages):
        elasticity, intercept, residuals = self.calculate_structural_elasticity(data[input_col], data[output_col])
        scenarios_input = {}
        scenarios_output = {}
        for change in price_change_percentages:
            adjusted_input = data[input_col] * (1 + change / 100)
            adjusted_elasticity = self.non_linear_elasticity(change / 100, elasticity)
            counterfactual_output = np.exp(intercept + adjusted_elasticity * np.log(adjusted_input) + residuals)
            counterfactual_output = self.apply_randomness(counterfactual_output, change, randomness_factor=0.1, cap=0.1)
            scenarios_input[f'{input_col} change {change}%'] = adjusted_input
            scenarios_output[f'{input_col} change {change}%'] = counterfactual_output
        return scenarios_input, scenarios_output

    # Forecast application to generated scenarios
    def apply_scenarios(self, data, scenarios, column_name, model, forecast_horizon=30):
        results = {}
        for scenario, value in scenarios.items():
            forecasted_data = model.forecast(value.values, forecast_horizon)
            forecasted_df = pd.DataFrame(data[column_name].iloc[-forecast_horizon:])
            forecasted_df[column_name] = forecasted_data
            forecasted_df.index = forecasted_df.index + pd.DateOffset(months=forecast_horizon)
            forecasted_df.loc[data.index[-1]] = data[column_name][-1]
            forecasted_df.sort_index(inplace=True)
            results[scenario] = forecasted_df
        return results