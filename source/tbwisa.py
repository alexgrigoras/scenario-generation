import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class TBWISAGenerator:
    """Scenario generation using structural causal inference"""

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

    def generate_scenarios(self, data, input_col, output_col, price_change_percentages):
        """Scenario generation using structural causal inference"""
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

    def apply_scenarios(self, data, scenarios, column_name, model, forecast_horizon=30):
        """Forecast application to generated scenarios"""
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
    
    def calculate_score_with_demand(self, revenue_window, demand_window, penalty_factor, demand_weight):
        """Compute score: revenue minus length penalty, plus reward for low demand"""

        avg_revenue = np.mean(revenue_window)
        window_size = len(revenue_window)
        penalty = penalty_factor * window_size

        # Normalize demand
        max_demand = np.max(demand_window) if np.max(demand_window) > 0 else 1
        avg_demand_norm = np.mean(demand_window) / max_demand

        # Invert demand - higher score for lower demand
        demand_boost = demand_weight * (1 - avg_demand_norm)

        return avg_revenue - penalty + demand_boost

    def find_optimal_window_with_demand(self, forecasted_result, forecasted_output, penalty_factor=0.1, demand_weight=1.0,
                                        min_window_size=2, max_window_size=5):
        """Find the best window for promotion considering both revenue and inverse demand"""

        best = {"scenario": None, "start": None, "end": None, "score": -np.inf}

        for scenario in forecasted_result:
            if scenario == "Baseline" or scenario == "Actuals":
                continue
            rev_series = forecasted_result[scenario].values
            demand_series = forecasted_output[scenario].values
            scenario_best_score = -np.inf
            scenario_best_range = (None, None)

            for window_size in range(min_window_size, min(len(rev_series), max_window_size) + 1):
                for start in range(len(rev_series) - window_size + 1):
                    rev_window = rev_series[start:start + window_size]
                    demand_window = demand_series[start:start + window_size]

                    score = self.calculate_score_with_demand(rev_window, demand_window, penalty_factor, demand_weight)

                    if score > scenario_best_score:
                        scenario_best_score = score
                        scenario_best_range = (start, start + window_size - 1)

            if scenario_best_score > best["score"]:
                best.update({
                    "scenario": scenario,
                    "start": scenario_best_range[0],
                    "end": scenario_best_range[1],
                    "score": scenario_best_score
                })

        return best