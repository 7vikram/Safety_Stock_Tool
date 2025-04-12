from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
import logging
import os

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)

def calculate_inventory_metrics(service_level, lead_time, historical_consumption, future_demand, current_inventory, moving_average_window, moq):
    try:
        # Parse input data
        hist_data = np.array(list(map(float, historical_consumption.split(","))))
        future_data = np.array(list(map(float, future_demand.split(","))))
        current_inventory = float(current_inventory)
        moving_average_window = int(moving_average_window)
        moq = float(moq)

        if len(hist_data) == 0 or len(future_data) == 0:
            raise ValueError("Historical consumption and future demand cannot be empty.")

        if service_level <= 0 or service_level >= 1:
            raise ValueError("Service level must be between 0 and 1 (exclusive).")

        if lead_time <= 0 or moving_average_window <= 0 or moq <= 0:
            raise ValueError("Lead time, moving average window, and MOQ must be positive numbers.")

        # Calculate safety stock for original service level
        sigma_demand = np.std(hist_data)
        z_score = norm.ppf(service_level)
        safety_stock = z_score * sigma_demand * np.sqrt(lead_time)

        # Simulate SS Min and SS Max
        ss_min_service_level = max(0.01, service_level - 0.1)  # Ensure service level doesn't go below 0%
        ss_max_service_level = min(0.99, service_level + 0.1)  # Ensure service level doesn't exceed 100%

        ss_min = norm.ppf(ss_min_service_level) * sigma_demand * np.sqrt(lead_time)
        ss_max = norm.ppf(ss_max_service_level) * sigma_demand * np.sqrt(lead_time)

        # Calculate stock-out probabilities
        stockout_prob_ss = 1 - norm.cdf(z_score)
        stockout_prob_ss_min = 1 - norm.cdf(norm.ppf(ss_min_service_level))
        stockout_prob_ss_max = 1 - norm.cdf(norm.ppf(ss_max_service_level))

        # Calculate moving average for future demand
        future_moving_avg = pd.Series(future_data).rolling(moving_average_window, min_periods=1).mean()

        # Calculate reorder point
        reorder_point = safety_stock + hist_data.mean() * lead_time

        # Predict inventory trajectory with replenishments
        inventory_levels = [current_inventory]
        replenishment_points = []

        for i, demand in enumerate(future_moving_avg):
            # Deduct demand from inventory
            current_inventory = inventory_levels[-1] - demand

            # Check if inventory falls below ROP and replenish
            if current_inventory < reorder_point:
                current_inventory += moq  # Replenish MOQ
                replenishment_points.append(i)  # Record the replenishment point

            inventory_levels.append(current_inventory)

        inventory_levels = np.array(inventory_levels)

        # Create replenishment cycle plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=inventory_levels, mode="lines+markers", name="Inventory Levels"))
        fig.add_trace(go.Scatter(y=[reorder_point] * len(inventory_levels), mode="lines", name="Reorder Point"))
        fig.add_trace(go.Scatter(y=[safety_stock] * len(inventory_levels), mode="lines", name="Safety Stock (SS)"))
        fig.add_trace(go.Scatter(y=[ss_min] * len(inventory_levels), mode="lines", name="Safety Stock Min (SS Min)", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(y=[ss_max] * len(inventory_levels), mode="lines", name="Safety Stock Max (SS Max)", line=dict(dash="dot")))

        # Mark replenishment points
        replenishment_y = [inventory_levels[i + 1] for i in replenishment_points]
        fig.add_trace(go.Scatter(
            x=replenishment_points,
            y=replenishment_y,
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Replenishment"
        ))

        # Determine max inventory level achieved
        max_inventory = np.max(inventory_levels)

        graph_json = fig.to_json()
        return {
            "safety_stock": safety_stock,
            "safety_stock_min": ss_min,
            "safety_stock_max": ss_max,
            "stockout_prob_ss": stockout_prob_ss,
            "stockout_prob_ss_min": stockout_prob_ss_min,
            "stockout_prob_ss_max": stockout_prob_ss_max,
            "reorder_point": reorder_point,
            "max_inventory": max_inventory,
            "graph": graph_json
        }
    except Exception as e:
        raise ValueError(f"Error in calculation: {str(e)}")

@app.route("/")
def index():
    return render_template("index.html")

logging.basicConfig(level=logging.DEBUG)

@app.route("/calculate-inventory-metrics/", methods=["POST"])
def calculate():
    try:
        # Get input data from the form
        service_level = float(request.form["service_level"])
        lead_time = int(request.form["lead_time"])
        historical_consumption = request.form["historical_consumption"]
        future_demand = request.form["future_demand"]
        current_inventory = request.form["current_inventory"]
        moving_average_window = request.form["moving_average_window"]
        moq = request.form["moq"]
        
        logging.debug(f"Input data: {request.form}")

        # Perform calculations
        result = calculate_inventory_metrics(
            service_level, lead_time, historical_consumption, future_demand, current_inventory, moving_average_window, moq
        )
        logging.debug(f"Calculation result: {result}")
        # Return results as JSON
        return jsonify(result)
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

if __name__ == "__main__":
    # Use dynamic port for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)