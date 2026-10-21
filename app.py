"""
app.py
======
Production Web Application & REST API for Sales & Demand Forecasting.
Exports top-level 'app', 'application', and 'handler' variables compatible
with Vercel Serverless, WSGI servers (Gunicorn/Waitress), and standard hosting platforms.
Future Interns - Machine Learning Task 1 (2026)
"""

import os
import json
from flask import Flask, render_template, jsonify, send_from_directory, request
import pandas as pd
import numpy as np

# Top-level application instances required by Vercel and WSGI servers
app = Flask(__name__, template_folder="templates")
application = app  # WSGI compatibility alias
handler = app      # Vercel / AWS Lambda serverless handler alias

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    """Serves the interactive executive sales & demand forecasting dashboard."""
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    if os.path.exists(template_path):
        return render_template("index.html")
    return "<h1>Sales & Demand Forecasting System</h1><p>Dashboard template initializing...</p>"


@app.route("/api/health")
def health():
    """System health check endpoint."""
    return jsonify({
        "status": "healthy",
        "system": "Sales & Demand Forecasting Intelligence",
        "author": "Srujan Shetty",
        "version": "1.0.0"
    })


@app.route("/api/forecast")
def get_forecast():
    """Returns 12-week forward demand forecast data."""
    csv_path = os.path.join(BASE_DIR, "outputs", "future_forecast_results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return jsonify(df.to_dict(orient="records"))
    return jsonify({"error": "Forecast data not found"}), 404


@app.route("/api/metrics")
def get_metrics():
    """Returns model benchmark tournament accuracy metrics."""
    csv_path = os.path.join(BASE_DIR, "outputs", "model_evaluation_metrics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return jsonify(df.to_dict(orient="records"))
    return jsonify({"error": "Metrics data not found"}), 404


@app.route("/api/inventory", methods=["GET"])
def calculate_inventory():
    """Calculates dynamic inventory parameters based on query parameters."""
    lead_time = int(request.args.get("lead_time", 2))
    service_level = float(request.args.get("service_level", 0.95))
    
    # Z-scores
    z_table = {0.90: 1.282, 0.95: 1.645, 0.98: 2.054, 0.99: 2.326}
    z_score = z_table.get(service_level, 1.645)

    csv_path = os.path.join(BASE_DIR, "outputs", "future_forecast_results.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        mean_demand = float(df["Forecast"].mean())
        std_demand = float(df["Forecast"].std()) if len(df) > 1 else (mean_demand * 0.228)
    else:
        mean_demand = 10711.52
        std_demand = 2446.22

    lead_time_demand = mean_demand * lead_time
    safety_stock = z_score * std_demand * np.sqrt(lead_time)
    reorder_point = lead_time_demand + safety_stock
    max_stock = reorder_point + (2 * mean_demand)

    return jsonify({
        "lead_time_weeks": lead_time,
        "service_level": service_level,
        "mean_weekly_demand": round(mean_demand, 2),
        "lead_time_demand": round(lead_time_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "max_stock_level": round(max_stock, 2),
        "working_capital_buffer": round(safety_stock * 0.65, 2)
    })


@app.route("/outputs/figures/<path:filename>")
def serve_figures(filename):
    """Serves high-resolution static charts."""
    figures_dir = os.path.join(BASE_DIR, "outputs", "figures")
    return send_from_directory(figures_dir, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
