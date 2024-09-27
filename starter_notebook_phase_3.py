# Phase 3: Model Monitoring using Evidently
# This starter notebook will help you set up model monitoring using the Evidently library.

# Import necessary libraries
import pandas as pd
import numpy as np
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, RegressionPerformanceTab

# Step 1: Load historical and current prediction data
def load_data():
    # Example data loading (replace with your real model predictions)
    reference_data = pd.DataFrame({'traffic_speed': np.random.normal(50, 5, 1000)})
    current_data = pd.DataFrame({'traffic_speed': np.random.normal(52, 5, 1000)})
    return reference_data, current_data

# Step 2: Create an Evidently dashboard
def create_dashboard(reference_data, current_data):
    # Set up a data drift and regression performance dashboard
    dashboard = Dashboard(tabs=[DataDriftTab(), RegressionPerformanceTab()])
    dashboard.calculate(reference_data, current_data)
    dashboard.save("evidently_report.html")
    print("Evidently report generated at 'evidently_report.html'")

if __name__ == "__main__":
    ref_data, curr_data = load_data()
    create_dashboard(ref_data, curr_data)
