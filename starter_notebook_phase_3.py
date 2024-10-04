import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently import ColumnMapping
from flask import Flask, send_file
import smtplib

# Step 1: Generate Dummy Data

# Generate dummy reference data (training data)
np.random.seed(0)
reference_data = pd.DataFrame({
    'feature_1': np.random.normal(0, 1, 1000),
    'feature_2': np.random.normal(0, 1, 1000),
    'target': np.random.choice([0, 1], 1000)
})

# Add a dummy prediction column to the reference data (NaN values)
reference_data['prediction'] = 0  # Dummy prediction column in reference data

# Generate dummy current data (production data, simulating drift)
current_data = pd.DataFrame({
    'feature_1': np.random.normal(0.5, 1, 1000),  # Simulated drift in feature_1
    'feature_2': np.random.normal(0, 1, 1000),
    'target': np.random.choice([0, 1], 1000)
})

# Simulate a simple prediction model (replace this with your actual model)
current_data['prediction'] = np.random.choice([0, 1], 1000)  # Simulated predictions for current data

# Step 2: Define the Column Mapping

# Create a column mapping to tell Evidently which columns represent predictions, targets, and features
column_mapping = ColumnMapping(
    prediction="prediction",
    target="target",
    numerical_features=["feature_1", "feature_2"]
)

# Step 3: Set Up Evidently Report for Monitoring Data Drift and Classification Performance

# Initialize the report with data drift and classification performance metrics
report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset()
])

# Calculate the report based on reference and current data, with column mapping
report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

# Save the report as an HTML file
report.save_html("evidently_model_report.html")

# Step 4: Set Up Flask App to Serve the Monitoring Dashboard

app = Flask(__name__)

@app.route('/monitoring')
def show_dashboard():
    # Serve the Evidently report as a static HTML file
    return send_file('evidently_model_report.html')

# Step 5: Function to Send Email Alerts for Data Drift

def send_email_alert(drift_score):
    sender = 'alert@yourdomain.com'
    receivers = ['team@yourdomain.com']
    message = f"""Subject: Data Drift Alert

    Data drift detected! The drift score is {drift_score}.
    Please check the monitoring dashboard for further details.
    """

    try:
        smtp_obj = smtplib.SMTP('localhost')  # Ensure a mail server is running locally or replace with a real SMTP server
        smtp_obj.sendmail(sender, receivers, message)
        print("Successfully sent email alert")
    except Exception as e:
        print(f"Error: unable to send email - {e}")

# Step 6: Check Data Drift and Trigger Email Alerts

# Get the drift results as a dictionary from Evidently report
drift_results = report.as_dict()
dataset_drift = drift_results['metrics'][0]['result']['dataset_drift']  # Get the dataset drift value

# Define a threshold for drift (e.g., 0.5 or 50%)
drift_threshold = 0.5

# Check if drift exceeds the threshold and trigger an email alert
if dataset_drift > drift_threshold:
    print(f"Data drift detected! Drift score: {dataset_drift}")
    send_email_alert(dataset_drift)
else:
    print(f"No significant data drift detected. Drift score: {dataset_drift}")

# Step 7: Run the Flask App

if __name__ == "__main__":
    print("Starting the Flask server for monitoring dashboard...")
    app.run(host="0.0.0.0", port=5001)
