# Phase 2: Model Deployment using Docker and Kubernetes
# This starter notebook will guide you through deploying your trained model as a RESTful API using Flask.
import mlflow
# Import necessary libraries
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf


# Load the trained model (ensure your model is saved from Phase 1)
def load_model():
    logged_model = 'runs:/74acccf0b4c6452f9ae2b9e943f8c5bf/traffic_prediction_model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model


# Initialize Flask app
app = Flask(__name__)
model = load_model()  # Replace 'lstm_model.h5' with the appropriate model file


@app.route('/predict', methods=['POST'])
def predict():
    # Parse input data
    data = request.get_json(force=True)
    prediction_input = np.array(data['input']).reshape(1, -1)  # Adjust input shape as per your model
    prediction = model.predict(prediction_input).tolist()

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)

# TODO: Build a Dockerfile for this Flask app and deploy using Kubernetes
