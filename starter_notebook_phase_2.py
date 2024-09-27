# Phase 2: Model Deployment using Docker and Kubernetes
# This starter notebook will guide you through deploying your trained model as a RESTful API using Flask.

# Import necessary libraries
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

# Load the trained model (ensure your model is saved from Phase 1)
def load_model(model_name='lstm_model.h5'):
    return tf.keras.models.load_model(model_name)

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
