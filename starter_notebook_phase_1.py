# Phase 1: Model Experimentation using Kubeflow
# This starter notebook will guide you through building LSTM and GRU models 
# and tracking the experiments using Kubeflow.

# Import essential libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Kubeflow SDK for creating a pipeline
import kfp
from kfp import dsl

# Step 1: Load and preprocess the METR-LA dataset
# Replace this with actual dataset loading logic
def load_data():
    # Example: Create a dummy dataset
    data = pd.DataFrame(np.sin(np.linspace(0, 100, 1000)), columns=['traffic_speed'])
    return data

# Step 2: Data preprocessing
def preprocess_data(data):
    # Use MinMaxScaler to normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

# Step 3: Create sequences for time-series forecasting
def create_sequences(data, time_steps=10):
    # Create sequences for LSTM or GRU input
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Step 4: Define LSTM and GRU models
def build_lstm_model(input_shape):
    # Basic LSTM architecture
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_gru_model(input_shape):
    # Basic GRU architecture
    model = Sequential()
    model.add(GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Step 5: Define the Kubeflow pipeline function
@dsl.pipeline(
    name='Traffic Prediction Pipeline',
    description='Train and evaluate LSTM and GRU models'
)
def traffic_prediction_pipeline(model_type: str):
    # The training function logic will be included here (refer to Phase 1 completion steps)
    pass  # TODO: Replace with the pipeline components

# Compile and run the Kubeflow pipeline
if __name__ == "__main__":
    # Compile pipeline definition
    kfp.compiler.Compiler().compile(traffic_prediction_pipeline, 'traffic_prediction_pipeline.yaml')
    
    # Upload and run pipeline (Make sure Kubeflow Pipelines is running)
    client = kfp.Client()
    experiment = client.create_experiment('Traffic_Prediction_Experiment')
    
    # Start a run for either LSTM or GRU
    run = client.run_pipeline(experiment.id, 'traffic_prediction_run', 'traffic_prediction_pipeline.yaml',
                              {"model_type": "LSTM"})  # Change to "GRU" for GRU model training
