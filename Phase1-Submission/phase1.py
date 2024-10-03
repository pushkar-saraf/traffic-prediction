# Phase 1: Model Experimentation using Kubeflow
# This starter notebook will guide you through building LSTM and GRU models 
# and tracking the experiments using Kubeflow.

import h5py
# Kubeflow SDK for creating a pipeline
import kfp
# Import essential libraries
import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import GRU, Dense, LSTM
from kfp import dsl, compiler
from kfp.dsl import Dataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Step 1: Load and preprocess the METR-LA dataset
# Replace this with actual dataset loading logic
def load_data():
    filename = 'METR-LA.h5'
    with h5py.File(filename, 'r') as f:
        timestamps = pd.to_datetime(f['df']['axis1'][:], unit='ns')
        sensor_ids = [s.decode('utf-8') for s in f['df']['axis0'][:]]
        data_values = f['df']['block0_values'][:]
    df = pd.DataFrame(data_values, index=timestamps, columns=sensor_ids)
    return df


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
        if i==0:
            plot_acf(data[0], lags=50)
            plt.show()
            exit()
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


# Step 4: Define LSTM and GRU models
def build_lstm_model(input_shape):
    # Basic LSTM architecture
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(207))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_gru_model(input_shape):
    # Basic GRU architecture
    model = Sequential()
    model.add(GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(207))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# Step 5: Define the Kubeflow pipeline function
def train_and_test(name: str, model: Sequential, sequences: Dataset):
    x = sequences[0]
    y = sequences[1]
    dataset_len = len(sequences[1])
    train_size = int(dataset_len * 0.7)
    x_train, x_test = x[0:train_size], x[train_size:dataset_len]
    y_train, y_test = y[0:train_size], y[train_size:dataset_len]
    model.fit(x_train, y_train, verbose=1)
    # make predictions
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)
    # calculate root mean squared error

    results = {}

    results['rmse'] = {}
    results['rmse']['train_score'] = np.sqrt(mean_squared_error(y_train, train_predict))
    print('Train Score: %.2f RMSE' % (results['rmse']['train_score']))
    results['rmse']['test_score'] = np.sqrt(mean_squared_error(y_test, test_predict))
    print('Test Score: %.2f RMSE' % (results['rmse']['test_score']))

    results['mae'] = {}
    results['mae']["train_score"] = mean_absolute_error(y_train, train_predict)
    print('Train Score: %.2f MAE' % (results['mae']["train_score"]))
    results['mae']["test_score"] = mean_absolute_error(y_test, test_predict)
    print('Test Score: %.2f MAE' % (results['mae']["train_score"]))

    results['r2'] = {}
    results['r2']["train_score"] = r2_score(y_train, train_predict)
    print('Train Score: %.2f R2' % (results['r2']["train_score"]))
    results['r2']["test_score"] = r2_score(y_test, test_predict)
    print('Test Score: %.2f R2' % (results['r2']["test_score"]))

    # Save the model into h5 format
    return results


@dsl.component
def print_result(model: str, result: str) -> str:
    result = f'{result}!'
    return result


@dsl.pipeline(name='ML Model', description='Train and test model')
def evaluate_model(model_type: str) -> str:
    data = load_data()
    data = preprocess_data(data)[0]
    sequences = create_sequences(data)
    input_shape = (10, 207)
    if model_type == 'LSTM':
        model = build_lstm_model(input_shape)
    elif model_type == 'GRU':
        model = build_gru_model(input_shape)
    else:
        raise 'Invalid model name'
    result = train_and_test(name=model_type, model=model, sequences=sequences)
    result = print_result(model=model_type, result=str(result))
    return result.output


# Compile and run the Kubeflow pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(evaluate_model, 'pipeline_lstm.yaml')
    client = kfp.Client(host=f"http://localhost:8080/pipeline")
    experiment = client.create_experiment('phase 1')
    run1 = client.run_pipeline(experiment.experiment_id, 'traffic_prediction_run', 'pipeline_lstm.yaml',
                               {'model_type': 'LSTM'})
    compiler.Compiler().compile(evaluate_model, 'pipeline_gru.yaml')
    client = kfp.Client(host=f"http://localhost:8080/pipeline")
    experiment = client.create_experiment('phase 1')
    run2 = client.run_pipeline(experiment.experiment_id, 'traffic_prediction_run', 'pipeline_gru.yaml',
                               {'model_type': 'GRU'})
