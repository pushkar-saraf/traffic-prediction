# Import essential libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Kubeflow SDK for creating a pipeline
import kfp
from kfp import dsl
from kfp.components import InputPath, OutputPath

# Step 1: Data preprocessing function
def preprocess_data(data: pd.DataFrame, scaler_path: OutputPath(str)) -> np.ndarray:
    # Use MinMaxScaler to normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    np.save(scaler_path, scaler)  # Save the scaler for later use
    return data_scaled

# Step 2: Create sequences for time-series forecasting
def create_sequences(data: np.ndarray, time_steps: int = 10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Step 3: Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Step 4: Build the GRU model
def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Step 5: Train model function
def train_model(model_type: str, data_path: InputPath(str), scaler_path: OutputPath(str)):
    # Load and preprocess data
    data = pd.read_csv(data_path)
    data_scaled = preprocess_data(data, scaler_path)
    
    # Create sequences
    time_steps = 10
    X, y = create_sequences(data_scaled, time_steps)
    
    # Build the appropriate model
    input_shape = (X.shape[1], X.shape[2])
    if model_type == 'LSTM':
        model = build_lstm_model(input_shape)
    elif model_type == 'GRU':
        model = build_gru_model(input_shape)
    
    # Train the model
    model.fit(X, y, epochs=5, batch_size=32)
    
    # Predict and evaluate
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Log results
    print(f'Model Type: {model_type}')
    print(f'MAE: {mae}, MSE: {mse}, R2 Score: {r2}')
    return mae, mse, r2

# Step 6: Kubeflow Pipeline definition
@dsl.pipeline(
    name='Traffic Prediction Pipeline',
    description='Train and evaluate LSTM and GRU models on traffic speed data'
)
def traffic_prediction_pipeline(model_type: str = 'LSTM'):
    # Training the model
    train_op = dsl.ContainerOp(
        name='Train Model',
        image='python:3.8-slim',
        command=['python3', '-c'],
        arguments=[
            f"""
            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, GRU, Dense
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            import pandas as pd
            import numpy as np

            def preprocess_data(data, scaler_path):
                scaler = MinMaxScaler()
                data_scaled = scaler.fit_transform(data)
                np.save(scaler_path, scaler)
                return data_scaled

            def create_sequences(data, time_steps=10):
                X, y = [], []
                for i in range(len(data) - time_steps):
                    X.append(data[i:i + time_steps])
                    y.append(data[i + time_steps])
                return np.array(X), np.array(y)

            def build_lstm_model(input_shape):
                model = Sequential()
                model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
                model.add(LSTM(32))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                return model

            def build_gru_model(input_shape):
                model = Sequential()
                model.add(GRU(64, input_shape=input_shape, return_sequences=True))
                model.add(GRU(32))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                return model

            # Load data
            data = pd.read_csv('{data_path}')
            data_scaled = preprocess_data(data, 'scaler.npy')
            X, y = create_sequences(data_scaled, time_steps=10)

            input_shape = (X.shape[1], X.shape[2])

            if '{model_type}' == 'LSTM':
                model = build_lstm_model(input_shape)
            else:
                model = build_gru_model(input_shape)

            # Train the model
            model.fit(X, y, epochs=5, batch_size=32)

            # Predict and evaluate
            y_pred = model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            print(f'Model Type: {model_type}')
            print(f'MAE: {{mae}}, MSE: {{mse}}, R2 Score: {{r2}}')
            """
        ],
        file_outputs={'metrics': '/output/metrics.txt'}
    )

# Step 7: Compile the pipeline
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(traffic_prediction_pipeline, 'traffic_prediction_pipeline.yaml')

    # Connect to the Kubeflow Pipelines instance (use Minikube URL)
    client = kfp.Client(host='http://localhost:8080')  # Ensure Minikube service is running and accessible

    # Create a new experiment
    experiment = client.create_experiment(name='Traffic_Prediction_Experiment')

    # Run the pipeline
    run = client.run_pipeline(
        experiment_id=experiment.id,
        job_name='Traffic_Prediction_Run',
        pipeline_package_path='traffic_prediction_pipeline.yaml',
        params={"model_type": "LSTM"}  # Change to "GRU" to use the GRU model
    )
