import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Input
# Helper Functions

# Convert multivariate time series into supervised learning samples
def create_time_series_data(inputs, output, num_timesteps):
    X, Y = [], []
    for i in range(len(inputs) - num_timesteps):
        X.append(inputs[i:i + num_timesteps])
        Y.append(output[i + num_timesteps])
    return np.array(X), np.array(Y)

# Compute and print evaluation metrics: RMSE, MAE, R²
def compute_metrics(y_true, y_pred, set_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{set_name} RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
    return rmse, mae, r2

# Visualize actual vs. predicted results using line plots and scatter plots
def visualize_results(y_true, y_pred, set_name):
    plt.figure(figsize=(12, 6))

    # Line plot: time series comparison
    plt.subplot(2, 1, 1)
    plt.plot(y_true, label='Actual Values', linewidth=1.5)
    plt.plot(y_pred, label='Predicted Values', linestyle='--', linewidth=1.5)
    plt.title(f'{set_name} Prediction Results Comparison')
    plt.xlabel('Time Points')
    plt.ylabel('PM2.5 Concentration')
    plt.legend()
    plt.grid(True)

    # Scatter plot: prediction accuracy
    plt.subplot(2, 1, 2)
    plt.scatter(y_true, y_pred, s=20)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', linewidth=2)
    plt.title(f'Scatter Plot of Predictions vs Observed ({set_name})')
    plt.xlabel('Actual PM2.5')
    plt.ylabel('Predicted PM2.5')
    r2 = r2_score(y_true, y_pred)
    plt.text(min(y_true), max(y_pred) * 0.9, f'R² = {r2:.2f}', fontsize=12, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Load Data 

# Path to the Excel dataset
file_path = 'Please replace this with your own file path'
data = pd.read_excel(file_path)

# Select input and output features
input_features = ['SO2', 'NO2', 'NO', 'NOx', 'CO', 'PM10']
output_feature = 'PM2.5'
inputs = data[input_features].values
output = data[output_feature].values

# Model Parameters
num_timesteps = 24           # Length of the input sequence window
num_hidden_units = 100       # Number of LSTM units
max_epochs = 200             # Max training epochs
batch_size = 64              # Mini-batch size

# Split Data into Train / Validation / Test
total_samples = len(inputs)
train_ratio, val_ratio = 0.7, 0.15
train_end = int(train_ratio * total_samples)
val_end = train_end + int(val_ratio * total_samples)

train_inputs = inputs[:train_end]
train_output = output[:train_end]
val_inputs = inputs[train_end:val_end]
val_output = output[train_end:val_end]
test_inputs = inputs[val_end:]
test_output = output[val_end:]

# Normalize Inputs
scaler = StandardScaler()
train_inputs_norm = scaler.fit_transform(train_inputs)
val_inputs_norm = scaler.transform(val_inputs)
test_inputs_norm = scaler.transform(test_inputs)

# Create Supervised Time Series Samples
X_train, y_train = create_time_series_data(train_inputs_norm, train_output, num_timesteps)
X_val, y_val = create_time_series_data(val_inputs_norm, val_output, num_timesteps)
X_test, y_test = create_time_series_data(test_inputs_norm, test_output, num_timesteps)

# Build LSTM Model
num_features = X_train.shape[2]
model = Sequential([
    Input(shape=(num_timesteps, num_features)),
    LSTM(num_hidden_units, return_sequences=False),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Callbacks: Early Stopping & Learning Rate Scheduler
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-5)
]

# Train the Model
model.fit(
    X_train, y_train,
    epochs=max_epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Make Predictions
y_pred_train = model.predict(X_train).flatten()
y_pred_val = model.predict(X_val).flatten()
y_pred_test = model.predict(X_test).flatten()

# Evaluate and Visualize Results
train_rmse, train_mae, train_r2 = compute_metrics(y_train, y_pred_train, "Training Set")
visualize_results(y_train, y_pred_train, "Training Set")

val_rmse, val_mae, val_r2 = compute_metrics(y_val, y_pred_val, "Validation Set")
visualize_results(y_val, y_pred_val, "Validation Set")

test_rmse, test_mae, test_r2 = compute_metrics(y_test, y_pred_test, "Test Set")
visualize_results(y_test, y_pred_test, "Test Set")
