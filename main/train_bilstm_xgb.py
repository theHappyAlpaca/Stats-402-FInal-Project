
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb

# Configuration
file_path = 'Please replace this with your own file path'
num_timesteps = 24
batch_size = 64
epochs = 200

# Helper Functions

# Create time series data for supervised learning
def create_time_series_data(inputs, output, num_timesteps):
    X, Y = [], []
    for i in range(len(inputs) - num_timesteps):
        X.append(inputs[i:i+num_timesteps])
        Y.append(output[i+num_timesteps])
    return np.array(X), np.array(Y)

# Print and return evaluation metrics
def compute_metrics(y_true, y_pred, set_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'[{set_name}] RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}')
    return rmse, mae, r2

# Visualize predictions vs. actual values
def visualize_results(y_true, y_pred, set_name):
    plt.figure(figsize=(10, 8))

    # Line plot comparison
    plt.subplot(2, 1, 1)
    plt.plot(y_true, label='Actual', linewidth=1.5)
    plt.plot(y_pred, label='Predicted', linestyle='--', linewidth=1.5)
    plt.legend()
    plt.title(f'{set_name} Prediction Results Comparison')
    plt.grid()

    # Scatter plot for correlation visualization
    plt.subplot(2, 1, 2)
    plt.scatter(y_true, y_pred, s=10, c='b')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    r2 = r2_score(y_true, y_pred)
    plt.text(min(y_true), max(y_pred)*0.9, f'R² = {r2:.2f}', fontsize=12)
    plt.title(f'Scatter Plot ({set_name})')
    plt.grid()
    plt.tight_layout()
    plt.show()

# Load and Preprocess Data

# Load Excel data
data = pd.read_excel(file_path)

# Select feature columns and target column
input_features = ['SO2', 'NO2', 'NO', 'NOx', 'CO', 'PM10']
output_feature = 'PM2.5'
inputs = data[input_features].values
output = data[output_feature].values

# Split dataset into training, validation, and test sets
train_ratio, val_ratio = 0.7, 0.15
total_samples = len(inputs)
train_end = int(train_ratio * total_samples)
val_end = train_end + int(val_ratio * total_samples)

train_inputs = inputs[:train_end]
val_inputs = inputs[train_end:val_end]
test_inputs = inputs[val_end:]

train_output = output[:train_end]
val_output = output[train_end:val_end]
test_output = output[val_end:]

# Normalize the input features using StandardScaler
scaler = StandardScaler()
train_inputs_norm = scaler.fit_transform(train_inputs)
val_inputs_norm = scaler.transform(val_inputs)
test_inputs_norm = scaler.transform(test_inputs)

# Generate time series windows
X_train, y_train = create_time_series_data(train_inputs_norm, train_output, num_timesteps)
X_val, y_val = create_time_series_data(val_inputs_norm, val_output, num_timesteps)
X_test, y_test = create_time_series_data(test_inputs_norm, test_output, num_timesteps)

# Build and Train LSTM Model

# Define LSTM model using TensorFlow/Keras
num_features = X_train.shape[2]
model = Sequential([
    Input(shape=(num_timesteps, num_features)),
    LSTM(100, return_sequences=False),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Define callbacks for early stopping and learning rate reduction
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-5)
]

# Train LSTM model
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# LSTM Output Predictions

# Predict outputs using trained LSTM model
lstm_train_pred = model.predict(X_train)
lstm_val_pred = model.predict(X_val)
lstm_test_pred = model.predict(X_test)

# XGBoost Layer

# Flatten time series features for XGBoost input
train_features = X_train.reshape(X_train.shape[0], -1)
val_features = X_val.reshape(X_val.shape[0], -1)
test_features = X_test.reshape(X_test.shape[0], -1)

# Concatenate LSTM predictions to original features as extra input for XGBoost
train_xgb_input = np.hstack((train_features, lstm_train_pred))
val_xgb_input = np.hstack((val_features, lstm_val_pred))
test_xgb_input = np.hstack((test_features, lstm_test_pred))

# Create XGBoost DMatrix objects
dtrain = xgb.DMatrix(train_xgb_input, label=y_train)
dval = xgb.DMatrix(val_xgb_input, label=y_val)
dtest = xgb.DMatrix(test_xgb_input, label=y_test)

# Define XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'eta': 0.1,
    'max_depth': 6,
    'verbosity': 0
}

# Train XGBoost model
xgb_model = xgb.train(params, dtrain, num_boost_round=100)

# Final Predictions and Evaluation

# Make predictions using trained XGBoost model
y_train_pred = xgb_model.predict(dtrain)
y_val_pred = xgb_model.predict(dval)
y_test_pred = xgb_model.predict(dtest)

# Evaluate performance on all datasets
compute_metrics(y_train, y_train_pred, 'Training Set')
visualize_results(y_train, y_train_pred, 'Training Set')

compute_metrics(y_val, y_val_pred, 'Validation Set')
visualize_results(y_val, y_val_pred, 'Validation Set')

compute_metrics(y_test, y_test_pred, 'Test Set')
visualize_results(y_test, y_test_pred, 'Test Set')
