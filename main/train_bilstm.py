import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Set device to CPU
device = torch.device("cpu")

# Custom dataset class for time series inputs
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# BiLSTM model definition
class BiLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMNet, self).__init__()
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # LSTM output: only keep the last timestep output
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Load data and interpolate missing values with cubic spline
def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    data = df.iloc[:, 1:]  # Skip date column if present
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data_num = data[numeric_cols].copy()
    data_cat = pd.DataFrame()

    # Interpolate all numeric columns
    data_all = pd.concat([data_cat, data_num], axis=1)
    for col in data_all.columns:
        series = data_all[col]
        if series.isnull().any():
            not_nan = series.dropna()
            f = interp1d(not_nan.index, not_nan.values, kind='cubic', fill_value='extrapolate')
            data_all[col] = f(range(len(series)))
    return data_all.values

# Create input-output sequences for supervised learning
def create_sequences(data, input_len, predict_len):
    x, y = [], []
    for i in range(len(data) - input_len - predict_len + 1):
        x.append(data[i:i + input_len])
        y.append(data[i + input_len:i + input_len + predict_len, -1])  # Target: last column (PM2.5)
    return np.array(x), np.array(y)

# Training loop for PyTorch model
def train_model(model, train_loader, num_epochs, criterion, optimizer):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
    return model

# Generate predictions for the entire dataset
def predict_all(model, x_array):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_array), 64):
            batch_x = torch.tensor(x_array[i:i+64], dtype=torch.float32).to(device)
            pred = model(batch_x)
            preds.append(pred.cpu().numpy())
    return np.vstack(preds)

# Evaluate model on given data split and plot results
def evaluate_from_raw(model, x_array, y_true, scaler_y, label="Set"):
    y_pred = predict_all(model, x_array)
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_true_inv = scaler_y.inverse_transform(y_true)

    # Compute evaluation metrics
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    mape = np.mean(np.abs((y_true_inv - y_pred_inv) / y_true_inv)) * 100
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_inv, y_pred_inv)

    print(f"\n{label} Evaluation:")
    print(f"MAE: {mae:.4f}, MAPE: {mape:.2f}%, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Plot prediction vs actual
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_inv, label="True")
    plt.plot(y_pred_inv, label="Predicted")
    plt.title(f"{label} Prediction")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function for training and evaluating BiLSTM on CPU
def main_cpu(file_path):
    # Load and preprocess data
    data = load_and_preprocess_data(file_path)
    input_len = 24
    predict_len = 1
    x, y = create_sequences(data, input_len, predict_len)

    # Train-validation-test split
    split1 = int(0.7 * len(x))
    split2 = int(0.85 * len(x))
    x_train, y_train = x[:split1], y[:split1]
    x_val, y_val = x[split1:split2], y[split1:split2]
    x_test, y_test = x[split2:], y[split2:]

    # Normalize inputs and outputs
    scaler_x = StandardScaler().fit(x_train.reshape(-1, x_train.shape[-1]))
    scaler_y = StandardScaler().fit(y_train)
    x_train = scaler_x.transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_val = scaler_x.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)
    x_test = scaler_x.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
    y_train = scaler_y.transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    # Wrap data in PyTorch DataLoader
    train_loader = DataLoader(TimeSeriesDataset(x_train, y_train), batch_size=64, shuffle=True)

    # Initialize and train BiLSTM model
    model = BiLSTMNet(input_size=x.shape[2], hidden_size=64, output_size=predict_len)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = train_model(model, train_loader, num_epochs=50, criterion=criterion, optimizer=optimizer)

    # Evaluate on all data splits
    evaluate_from_raw(model, x_train, y_train, scaler_y, label="Train Set")
    evaluate_from_raw(model, x_val, y_val, scaler_y, label="Validation Set")
    evaluate_from_raw(model, x_test, y_test, scaler_y, label="Test Set")

