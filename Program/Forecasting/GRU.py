import os
import random
from distutils.command.config import config

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from Forecasting.EvaluationModelBase import ForecastingModelBase

# --- 0. Configuration & Reproducibility ---
CONFIG = {
    'seq_length': 7,
    'batch_size': 64,
    'hidden_size': 64,  # 128 might be overkill for only 4 features
    'num_layers': 2,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'weight_decay': 0,   # L2 Regularization (Crucial new addition)
    'epochs': 250,  # High number, controlled by Early Stopping
    'patience': 100000,  # Stop if validation loss doesn't improve for 10 epochs
    'model_path': 'gru_model.pth',
    'scaler_path': 'scalers.pkl'
}

def set_seed(seed=42):
    # 1. Python & OS
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # 4. Deterministic algorithms (Crucial for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 4. Model Definition ---
class GRURegression(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super().__init__()
        # CHANGE 1: Use nn.GRU instead of nn.LSTM
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)

        # CHANGE 2: GRU returns (output, h_n).
        # Unlike LSTM, there is no cell state (c_n).
        gru_out, _ = self.gru(x)

        # Take the output of the last time step
        # shape: (batch, hidden_size)
        last_time_step = gru_out[:, -1, :]

        out = self.fc(last_time_step)
        return out

class GRUForecastingModel(ForecastingModelBase):
    def __init__(self):

        self.model = None
        super().__init__()

    def train(self, x_train: pd.DataFrame, y_train: pd.Series):
        set_seed(42)

        # Set device (GPU/MPS/CPU)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")

        # create validation split from train data
        split_idx = int(len(x_train) * 0.8)
        X_val = x_train.iloc[split_idx:].copy()
        y_val = y_train.iloc[split_idx:].copy()
        x_train = x_train.iloc[:split_idx].copy()
        y_train = y_train.iloc[:split_idx].copy()

        X_train, y_train = self._create_sequences( x_train, y_train, CONFIG['seq_length'])
        X_val, y_val = self._create_sequences(X_val, y_val, CONFIG['seq_length'])

        # Convert to Tensors
        # X_train is already (Samples, Seq_Len, Features).
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)  # Target needs (Batch, 1)

        X_test_tensor = torch.FloatTensor(X_val).to(device)
        y_test_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)

        # Dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

        model = GRURegression(
            input_size=len(x_train.columns),
            hidden_size=CONFIG['hidden_size'],
            num_layers=CONFIG['num_layers'],
            dropout_prob=CONFIG['dropout']
        ).to(device)

        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'],
                                     weight_decay=CONFIG['weight_decay'])

        # --- 5. Training with Early Stopping ---
        best_val_loss = float('inf')
        patience_counter = 0

        print("Starting training...")
        for epoch in range(CONFIG['epochs']):
            model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation step
            model.eval()
            with torch.no_grad():
                val_preds = model(X_test_tensor)
                val_loss = criterion(val_preds, y_test_tensor).item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{CONFIG['epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), CONFIG['model_path'])  # Save best model
            else:
                patience_counter += 1
                if patience_counter >= CONFIG['patience']:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        # Load best model weights before exiting train
        self.model = model
        self.model.load_state_dict(torch.load(CONFIG['model_path']))
        pass

    def predict(self, x_test: pd.DataFrame, x_gap : pd.DataFrame) -> pd.Series:
        set_seed(42)
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Set device (GPU/MPS/CPU)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")

        xs = self._create_sequences_predict(x_test, CONFIG['seq_length'])

        # GRU prediction
        xs_tensor = torch.FloatTensor(xs).to(device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(xs_tensor)

        # Convert back to CPU numpy and then Series
        preds_np = preds.cpu().numpy().flatten()
        return pd.Series(preds_np)

    def _create_sequences(self, data_features, data_target, seq_length):
        # convert to numpy arrays
        data_features_np = data_features.values
        data_target_np = data_target.values

        xs, ys = [], []
        # Use len(data_features) correctly
        for i in range(len(data_features_np) - seq_length):
            x = data_features_np[i:(i + seq_length)]
            y = data_target_np[i + seq_length - 1] # target value is in the same data row as feature
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def _create_sequences_predict(self, data_features, seq_length):
        # convert to numpy arrays
        data_features_np = data_features.values

        xs = []
        # Use len(data_features) correctly
        for i in range(len(data_features_np) - seq_length + 1):
            x = data_features_np[i:(i + seq_length)]
            xs.append(x)

        return np.array(xs)

    def experiment(self, feature_matrix: pd.DataFrame, predictor_cols: list[str], target_col: str):
        pass

    def plot_results(self):
        pass