import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from Forecasting.EvaluationModelBase import ForecastingModelBase

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

# --- 0. Configuration & Reproducibility ---
CONFIG = {
    'seq_length': 7,
    'batch_size': 32,
    'hidden_size': 64,  # 128 might be overkill for only 4 features
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'weight_decay': 0,   # L2 Regularization (Crucial new addition)
    'epochs': 250,  # High number, controlled by Early Stopping
    'patience': 100000,  # Stop if validation loss doesn't improve for 10 epochs
    'model_path': 'lstm_model.pth',
    'scaler_path': 'scalers.pkl'
}

# --- 4. Model Definition ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)

        # Take the output of the last time step
        # shape: (batch, hidden_size)
        last_time_step = lstm_out[:, -1, :]

        out = self.fc(last_time_step)
        return out

class LSTMForecastingModel(ForecastingModelBase):
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
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

        model = LSTMRegressor(
            input_size=len(x_train.columns),
            hidden_size=CONFIG['hidden_size'],
            num_layers=CONFIG['num_layers'],
            dropout_prob=CONFIG['dropout']
        ).to(device)

        criterion = nn.MSELoss()
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

        x_test_tensor = torch.FloatTensor(xs).to(device)
        with torch.no_grad():
            y_pred_scaled = self.model(x_test_tensor).cpu().numpy()

        return pd.Series(y_pred_scaled.flatten())

    # --- 3. Sequence Creation ---
    def _create_sequences(self, data_features, data_target, seq_length):
        # convert to numpy arrays
        data_features_np = data_features.values
        data_target_np = data_target.values

        xs, ys = [], []
        # Use len(data_features) correctly
        for i in range(len(data_features_np) - seq_length):
            x = data_features_np[i:(i + seq_length)]
            y = data_target_np[i + seq_length]
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def _create_sequences_predict(self, data_features, seq_length):
        # convert to numpy arrays
        data_features_np = data_features.values

        xs = []
        # Use len(data_features) correctly
        for i in range(len(data_features_np) - seq_length):
            x = data_features_np[i:(i + seq_length)]
            xs.append(x)

        return np.array(xs)

    def experiment(self, feature_matrix: pd.DataFrame, predictor_cols: list[str], target_col: str):
        set_seed(42)
        # Set device (GPU/MPS/CPU)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")

        # --- 1. Data Preparation ---
        # Mocking df_combined for the sake of the example to make it runnable
        # Replace this with your actual dataframe load
        # df_combined = pd.read_csv('your_data.csv')
        feature_cols = predictor_cols

        # FIX: Handle NaNs upfront in the DataFrame before splitting
        df_combined = feature_matrix.fillna(0.0)

        # FIX: Split Data into Train/Test BEFORE Scaling to prevent Data Leakage
        split_idx = int(len(df_combined) * 0.8)
        train_df = df_combined.iloc[:split_idx].copy()
        test_df = df_combined.iloc[split_idx:].copy()

        print("train set: ", train_df.shape)
        print("test set: ", test_df.shape)

        # --- 2. Scaling ---
        # Initialize scalers
        scaler_x = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))

        # FIX: Fit ONLY on Training data
        scaler_x.fit(train_df[feature_cols])
        scaler_y.fit(train_df[[target_col]])

        # Transform both Train and Test using the Train-fitted scalers
        train_df[feature_cols] = scaler_x.transform(train_df[feature_cols])
        train_df[target_col] = scaler_y.transform(train_df[[target_col]])

        test_df[feature_cols] = scaler_x.transform(test_df[feature_cols])
        test_df[target_col] = scaler_y.transform(test_df[[target_col]])

        print(train_df.head(10))

        X_train, y_train = self._create_sequences(train_df[feature_cols], train_df[target_col], CONFIG['seq_length'])
        X_test, y_test = self._create_sequences(test_df[feature_cols], test_df[target_col], CONFIG['seq_length'])

        # Convert to Tensors
        # FIX: Removed .unsqueeze(1). LSTM expects (Batch, Seq_Len, Features).
        # X_train is already (Samples, Seq_Len, Features).
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)  # Target needs (Batch, 1)

        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)

        # Dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)


        model = LSTMRegressor(
            input_size=len(feature_cols),
            hidden_size=CONFIG['hidden_size'],
            num_layers=CONFIG['num_layers'],
            dropout_prob=CONFIG['dropout']
        ).to(device)

        criterion = nn.MSELoss()
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

        # --- 6. Final Evaluation ---
        # Load the best model (not necessarily the last one)
        model.load_state_dict(torch.load(CONFIG['model_path']))
        model.eval()

        with torch.no_grad():
            y_pred_scaled = model(X_test_tensor).cpu().numpy()
            y_test_actual_scaled = y_test_tensor.cpu().numpy()

        # Inverse transform to get real units (Pct_Change)
        y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
        y_test_real = scaler_y.inverse_transform(y_test_actual_scaled)

        # Metrics
        mse = np.mean((y_pred_real - y_test_real) ** 2)
        rmse = np.sqrt(mse)

        print(f"\nFinal Test RMSE (Original Units): {rmse:.6f}")
        print("\nSample predictions vs actual (Original Units):")
        for i in range(5):
            print(f"Pred: {y_pred_real[i][0]:.6f}, Actual: {y_test_real[i][0]:.6f}")
        pass

    def plot_results(self):
        pass