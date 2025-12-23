import os
from distutils.command.config import config

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from FeatureMatrixPipeline import get_feature_matrix
from Impact.ImpactScoreAnalyzerEnums import ImpactModel
from SP500_Prices.PriceAnalyzer import TechnicalIndicators
from Sentiment.SentimentAnalyzer import DatasetSources, SentimentModel, GranularityLevel

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"  # prevent OpenMP conflict early

# Change working directory to project root
os.chdir("D:/Studium/Master/Masterarbeit/MAI_FinancialForecasting/Program")

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

start_date = "17/12/2010"
end_date = "18/07/2019"
impact_model = ImpactModel.NONE
df_combined = get_feature_matrix(
    start_date=start_date,
    end_date=end_date,
    impact_model=impact_model,
    tech_indicators=[TechnicalIndicators.VOLATILITY],
    sentiment_sources=[DatasetSources.LUCASPHAM, DatasetSources.NIFTY],
    sentiment_model=SentimentModel.FINBERT,
    granularity_level=GranularityLevel.DAILY
)
print(df_combined.head(20))
print(df_combined.info())

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- 0. Configuration & Reproducibility ---
CONFIG = {
    'seq_length': 30,
    'batch_size': 32,
    'hidden_size': 64,  # 128 might be overkill for only 4 features
    'num_layers': 2,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'weight_decay': 0,   # L2 Regularization (Crucial new addition)
    'epochs': 1000,  # High number, controlled by Early Stopping
    'patience': 1000000,  # Stop if validation loss doesn't improve for 10 epochs
    'model_path': 'lstm_model.pth',
    'scaler_path': 'scalers.pkl'
}

# Set device (GPU/MPS/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Data Preparation ---
# Mocking df_combined for the sake of the example to make it runnable
# Replace this with your actual dataframe load
# df_combined = pd.read_csv('your_data.csv')
feature_cols = ['Pct_Change', 'Volume', 'Volatility', 'sentiment']
target_col = 'Pct_Change_next'

# FIX: Handle NaNs upfront in the DataFrame before splitting
df_combined = df_combined.fillna(0.0)

# FIX: Split Data into Train/Test BEFORE Scaling to prevent Data Leakage
split_idx = int(len(df_combined) * 0.8)
train_df = df_combined.iloc[:split_idx].copy()
test_df = df_combined.iloc[split_idx:].copy()

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

# --- 3. Sequence Creation ---
def create_sequences(data_features, data_target, seq_length):
    xs, ys = [], []
    # Use len(data_features) correctly
    for i in range(len(data_features) - seq_length):
        x = data_features[i:(i + seq_length)]
        y = data_target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


X_train, y_train = create_sequences(train_df[feature_cols].values, train_df[target_col].values, CONFIG['seq_length'])
X_test, y_test = create_sequences(test_df[feature_cols].values, test_df[target_col].values, CONFIG['seq_length'])

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


model = LSTMRegressor(
    input_size=len(feature_cols),
    hidden_size=CONFIG['hidden_size'],
    num_layers=CONFIG['num_layers'],
    dropout_prob=CONFIG['dropout']
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])

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
        print(f"Epoch {epoch + 1}/{CONFIG['epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

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

