import os
from distutils.command.config import config

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import EvaluationPipeline
from FeatureMatrixPipeline import get_feature_matrix
from Forecasting.LSTM import LSTMForecastingModel
from Impact.ImpactScoreAnalyzerEnums import ImpactModel
from SP500_Prices.PriceAnalyzer import TechnicalIndicators
from Sentiment.SentimentAnalyzer import DatasetSources, SentimentModel, GranularityLevel

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"  # prevent OpenMP conflict early

# Change working directory to project root
os.chdir("D:/Studium/Master/Masterarbeit/MAI_FinancialForecasting/Program")

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

start_date = "17/12/2017"
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

# with lookahead bias
df_combined['sentiment_tomorrow'] = df_combined['sentiment'].shift(-1)
df_combined.dropna(subset=['sentiment_tomorrow'], inplace=True)

lstm_model = LSTMForecastingModel()
feature_cols = ['sentiment_tomorrow']
target_col = 'Pct_Change_next'
# lstm_model.experiment(df_combined, feature_cols, target_col)

lstm_model_results = EvaluationPipeline.evaluate_model_on_regression(
    model=lstm_model,
    feature_matrix=df_combined,
    predictor_cols=feature_cols,
    target_col=target_col,
    target_horizon_in_days=1
)

# FIX: Handle NaNs upfront in the DataFrame before splitting
df_combined = df_combined.fillna(0.0)
# FIX: Split Data into Train/Test BEFORE Scaling to prevent Data Leakage
split_idx = int(len(df_combined) * 0.8)
train_df = df_combined.iloc[:split_idx].copy()
test_df = df_combined.iloc[split_idx:].copy()

# --- 2. Scaling ---
# # Initialize scalers
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

train_df_copy = train_df.copy()
test_df_copy = test_df.copy()

# Experiment 1
print("train set: ", train_df.shape)
print("test set: ", test_df.shape)
lstm_model.train(train_df[feature_cols], train_df[target_col])
preds = lstm_model.predict(test_df[feature_cols], x_gap=pd.DataFrame())

# Calculate Metrics
mse = mean_squared_error(test_df[target_col][6:], preds)
rmse = np.sqrt(mse)

concatenated = pd.concat([train_df_copy, test_df_copy], ignore_index=True)

# Experiment 2
lstm_model.experiment(concatenated, feature_cols, target_col)

