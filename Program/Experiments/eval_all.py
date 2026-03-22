import os

import numpy as np
import pandas as pd
from sympy.core.random import random

import EvaluationPipeline
from FeatureMatrixPipeline import get_feature_matrix
from Forecasting.ARMA import ARMAForecastingModel
from Forecasting.GRU import GRUForecastingModel
from Forecasting.LSTM import LSTMForecastingModel
from Forecasting.MLR import MLRForecastingModel
from Forecasting.MLogR import MLogRForecastingModel
from Forecasting.AlwaysMean import MeanForecastingModel

from Forecasting.MomentumBased import MomentumBasedForecastingModel
from Forecasting.XGBoost import XGBoostForecastingModel
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
end_date = "06/11/2020"
impact_model = ImpactModel.NONE
df_combined = get_feature_matrix(
    start_date=start_date,
    end_date=end_date,
    impact_model=impact_model,
    tech_indicators=[TechnicalIndicators.VOLATILITY, TechnicalIndicators.VIX, TechnicalIndicators.MOVING_AVERAGE_30, TechnicalIndicators.US1Y_YIELD],
    sentiment_sources=[DatasetSources.LUCASPHAM, DatasetSources.AENLLE],
    sentiment_model=SentimentModel.FINBERT,
    granularity_level=GranularityLevel.DAILY
)
print(df_combined.head(20))
print(df_combined.info())

# print("count of sentiment 0s:", (df_combined['weighted_sentiment'] == 0).sum())
#
# # with lookahead bias
df_combined['sentiment_tomorrow'] = df_combined['sentiment'].shift(-1)
df_combined.dropna(subset=['sentiment_tomorrow'], inplace=True)
# #
# df_combined['rolling_weighted_sentiment_3day'] = df_combined['weighted_sentiment'].rolling(window=3, min_periods=1).mean()
# df_combined['Target_60d_Return'] = df_combined['Close'].pct_change(periods=60).shift(-60)
# df_combined.dropna(subset=['Target_60d_Return'], inplace=True)

# fill series with random values
# df_combined['noise'] = np.random.uniform(-0.05, 0.05, size=len(df_combined))

feature_cols = ['Pct_Change']
#feature_cols = ['Log_Pct_Change', 'sentiment', 'VIX', 'US1Y_Yield', 'Volume']
# feature_cols = ['weighted_sentiment']
# feature_cols = ['weighted_sentiment', 'VIX']
# feature_cols = ['Pct_Change', 'VIX', 'Volume', 'Moving_Average_30']
# feature_cols = ['weighted_sentiment', 'Pct_Change', 'VIX', 'Volume', 'Moving_Average_30']
# feature_cols = ['Pct_Change_next']
# feature_cols = ['rolling_sentiment_30day']
target_col = 'Pct_Change_next'

mean_model = MeanForecastingModel()
mean_model_results = EvaluationPipeline.evaluate_model_on_regression(
    model=mean_model,
    feature_matrix=df_combined,
    predictor_cols=feature_cols,
    target_col=target_col,
    target_horizon_in_days=1
)

# momentum_model_results = None
# if 'Pct_Change' in df_combined.columns:
#     feature_cols_copy = feature_cols.copy()
#     feature_cols_copy.append('Pct_Change') # momentum model need this column (yt-1)
#     momentum_model = MomentumBasedForecastingModel()
#     momentum_model_results = EvaluationPipeline.evaluate_model_on_regression(
#         model=momentum_model,
#         feature_matrix=df_combined,
#         predictor_cols=feature_cols_copy,
#         target_col=target_col,
#         target_horizon_in_days=1
#     )

arima_model = ARMAForecastingModel()
arima_results = EvaluationPipeline.evaluate_model_on_regression(
    model=arima_model,
    feature_matrix=df_combined,
    predictor_cols=feature_cols,
    target_col=target_col,
    target_horizon_in_days=1
)

gru_model = GRUForecastingModel()
gru_model_results = EvaluationPipeline.evaluate_model_on_regression(
    model=gru_model,
    feature_matrix=df_combined,
    predictor_cols=feature_cols,
    target_col=target_col,
    target_horizon_in_days=1
)

mlr_model = MLRForecastingModel()
mlr_model_results = EvaluationPipeline.evaluate_model_on_regression(
    model=mlr_model,
    feature_matrix=df_combined,
    predictor_cols=feature_cols,
    target_col=target_col,
    target_horizon_in_days=1
)

lstm_model = LSTMForecastingModel()
lstm_model_results = EvaluationPipeline.evaluate_model_on_regression(
    model=lstm_model,
    feature_matrix=df_combined,
    predictor_cols=feature_cols,
    target_col=target_col,
    target_horizon_in_days=1
)

# Change target to next day up and down (1 if up, 0 if down)
df_combined['Target'] = (df_combined["Pct_Change_next"] > 0).astype(int) # 1 if next day's pct change > 0 else 0
target_col = 'Target'

xGBoost_model = XGBoostForecastingModel()
xGBoost_model_results = EvaluationPipeline.evaluate_model_on_classification(
    model=xGBoost_model,
    feature_matrix=df_combined,
    predictor_cols=feature_cols,
    target_col=target_col,
    target_horizon_in_days=1
)

mlogr_model = MLogRForecastingModel()
mlogr_model_results = EvaluationPipeline.evaluate_model_on_classification(
    model=mlogr_model,
    feature_matrix=df_combined,
    predictor_cols=feature_cols,
    target_col=target_col,
    target_horizon_in_days=1
)

# plot the results
print("-----------------------REGRESSION RESULTS-----------------------")
print("ARIMA Results:", arima_results)
print("LSTM Results:", lstm_model_results)
print("GRU Results:", gru_model_results)
print("MLR Results:", mlr_model_results)
print("Mean-Baseline:", mean_model_results)
# if momentum_model_results is not None:
#     print("Momentum-Based Results:", momentum_model_results)

print("-----------------------CLASSIFICATION RESULTS-----------------------")
print("XGBoost Results:", xGBoost_model_results)
print("MLogR Results:", mlogr_model_results)