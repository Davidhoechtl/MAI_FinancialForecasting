import os
from distutils.command.config import config

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import EvaluationPipeline
from FeatureMatrixPipeline import get_feature_matrix
from Forecasting.ARMA import ARMAForecastingModel
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
end_date = "06/11/2020"
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

print("count of sentiment 0s:", (df_combined['sentiment'] == 0).sum())

arima_model = ARMAForecastingModel()
lstm_model = LSTMForecastingModel()

feature_cols = ['Pct_Change', 'sentiment']
target_col = 'Pct_Change_next'

arima_results = EvaluationPipeline.evaluate_model_on_regression(
    model=arima_model,
    feature_matrix=df_combined,
    predictor_cols=feature_cols,
    target_col=target_col,
    target_horizon_in_days=1
)

lstm_model_results = EvaluationPipeline.evaluate_model_on_regression(
    model=lstm_model,
    feature_matrix=df_combined,
    predictor_cols=feature_cols,
    target_col=target_col,
    target_horizon_in_days=1
)

# plot the results
print("ARIMA Results:", arima_results)
print("LSTM Results:", lstm_model_results)