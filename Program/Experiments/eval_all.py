import os
import pandas as pd
import EvaluationPipeline
from FeatureMatrixPipeline import get_feature_matrix
from Forecasting.ARMA import ARMAForecastingModel
from Forecasting.GRU import GRUForecastingModel
from Forecasting.LSTM import LSTMForecastingModel
from Forecasting.MLR import MLRForecastingModel
from Forecasting.MLogR import MLogRForecastingModel
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
impact_model = ImpactModel.LLAMA_3_1_Instruct
df_combined = get_feature_matrix(
    start_date=start_date,
    end_date=end_date,
    impact_model=impact_model,
    tech_indicators=[TechnicalIndicators.VOLATILITY, TechnicalIndicators.MOVING_AVERAGE_30],
    sentiment_sources=[DatasetSources.LUCASPHAM, DatasetSources.NIFTY],
    sentiment_model=SentimentModel.FINBERT,
    granularity_level=GranularityLevel.DAILY
)
print(df_combined.head(20))
print(df_combined.info())

print("count of sentiment 0s:", (df_combined['weighted_sentiment'] == 0).sum())

feature_cols = ['Pct_Change', 'weighted_sentiment', 'Volatility']
target_col = 'Pct_Change_next'

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

print("-----------------------CLASSIFICATION RESULTS-----------------------")
print("XGBoost Results:", xGBoost_model_results)
print("MLogR Results:", mlogr_model_results)