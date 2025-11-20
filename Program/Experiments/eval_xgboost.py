import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from FeatureMatrixPipeline import get_feature_matrix
from Forecasting.ARMA import ARMAForecastingModel
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
end_date = "18/07/2020"
impact_model = ImpactModel.NONE
df_combined = get_feature_matrix(
    start_date=start_date,
    end_date=end_date,
    impact_model=impact_model,
    tech_indicators=[TechnicalIndicators.VOLATILITY],
    sentiment_sources=[DatasetSources.LUCASPHAM],
    sentiment_model=SentimentModel.FINBERT,
    granularity_level=GranularityLevel.DAILY
)

df_combined['rolling_sentiment_3'] = df_combined['sentiment'].rolling(window=3, min_periods=1).mean()

# --- Create target (next-day direction) ---
df_combined["Target"] = (df_combined["Pct_Change_next"] > 0).astype(int) # 1 if next day's pct change > 0 else 0
df = df_combined.dropna(subset=["Target"])

# is weighted sentiment benefitial
eval_model = XGBoostForecastingModel()
eval_model.evaluate(df_combined, target_col="Target", predictor_cols=['Pct_Change'])
eval_model.plot_results()
eval_model = XGBoostForecastingModel()
eval_model.evaluate(df_combined, target_col="Target", predictor_cols=['Pct_Change', 'sentiment'])
eval_model.plot_results()
eval_model = XGBoostForecastingModel()
eval_model.evaluate(df_combined, target_col="Target", predictor_cols=['Pct_Change', 'weighted_sentiment'])
eval_model.plot_results()

# is volatility and sentiment in combination benefitial
eval_model = XGBoostForecastingModel()
eval_model.evaluate(df_combined, target_col="Target", predictor_cols=['Pct_Change', 'Volatility'])
eval_model.plot_results()
eval_model = XGBoostForecastingModel()
eval_model.evaluate(df_combined, target_col="Target", predictor_cols=['Pct_Change', 'sentiment', 'Volatility'])
eval_model.plot_results()

# is rolling sentiment benefitial
eval_model = XGBoostForecastingModel()
eval_model.evaluate(df_combined, target_col="Target", predictor_cols=['Pct_Change', 'sentiment', 'Volatility', 'Volume'])
eval_model.plot_results()
eval_model = XGBoostForecastingModel()
eval_model.evaluate(df_combined, target_col="Target", predictor_cols=['Pct_Change', 'rolling_sentiment_3', 'Volatility', 'Volume'])
eval_model.plot_results()