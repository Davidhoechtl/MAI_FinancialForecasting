import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import EvaluationPipeline
from FeatureMatrixPipeline import get_feature_matrix
from Forecasting.ARMA import ARMAForecastingModel
from Forecasting.MLR import MLRForecastingModel
from Forecasting.MLogR import MLogRForecastingModel
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
impact_model = ImpactModel.LLAMA_3_1_Instruct
df_combined = get_feature_matrix(
    start_date=start_date,
    end_date=end_date,
    impact_model=impact_model,
    tech_indicators=[TechnicalIndicators.VOLATILITY, TechnicalIndicators.MOVING_AVERAGE_30],
    sentiment_sources=[DatasetSources.LUCASPHAM],
    sentiment_model=SentimentModel.FINBERT,
    granularity_level=GranularityLevel.DAILY
)

df_combined["Target"] = (df_combined["Pct_Change_next"] > 0).astype(int) # 1 if next day's pct change > 0 else 0
df = df_combined.dropna(subset=["Target"])

# df_combined['rolling_sentiment_3'] = df_combined['weighted_sentiment'].rolling(window=3, min_periods=1).mean()
feature_cols = ['Pct_Change', 'sentiment', 'Volatility']
target_col = 'Target'

eval_model = MLogRForecastingModel()
eval_model.experiment(df_combined, target_col=target_col, predictor_cols=feature_cols)
eval_model.plot_results()

# train test split
EvaluationPipeline.evaluate_model_on_classification(
    model=eval_model,
    feature_matrix=df_combined,
    predictor_cols=feature_cols,
    target_col=target_col,
    target_horizon_in_days=1
)