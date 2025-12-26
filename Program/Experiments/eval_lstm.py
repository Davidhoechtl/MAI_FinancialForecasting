import os
from distutils.command.config import config

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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

lstm_model = LSTMForecastingModel()
feature_cols = ['Pct_Change', 'Volume', 'Volatility', 'sentiment']
target_col = 'Pct_Change_next'
lstm_model.experiment(df_combined, feature_cols, target_col)