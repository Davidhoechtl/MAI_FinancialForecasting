import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from FeatureMatrixPipeline import get_feature_matrix
from Forecasting.ARMA import ARMAForecastingModel
from Impact.ImpactScoreAnalyzerEnums import ImpactModel
from SP500_Prices.PriceAnalyzer import TechnicalIndicators
from Sentiment.SentimentAnalyzer import DatasetSources, SentimentModel, GranularityLevel

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"  # prevent OpenMP conflict early

# Change working directory to project root
os.chdir("D:/Studium/Master/Masterarbeit/MAI_FinancialForecasting/Program")

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

start_date = "17/12/2019"
end_date = "18/07/2020"
impact_model = ImpactModel.GPT_OSS_20B
df_combined = get_feature_matrix(
    start_date=start_date,
    end_date=end_date,
    impact_model=impact_model,
    tech_indicators=[TechnicalIndicators.VOLATILITY],
    sentiment_sources=[DatasetSources.LUCASPHAM],
    sentiment_model=SentimentModel.FINBERT,
    granularity_level=GranularityLevel.DAILY
)

# df_combined['rolling_sentiment_3'] = df_combined['weighted_sentiment'].rolling(window=3, min_periods=1).mean()

eval_model = ARMAForecastingModel()
eval_model.experiment(df_combined, target_col="Pct_Change", predictor_cols=['sentiment'])
eval_model.plot_results()

# eval_model = ARMAForecastingModel()
# eval_model.evaluate(df_combined, target_col="Pct_Change_next", predictor_cols=['weighted_sentiment'])
# eval_model.plot_results()

eval_model = ARMAForecastingModel()
eval_model.experiment(df_combined, target_col="Pct_Change_next", predictor_cols=['rolling_sentiment_3'])
eval_model.plot_results()