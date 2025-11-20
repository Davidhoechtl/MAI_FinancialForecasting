import os

import pandas as pd

import Sentiment.SentimentAnalyzer
from FeatureMatrixPipeline import get_feature_matrix
from Forecasting.ARMA import ARMAForecastingModel
from SP500_Prices.PriceAnalyzer import TechnicalIndicators

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"  # prevent OpenMP conflict early
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

start_date = "17/12/2017"
end_date = "18/07/2020"
impact_model = Sentiment.SentimentAnalyzer.ImpactModel.LLAMA_3_1_Instruct
df_combined = get_feature_matrix(
    start_date=start_date,
    end_date=end_date,
    impact_model=impact_model,
    tech_indicators=[TechnicalIndicators.VOLATILITY],
    sentiment_sources=[Sentiment.SentimentAnalyzer.DatasetSources.LUCASPHAM],
    sentiment_model=Sentiment.SentimentAnalyzer.SentimentModel.FINBERT,
    granularity_level=Sentiment.SentimentAnalyzer.GranularityLevel.DAILY
)

if impact_model == Sentiment.SentimentAnalyzer.ImpactModel.NONE:
    df_combined["sentiment_lag0"] = df_combined["sentiment"].shift(0)
    # df_combined["sentiment_lag0"] = df_combined["sentiment"].rolling(window=3, min_periods=1).mean()
    # df_combined["sentimentxVolatility"] = df_combined["sentiment"] * df_combined["Volatility"]
    # df_combined["sentiment_lag1"] = df_combined["sentiment"].shift(1).fillna(0)
    # df_combined["sentiment_lag2"] = df_combined["sentiment"].shift(2).fillna(0)
    # df_combined["sentiment_lag3"] = df_combined["sentiment"].shift(3).fillna(0)
else:
    df_combined["sentiment_lag0"] = df_combined["weighted_sentiment"].shift(0)
    # df_combined["sentiment_lag0"] = df_combined["weighted_sentiment"].rolling(window=3, min_periods=1).mean()
    # df_combined["sentimentxVolatility"] = df_combined["weighted_sentiment"] * df_combined["Volatility"]
    # df_combined["sentiment_lag1"] = df_combined["weighted_sentiment"].shift(1).fillna(0)
    # df_combined["sentiment_lag2"] = df_combined["weighted_sentiment"].shift(2).fillna(0)
    # df_combined["sentiment_lag3"] = df_combined["weighted_sentiment"].shift(3).fillna(0)

df_combined.drop(columns=["sentiment"], inplace=True)
print(df_combined.head(40))

eval_model = ARMAForecastingModel()
# eval_model = XGBoostForecastingModel()
eval_model.evaluate(df_combined)
eval_model.plot_results()

from Utils import result_plots as rp
rp.plot_price_change_sentiment_scatter(df_combined, 0)
# rp.plot_arima_pvalues(best_sentiment_model)
rp.sentiment_price_plot(df_combined)
# rp.prediction_vs_real_priceChange(df_combined, best_model, best_sentiment_model, start_date='03/05/2020', end_date='19/07/2020')