import pandas as pd
from fontTools.subset import subset

import SP500_Prices.Sources.InvestPy_UsEastern.scrape as investpy_sp500_scrape
import Sentiment.SentimentAnalyzer
import Sentiment.SentimentLoader as sentiment_loader
from Forecasting.ARMA import ARMAForecastingModel
import os

import SP500_Prices.PriceAnalyzer as technicals_loader
from Forecasting.XGBoost import XGBoostForecastingModel
from SP500_Prices.PriceAnalyzer import TechnicalIndicators
from Sentiment.SentimentAnalyzer import DatasetSources

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"  # prevent OpenMP conflict early

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def join_sentiment_to_prices(df_prices, df_sentiment):
    print(df_prices['Date'].dt.tz)  # shows UTC
    print(df_sentiment['date'].dt.tz)  # might be None or something else

    print(df_prices.head())
    print(df_sentiment.head())

    # Ensure Date columns are aligned
    df_prices['Date_only'] = df_prices['Date'].dt.date
    df_sentiment['Date_only'] = df_sentiment['date'].dt.date
    # Merge
    df_combined = pd.merge(df_prices, df_sentiment[['Date_only', 'sentiment']],
                           on='Date_only', how='left')

    #df_combined = pd.merge(df_prices, df_sentiment, on='Date', how='left')
    # Fill missing sentiment values with 0
    df_combined['sentiment'] = df_combined['sentiment'].fillna(0)
    return df_combined

start_date = "17/12/2017"
end_date = "18/07/2020"
df_prices = investpy_sp500_scrape.get_sp500_data(start_date, end_date)
df_prices = df_prices.reset_index()
# print(df_prices.head())
df_technicals = technicals_loader.analyze_price(
    df_price=df_prices,
    indicators=[TechnicalIndicators.VOLATILITY]
)
df_sentiment = sentiment_loader.load(
    datasets= [DatasetSources.LUCASPHAM],
    sentiment_model= Sentiment.SentimentAnalyzer.SentimentModel.FINBERT,
    granularity_level = Sentiment.SentimentAnalyzer.GranularityLevel.DAILY,
    start_date=start_date,
    end_date=end_date
)
# print(df_sentiment.head())
df_combined = join_sentiment_to_prices(df_technicals, df_sentiment)
df_combined["sentiment_lag0"] = df_combined["sentiment"].shift(0)
df_combined["sentiment_lag1"] = df_combined["sentiment"].shift(1).fillna(0)
df_combined["sentiment_lag2"] = df_combined["sentiment"].shift(2).fillna(0)
df_combined["sentiment_lag3"] = df_combined["sentiment"].shift(3).fillna(0)
df_combined.drop(columns=["sentiment"], inplace=True)
print(df_combined.head(40))

# eval_model = ARMAForecastingModel()
eval_model = XGBoostForecastingModel()
eval_model.evaluate(df_combined)
eval_model.plot_results()

from Utils import result_plots as rp
rp.plot_price_change_sentiment_scatter(df_combined, 1)
# rp.plot_arima_pvalues(best_sentiment_model)
rp.sentiment_price_plot(df_combined)
# rp.prediction_vs_real_priceChange(df_combined, best_model, best_sentiment_model, start_date='03/05/2020', end_date='19/07/2020')