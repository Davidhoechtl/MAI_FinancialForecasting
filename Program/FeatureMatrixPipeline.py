import pandas as pd
import numpy as np
import SP500_Prices.Sources.InvestPy_UsEastern.scrape as investpy_sp500_scrape
import Sentiment.SentimentAnalyzer
import Sentiment.SentimentLoader as sentiment_loader
import SP500_Prices.PriceAnalyzer as technicals_loader
from Impact.ImpactScoreAnalyzerEnums import EvaluationMode
from SP500_Prices.PriceAnalyzer import TechnicalIndicators
from Sentiment.SentimentAnalyzer import DatasetSources
import os

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def join_sentiment_to_prices(df_prices, df_sentiment, impact_model):
    print(df_prices['Date'].dt.tz)  # shows UTC
    print(df_sentiment['date'].dt.tz)  # might be None or something else

    print(df_prices.head())
    print(df_sentiment.head())

    # Ensure Date columns are aligned
    df_prices['Date_only'] = df_prices['Date'].dt.date
    df_sentiment['Date_only'] = df_sentiment['date'].dt.date
    # Merge

    if impact_model == Sentiment.SentimentAnalyzer.ImpactModel.NONE:
        df_combined = pd.merge(df_prices, df_sentiment[['Date_only', 'sentiment']],
                               on='Date_only', how='left')

    else:
        df_combined = pd.merge(df_prices, df_sentiment[['Date_only', 'sentiment', 'weighted_sentiment']],
                           on='Date_only', how='left')
        df_combined['weighted_sentiment'] = df_combined['weighted_sentiment'].fillna(0)

    # Fill missing sentiment values with 0
    df_combined['sentiment'] = df_combined['sentiment'].fillna(0)
    return df_combined

def get_feature_matrix(
        start_date: str,
        end_date: str,
        impact_model: Sentiment.SentimentAnalyzer.ImpactModel,
        tech_indicators: list[TechnicalIndicators],
        sentiment_sources: list[DatasetSources],
        sentiment_model: Sentiment.SentimentAnalyzer.SentimentModel,
        granularity_level: Sentiment.SentimentAnalyzer.GranularityLevel,
        impact_model_evaluation_mode: EvaluationMode = EvaluationMode.CLASSIFICATION
):
    df_prices = investpy_sp500_scrape.get_sp500_data(start_date, end_date)
    df_prices = df_prices.reset_index()
    df_technicals = technicals_loader.analyze_price(
        df_price=df_prices,
        indicators=tech_indicators,
        start_date = start_date,
        end_date = end_date
    )

    df_sentiment = sentiment_loader.load(
        datasets= sentiment_sources,
        sentiment_model= sentiment_model,
        granularity_level = granularity_level,
        impact_model=impact_model,
        impact_model_evaluation_mode=impact_model_evaluation_mode,
        start_date=start_date,
        end_date=end_date
    )

    df_combined = join_sentiment_to_prices(df_technicals, df_sentiment, impact_model)
    df_combined["Pct_Change_next"] = df_combined["Pct_Change"].shift(-1)
    df_combined = df_combined.dropna(subset=['Pct_Change_next'])

    return df_combined