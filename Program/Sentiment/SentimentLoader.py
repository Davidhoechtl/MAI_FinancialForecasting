from Sentiment.Datasets.Headlines_2017_12_to_2020_7_USEastern.dataset_adapter import Adapter1
from Sentiment.Datasets.NIFTY.nifty_adapter import NiftyAdapter
from Sentiment.SentimentAnalyzer import analyze_sentiment, SentimentModel, GranularityLevel
import pandas as pd

def filter_dataset_by_dates(df: pd.DataFrame, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Filters the dataset between start and end date (inclusive).
    """
    if 'date' not in df.columns:
        raise KeyError("Expected column 'date' not found in DataFrame")

    # Ensure datetime dtype
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    if start is not None:
        start = pd.to_datetime(start)
        start = start.tz_localize("US/Eastern")
        df = df[df['date'] >= start]
    if end is not None:
        end = pd.to_datetime(end)
        end = end.tz_localize("US/Eastern")
        df = df[df['date'] <= end]

    return df

def load(
    sentiment_model:SentimentModel,
    granularity_level: GranularityLevel,
    start_date: str = None,
    end_date: str = None ):
    """
    Loads all headline datasets, filters by date, and runs sentiment analysis.
    """

    # First Dataset: https://www.kaggle.com/datasets/notlucasp/financial-news-headlines
    headlines_2017_12_to_2020_7_loader = Adapter1()
    if not headlines_2017_12_to_2020_7_loader.try_load_preprocessed():
        headlines_2017_12_to_2020_7_loader.load()
    headlines_2017_12_to_2020_7_data = headlines_2017_12_to_2020_7_loader.to_standard_format()
    headlines_2017_12_to_2020_7_data = filter_dataset_by_dates(headlines_2017_12_to_2020_7_data, start_date, end_date)

    nifty_adapter = NiftyAdapter()
    if not nifty_adapter.try_load_preprocessed():
        nifty_adapter.load()
    df_nifty = nifty_adapter.to_standard_format()
    df_nifty = filter_dataset_by_dates(df_nifty, start_date, end_date)

    df_list = [
        headlines_2017_12_to_2020_7_data,
        # df_nifty
    ]

    sentiment_scored = analyze_sentiment(
        datasets=df_list,
        sentiment_model=sentiment_model,
        granuality_level=granularity_level
    )

    return sentiment_scored

