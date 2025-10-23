import pandas as pd
from enum import Enum

class SentimentModel(Enum):
    VADER = 1

class GranularityLevel(Enum):
    DAILY = 1
    WEEKLY = 2

def analyze_sentiment(datasets: list[pd.DataFrame], sentiment_model: SentimentModel, granuality_level: GranularityLevel) -> pd.DataFrame:
    """
    Analyze sentiment of headlines in the combined DataFrame using the specified sentiment model.

    Parameters:
        datasets (list[pd.DataFrame]): List of DataFrames to analyze.
        sentiment_model (SentimentModel): The sentiment analysis model to use.

    Returns:
        pd.DataFrame: DataFrame with sentiment analysis results.
    """
    # Merge datasets
    combined = merge(datasets)

    # Preprocess data
    combined = preprocess(combined, sentiment_model)

    # Deduplicate data
    combined = deduplicate(combined)

    if sentiment_model == SentimentModel.VADER:
        combined['sentiment'] = analyze_with_vader(combined['headline'])

    mapped_to_timeseries = group_by_granularity(combined, granuality_level)

    return mapped_to_timeseries

def group_by_granularity(combined: pd.DataFrame, granularity_level: GranularityLevel) -> pd.DataFrame:
    """
    Create a timeseries with the given GranularityLevel. Then group the sentiment data by GranularityLevel
    and map it to the timeseries.

    Where NaN add 0 for the sentiment score

    Parameters:
        combined (pd.DataFrame): The DataFrame to group.
        granularity_level (GranularityLevel): The granularity level to group by (daily, weekly).

    Returns:
        pd.DataFrame: Grouped DataFrame.
    """

    if 'date' not in combined.columns or 'sentiment' not in combined.columns:
        raise ValueError("combined DataFrame must contain 'date' and 'sentiment' columns.")

    # Ensure datetime is timezone-aware and in US/Eastern
    if not pd.api.types.is_datetime64_any_dtype(combined['date']):
        raise TypeError("'date' column must be datetime type.")

    if combined['date'].dt.tz is None or str(combined['date'].dt.tz).lower() != 'us/eastern':
        raise ValueError(f"Expected timezone 'US/Eastern', got '{combined['date'].dt.tz}'.")

    # Determine date range based on existing data
    start_date = combined['date'].min().normalize()
    end_date = combined['date'].max().normalize()
    tz = combined['date'].dt.tz  # preserve timezone

    # Choose frequency based on granularity
    if granularity_level == GranularityLevel.DAILY:
        freq = 'D'
    elif granularity_level == GranularityLevel.WEEKLY:
        freq = 'W'
    else:
        raise ValueError(f"Unsupported granularity level: {granularity_level}")

    # Create a continuous date range with timezone awareness
    full_range = pd.date_range(start=start_date, end=end_date, freq=freq, tz=tz)

    # Group by chosen granularity and compute mean sentiment per period
    grouped = (
        combined
        .groupby(pd.Grouper(key='date', freq=freq))
        .agg({'sentiment': 'mean'})
        .reset_index()
    )

    # Reindex to ensure full coverage of time series
    grouped = (
        grouped
        .set_index('date')
        .reindex(full_range)
        .fillna({'sentiment': 0})
        .rename_axis('date')
        .reset_index()
    )

    # Count how many rows have sentiment == 0
    zero_count = (grouped['sentiment'] == 0).sum()
    print(f"Number of rows with sentiment == 0: {zero_count}")

    return grouped

def analyze_with_vader(headlines: pd.Series) -> pd.Series:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk

    nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()
    def classify_sentiment_polarity(text):
        if pd.isna(text):
            return None
        sentiment_score = sia.polarity_scores(text)['compound']
        return sentiment_score

    return headlines.apply(classify_sentiment_polarity)


def preprocess(combined: pd.DataFrame, sentiment_model: SentimentModel) -> pd.DataFrame:
    """
    Not yet implemented preprocessing function.
    """
    return combined

def merge(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple pandas DataFrames into a single DataFrame.

    Parameters:
        datasets (list[pd.DataFrame]): List of DataFrames to merge.

    Returns:
        pd.DataFrame: A single merged DataFrame.
    """
    if not datasets:
        return pd.DataFrame()

    # Concatenate all datasets into one
    combined = pd.concat(datasets, ignore_index=True)

    combined = combined.sort_values('date')

    return combined

def deduplicate(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate the combined DataFrame based on 'source' and 'headline' columns.
    Parameters:
        combined (pd.DataFrame): The combined DataFrame to deduplicate.
    Returns:
        pd.DataFrame: Deduplicated DataFrame.
    """

    # Remove duplicates (keep the first occurrence)
    deduped = combined.drop_duplicates(subset=['source', 'headline'], keep='first')

    # Reset index for cleanliness
    deduped.reset_index(drop=True, inplace=True)

    return deduped
