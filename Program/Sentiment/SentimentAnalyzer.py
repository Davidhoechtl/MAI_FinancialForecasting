import pandas as pd
from enum import Enum

from Impact.ImpactScoreAnalyzer import load_impact_score
from Impact.ImpactScoreAnalyzerEnums import EvaluationMode, ImpactModel
from Sentiment.Models.FinBERT import FinBERTSentimentModel
from Sentiment.Models.Vader import VaderSentimentModel
from Sentiment.Models.SentimentModelBase import SentimentModelBase
from Utils.sentiment_plots import show_daily_sentiment, plot_sentiment_distribution

class SentimentModel(Enum):
    VADER = 1,
    FINBERT = 2

class GranularityLevel(Enum):
    DAILY = 1,
    WEEKLY = 2

class AggregationMethod(Enum):
    MEAN = 1,
    SUM = 2

class DatasetSources(Enum):
    NIFTY = 1,
    LUCASPHAM = 2,
    FNSPID = 3,
    AENLLE = 4

def analyze_sentiment(
        datasets: list[pd.DataFrame],
        sentiment_model: SentimentModel,
        granuality_level: GranularityLevel,
        impact_model: ImpactModel = ImpactModel.NONE,
        impact_model_evaluation_mode: EvaluationMode = EvaluationMode.CLASSIFICATION,
        aggregation_function: AggregationMethod = AggregationMethod.MEAN) -> pd.DataFrame:
    """
    Analyze sentiment of headlines in the combined DataFrame using the specified sentiment model.

    Parameters:
        datasets (list[pd.DataFrame]): List of DataFrames to analyze.
        sentiment_model (SentimentModel): The sentiment analysis model to use.
        granuality_level (GranularityLevel): The granularity level for grouping results.
        impact_model (ImpactModel): The impact model to use (default is NONE).
        impact_model_evaluation_mode (Impact Model Evaluation Mode): The impact models evaluation function (prompt to use).
        aggregation_function (AggregationMethod): The method to aggregate sentiment scores when grouping by granularity (default is MEAN).

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
        model = VaderSentimentModel()
    elif sentiment_model == SentimentModel.FINBERT:
        model = FinBERTSentimentModel()
    else:
        raise ValueError(f"Unknown sentiment model: {sentiment_model}")

    combined["sentiment"] = get_sentiment(combined['headline'], model)
    # show_daily_sentiment(combined)
    # plot_sentiment_distribution(combined)

    if impact_model != ImpactModel.NONE:
        combined["impact_score"] = get_impact_scores(combined['headline'], impact_model, impact_model_evaluation_mode)

        # filtering low impact scores
        threshold = 0.1 # Example threshold
        mask = combined["impact_score"] < threshold
        print(f"Filtering out {mask.sum()} headlines with impact score below {threshold}")
        combined = combined[~mask]

        # created weighted sentiment feature
        combined["weighted_sentiment"] = get_weighted_sentiment(combined["sentiment"], combined["impact_score"])
        # plot_sentiment_distribution(combined, sentiment_col="weighted_sentiment")
        # show_daily_sentiment(combined, sentiment_col="weighted_sentiment")

    mapped_to_timeseries = group_by_granularity(combined, granuality_level, aggregation_function)

    return mapped_to_timeseries

def group_by_granularity(combined: pd.DataFrame, granularity_level: GranularityLevel, aggregation_function: AggregationMethod) -> pd.DataFrame:
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

    aggregation_function = 'mean' if aggregation_function == AggregationMethod.MEAN else 'sum'

    if "weighted_sentiment" not in combined.columns:
        # Group by chosen granularity and compute mean sentiment per period
        grouped = (
            combined
            .groupby(pd.Grouper(key='date', freq=freq))
            .agg({'sentiment': aggregation_function})
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
    else:
        # Group by chosen granularity and compute mean sentiment per period
        grouped = (
            combined
            .groupby(pd.Grouper(key='date', freq=freq))
            .agg({'sentiment': aggregation_function, 'weighted_sentiment': aggregation_function})
            .reset_index()
        )

        # Reindex to ensure full coverage of time series
        grouped = (
            grouped
            .set_index('date')
            .reindex(full_range)
            .fillna({'sentiment': 0, 'weighted_sentiment': 0})
            .rename_axis('date')
            .reset_index()
        )

    # Count how many rows have sentiment == 0
    zero_count = (grouped['sentiment'] == 0).sum()
    print(f"Number of rows with sentiment == 0: {zero_count}")

    return grouped

def get_sentiment( headlines: pd.Series, sentiment_model: SentimentModelBase) -> pd.Series:
    # could not load from file -> analyze and write to file
    headlines = sentiment_model.preprocess(headlines)

    sentiment = sentiment_model.analyze(headlines)

    # sanity check sentiment must have the same size as headlines
    if len(headlines) != len(sentiment):
        raise Exception("The retrieved sentiment series length mismatches with the headline series")

    # return the sentiment pd.Series, that was either loaded or analyzed
    return sentiment

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

    # Find duplicate headlines per source
    duplicates = combined[combined.duplicated(subset=['source', 'headline'], keep='first')]

    print(f"🔁 Found {len(duplicates)} duplicate rows")
    print(duplicates.head(10))

    # Remove duplicates (keep the first occurrence)
    deduped = combined.drop_duplicates(subset=['source', 'headline'], keep='first')

    # Reset index for cleanliness
    deduped.reset_index(drop=True, inplace=True)

    return deduped

def get_impact_scores(headlines: pd.Series, impact_model: ImpactModel, impact_model_evaluation_mode: EvaluationMode) -> pd.Series:
    return load_impact_score(headlines, impact_model, evaluation_mode=impact_model_evaluation_mode)

def get_weighted_sentiment(sentiment: pd.Series, impact_score: pd.Series) -> pd.Series:
    """
    Calculate weighted sentiment by multiplying sentiment scores with impact scores.

    Parameters:
        sentiment (pd.Series): Series of sentiment scores.
        impact_score (pd.Series): Series of impact scores.

    Returns:
        pd.Series: Series of weighted sentiment scores.
    """
    if len(sentiment) != len(impact_score):
        raise ValueError("Sentiment and impact score series must have the same length.")

    weighted_sentiment = sentiment * impact_score
    return weighted_sentiment