from Impact.ImpactScoreAnalyzerEnums import EvaluationMode
from Sentiment.Datasets.FNSPID.FnspidAdapter import FnspidAdapter
from Sentiment.Datasets.Headlines_2017_12_to_2020_7_USEastern.dataset_adapter import Adapter1
from Sentiment.Datasets.Miguel_Aenlle.AenlleAdapter import AenlleAdapter
from Sentiment.Datasets.NIFTY.nifty_adapter import NiftyAdapter
from Sentiment.Datasets.dataset_adapter_base import DatasetAdapterBase
from Sentiment.SentimentAnalyzer import analyze_sentiment, SentimentModel, GranularityLevel, DatasetSources, \
    ImpactModel, AggregationMethod
import pandas as pd
from Utils.pandas_helper import filter_dataset_by_dates

def load_dataset(dataset_adapter: DatasetAdapterBase, start_date: str, end_date: str) -> pd.DataFrame:
    if not dataset_adapter.try_load_preprocessed():
        dataset_adapter.load()
    df = dataset_adapter.to_standard_format()
    df = filter_dataset_by_dates(df, start_date, end_date)
    return df

def load(
    datasets: list[DatasetSources],
    sentiment_model:SentimentModel,
    granularity_level: GranularityLevel,
    impact_model: ImpactModel = ImpactModel.NONE,
    impact_model_evaluation_mode: EvaluationMode = EvaluationMode.CLASSIFICATION,
    aggregation_function: AggregationMethod = AggregationMethod.MEAN,
    start_date: str = None,
    end_date: str = None ):
    """
    Loads all headline datasets, filters by date, and runs sentiment analysis.
    """

    all_dataframes = []
    for source in datasets:
        # Instantiate adapter depending on dataset
        if source == DatasetSources.LUCASPHAM:
            adapter = Adapter1()
        elif source == DatasetSources.NIFTY:
            adapter = NiftyAdapter()
        elif source == DatasetSources.FNSPID:
            adapter = FnspidAdapter()
        elif source == DatasetSources.AENLLE:
            adapter = AenlleAdapter()
        else:
            print(f"⚠️ Unknown dataset: {source}, skipping.")
            continue

        df = load_dataset(adapter, start_date, end_date)
        all_dataframes.append(df)

    # Concatenate all selected datasets
    if not all_dataframes:
        raise Exception("⚠️ No datasets loaded. Check your list.")

    sentiment_scored = analyze_sentiment(
        datasets=all_dataframes,
        sentiment_model=sentiment_model,
        granuality_level=granularity_level,
        impact_model=impact_model,
        impact_model_evaluation_mode=impact_model_evaluation_mode,
        aggregation_function=aggregation_function
    )

    return sentiment_scored

