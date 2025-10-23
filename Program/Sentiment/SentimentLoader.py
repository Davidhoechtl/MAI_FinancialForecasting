from Sentiment.Datasets.Headlines_2017_12_to_2020_7_USEastern.dataset_adapter import Adapter1
from Sentiment.SentimentAnalyzer import analyze_sentiment, SentimentModel, GranularityLevel

def load(sentiment_model:SentimentModel, granularity_level: GranularityLevel):
    # First Dataset: https://www.kaggle.com/datasets/notlucasp/financial-news-headlines
    headlines_2017_12_to_2020_7_loader = Adapter1()
    if not headlines_2017_12_to_2020_7_loader.try_load_preprocessed():
        headlines_2017_12_to_2020_7_loader.load()
    headlines_2017_12_to_2020_7_data = headlines_2017_12_to_2020_7_loader.to_standard_format()

    df_list = [
        headlines_2017_12_to_2020_7_data
    ]

    sentiment_scored = analyze_sentiment(
        datasets=df_list,
        sentiment_model=sentiment_model,
        granuality_level=granularity_level
    )

    return sentiment_scored

