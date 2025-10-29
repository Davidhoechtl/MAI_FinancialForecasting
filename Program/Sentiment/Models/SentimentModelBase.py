from abc import ABC, abstractmethod
import pandas as pd

class SentimentModelBase(ABC):
    def __init__(self):
        self.sentiment = pd.Series(dtype="float64")  # define df series in __init__

    @abstractmethod
    def try_load_preprocessed(self, headline_column_hash: str) -> bool:
        """
        Sentiment was already analyzed. Try to load pd.Series from file
        :param headline_column_hash: hash that checks order and content of headline column
        :return: True if the pd.Series could be loaded successfully
        """
        pass

    @abstractmethod
    def preprocess(self, headlines: pd.Series) -> pd.Series:
        """
        Prepare the headline text for sentiment analysis.
        Example: lowercasing, removing punctuation, stopwords, etc.
        :param headlines: Series of raw headline strings
        :return: Series of cleaned/preprocessed headline strings
        """
        pass

    @abstractmethod
    def analyze(self, headlines: pd.Series):
        """
        Analyze the preprocessed headlines for sentiment polarity.
        Example output: Series of sentiment scores (-1, 0, 1)
        :param headlines: Series of preprocessed headline strings
        :return: Series of sentiment polarity values
        """
        pass