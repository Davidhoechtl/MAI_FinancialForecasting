from abc import ABC, abstractmethod
import pandas as pd

class SentimentModelBase(ABC):
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