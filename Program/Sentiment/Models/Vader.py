import hashlib

import pandas as pd

from Sentiment.Models.SentimentMapUtils import save_sentiment_map, load_sentiment_map
from Sentiment.Models.SentimentModelBase import SentimentModelBase
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from pathlib import Path
from typing import Optional
import re

# Always relative to this script (analyze.py)
BASE_PATH = Path(__file__).resolve().parent
CACHE_FILE_PATH = BASE_PATH / "Vader_sentiment_map.csv"

class VaderSentimentModel(SentimentModelBase):
    def preprocess(self, headlines: pd.Series) -> pd.Series:
        def clean_text(text: Optional[str]) -> str:
            if pd.isna(text):
                return ""
            text = str(text)

            # Remove HTML tags
            text = re.sub(r"<.*?>", "", text)

            # Remove URLs
            text = re.sub(r"http\S+|www\S+", "", text)

            # Remove emojis / non-ASCII
            text = text.encode("ascii", errors="ignore").decode()

            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()

            # Lowercase (VADER handles uppercase emphasis, but headlines are usually normalized)
            text = text.lower()

            return text

        return headlines.apply(clean_text)

    def compute(self, headlines: pd.Series):
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()

        def classify_sentiment_polarity(text):
            if pd.isna(text):
                return None
            return sia.polarity_scores(text)["compound"]

        # Compute sentiment scores
        sentiment = headlines.apply(classify_sentiment_polarity)

        return sentiment

    def analyze(self, headlines: pd.Series):
        """
        Analyze headlines with VADER and save results to a file.
        :param headlines: preprocessed headlines
        :return: Series of sentiment compound scores
        """
        # 1) load map
        sentiment_map = load_sentiment_map(CACHE_FILE_PATH) or {}

        # 2) collect missing headlines
        missing_idx = []
        missing_texts = []
        for idx, h in headlines.items():
            text = "" if pd.isna(h) else str(h)
            h_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h_hash not in sentiment_map:
                missing_idx.append(idx)
                missing_texts.append(text)

        # 3) compute missing via model
        if missing_texts:
            missing_series = pd.Series(missing_texts, index=missing_idx)
            computed = self.compute(missing_series)  # should return Series aligned to missing_series.index

            # 4) update map with computed values
            for idx, val in computed.items():
                text = missing_series.at[idx]
                h_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                sentiment_map[h_hash] = {"headline": missing_series.at[idx], "sentiment": float(val)}

            # save updated map
            save_sentiment_map(sentiment_map, CACHE_FILE_PATH)

        # 5) build full result Series from map
        result = pd.Series([float("nan")] * len(headlines), index=headlines.index, dtype=float)
        for idx, h in headlines.items():
            text = "" if pd.isna(h) else str(h)
            h_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            entry = sentiment_map.get(h_hash)
            if entry is not None:
                result.at[idx] = float(entry.get("sentiment"))
            else:
                raise Exception("Sentiment entry missing after computation")

        return result