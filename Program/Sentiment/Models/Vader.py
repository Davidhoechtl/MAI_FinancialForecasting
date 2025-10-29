import pandas as pd
from Sentiment.Models.SentimentModelBase import SentimentModelBase
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from pathlib import Path
from typing import Optional
import re

from Utils.pandas_helper import hash_headline_column

# Always relative to this script (analyze.py)
BASE_PATH = Path(__file__).resolve().parent

class VaderSentimentModel(SentimentModelBase):
    def __init__(self):
        super().__init__()

    def try_load_preprocessed(self, headline_column_hash: str) -> bool:
        file = BASE_PATH / f"Vader_{str(headline_column_hash)}.csv"
        if not file.exists():
            return False

        try:
            df = pd.read_csv(file)

            # Assume there's a 'sentiment' column (adapt if different)
            if "sentiment" not in df.columns:
                print(f"[WARN] 'sentiment' column missing in {file}")
                return False

            # Save the sentiment column as Series
            self.sentiment = df["sentiment"].astype(float)

            print(f"[INFO] Loaded {len(self.sentiment)} sentiment values from {file}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load preprocessed file: {e}")
            return False
        pass

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

    def analyze(self, headlines: pd.Series):
        """
        Analyze headlines with VADER and save results to a file.
        :param headlines: preprocessed headlines
        :return: Series of sentiment compound scores
        """
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()

        def classify_sentiment_polarity(text):
            if pd.isna(text):
                return None
            return sia.polarity_scores(text)["compound"]

        # Compute sentiment scores
        self.sentiment = headlines.apply(classify_sentiment_polarity)

        # Save to file
        output_path = BASE_PATH / f"Vader_{hash_headline_column(headlines)}.csv"
        try:
            df = pd.DataFrame({"sentiment": self.sentiment})
            df.to_csv(output_path, index=False)
            print(f"[INFO] Saved {len(df)} sentiment values to {output_path}")
        except Exception as e:
            print(f"[ERROR] Could not save sentiment file: {e}")

        return self.sentiment