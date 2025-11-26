import hashlib

import pandas as pd

from Sentiment.Models.SentimentMapUtils import load_sentiment_map, save_sentiment_map
from Sentiment.Models.SentimentModelBase import SentimentModelBase
from pathlib import Path
from typing import Optional
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch


# Always relative to this script (analyze.py)
BASE_PATH = Path(__file__).resolve().parent
CACHE_FILE_PATH = BASE_PATH / "FinBERT_sentiment_map.csv"

MAX_TOKEN_LENGTH = 512
MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class FinBERTSentimentModel(SentimentModelBase):
    def preprocess(self, headlines: pd.Series) -> pd.Series:
        """
        Light normalization for FinBERT input:
        - Remove HTML, URLs, and non-ASCII characters
        - Normalize whitespace
        - Keep financial abbreviations and punctuation
        """
        def clean_text(text: Optional[str]) -> str:
            if pd.isna(text):
                return ""
            text = str(text)
            text = re.sub(r"<.*?>", "", text)               # remove HTML
            text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs
            text = text.encode("ascii", errors="ignore").decode()
            text = re.sub(r"\s+", " ", text).strip()        # normalize spaces
            return text

        headlines = headlines.apply(clean_text)

        return headlines

    def compute(self, headlines: pd.Series):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] FinBERT running on {device}")

        # check all headlines that exceed the max input length of the Transformer
        token_lengths = headlines.apply(lambda x: len(tokenizer.encode(x, add_special_tokens=True)))
        valid_mask = token_lengths <= MAX_TOKEN_LENGTH
        removed_count = len(headlines) - valid_mask.sum()
        if removed_count > 0:
            print(f"[WARN] {removed_count} headlines exceeded 512 tokens and will be truncated")

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
        model.eval()

        sentiments = []
        batch_size = 32

        print(f"[INFO] Evaluating {len(headlines)} headlines with FinBERT (batch size={batch_size})...")

        # with torch.no_grad():
        #     total = len(headlines)
        #     for i in range(0, len(headlines), batch_size):
        #         batch_texts = headlines[i:i + batch_size].tolist()
        #
        #         # --- Progress info ---
        #         progress = min(i + batch_size, total)
        #         print(f"\r[INFO] Processing FinBERT batch {progress}/{total} ({progress / total:.1%})", end="",
        #               flush=True)
        #
        #         # Tokenize batch
        #         inputs = tokenizer(
        #             batch_texts,
        #             padding=True,
        #             truncation=True,
        #             max_length=512,
        #             return_tensors="pt"
        #         ).to(device)
        #
        #         # Forward pass
        #         logits = model(**inputs).logits
        #         probs = F.softmax(logits, dim=-1).cpu().numpy()  # shape: (batch_size, 3)
        #
        #         # FinBERT order: [positive, neutral, negative]
        #         for p_pos, p_neu, p_neg in probs:
        #             # score = (p_pos - p_neg) * (1 - p_neu)  # continuous ∈ [-1, 1]
        #             score = (p_pos - p_neg)  # continuous ∈ [-1, 1]
        #             sentiments.append(score)
        #
        # # Store results
        # self.sentiment = pd.Series(sentiments, index=headlines.index)

        # # === Sentiment berechnen ===
        print("Analysiere Sentiment ...")
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
        results = nlp(headlines.tolist(), batch_size=batch_size)

        # === Ergebnisse zuordnen ===
        sentiment_mapping = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
        labels = [r['label'] for r in results]
        sentiment = pd.Series(labels).map(sentiment_mapping)

        return sentiment

    def analyze(self, headlines: pd.Series):
        """
        Analyze headlines using FinBERT and save results to file.
        :param headlines: preprocessed headlines
        :return: Series of sentiment scores (-1=negative, 0=neutral, 1=positive)
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

            # SAFETY CHECK: Ensure lengths match
            if len(computed) != len(missing_texts):
                raise ValueError(f"Mismatch! Input: {len(missing_texts)}, Output: {len(computed)}")

            # 4) update map with computed values
            for text, score in zip(missing_texts, computed):
                h_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                sentiment_map[h_hash] = {"headline": text, "sentiment": float(score)}

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