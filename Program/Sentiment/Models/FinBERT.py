import pandas as pd
from Sentiment.Models.SentimentModelBase import SentimentModelBase
from pathlib import Path
from typing import Optional
import re
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from Utils.pandas_helper import hash_headline_column

# Always relative to this script (analyze.py)
BASE_PATH = Path(__file__).resolve().parent

MAX_TOKEN_LENGTH = 512
MODEL_NAME = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class FinBERTSentimentModel(SentimentModelBase):
    def __init__(self):
        super().__init__()

    def try_load_preprocessed(self, headline_column_hash: str) -> bool:
        file = BASE_PATH / f"FinBERT_{str(headline_column_hash)}.csv"
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

    def analyze(self, headlines: pd.Series):
        """
        Analyze headlines using FinBERT and save results to file.
        :param headlines: preprocessed headlines
        :return: Series of sentiment scores (-1=negative, 0=neutral, 1=positive)
        """

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
        self.sentiment = pd.Series(labels).map(sentiment_mapping)

        # === Save to cache ===
        output_path = BASE_PATH / f"FinBERT_{hash_headline_column(headlines)}.csv"
        try:
            df = pd.DataFrame({"headlines": headlines, "sentiment": self.sentiment})
            df.to_csv(output_path, index=False)
            print(f"[INFO] Saved {len(df)} sentiment values to {output_path}")
        except Exception as e:
            print(f"[ERROR] Could not save sentiment file: {e}")

        return self.sentiment
