import os
import pandas as pd

def load_sentiment_map(file_path):
    """Load saved impact map from CSV (if it exists)."""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return {
            row["hash"]: {"headline": row["headline"], "sentiment": row["sentiment"]}
            for _, row in df.iterrows()
        }
    return {}

def save_sentiment_map(impact_map, file_path):
    """Save impact map to CSV."""
    df = pd.DataFrame([
        {"hash": h, "headline": v["headline"], "sentiment": v["sentiment"]}
        for h, v in impact_map.items()
    ])
    df.to_csv(file_path, index=False)
    print(f"✅ Saved {len(df)} sentiment entries to {file_path}")