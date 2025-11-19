import os
import pandas as pd

def load_impact_map(file_path):
    """Load saved impact map from CSV (if it exists)."""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return {
            row["hash"]: {"headline": row["headline"], "impact": row["impact"]}
            for _, row in df.iterrows()
        }
    return {}

def save_impact_map(impact_map, file_path):
    """Save impact map to CSV."""
    df = pd.DataFrame([
        {"hash": h, "headline": v["headline"], "impact": v["impact"]}
        for h, v in impact_map.items()
    ])
    df.to_csv(file_path, index=False)
    print(f"✅ Saved {len(df)} impact entries to {file_path}")