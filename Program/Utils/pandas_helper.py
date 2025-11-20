import pandas as pd
import hashlib

def filter_dataset_by_dates(df: pd.DataFrame, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Filters the dataset between start and end date (inclusive).
    """
    if 'date' not in df.columns:
        raise KeyError("Expected column 'date' not found in DataFrame")

    # Ensure datetime dtype
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    if start is not None:
        start = pd.to_datetime(start)
        start = start.tz_localize("US/Eastern")
        df = df[df['date'] >= start]
    if end is not None:
        end = pd.to_datetime(end)
        end = end.tz_localize("US/Eastern")
        df = df[df['date'] <= end]

    return df
