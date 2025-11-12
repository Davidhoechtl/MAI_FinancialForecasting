import pandas as pd
import hashlib

def hash_headline_column(headlines: pd.Series) -> str:
    """
    Compute a stable hash for the entire 'Headline' column of a DataFrame.
    The hash changes if any headline text or order changes.
    """
    # Ensure consistent string representation
    col_series = headlines.astype(str)

    # Use pandas built-in fast row hashing
    hash_values = pd.util.hash_pandas_object(col_series, index=True)

    # Combine all row hashes into one SHA256
    sha = hashlib.sha256()
    sha.update(hash_values.values.tobytes())
    return sha.hexdigest()

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
