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