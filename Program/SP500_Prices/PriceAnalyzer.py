import pandas as pd
from enum import Enum
import numpy as np

class TechnicalIndicators(Enum):
    VOLUME = "Volume",
    VOLATILITY = "Volatility",
    MOVING_AVERAGE_30 = "Moving_Average_30",
    MOVING_AVERAGE_60 = "Moving_Average_60"

def analyze_price(df_price: pd.DataFrame, indicators: list[TechnicalIndicators]) -> pd.DataFrame:
    """
    Analyse the time series of prices and calculate technical indicators
    :param df_price:
        dataframe with the minimum format: date, price, volume
    :param indicators:
        indicators that should be analysed
    :return:
        dataframe with price and technical indicators
    """

    df_enriched = df_price.copy()
    for indicator in indicators:
        if indicator == TechnicalIndicators.VOLUME:
            df_enriched[indicator.value] = get_volume_feature(df_price)
        elif indicator == TechnicalIndicators.VOLATILITY:
            df_enriched[indicator.value] = get_volatility_feature(df_price)
        elif indicator == TechnicalIndicators.MOVING_AVERAGE_30:
            df_enriched[indicator.value] = get_moving_average_feature(df_price, window=30)
        elif indicator == TechnicalIndicators.MOVING_AVERAGE_60:
            df_enriched[indicator.value] = get_moving_average_feature(df_price, window=60)
        else:
            print(f"[WARN]: The technical indicator {indicator.value} is unknown and will not be included.")

    return df_enriched

# --------------------------------------------------
# 💡 Feature: Volatility (Rolling standard deviation)
# --------------------------------------------------
def get_volatility_feature(df_price: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Compute rolling volatility (standard deviation of log returns)
    :param df_price: DataFrame with 'Close' prices
    :param window: Rolling window size (default 30)
    :return: pd.Series with rolling volatility
    """
    if "Close" not in df_price.columns:
        raise ValueError("DataFrame must contain 'Close' column for volatility calculation.")

    # Compute log returns
    log_returns = np.log(df_price["Close"] / df_price["Close"].shift(1))

    # Rolling standard deviation (volatility)
    volatility = log_returns.rolling(window=window, min_periods=1).std()

    # Replace NaN (first two values) with 0
    volatility = volatility.fillna(0.0)

    return volatility

def get_moving_average_feature(df_price: pd.DataFrame, window: int = 30, column: str = "Close") -> pd.Series:
    """
    Compute rolling moving average with min_periods=1.
    :param df_price: DataFrame with the target column (default 'Close')
    :param window: Rolling window size (default 30)
    :param column: Column to compute the moving average on
    :return: pd.Series with rolling mean (min_periods=1)
    """
    if column not in df_price.columns:
        raise ValueError(f"DataFrame must contain '{column}' column for moving average calculation.")

    moving_avg = df_price[column].rolling(window=window, min_periods=1).mean()
    return moving_avg

# --------------------------------------------------
# 💡 Feature: Volume (normalized or simple)
# --------------------------------------------------
def get_volume_feature(df_price: pd.DataFrame, window: int = 30) -> pd.Series:
    """
    Compute normalized volume (z-score within rolling window)
    :param df_price: DataFrame with 'Volume'
    :param window: Rolling window size (default 30)
    :return: pd.Series with normalized volume
    """
    if "Volume" not in df_price.columns:
        raise ValueError("DataFrame must contain 'Volume' column for volume calculation.")

    # Rolling z-score normalization: (x - mean) / std
    rolling_mean = df_price["Volume"].rolling(window=window, min_periods=window).mean()
    rolling_std = df_price["Volume"].rolling(window=window, min_periods=window).std()
    zscore_volume = (df_price["Volume"] - rolling_mean) / rolling_std

    return zscore_volume