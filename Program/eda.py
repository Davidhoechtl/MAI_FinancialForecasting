import pandas as pd
import SP500_Prices.Sources.InvestPy_UsEastern.scrape as investpy_sp500_scrape
import Sentiment.SentimentAnalyzer
import Sentiment.SentimentLoader as sentiment_loader
import SP500_Prices.PriceAnalyzer as technicals_loader
from SP500_Prices.PriceAnalyzer import TechnicalIndicators
from Sentiment.SentimentAnalyzer import DatasetSources
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"  # prevent OpenMP conflict early

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

from scipy.stats import spearmanr

def calc_spearman_corr(df: pd.DataFrame, days: int):
    """
    Calculate the Spearman correlation between Pct_Change and rolling_sentiment_{days}.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Pct_Change' and the rolling sentiment column.
    days : int
        Window size used for rolling sentiment (must match column name).

    Returns
    -------
    float
        Spearman correlation coefficient.
    """
    col_name = f"rolling_sentiment_{days}"
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found. Make sure to add it first.")

    # Create shifted copy so each sentiment_t predicts return_{t+1}
    df_valid = df.copy()

    # Drop NaNs from both columns before computing correlation
    df_valid = df_valid.dropna(subset=["Pct_Change_next", col_name])

    corr, p_value = spearmanr(df_valid["Pct_Change_next"], df_valid[col_name])
    return corr, p_value


def add_rolling_sentiment(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Adds a rolling sentiment feature to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least the 'Date_only' and 'sentiment' columns.
    days : int
        Rolling window size (in days).

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column 'rolling_sentiment_{days}'.
    """
    df_sorted = df.sort_values(by="Date_only").copy()
    col_name = f"rolling_sentiment_{days}"
    df_sorted[col_name] = df_sorted["sentiment"].rolling(window=days, min_periods=1).mean()
    return df_sorted

def join_sentiment_to_prices(df_prices, df_sentiment):
    print(df_prices['Date'].dt.tz)  # shows UTC
    print(df_sentiment['date'].dt.tz)  # might be None or something else

    print(df_prices.head())
    print(df_sentiment.head())

    # Ensure Date columns are aligned
    df_prices['Date_only'] = df_prices['Date'].dt.date
    df_sentiment['Date_only'] = df_sentiment['date'].dt.date
    # Merge
    df_combined = pd.merge(df_prices, df_sentiment[['Date_only', 'sentiment']],
                           on='Date_only', how='left')

    #df_combined = pd.merge(df_prices, df_sentiment, on='Date', how='left')
    # Fill missing sentiment values with 0
    df_combined['sentiment'] = df_combined['sentiment'].fillna(0)
    return df_combined

start_date = "17/12/2017"
end_date = "18/07/2020"
df_prices = investpy_sp500_scrape.get_sp500_data(start_date, end_date)
df_prices = df_prices.reset_index()
# print(df_prices.head())
df_technicals = technicals_loader.analyze_price(
    df_price=df_prices,
    indicators=[TechnicalIndicators.VOLATILITY]
)
df_sentiment = sentiment_loader.load(
    datasets= [DatasetSources.LUCASPHAM],
    sentiment_model= Sentiment.SentimentAnalyzer.SentimentModel.FINBERT,
    granularity_level = Sentiment.SentimentAnalyzer.GranularityLevel.DAILY,
    start_date=start_date,
    end_date=end_date
)

df_combined = join_sentiment_to_prices(df_technicals, df_sentiment)
df_combined["Pct_Change_next"] = df_combined["Pct_Change"].shift(-1)
print(df_combined.head(10))

# === Run correlations for windows 1–30 ===
results = []
for days in range(1, 31):
    df_temp = add_rolling_sentiment(df_combined, days)
    corr, p_value = calc_spearman_corr(df_temp, days)
    results.append({"days": days, "spearman_corr": corr, "p_value": p_value})

# Convert to DataFrame
df_corr = pd.DataFrame(results)

# === Plot correlation with significance highlighting ===
plt.figure(figsize=(10, 5))
plt.plot(df_corr["days"], df_corr["spearman_corr"], marker="o", label="Spearman correlation")

# Highlight statistically significant points
sig_points = df_corr[df_corr["p_value"] < 0.05]
plt.scatter(sig_points["days"], sig_points["spearman_corr"], color="red", label="p < 0.05", zorder=5)

plt.axhline(0, color="gray", linestyle="--", alpha=0.7)
plt.title("Spearman Correlation vs. Rolling Sentiment Window (Significance Highlighted)")
plt.xlabel("Rolling Window Size (days)")
plt.ylabel("Spearman Correlation (Pct_Change vs Rolling Sentiment)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Optional: print table of significant results
print(df_corr[df_corr["p_value"] < 0.05])