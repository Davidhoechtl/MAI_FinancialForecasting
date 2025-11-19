import pandas as pd
import numpy as np
import SP500_Prices.Sources.InvestPy_UsEastern.scrape as investpy_sp500_scrape
import Sentiment.SentimentAnalyzer
import Sentiment.SentimentLoader as sentiment_loader
import SP500_Prices.PriceAnalyzer as technicals_loader
from SP500_Prices.PriceAnalyzer import TechnicalIndicators
from Sentiment.SentimentAnalyzer import DatasetSources
import os

from Utils.eda_plots import plot_rolling_sentiment_correlations, add_rolling_sentiment, plot_sentiment_histograms

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"  # prevent OpenMP conflict early

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def join_sentiment_to_prices(df_prices, df_sentiment, impact_model):
    print(df_prices['Date'].dt.tz)  # shows UTC
    print(df_sentiment['date'].dt.tz)  # might be None or something else

    print(df_prices.head())
    print(df_sentiment.head())

    # Ensure Date columns are aligned
    df_prices['Date_only'] = df_prices['Date'].dt.date
    df_sentiment['Date_only'] = df_sentiment['date'].dt.date
    # Merge

    if impact_model == Sentiment.SentimentAnalyzer.ImpactModel.NONE:
        df_combined = pd.merge(df_prices, df_sentiment[['Date_only', 'sentiment']],
                               on='Date_only', how='left')

    else:
        df_combined = pd.merge(df_prices, df_sentiment[['Date_only', 'sentiment', 'weighted_sentiment']],
                           on='Date_only', how='left')
        df_combined['weighted_sentiment'] = df_combined['weighted_sentiment'].fillna(0)

    # Fill missing sentiment values with 0
    df_combined['sentiment'] = df_combined['sentiment'].fillna(0)
    return df_combined


start_date = "17/12/2017"
end_date = "18/07/2020"
impact_model = Sentiment.SentimentAnalyzer.ImpactModel.LLAMA_3_1_Instruct
sentiment_col = "weighted_sentiment" if impact_model != Sentiment.SentimentAnalyzer.ImpactModel.NONE else "sentiment"

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
    impact_model=impact_model,
    start_date=start_date,
    end_date=end_date
)

df_combined = join_sentiment_to_prices(df_technicals, df_sentiment, impact_model)
df_combined["Pct_Change_next"] = df_combined["Pct_Change"].shift(-1)
print(df_combined.head(10))

# tries 1-30 days of rolling sentiment and plot the spearman and pearson correlations (checks if covariance is significant)
df_corr = plot_rolling_sentiment_correlations(df_combined, sentiment_col)

# Optional: print table of significant results
print(df_corr[df_corr["spearman_p"] < 0.05])

# Pick best correlation (by absolute value and p < 0.05)
significant = df_corr[df_corr["spearman_p"] < 0.05]
if not significant.empty:
    best_row = significant.loc[significant["spearman_corr"].abs().idxmax()]
else:
    best_row = df_corr.loc[df_corr["spearman_corr"].abs().idxmax()]

best_days = int(best_row["days"])
print(f"Best correlation at {best_days} days: Spearman={best_row['spearman_corr']:.3f}, p={best_row['spearman_p']:.4f}")

col_name = f"rolling_sentiment_{best_days}"
df_best = add_rolling_sentiment(df_combined, best_days, sentiment_col)

def find_best_split_ks(df: pd.DataFrame,
                       col_name: str,
                       target_col: str = "Pct_Change_next"):
    """
    Find the threshold on `col_name` that best separates the sentiment
    distributions for up vs down days using a KS / Youden J criterion.

    Returns
    -------
    best_t : float or None
        Threshold on sentiment. None if no meaningful split is possible.
    best_J : float or None
        Maximal (signed) Youden J statistic (TPR - FPR). None if no split.
    """

    # Extract arrays
    x = df[col_name].to_numpy()
    y = (df[target_col] > 0).astype(int).to_numpy()  # 1 = up, 0 = down/flat

    # Remove NaNs / infs
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) == 0:
        print("No valid rows (after filtering NaNs).")
        return None, None

    # Need both classes
    if len(np.unique(y)) < 2:
        print("Only one class present in y, can't define a separating threshold.")
        return None, None

    # Unique sentiment values
    uniq = np.unique(x)
    if len(uniq) < 2:
        print("Not enough unique sentiment values for a meaningful split.")
        print(f"Unique values in {col_name}: {uniq}")
        return None, None

    # Split into two distributions
    x_pos = x[y == 1]
    x_neg = x[y == 0]

    # Candidate thresholds = midpoints between sorted unique values
    thresholds = (uniq[:-1] + uniq[1:]) / 2.0

    best_t = None
    best_J = None

    for t in thresholds:
        # PRED: "positive" side = x > t
        tpr = (x_pos > t).mean()   # True positive rate
        fpr = (x_neg > t).mean()   # False positive rate
        J = tpr - fpr              # Youden's J

        if best_J is None or abs(J) > abs(best_J):
            best_J = J
            best_t = t

    if best_t is None:
        print("Could not find a separating threshold (distributions identical?).")
        return None, None

    print(f"Best threshold t = {best_t:.4f}, Youden J = {best_J:.4f}")
    return best_t, best_J

best_t, best_acc = find_best_split_ks(df_best, col_name)

# Finally for the best correlation create a histogram plot
plot_sentiment_histograms(df_best, best_days, best_t)

