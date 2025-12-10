import pandas as pd
import numpy as np
import Sentiment.SentimentAnalyzer
from FeatureMatrixPipeline import get_feature_matrix
from SP500_Prices.PriceAnalyzer import TechnicalIndicators
import os

from Utils.eda_plots import plot_rolling_sentiment_correlations, add_rolling_sentiment, plot_sentiment_histograms, \
    calc_spearman_corr, calc_pearson_corr

# Change working directory to project root
os.chdir("D:/Studium/Master/Masterarbeit/MAI_FinancialForecasting/Program")

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"  # prevent OpenMP conflict early

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

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


start_date = "17/12/2017"
end_date = "18/07/2020"
impact_model = Sentiment.SentimentAnalyzer.ImpactModel.LLAMA_3_1_Instruct
sentiment_col = "weighted_sentiment" if impact_model != Sentiment.SentimentAnalyzer.ImpactModel.NONE else "sentiment"

start_date = "17/12/2017"
end_date = "18/07/2020"
df_combined = get_feature_matrix(
    start_date=start_date,
    end_date=end_date,
    impact_model=impact_model,
    tech_indicators=[TechnicalIndicators.VOLATILITY],
    sentiment_sources=[Sentiment.SentimentAnalyzer.DatasetSources.LUCASPHAM, Sentiment.SentimentAnalyzer.DatasetSources.NIFTY],
    sentiment_model=Sentiment.SentimentAnalyzer.SentimentModel.FINBERT,
    granularity_level=Sentiment.SentimentAnalyzer.GranularityLevel.DAILY
)

best_day_correlations = {}

for i in range(1,100):
    print("Target: next ", i, "days price change percentage")
    target_col = f'Target_{i}'

    # 1. Create a COPY for this iteration so we don't destroy the original data
    df_epoch = df_combined.copy()

    # Create the target -> the next i-days price change percentage
    df_epoch[target_col] = df_epoch['Close'].pct_change(periods=i).shift(-i)
    df_epoch.dropna(subset=[target_col], inplace=True)

    # === Run correlations for windows 1–30 ===
    results = []
    for days in range(1, 61):
        df_temp = add_rolling_sentiment(df_epoch, days, sentiment_col)
        spearman_corr, spearman_p = calc_spearman_corr(df_temp, days, target_col)
        pearson_corr, pearson_p = calc_pearson_corr(df_temp, days, target_col)

        results.append({
            "days": days,
            "spearman_corr": spearman_corr,
            "spearman_p": spearman_p,
            "pearson_corr": pearson_corr,
            "pearson_p": pearson_p,
        })

    # Convert to DataFrame
    df_corr = pd.DataFrame(results)

    # # Optional: print table of significant results
    # print(df_corr[df_corr["spearman_p"] < 0.05])

    # Pick best correlation (by absolute value and p < 0.05)
    best_row = df_corr.loc[df_corr["spearman_corr"].abs().idxmax()]

    best_days = int(best_row["days"])
    print(f"Best correlation for next {i} days price change at {best_days} days rolling sentiment: Spearman={best_row['spearman_corr']:.3f}, p={best_row['spearman_p']:.7f}")

    col_name = f"rolling_sentiment_{best_days}"
    df_best = add_rolling_sentiment(df_epoch, best_days, sentiment_col)

    best_t, best_acc = find_best_split_ks(df_best, col_name, target_col=target_col)

    best_day_correlations[i] = {
        "Target_Days": i,
        "Best_Rolling_Sentiment_Days": best_days,
        "Spearman_Correlation": best_row['spearman_corr'],
        "Spearman_p_value": best_row['spearman_p'],
        "Best_Threshold": best_t,
        "Best_Youden_J": best_acc
    }

# Convert results to DataFrame for easier viewing
df_best_day_correlations = pd.DataFrame.from_dict(best_day_correlations, orient='index')

# plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df_best_day_correlations["Target_Days"], df_best_day_correlations["Spearman_Correlation"], marker='o')
plt.title("Best Spearman Correlation vs. Target Days")
plt.xlabel("Target Days")
plt.ylabel("Best Spearman Correlation")
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.show()

best_target = df_best_day_correlations.loc[df_best_day_correlations["Spearman_Correlation"].abs().idxmax()]
for i in [best_target["Target_Days"]]:
    i = int(i)
    print("Target: next ", i, "days price change percentage")
    target_col = f'Target_{i}'

    # 1. Create a COPY for this iteration so we don't destroy the original data
    df_epoch = df_combined.copy()

    # Create the target -> the next i-days price change percentage
    df_epoch[target_col] = df_epoch['Close'].pct_change(periods=i).shift(-i)
    df_epoch.dropna(subset=[target_col], inplace=True)

    # tries 1-30 days of rolling sentiment and plot the spearman and pearson correlations (checks if covariance is significant)
    df_corr = plot_rolling_sentiment_correlations(df_epoch, sentiment_col, target_col)

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
    df_best = add_rolling_sentiment(df_epoch, best_days, sentiment_col)

    best_t, best_acc = find_best_split_ks(df_best, col_name, target_col=target_col)

    # Finally for the best correlation create a histogram plot
    plot_sentiment_histograms(df_best, best_days, best_t)
