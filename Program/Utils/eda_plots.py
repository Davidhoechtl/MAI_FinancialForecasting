import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr, pearsonr

def plot_sentiment_histograms(df, best_days, best_threshold=None, target_col="Pct_Change_next"):
    df_tmp = df.copy()

    col_name = f"rolling_sentiment_{best_days}"
    df_tmp = df_tmp.dropna(subset=[target_col, col_name])
    # Ground truth: did the next day go up or down?
    df_tmp["up"] = (df_tmp[target_col] > 0).astype(int)

    sent_up = df_tmp.loc[df_tmp["up"] == 1, col_name]
    sent_down = df_tmp.loc[df_tmp["up"] == 0, col_name]

    plt.figure(figsize=(8, 5))
    plt.hist(df_tmp[col_name], bins=50, alpha=0.5, density=True, label="sentiment")
    plt.xlabel("Sentiment score")
    plt.ylabel("Density")
    plt.title("Sentiment distribution")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(sent_up, bins=50, alpha=0.5, density=True, label="Up next day")
    plt.hist(sent_down, bins=50, alpha=0.5, density=True, label="Down next day")
    if best_threshold is not None:
        plt.axvline(best_threshold, linestyle="--")
    plt.xlabel(f"Sentiment score (window = {best_days} days)")
    plt.ylabel("Density")
    plt.title("Sentiment distribution for up vs. down next-day returns")
    plt.legend()
    plt.show()


def calc_pearson_corr(df: pd.DataFrame, days: int, target_col: str = "Pct_Change_next"):
    """
    Calculate the Pearson (linear) correlation between Pct_Change_next and rolling_sentiment_{days}.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Pct_Change_next' and the rolling sentiment column.
    days : int
        Rolling window size (must match column name, e.g. 'rolling_sentiment_5').

    Returns
    -------
    tuple[float, float]
        (Pearson correlation coefficient, p-value)
    """
    col_name = f"rolling_sentiment_{days}"
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found. Make sure to add it first.")

    # Drop NaN values for valid correlation computation
    df_valid = df.dropna(subset=[target_col, col_name])

    corr, p_value = pearsonr(df_valid[target_col], df_valid[col_name])
    return corr, p_value

def calc_spearman_corr(df: pd.DataFrame, days: int, target_col: str = "Pct_Change_next"):
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
    df_valid = df_valid.dropna(subset=[target_col, col_name])

    corr, p_value = spearmanr(df_valid[target_col], df_valid[col_name])
    return corr, p_value


def add_rolling_sentiment(df: pd.DataFrame, days: int, sentiment_col: str) -> pd.DataFrame:
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
    df_sorted[col_name] = df_sorted[sentiment_col].rolling(window=days, min_periods=1).mean()
    return df_sorted

def plot_rolling_sentiment_correlations(df: pd.DataFrame, sentiment_col: str, target_col: str = "Pct_Change_next") -> pd.DataFrame:
    # === Run correlations for windows 1–30 ===
    results = []
    for days in range(1, 61):
        df_temp = add_rolling_sentiment(df, days, sentiment_col)
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

    # # === Plot correlation with significance highlighting ===
    plt.figure(figsize=(10, 5))
    plt.plot(df_corr["days"], df_corr["spearman_corr"], label="Spearman")
    plt.plot(df_corr["days"], df_corr["pearson_corr"], label="Pearson")

    # Highlight statistically significant points
    sig_points_spearman = df_corr[df_corr["spearman_p"] < 0.05]
    sig_points_pearson = df_corr[df_corr["pearson_p"] < 0.05]
    plt.scatter(sig_points_spearman["days"], sig_points_spearman["spearman_corr"], color="red", label="p < 0.05", zorder=5)
    plt.scatter(sig_points_pearson["days"], sig_points_pearson["pearson_corr"], color="red", label="p < 0.05", zorder=5)

    plt.axhline(0, color="gray", linestyle="--", alpha=0.7)
    plt.title("Spearman/Pearson Correlation: Rolling Sentiment vs Pct_Ch (Significance Highlighted)")
    plt.xlabel("Rolling Window Size (days)")
    plt.ylabel(f"Spearman Correlation ({target_col} vs Rolling Sentiment)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    return df_corr