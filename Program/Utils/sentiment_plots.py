import pandas as pd
import matplotlib.pyplot as plt

def plot_sentiment_distribution(df: pd.DataFrame, sentiment_col: str = "sentiment"):
    """
    Plots the distribution of positive, neutral, and negative sentiments.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a sentiment column.
    sentiment_col : str, optional
        Name of the sentiment column (default = 'sentiment').
    """
    if sentiment_col not in df.columns:
        raise KeyError(f"Column '{sentiment_col}' not found in DataFrame.")

    # --- Map continuous scores to -1 / 0 / 1 ---
    raw_sent = df[sentiment_col]

    categorized = raw_sent.apply(
        lambda x: -1 if x < 0 else (1 if x > 0 else 0)
    )

    # --- Count each sentiment ---
    sentiment_counts = categorized.value_counts().sort_index()

    # Map to readable labels
    sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    sentiment_counts.index = sentiment_counts.index.map(sentiment_labels)

    # --- Bar Chart ---
    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind='bar', color=['red', 'gray', 'green'])
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Headlines")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def show_daily_sentiment(df, sentiment_col: str = "sentiment"):
    df = df.copy()

    if sentiment_col not in df.columns:
        raise KeyError(f"Column '{sentiment_col}' not found in DataFrame.")

    # Assume `combined` has columns: ['date', 'headline', 'sentiment']
    # Ensure date is proper datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop rows without valid date/sentiment
    df = df.dropna(subset=['date', sentiment_col])

    # --- 1️⃣ Group by day and compute average sentiment ---
    daily_sentiment = (
        df
        .groupby(df['date'].dt.date)[sentiment_col]
        .mean()
        .reset_index()
        .rename(columns={'date': 'Date', sentiment_col: 'Daily_Sentiment'})
    )

    # --- 2️⃣ Plot the daily average sentiment ---
    plt.figure(figsize=(10,5))
    plt.plot(daily_sentiment['Date'], daily_sentiment['Daily_Sentiment'], marker='o')
    plt.title('Daily Average Sentiment')
    plt.xlabel('Date')
    plt.ylabel('Sentiment (avg per day)')
    plt.grid(True)
    plt.tight_layout()