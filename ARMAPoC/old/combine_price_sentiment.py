import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load both CSVs
df_prices = pd.read_csv('sp500_2023.csv')
df_prices['Date'] = pd.to_datetime(df_prices['Date'])

df_news = pd.read_csv('processed_headlines.csv')  # Replace with your actual news CSV filename
df_news['Date'] = pd.to_datetime(df_news['Date'])
# Aggregate sentiment scores by date
df_new_aggregates = (
    df_news.groupby(df_news['Date'].dt.date)['sentiment']
    .mean()   # you can also use .sum() if you want total sentiment instead of average
    .reset_index()
)

# Rename columns
df_new_aggregates.columns = ['Date', 'daily_sentiment']
# Convert Date back to datetime64[ns]
df_new_aggregates['Date'] = pd.to_datetime(df_new_aggregates['Date'])

# Merge, adding only the sentiment score
df_combined = pd.merge(df_prices, df_new_aggregates, on='Date', how='left')
# Fill missing sentiment values with 0
df_combined['daily_sentiment'] = df_combined['daily_sentiment'].fillna(0)

def show_headlines_per_day_plot(df):
    # Group by date and count datapoints
    month_df = df[(df['Date'].dt.year == 2023) & (df['Date'].dt.month <= 4)]
    counts_per_day = month_df.groupby(month_df['Date'].dt.date).size()

    # Plot the counts
    plt.figure(figsize=(12, 6))
    counts_per_day.plot(kind='bar')
    plt.xlabel('Date')
    plt.ylabel('Number of Datapoints')
    plt.title('Datapoints per Day')
    plt.tight_layout()
    plt.show()

def show_sentiment_over_time(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['daily_sentiment'], marker='o', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Daily Sentiment (avg)')
    plt.title('Daily Aggregated Sentiment Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Call it
show_sentiment_over_time(df_new_aggregates)
show_headlines_per_day_plot(df_combined)

print(df_combined.head())
print(df_combined[df_combined['daily_sentiment'].isna()].head(100))
df_combined.to_csv('combined_2023.csv')
# show_headlines_per_day_plot(df_prices)
show_headlines_per_day_plot(df_news)
# Save combined CSV
# df_combined.to_csv('combined.csv', index=False)