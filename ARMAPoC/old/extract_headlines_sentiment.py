import pandas as pd
import json
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def sentiment_to_score(sentiment):
    if isinstance(sentiment, dict):
        return sentiment.get('positive', 0) - sentiment.get('negative', 0)
    return None

# Load the JSON file
with open(r'/2023_processed (1).json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# If the file contains a list of datapoints
df = pd.DataFrame([
    {
        'headlines': item.get('title'),
        'Date': item.get('date_publish'),
        'mentioned_companies': item.get('mentioned_companies'),
        'sentiment': item.get('sentiment')
    }
    for item in data
])

df['Date'] = pd.to_datetime(df['Date'])
df['sentiment'] = df['sentiment'].apply(sentiment_to_score)
df.to_csv(r'D:\Studium\Master\MAI\Masterarbeit\ARMAPoC\processed_headlines.csv', index=False)

def show_headlines_per_day_plot(df):
    # Group by date and count datapoints
    month_df = df[(df['Date'].dt.year == 2023) & (df['Date'].dt.month <= 3)]
    counts_per_day = month_df.groupby(month_df['Date'].dt.date).size()

    # Plot the counts
    plt.figure(figsize=(12, 6))
    counts_per_day.plot(kind='bar')
    plt.xlabel('Date')
    plt.ylabel('Number of Datapoints')
    plt.title('Datapoints per Day')
    plt.tight_layout()
    plt.show()

def show_sentiment_score_distribution(df):
    # Count positive and negative sentiment
    positive_count = (df['sentiment'] > 0).sum()
    negative_count = (df['sentiment'] < 0).sum()

    # Prepare data for plotting
    sentiment_counts = {'Positive (>0)': positive_count, 'Negative (<0)': negative_count}

    plt.figure(figsize=(6, 4))
    plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['green', 'red'])
    plt.ylabel('Number of Headlines')
    plt.title('Count of Positive and Negative Sentiments')
    plt.tight_layout()
    plt.show()

show_headlines_per_day_plot(df)
# show_sentiment_score_distribution(df)

print(df.head(20))