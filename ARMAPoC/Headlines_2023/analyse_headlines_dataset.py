import json
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

# https://huggingface.co/datasets/luckycat37/financial-news-dataset

# def get_price_per_hour_sp500(start, end):
    # """
    # Fetch hourly S&P 500 prices using yfinance.
    #
    # Parameters:
    #     start (str): Start date in 'YYYY-MM-DD' format
    #     end (str): End date in 'YYYY-MM-DD' format
    #
    # Returns:
    #     pd.DataFrame: DataFrame with ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    # """
    # ticker = yf.Ticker("^GSPC")  # S&P 500 Index
    # df = ticker.history(interval="1h", start=start, end=end)
    #
    # if df.empty:
    #     print("No price data found for given period.")
    #     return pd.DataFrame()
    #
    # df = df.reset_index()
    # df.rename(columns={'Datetime': 'Date'}, inplace=True)
    #
    # return df

def plot_headline_counts(df_headlines, start_date, end_date):
    """
    Plots the number of headlines per hour between start_date and end_date.

    Parameters:
        df_headlines (pd.DataFrame): DataFrame with columns ['Date', 'headlines']
        start_date (str or pd.Timestamp): Start of the window
        end_date (str or pd.Timestamp): End of the window
    """
    ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
    data, meta = ts.get_intraday(symbol='^GSPC', interval='60min', outputsize='full')
    return data

    # Plot
    plt.figure(figsize=(15, 5))
    plt.bar(df_counts['Date'], df_counts['headline_count'], width=0.03, align='center')
    plt.title(f'Number of Headlines per Hour from {start_date} to {end_date}')
    plt.xlabel('Date')
    plt.ylabel('Headline Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

with open(r'2023_processed (1).json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# If the file contains a list of datapoints
df_headlines = pd.DataFrame([
    {
        'headlines': item.get('title'),
        'Date': item.get('date_publish'),
        'mentioned_companies': item.get('mentioned_companies'),
        'sentiment': item.get('sentiment')
    }
    for item in data
])

# Make sure 'Date' is datetime
df_headlines['Date'] = pd.to_datetime(df_headlines['Date'])
# Sort by date
df_headlines_sorted = df_headlines.sort_values(by='Date')
print(df_headlines_sorted.head(24))

# Create an hourly time series covering the full range of your headlines
start_time = df_headlines['Date'].min().floor('h')
end_time = df_headlines['Date'].max().ceil('h')
df_hourly = pd.DataFrame({'Date': pd.date_range(start=start_time, end=end_time, freq='h')})

# Left join headlines to the hourly time series
df_merged = df_hourly.merge(df_headlines, on='Date', how='left')

# Optional: fill NaNs for sentiment or headlines if needed
df_merged['sentiment'] = df_merged['sentiment'].fillna(0)
df_merged['headlines'] = df_merged['headlines'].fillna('')

print(df_merged.head(24))  # first 24 hours
# print(get_price_per_hour_sp500('2024-03-01', '2024-04-01').head(24))  # first 24 hours
#
# plot_headline_counts(df_merged, '2023-03-01', '2023-04-01')

def sentiment_to_score(sentiment):
    if isinstance(sentiment, dict):
        pos = sentiment.get('positive', 0)
        neg = sentiment.get('negative', 0)
        neu = sentiment.get('neutral', 0)

        # Weighted sum normalized to [-1, 1]
        total = pos + neg + neu
        if total == 0:
            return 0
        return (pos - neg) / total
    return None

def get_sentiment_data(timezone='US/Eastern'):
    import pytz
    with open(r'./Headlines_2023/2023_processed (1).json', 'r', encoding='utf-8') as f:
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
    # Localize datetime to the specified timezone
    tz = pytz.timezone(timezone)
    # If dates are naive, assume UTC and then convert to target timezone
    if df['Date'].dt.tz is None:
        df['Date'] = df['Date'].dt.tz_localize('UTC').dt.tz_convert(tz)
    else:
        df['Date'] = df['Date'].dt.tz_convert(tz)

    df['sentiment'] = df['sentiment'].apply(sentiment_to_score)

    df_new_aggregates = (
        df.groupby(df['Date'].dt.date)['sentiment']
        .mean()  # you can also use .sum() if you want total sentiment instead of average
        .reset_index()
    )
    # Rename columns
    df_new_aggregates.columns = ['Date', 'daily_sentiment']
    # Convert Date back to datetime64[ns]
    df_new_aggregates['Date'] = pd.to_datetime(df_new_aggregates['Date'])
    return df_new_aggregates