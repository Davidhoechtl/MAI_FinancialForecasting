
import pandas as pd

def get_daily_aggregated_sentiment(df):
    df_daily_aggregates = (
        df.groupby(df['Date'].dt.date)['sentiment']
        .mean()   # you can also use .sum() if you want total sentiment instead of average
        .reset_index()
    )
    # Rename columns
    df_daily_aggregates.columns = ['Date', 'sentiment']

    return df_daily_aggregates

def get_hourly_aggregated_sentiment(df):
    df_hourly_aggregates = (
        df.groupby(df['Date'].dt.floor('H'))['sentiment']
        .mean()   # you can also use .sum() if you want total sentiment instead of average
        .reset_index()
    )
    # Rename columns
    df_hourly_aggregates.columns = ['Date', 'sentiment']
    return df_hourly_aggregates