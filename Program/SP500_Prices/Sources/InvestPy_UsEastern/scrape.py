import pandas as pd
import investpy

def get_sp500_data(start, end):
    df = investpy.indices.get_index_historical_data(
        index="S&P 500",
        country="United States",
        from_date=start,
        to_date=end
    )

    # Ensure the index is datetime and localize to US/Eastern
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize('US/Eastern')  # directly localize since it's naive
    # Convert to UTC
    # df.index = df.index.tz_convert('UTC')

    # Calculate daily percent change
    df['Pct_Change'] = df['Close'].pct_change()

    return df[['Close', 'Pct_Change']]