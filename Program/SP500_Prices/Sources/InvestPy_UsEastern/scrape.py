import pandas as pd
import investpy

def get_sp500_data(start, end):
    # df = investpy.indices.get_index_historical_data(
    #     index="S&P 500",
    #     country="United States",
    #     from_date=start,
    #     to_date=end
    # )
    df = investpy.etfs.get_etf_historical_data(
        etf='SPDR S&P 500',
        country='United States',
        from_date=start,
        to_date=end
    )
    print(df.head(10))
    # Ensure the index is datetime and localize to US/Eastern
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize('US/Eastern')  # directly localize since it's naive
    # Convert to UTC
    # df.index = df.index.tz_convert('UTC')

    # Calculate daily percent change
    df['Pct_Change'] = df['Close'].pct_change()
    return df[['Close', 'Pct_Change', 'Volume']]

def get_sp500_data_weekly(start, end):
    import investpy
    import pandas as pd

    df = investpy.indices.get_index_historical_data(
        index="S&P 500",
        country="United States",
        from_date=start,
        to_date=end
    )

    # Ensure DateTimeIndex with timezone
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize('US/Eastern')

    # Daily percent change
    df["Pct_Change"] = df["Close"].pct_change()

    # Weekly resample: take last available close per week (Friday or last trading day)
    df_weekly = df.resample("W").last()

    # Compute weekly percent change if desired
    df_weekly["Pct_Change"] = df_weekly["Close"].pct_change()

    return df_weekly[["Close", "Pct_Change"]]