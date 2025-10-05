import pandas as pd
import matplotlib.pyplot as plt

def visualize_headline_count_daily(df, start_date, end_date):
    """
    Visualizes the number of headlines per day within a specified date range.
    Includes days with zero headlines.
    """
    # Ensure datetime and strip timezone info
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None) # for plotting timezone is not relevant

    # Filter DataFrame for the specified date range
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    filtered_df = df.loc[mask]

    # Group by date and count headlines
    headline_counts = filtered_df.groupby(filtered_df['Date'].dt.date).size()

    # Create a full date range
    full_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Reindex to include missing dates (fill with 0)
    headline_counts = headline_counts.reindex(full_range.date, fill_value=0)

    # Plotting
    plt.figure(figsize=(15, 5))
    plt.bar(headline_counts.index, headline_counts.values, width=0.8, align='center')
    plt.title(f'Number of Headlines per Day from {start_date} to {end_date}')
    plt.xlabel('Date')
    plt.ylabel('Headline Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualize_headline_count_hourly(df, start_date, end_date):
    """
    Visualizes the number of headlines per hour within a specified date range.
    Includes hours with zero headlines.
    """
    # Ensure datetime and strip timezone info
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None) # for plotting timezone is not relevant

    # Filter DataFrame for the specified date range
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    filtered_df = df.loc[mask]

    # Group by hour and count headlines
    headline_counts = filtered_df.groupby(filtered_df['Date'].dt.floor('H')).size()

    # Create full hourly range
    full_range = pd.date_range(start=start_date, end=end_date, freq='H')

    # Reindex to include missing hours (fill with 0)
    headline_counts = headline_counts.reindex(full_range, fill_value=0)

    # Plotting
    plt.figure(figsize=(15, 5))
    plt.bar(headline_counts.index, headline_counts.values, width=0.03, align='center')
    plt.title(f'Number of Headlines per Hour from {start_date} to {end_date}')
    plt.xlabel('Date')
    plt.ylabel('Headline Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()