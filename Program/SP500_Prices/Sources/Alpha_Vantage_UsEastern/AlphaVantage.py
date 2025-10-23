from alpha_vantage.timeseries import TimeSeries
import os
from datetime import datetime
import pandas as pd
import pytz  # for timezone conversion

current_dateTime = datetime.now()


apikey = os.getenv('AVKey')

months = [
    '2023-01',
    # '2023-02',
    # '2023-03',
    # '2023-04',
    # '2023-05',
    # '2023-06',
    # '2023-07',
    # '2023-08',
    # '2023-09',
    # '2023-10',
    # '2023-11',
    # '2023-12'
]

# {
#     "1. symbol": "500.PAR",
#     "2. name": "Amundi Index Solutions - Amundi S&P 500 UCITS ETF C EUR",
#     "3. type": "ETF",
#     "4. region": "Paris",
#     "5. marketOpen": "09:00",
#     "6. marketClose": "17:30",
#     "7. timezone": "UTC+02",
#     "8. currency": "EUR",
#     "9. matchScore": "0.6667"
# },

ts = TimeSeries(key='JG2MKU2L0RUB8CM7', output_format='pandas')
month_data = {}
for month in months:
    print(f"Fetching data for month: {month}")
    # Fetch intraday data for SPY (S&P 500 ETF)
    data, meta_data = ts.get_intraday(
        symbol="500.PAR",          # use SPY instead of ^GSPC
        interval="60min",      # hourly data (choose "1min" if you want minutely)
        outputsize="full",      # full month if month param is supported, else last ~30 days,
        month = month
    )

    # Convert index to datetime
    data.index = pd.to_datetime(data.index)

    # Localize to US Eastern Time (Alpha Vantage default)
    data.index = data.index.tz_localize('US/Eastern')

    # Convert to UTC
    data.index = data.index.tz_convert('UTC')

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data.to_csv(f"SPY_intraday_{month}_UTC.csv")

    month_data[month] = data


#concat data for all months
df = pd.concat(month_data.values())
# Sort the DataFrame by datetime index
df.sort_index(inplace=True)

# Save the data to a CSV file
df.to_csv("SPY_hourly_2023_UTC.csv")

# Display the first few rows of the data
print(df.head())
