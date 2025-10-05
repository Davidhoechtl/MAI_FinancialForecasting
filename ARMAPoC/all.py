import investpy
import pandas as pd
import json
import statsmodels.api as sm
import pytz  # For timezone handling
from Headlines_2017_12_to_2020_7_USEastern import utils as prep

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

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
    df.index = df.index.tz_convert('UTC')

    # Calculate daily percent change
    df['Pct_Change'] = df['Close'].pct_change()

    return df[['Close', 'Pct_Change']]

def join_sentiment_to_prices(df_prices, df_sentiment):
    print(df_prices['Date'].dt.tz)  # shows UTC
    print(df_sentiment['Date'].dt.tz)  # might be None or something else

    print(df_prices.head())
    print(df_sentiment.head())

    # Ensure Date columns are aligned
    df_prices['Date_only'] = df_prices['Date'].dt.date
    df_sentiment['Date_only'] = df_sentiment['Date'].dt.date
    # Merge
    df_combined = pd.merge(df_prices, df_sentiment[['Date_only', 'sentiment']],
                           on='Date_only', how='left')

    #df_combined = pd.merge(df_prices, df_sentiment, on='Date', how='left')
    # Fill missing sentiment values with 0
    df_combined['daily_sentiment'] = df_combined['sentiment'].fillna(0)
    return df_combined

def eval_on_arima(df):
    # Drop NaNs from returns
    df_model = df.dropna(subset=['Pct_Change'])

    # Shift sentiment backward by 1 day for lookahead
    #df_model['daily_sentiment'] = df_model['daily_sentiment'].shift(-3).fillna(0)

    # Fit ARMA(p,q) on returns
    arma_model = sm.tsa.ARIMA(df_model['Pct_Change'], order=(3, 0, 1)).fit()
    print(arma_model.summary())

    # Fit ARMAX with sentiment as exogenous regressor
    armax_model = sm.tsa.ARIMA(df_model['Pct_Change'], order=(3, 0, 1), exog=df_model[['daily_sentiment']]).fit()
    print(armax_model.summary())

    print("ARMA AIC:", arma_model.aic, "BIC:", arma_model.bic)
    print("ARMAX AIC:", armax_model.aic, "BIC:", armax_model.bic)

df_prices = get_sp500_data("01/01/2018", "19/07/2020")
# df_prices = get_sp500_data("01/01/2018", "31/12/2018")
# df_prices = get_sp500_data("01/01/2019", "31/12/2019")
df_prices = df_prices.reset_index()
# print(df_prices.head())
df_headlines = pd.read_csv("./Headlines_2017_12_to_2020_7_USEastern/processed_headlines.csv")
df_headlines['Date'] = pd.to_datetime(df_headlines['Date'], utc=True)
df_sentiment = prep.get_daily_aggregated_sentiment(df_headlines)
df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date'], utc=True)

# print(df_sentiment.head())

df_combined = join_sentiment_to_prices(df_prices, df_sentiment)
print(df_combined.head(20))

eval_on_arima(df_combined)

# Load the JSON file

