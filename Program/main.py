import pandas as pd
import statsmodels.api as sm
import SP500_Prices.Sources.InvestPy_UsEastern.scrape as investpy_sp500_scrape
import Sentiment.SentimentAnalyzer
import Sentiment.SentimentLoader as sentiment_loader
from Sentiment.SentimentAnalyzer import SentimentModel

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def join_sentiment_to_prices(df_prices, df_sentiment):
    print(df_prices['Date'].dt.tz)  # shows UTC
    print(df_sentiment['date'].dt.tz)  # might be None or something else

    print(df_prices.head())
    print(df_sentiment.head())

    # Ensure Date columns are aligned
    df_prices['Date_only'] = df_prices['Date'].dt.date
    df_sentiment['Date_only'] = df_sentiment['date'].dt.date
    # Merge
    df_combined = pd.merge(df_prices, df_sentiment[['Date_only', 'sentiment']],
                           on='Date_only', how='left')

    #df_combined = pd.merge(df_prices, df_sentiment, on='Date', how='left')
    # Fill missing sentiment values with 0
    df_combined['daily_sentiment'] = df_combined['sentiment'].fillna(0)
    return df_combined

def eval_on_arma(df, p=3, q=1):
    # Drop NaNs from returns
    df_model = df.dropna(subset=['Pct_Change'])

    # Fit ARMA(p,q) on returns
    arma_model = sm.tsa.ARIMA(df_model['Pct_Change'], order=(p, 0, q)).fit()

    print("ARMA AIC:", arma_model.aic, "BIC:", arma_model.bic)
    return arma_model

def eval_on_arma_with_sentiment(df, p=3, q=1):
    # Drop NaNs from returns
    df_model = df.dropna(subset=['Pct_Change'])

    # Shift sentiment backward by 1 day for lookahead
    # df_model['daily_sentiment'] = df_model['daily_sentiment'].shift(-1).fillna(0)

    # Fit ARMAX with sentiment as exogenous regressor
    armax_model = sm.tsa.ARIMA(df_model['Pct_Change'], order=(p, 0, q), exog=df_model[['daily_sentiment']]).fit()
    return armax_model

df_prices = investpy_sp500_scrape.get_sp500_data("18/12/2017", "19/07/2020")
#df_prices = investpy_sp500_scrape.get_sp500_data("01/01/2018", "31/12/2018")
#df_prices = investpy_sp500_scrape.get_sp500_data("01/01/2019", "31/12/2019")
#df_prices = investpy_sp500_scrape.get_sp500_data("01/01/2019", "19/07/2020")
df_prices = df_prices.reset_index()
# print(df_prices.head())

df_sentiment = sentiment_loader.load(
    sentiment_model= Sentiment.SentimentAnalyzer.SentimentModel.VADER,
    granularity_level = Sentiment.SentimentAnalyzer.GranularityLevel.DAILY
)
# print(df_sentiment.head())

df_combined = join_sentiment_to_prices(df_prices, df_sentiment)
print(df_combined.head(20))

# Grid search for best p and q - normal ARMA
best_aic = float('inf')
best_bic = float('inf')
best_p = 0
best_q = 0
best_model = None
for p in range(1, 4):
    for q in range (1, 4):
        print(f"Evaluating ARMA({p},{q})")
        model = eval_on_arma(df_combined, p, q)
        print(f"ARMA({p},{q}) AIC: {model.aic}, BIC: {model.bic}")
        print("-----------------------------------")
        if model.aic < best_aic :
            best_aic = model.aic
            best_bic = model.bic
            best_model = model
            best_p = p
            best_q = q
print(f"Best ARMA({best_p},{best_q}) AIC: {best_aic}, BIC: {best_bic}")
print(best_model.summary())

best_sentiment_aic = float('inf')
best_sentiment_bic = float('inf')
best_sentiment_p = 0
best_sentiment_q = 0
best_sentiment_model = None
for p in range(1, 4):
    for q in range (1, 4):
        print(f"Evaluating ARMA with sentiment ({p},{q})")
        model = eval_on_arma_with_sentiment(df_combined, p, q)
        print(f"ARMA with sentiment ({p},{q}) AIC: {model.aic}, BIC: {model.bic}")
        print("-----------------------------------")
        if model.aic < best_sentiment_aic:
            best_sentiment_aic = model.aic
            best_sentiment_bic = model.bic
            best_sentiment_model = model
            best_sentiment_p = p
            best_sentiment_q = q

print(best_model.summary())
print(f"Best ARMA({best_p},{best_q}) AIC: {best_model.aic}, BIC: {best_model.bic}")
print(f"Best ARMA with sentiment ({best_sentiment_p},{best_sentiment_q}) AIC: {best_sentiment_model.aic}, BIC: {best_sentiment_model.bic}")
print(best_sentiment_model.summary())

from Utils import result_plots as rp

# rp.plot_price_change_sentiment_scatter(df_combined)
# rp.plot_arima_pvalues(best_sentiment_model)
# rp.sentiment_price_plot(df_combined)
rp.prediction_vs_real_priceChange(df_combined, best_model, best_sentiment_model, start_date='03/05/2020', end_date='19/07/2020')