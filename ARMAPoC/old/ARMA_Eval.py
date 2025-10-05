import statsmodels.api as sm
import pandas as pd

df_combined = pd.read_csv('combined_2023.csv')
df_combined['Date'] = pd.to_datetime(df_combined['Date'])

# Target = log returns (better for stationarity)
df_combined['returns'] = df_combined['Close'].pct_change().dropna()

print(df_combined.head(20))

# Drop NaNs from returns
df_model = df_combined.dropna(subset=['returns'])

# Fit ARMA(p,q) on returns
arma_model = sm.tsa.ARIMA(df_model['returns'], order=(2, 0, 1)).fit()
print(arma_model.summary())

# Fit ARMAX with sentiment as exogenous regressor
armax_model = sm.tsa.ARIMA(df_model['returns'], order=(2, 0, 1), exog=df_model[['daily_sentiment']]).fit()
print(armax_model.summary())

print("ARMA AIC:", arma_model.aic, "BIC:", arma_model.bic)
print("ARMAX AIC:", armax_model.aic, "BIC:", armax_model.bic)