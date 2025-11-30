from Forecasting.EvaluationModelBase import ForecastingModelBase
import statsmodels.api as sm
import pandas as pd
from Utils import result_plots as rp

P_RANGE = range(1, 2)
Q_RANGE = range(1, 2)

class ARMAForecastingModel(ForecastingModelBase):
    def __init__(self):
        self.result_arima = []
        self.result_arima_with_sentiment = []
        super().__init__()

    def evaluate(self, feature_matrix: pd.DataFrame, predictor_cols: list[str], target_col: str):
        # Grid search for best p and q - normal ARMA
        best_aic = float('inf')
        best_bic = float('inf')
        best_p = 0
        best_q = 0
        best_model = None
        for p in P_RANGE:
            for q in Q_RANGE:
                print(f"Evaluating ARMA({p},{q})")
                model = self.eval_on_arma(feature_matrix, target_col, p, q)
                self.result_arima.append({'p': p, 'q': q, 'AIC': model.aic, 'BIC': model.bic})
                print(f"ARMA({p},{q}) AIC: {model.aic}, BIC: {model.bic}")
                print("-----------------------------------")
                if model.aic < best_aic:
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
        for p in P_RANGE:
            for q in Q_RANGE:
                print(f"Evaluating ARMA with sentiment ({p},{q})")
                model = self.eval_on_arma_with_sentiment(feature_matrix, target_col, predictor_cols, p, q)
                self.result_arima_with_sentiment.append({'p': p, 'q': q, 'AIC': model.aic, 'BIC': model.bic})
                print(f"ARMA with sentiment ({p},{q}) AIC: {model.aic}, BIC: {model.bic}")
                # print(model.summary())
                print("-----------------------------------")
                if model.aic < best_sentiment_aic:
                    best_sentiment_aic = model.aic
                    best_sentiment_bic = model.bic
                    best_sentiment_model = model
                    best_sentiment_p = p
                    best_sentiment_q = q

        print(best_model.summary())
        print(f"Best ARMA({best_p},{best_q}) AIC: {best_model.aic}, BIC: {best_model.bic}")
        print(
            f"Best ARMA with sentiment ({best_sentiment_p},{best_sentiment_q}) AIC: {best_sentiment_model.aic}, BIC: {best_sentiment_model.bic}")
        print(best_sentiment_model.summary())
        pass

    def plot_results(self):
        rp.plot_arma_aic_heatmap(self.result_arima, self.result_arima_with_sentiment)
        pass

    def eval_on_arma(self, df, target, p=3, q=1):
        # Drop NaNs from returns
        df_model = df.dropna(subset=[target])

        # Fit ARMA(p,q) on returns
        arma_model = sm.tsa.ARIMA(df_model[target], order=(p, 0, q)).fit()

        print("ARMA AIC:", arma_model.aic, "BIC:", arma_model.bic)
        return arma_model

    def eval_on_arma_with_sentiment(self, df, target_col, predictors, p=3, q=1):
        # Drop NaNs from returns
        df_model = df.dropna(subset=[target_col])

        armax_model = sm.tsa.ARIMA(df_model[target_col], order=(p, 0, q), exog=df_model[predictors]).fit()

        return armax_model