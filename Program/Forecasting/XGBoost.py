from Forecasting.EvaluationModelBase import ForecastingModelBase
from Utils import result_plots as rp
import pandas as pd
import xgboost as xgb
import numpy as np

from Utils.eval_helper import evaluate_classification


class XGBoostForecastingModel(ForecastingModelBase):
    def __init__(self):
        self.result= []
        self.result_with_sentiment = []
        super().__init__()

    def evaluate(self, feature_matrix):
        # --- Features ---
        # X1 = ["Pct_Change", "sentiment_lag0"]
        X2 = ["Pct_Change", "sentiment_lag0"]
        X3 = ["Pct_Change"]
        X4 = ["Pct_Change", "Volatility", "sentiment_lag0"]
        X5 = ["Pct_Change", "Volatility", "Volume", "sentiment_lag0"]
        X6 = ["Pct_Change", "Volatility", "Volume"]
        # X5 = ["Pct_Change", "Volatility", "sentiment_lag0", "sentiment_lag1"]
        # X6 = ["Volume", "sentiment_lag0"]
        # X7 = ["Volume", "sentiment_lag1"]
        # X8 = ["Volume", "Volatility", "sentiment_lag0"]
        # X9 = ["Volume", "Volatility", "sentiment_lag1"]
        # X10 = ["Volume", "Volatility", "sentiment_lag0", "sentiment_lag1"]
        # results = self.train_xgboost_classifier(feature_matrix, X1)
        results = self.train_xgboost_classifier(feature_matrix, X2)
        results = self.train_xgboost_classifier(feature_matrix, X3)
        results = self.train_xgboost_classifier(feature_matrix, X4)
        results = self.train_xgboost_classifier(feature_matrix, X5)
        results = self.train_xgboost_classifier(feature_matrix, X6 )
        # results = self.train_xgboost_classifier(feature_matrix, X7 )
        # results = self.train_xgboost_classifier(feature_matrix, X8 )
        # results = self.train_xgboost_classifier(feature_matrix, X9 )
        # results = self.train_xgboost_classifier(feature_matrix, X10 )

    def train_xgboost_classifier(self, feature_matrix: pd.DataFrame, feature_cols: list[str], num_round: int = 500,
                                 verbose: bool = True):
        """
        Train an XGBoost binary classifier on the given feature matrix.

        Args:
            feature_matrix (pd.DataFrame): Must contain 'Pct_Change', 'Volatility', 'sentiment', etc.
            feature_cols (list[str]): List of column names to use as model features.
            num_round (int): Number of boosting rounds.
            verbose (bool): If True, prints evaluation metrics.

        Returns:
            dict: Evaluation metrics (accuracy, precision, recall, f1, confusion_matrix)
        """

        # --- Copy to avoid mutating original ---
        df = feature_matrix.copy()

        # --- Drop missing core features ---
        df = df.dropna(subset=feature_cols + ["Pct_Change"])

        # --- Create target (next-day direction) ---
        df["Target"] = (df["Pct_Change"].shift(-1) > 0).astype(int)
        df = df.dropna(subset=["Target"])

        # --- Features & target ---
        X = df[feature_cols]
        y = df["Target"]

        # --- Train/test split (time-based) ---
        split_idx = int(len(df) * 0.8)
        dtrain = xgb.DMatrix(X.iloc[:split_idx], label=y.iloc[:split_idx])
        dtest = xgb.DMatrix(X.iloc[split_idx:], label=y.iloc[split_idx:])

        # --- XGBoost parameters ---
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42
        }

        # --- Train model ---
        bst = xgb.train(params, dtrain, num_boost_round=num_round)

        # --- Predict ---
        y_pred_prob = bst.predict(dtest)
        y_pred = (y_pred_prob > 0.5).astype(int)
        y_true = y.iloc[split_idx:]

        # --- Evaluate ---
        print("Features " + " ".join(feature_cols))
        print("---------------------------------------")
        results = evaluate_classification(y_true, y_pred, verbose=verbose)

        # --- Return results and model ---
        return {
            "model": bst,
            "metrics": results,
            "features_used": feature_cols,
            "train_size": split_idx,
            "test_size": len(df) - split_idx
        }

    def plot_results(self):
        # rp.plot_arma_aic_heatmap(self.result_arima, self.result_arima_with_sentiment)
        pass