from Forecasting.EvaluationModelBase import ForecastingModelBase
from Utils import result_plots as rp
import pandas as pd
import xgboost as xgb
import numpy as np
from Utils.eval_helper import evaluate_classification

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

class XGBoostForecastingModel(ForecastingModelBase):
    def __init__(self):
        self.result= []
        self.result_with_sentiment = []

        self.model = None
        super().__init__()

    def train(self, x_train: pd.DataFrame, y_train: pd.Series):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        bst = xgb.train(params, dtrain, num_boost_round=500)
        self.model = bst
        pass

    def predict(self, x_test: pd.DataFrame, x_gap: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        #convert to d matrix
        dtest = xgb.DMatrix(x_test)
        y_pred_prob = self.model.predict(dtest)
        yhat = (y_pred_prob > 0.5).astype(int)

        return yhat

    def experiment(self, feature_matrix, target_col: str, predictor_cols: list[str]):
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
        results = self.train_xgboost_classifier(feature_matrix, target_col, predictor_cols)

    def train_xgboost_classifier(self, feature_matrix: pd.DataFrame, target_col:str, feature_cols: list[str], num_round: int = 500,
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

        # --- Features & target ---
        X = df[feature_cols]
        y = df[target_col]

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