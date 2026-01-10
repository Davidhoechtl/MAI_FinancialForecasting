import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Forecasting.EvaluationModelBase import ForecastingModelBase

class MLogRForecastingModel(ForecastingModelBase):
    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, x_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains a Logistic Regression model to predict market direction (Up/Down).
        """
        # 1. Align indices and clean data
        common_index = x_train.index.intersection(y_train.index)
        x_aligned = x_train.loc[common_index]
        y_aligned = y_train.loc[common_index]

        # Drop rows with NaNs
        mask = x_aligned.notna().all(axis=1) & y_aligned.notna()
        x_clean = x_aligned[mask]
        y_clean = y_aligned[mask]

        # 2. Convert continuous target to Binary Class (1: Up, 0: Down/Flat)
        # Note: If y_train is already binary, this doesn't hurt, but usually it's pct_change
        y_binary = (y_clean > 0).astype(int)

        self.model = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000)
        self.model.fit(x_clean, y_binary)

    def predict(self, x_test: pd.DataFrame, x_gap: DataFrame = None) -> pd.Series:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Predict class labels directly (0 or 1)
        predictions = self.model.predict(x_test)
        return pd.Series(predictions, index=x_test.index)

    def experiment(self, feature_matrix: pd.DataFrame, predictor_cols: list[str], target_col: str):
        # Analyze coefficients and simple classification metrics
        print("Running MLogR Experiment (Classification Analysis):")

        df_clean = feature_matrix.dropna(subset=[target_col] + predictor_cols)
        X = df_clean[predictor_cols]
        # Convert target to binary
        y = (df_clean[target_col] > 0).astype(int)

        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X, y)

        # 1. Coefficient Analysis
        print(f"Intercept: {model.intercept_[0]:.6f}")
        print("Coefficients (Log-Odds):")
        for feature, coef in zip(predictor_cols, model.coef_[0]):
            print(f"{feature}: {coef:.6f}")

        # 2. In-sample Performance
        y_pred = model.predict(X)
        print("\nIn-Sample Classification Report:")
        print(classification_report(y, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))

    def plot_results(self):
        # Implementation depends on specific plotting needs for classification
        pass
