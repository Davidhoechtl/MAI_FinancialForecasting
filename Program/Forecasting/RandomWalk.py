import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from Forecasting.EvaluationModelBase import ForecastingModelBase
import numpy as np

class RandomWalkForecastingModel(ForecastingModelBase):
    def __init__(self):
        super().__init__()
        self.model = None
        self.mean = None
        self.std = None

    def train(self, x_train: pd.DataFrame, y_train: pd.Series):
        # 1. Align indices (just in case)
        mean, std = y_train.mean(), y_train.std()
        self.mean = mean
        self.std = std

    def predict(self, x_test: pd.DataFrame, x_gap: DataFrame) -> pd.Series:
        if self.mean is None:
            raise ValueError("Model has not been trained yet.")

        # sample from normal distribution with mean and std
        predictions = pd.Series(
            np.random.normal(loc=self.mean, scale=self.std, size=len(x_test)),
            index=x_test.index
        )

        return predictions

    def experiment(self, feature_matrix: pd.DataFrame, predictor_cols: list[str], target_col: str):
        # Fit on the entire provided dataset to analyze coefficients
        pass

    def plot_results(self):
        pass
