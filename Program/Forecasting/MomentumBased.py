import pandas as pd
from pandas import DataFrame
from Forecasting.EvaluationModelBase import ForecastingModelBase

class MomentumBasedForecastingModel(ForecastingModelBase):
    def __init__(self):
        super().__init__()

    def train(self, x_train: pd.DataFrame, y_train: pd.Series):
        pass

    def predict(self, x_test: pd.DataFrame, x_gap: DataFrame) -> pd.Series:
        if 'Pct_Change' not in x_test.columns:
            raise ValueError("Input data must contain 'Pct_Change' column for momentum-based prediction.")

        #y = yt-1
        return x_test['Pct_Change'] # just use the last pct_change as prediction

    def experiment(self, feature_matrix: pd.DataFrame, predictor_cols: list[str], target_col: str):
        # Fit on the entire provided dataset to analyze coefficients
        pass

    def plot_results(self):
        pass
