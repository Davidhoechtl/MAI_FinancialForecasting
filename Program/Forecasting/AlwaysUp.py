from Forecasting.EvaluationModelBase import ForecastingModelBase
import pandas as pd

# Bette classification baseline than random guessing: Markets have a tendency to go up, majority class is up
class AlwaysUpModel(ForecastingModelBase):
    def __init__(self):
        super().__init__()

    def train(self, x_train: pd.DataFrame, y_train: pd.Series):
        # noop
        pass

    def predict(self, x_test: pd.DataFrame, x_gap: pd.DataFrame) -> pd.Series:
        predictions = pd.Series(
            1, # (1: Up, 0: Down/Flat)
            index=x_test.index
        )
        return predictions

    def experiment(self, feature_matrix: pd.DataFrame, predictor_cols: list[str], target_col: str):
        # Fit on the entire provided dataset to analyze coefficients
        pass

    def plot_results(self):
        pass
