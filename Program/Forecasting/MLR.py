import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from Forecasting.EvaluationModelBase import ForecastingModelBase

class MLRForecastingModel(ForecastingModelBase):
    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, x_train: pd.DataFrame, y_train: pd.Series):
        # 1. Align indices (just in case)
        common_index = x_train.index.intersection(y_train.index)
        x_aligned = x_train.loc[common_index]
        y_aligned = y_train.loc[common_index]

        # 2. Drop rows with NaNs in either features or target
        # Sklearn cannot handle NaNs natively
        mask = x_aligned.notna().all(axis=1) & y_aligned.notna()
        x_clean = x_aligned[mask]
        y_clean = y_aligned[mask]

        self.model = LinearRegression()
        self.model.fit(x_clean, y_clean)

    def predict(self, x_test: pd.DataFrame, x_gap: DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        predictions = self.model.predict(x_test)
        return pd.Series(predictions, index=x_test.index)

    def experiment(self, feature_matrix: pd.DataFrame, predictor_cols: list[str], target_col: str):
        # Fit on the entire provided dataset to analyze coefficients
        print("Running MLR Experiment (Coefficient Analysis):")

        df_clean = feature_matrix.dropna(subset=[target_col] + predictor_cols)
        X = df_clean[predictor_cols]
        y = df_clean[target_col]

        model = LinearRegression()
        model.fit(X, y)

        print(f"Intercept: {model.intercept_:.6f}")
        for feature, coef in zip(predictor_cols, model.coef_):
            print(f"{feature}: {coef:.6f}")

    def plot_results(self):
        pass
