from abc import ABC, abstractmethod

import pandas as pd


class ForecastingModelBase(ABC):
    def __init__(self):
        self.results = []

    @abstractmethod
    def evaluate(self, feature_matrix: pd.DataFrame, predictor_cols: list[str], target_col: str):
        pass

    @abstractmethod
    def plot_results(self):
        pass