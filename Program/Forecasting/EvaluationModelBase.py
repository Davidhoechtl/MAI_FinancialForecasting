from abc import ABC, abstractmethod

import pandas as pd
from pandas import DataFrame


class ForecastingModelBase(ABC):
    def __init__(self):
        self.results = []
        self.name = self.__class__.__name__

    @abstractmethod
    def train(self, x_train : pd.DataFrame, y_train : pd.Series):
        pass

    @abstractmethod
    def predict(self, x_test : pd.DataFrame, x_gap : DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def experiment(self, feature_matrix: pd.DataFrame, predictor_cols: list[str], target_col: str):
        pass

    @abstractmethod
    def plot_results(self):
        pass