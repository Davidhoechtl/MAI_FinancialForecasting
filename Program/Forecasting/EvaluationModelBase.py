from abc import ABC, abstractmethod

class ForecastingModelBase(ABC):
    def __init__(self):
        self.results = []

    @abstractmethod
    def evaluate(self, feature_matrix):
        pass

    @abstractmethod
    def plot_results(self):
        pass