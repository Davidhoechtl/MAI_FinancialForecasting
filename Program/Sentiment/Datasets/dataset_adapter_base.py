from abc import ABC, abstractmethod

import pandas as pd


class DatasetAdapterBase(ABC):
    @abstractmethod
    def try_load_preprocessed(self) -> bool:
        """Load data from a preprocessed file"""
        pass

    @abstractmethod
    def load(self):
        """Load data and create preprocessed file if not exists."""
        pass

    @abstractmethod
    def to_standard_format(self) -> pd.DataFrame:
        """Convert raw data into a unified format like source;date;headline."""
        pass