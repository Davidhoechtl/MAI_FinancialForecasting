import os
import pandas as pd
from datasets import concatenate_datasets
from datasets import load_dataset

from Sentiment.Datasets.dataset_adapter_base import DatasetAdapterBase

BASE_PATH = "Sentiment/Datasets/Ferrell/"

# First Dataset: https://www.kaggle.com/datasets/notlucasp/financial-news-headlines
class Adapter1 (DatasetAdapterBase):
    def __init__(self):
        self.df = pd.DataFrame()  # define df in __init__

    def try_load_preprocessed(self) -> bool:
        # check if processed_headlines.csv exists
        file_path = BASE_PATH + "processed_headlines.csv"

        # Check if file exists
        if not os.path.exists(file_path):
            return False

        self.df = pd.read_csv(file_path, parse_dates=['Date'])
        print(f"Loaded preprocessed data with {len(self.df)} records.")

        # Force conversion to datetime, just in case
        self.df["Date"] = pd.to_datetime(self.df["Date"], utc=True)
        date_nat_count = self.df['Date'].isna().sum()
        if date_nat_count > 0:
            raise Exception("Error reading the preprocessed dataframe file, there where invalids dates.")

        # Convert from UTC to US/Eastern
        self.df["Date"] = self.df["Date"].dt.tz_convert("US/Eastern")

        return True

    def load(self):
        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset("Brianferrell787/financial-news-multisource")

        # 2️. concatenate splits
        data = concatenate_datasets([
            ds["train"],
            ds["valid"],
            ds["test"]
        ])

        self.df = df

    def to_standard_format(self) -> pd.DataFrame:
        if self.df.empty:
            raise ValueError("DataFrame is empty. Call load() before to_standard_format().")

        return self.df