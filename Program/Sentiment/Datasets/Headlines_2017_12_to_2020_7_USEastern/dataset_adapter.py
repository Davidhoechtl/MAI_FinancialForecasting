import os
import pandas as pd
from Sentiment.Datasets.dataset_adapter_base import DatasetAdapterBase

BASE_PATH = "Sentiment/Datasets/Headlines_2017_12_to_2020_7_USEastern/"

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
        cnbc = pd.read_csv(BASE_PATH + "cnbc_headlines.csv")
        cnbc['Time'] = pd.to_datetime(cnbc['Time'].str.replace('ET', '', regex=False).str.strip())
        cnbc['Source'] = 'CNBC'
        print(cnbc.head())

        reuters = pd.read_csv(BASE_PATH + "reuters_headlines.csv")
        reuters['Source'] = 'REUTERS'
        print(reuters.head())

        guardian = pd.read_csv(BASE_PATH + "guardian_headlines.csv")
        guardian['Source'] = 'GUARDIAN'
        print(guardian.head())

        df = pd.concat([cnbc, guardian, reuters], ignore_index=True)

        # Use 'Description' if available, otherwise 'Headlines'
        df['Headlines'] = df['Description'].combine_first(df['Headlines'])

        # Drop the now-redundant Description column
        if 'Description' in df.columns:
            df = df.drop(columns=['Description'])

        # Count missing headlines
        headline_nan_count = df['Headlines'].isna().sum()
        print(f"Number of NA values in Headlines: {headline_nan_count}")
        df = df.dropna(subset=['Headlines'])

        # Convert 'Time' to datetime and localize to US/Eastern
        df['Date'] = pd.to_datetime(df['Time'], errors='coerce')
        date_nat_count = df['Date'].isna().sum()
        print(f"Number of NA values in Date after conversion: {date_nat_count}")
        df = df.dropna(subset=['Date'])
        df['Date'] = df['Date'].apply(lambda ts: ts.tz_localize('US/Eastern'))

        # Sort df by 'Date' in ascending order
        df = df.sort_values(by='Date')

        df.drop(columns=['Time'], inplace=True)

        # export to CSV
        df.to_csv(BASE_PATH + 'processed_headlines.csv', index=False)

        self.df = df

    def to_standard_format(self) -> pd.DataFrame:
        if self.df.empty:
            raise ValueError("DataFrame is empty. Call load() before to_standard_format().")

        standardized_df = self.df.rename(columns={
            'Source': 'source',
            'Date': 'date',
            'Headlines': 'headline',
        })[['source', 'date', 'headline']]

        return standardized_df