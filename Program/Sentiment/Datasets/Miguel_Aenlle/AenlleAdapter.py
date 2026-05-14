import os
import pandas as pd
from Sentiment.Datasets.dataset_adapter_base import DatasetAdapterBase

BASE_PATH = "Sentiment/Datasets/Miguel_Aenlle/"
CSV_NAME = "raw_partner_headlines.csv"
CSV_NAME_PROCESSED = "processed_headlines.csv"

# Dataset: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
# it is not stated that the timezone is us/eastern, i checked some of the url and it seams to be the case
class AenlleAdapter (DatasetAdapterBase):
    def __init__(self):
        self.df = pd.DataFrame()

    def try_load_preprocessed(self) -> bool:
        # check if processed_headlines.csv exists
        file_path = BASE_PATH + CSV_NAME_PROCESSED

        # Check if file exists
        if not os.path.exists(file_path):
            return False

        self.df = pd.read_csv(file_path, parse_dates=['date'])
        print(f"Loaded preprocessed data with {len(self.df)} records.")

        # Force conversion to datetime, just in case
        self.df["date"] = pd.to_datetime(self.df["date"], utc=True)
        date_nat_count = self.df['date'].isna().sum()
        if date_nat_count > 0:
            raise Exception("Error reading the preprocessed dataframe file, there where invalids dates.")

        # Convert from UTC to US/Eastern
        self.df["date"] = self.df["date"].dt.tz_convert("US/Eastern")

        return True

    def load(self):
        # layout: index,headline,url,publisher,date,stock
        df = pd.read_csv(BASE_PATH + CSV_NAME)
        df['source'] = df['publisher']
        print(df.head())

        # Count missing headlines
        headline_nan_count = df['headline'].isna().sum()
        print(f"Number of NA values in Headlines: {headline_nan_count}")
        df = df.dropna(subset=['headline'])

        # Clean headline column
        df["headline"] = df["headline"].str.replace('\n', ' ').str.replace('\r', ' ')
        df["headline"] = df["headline"].str.replace(r'\s+', ' ', regex=True).str.strip()

        # Convert 'Time' to datetime and localize to US/Eastern
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        date_nat_count = df['date'].isna().sum()
        print(f"Number of NA values in Date after conversion: {date_nat_count}")
        df = df.dropna(subset=['date'])
        df['date'] = df['date'].apply(lambda ts: ts.tz_localize('US/Eastern'))

        # Sort df by 'Date' in ascending order
        df = df.sort_values(by='date')

        # export to CSV
        df.to_csv(BASE_PATH + 'processed_headlines.csv', index=False)

        self.df = df

    def to_standard_format(self) -> pd.DataFrame:
        if self.df.empty:
            raise ValueError("DataFrame is empty. Call load() before to_standard_format().")

        standardized_df = self.df[['source', 'date', 'headline']]

        return standardized_df