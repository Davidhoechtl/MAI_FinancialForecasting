import pandas as pd
import numpy as np
from Sentiment.Datasets.dataset_adapter_base import DatasetAdapterBase
from datasets import load_dataset, concatenate_datasets
from pathlib import Path

# Always relative to this script (analyze.py)
BASE_PATH = Path(__file__).resolve().parent
PROCESSED_FILE = BASE_PATH / "processed_headlines.csv"
BROKEN_FILE = BASE_PATH / "broken_headlines.csv"

CUT_START = 2010
CUT_END = 2020

class FnspidAdapter (DatasetAdapterBase):
    def __init__(self):
        self.df = pd.DataFrame()  # define df in __init__

    def try_load_preprocessed(self) -> bool:
        if not PROCESSED_FILE.exists():
            return False

        self.df = pd.read_csv(PROCESSED_FILE, parse_dates=['Date'])
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
        # 1️. Load the dataset from Hugging Face
        df = pd.read_csv(BASE_PATH / "All_external.csv")
        print(df.head(10))

        # 2. Extract specific columns
        df = df[['Date', 'Article_title', 'Publisher']].copy()

        # 3. Rename columns to your requirements
        df.columns = ['Date', 'Headline', 'Source']

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # the dataset has utc dates that have the timestamp truncated. We may not convert to us/eastern and treat it as naive
        date_nat_count = df['Date'].isna().sum()
        print(f"Number of NA values in Date after conversion: {date_nat_count}")
        df = df.dropna(subset=['Date'])
        # 3. Normalize (Truncate time to 00:00:00) and Localize
        # .dt.normalize() sets the time to midnight, effectively "truncating" the time part
        # .dt.tz_localize(None) ensures we start with a naive timestamp (strips existing TZ if any)
        # .dt.tz_localize('US/Eastern') applies the Eastern timezone
        df['Date'] = (df['Date']
                      .dt.normalize()
                      .dt.tz_localize(None)
                      .dt.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='shift_forward')
                      )

        # 4. Filter Date: Keep only years 2000 to 2020 (inclusive)
        #    Note: "outside of 2000-2020 away" means keep inside.
        df = df[(df['Date'].dt.year >= CUT_START) & (df['Date'].dt.year <= CUT_END)]

        # 5. Clean Headline: Remove rows with broken headlines
        mask_na = df['Headline'].isna()
        mask_short = df['Headline'].str.len() < 5
        mask_whitespace = df['Headline'].str.strip() == ""
        # We calculate the ratio for the whole column at once using a list comprehension
        def calculate_ascii_ratio(text):
            if not isinstance(text, str): return 0
            return len(text.encode('ascii', 'ignore')) / len(text)

        ascii_ratios = [calculate_ascii_ratio(x) for x in df['Headline']]
        mask_bad_ascii = np.array(ascii_ratios) < 1

        # Combine all "Bad" conditions
        mask_broken = mask_na | mask_short | mask_whitespace | mask_bad_ascii

        # Separate the data
        broken = df[mask_broken]
        df = df[~mask_broken].reset_index(drop=True)

        print("🧩 Broken rows detected:", len(broken))
        if len(broken) > 0:
            broken.to_csv(BROKEN_FILE, index=False)

        df['Source'] = df['Source'].fillna('unknown')
        df['Headline'] = df['Headline'].str.replace(r'["\,]', '', regex=True)

        # Sort df by 'Date' in ascending order
        df = df.sort_values(by='Date')

        # drop duplicates
        df.drop_duplicates(subset=['Headline'], keep='first', inplace=True)

        # export to CSV
        df.to_csv(PROCESSED_FILE, index=False)

        # Display the result
        print(df)

        # Verification of data types
        print("\nData Types:")
        print(df.dtypes)

        self.df = df

    def to_standard_format(self) -> pd.DataFrame:
        if self.df.empty:
            raise ValueError("DataFrame is empty. Call load() before to_standard_format().")

        standardized_df = self.df.rename(columns={
            'Source': 'source',
            'Date': 'date',
            'Headline': 'headline',
        })[['source', 'date', 'headline']]

        return standardized_df

if __name__ == "__main__":
    adapter = FnspidAdapter()
    # if not adapter.try_load_preprocessed():
    #     adapter.load()
    #
    adapter.load()
    df = adapter.to_standard_format()
    print(df.head(10))