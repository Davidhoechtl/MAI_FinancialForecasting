import pandas as pd
from Sentiment.Datasets.dataset_adapter_base import DatasetAdapterBase
from datasets import load_dataset, concatenate_datasets
from pathlib import Path

# Always relative to this script (analyze.py)
BASE_PATH = Path(__file__).resolve().parent
PROCESSED_FILE = BASE_PATH / "processed_headlines.csv"
BROKEN_FILE = BASE_PATH / "broken_headlines.csv"

class NiftyAdapter (DatasetAdapterBase):
    def __init__(self):
        self.df = pd.DataFrame()  # define df in __init__

    def try_load_preprocessed(self) -> bool:
        if not PROCESSED_FILE.exists():
            return False

        self.df = pd.read_csv(PROCESSED_FILE, parse_dates=['date'])
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
        # 1️. Load the dataset from Hugging Face
        dataset = load_dataset("raeidsaqur/nifty")

        # 2️. concatenate splits
        data = concatenate_datasets([
            dataset["train"],
            dataset["valid"],
            dataset["test"]
        ])

        # 3️. Extract only the needed columns and add the source
        df = pd.DataFrame({
            "source": ["NIFTY"] * len(data),
            "date": data["date"],
            "headline": data["news"]
        })

        # 4️⃣ Split headlines on line breaks and explode to individual rows
        # - split by '\n'
        # - expand=False so each cell becomes a list
        # - explode() creates a new row for each list element
        df["headline"] = df["headline"].str.split("\n")
        df = df.explode("headline")

        # 5️⃣ Clean up whitespace and drop empty lines
        df["headline"] = df["headline"].str.strip()

        broken = df[
            df["headline"].isna() |  # NaN values
            (df["headline"].str.len() < 5) |  # too short to be meaningful
            (df["headline"].str.contains(r"^[\s\n\r]*$"))  # only whitespace/newline
        ]
        print("🧩 Broken rows detected:", len(broken))
        if len(broken) > 0:
            print("Example broken entries:")
            print(broken.head(10))
            broken.to_csv(BROKEN_FILE, index=False)

        # 7️⃣ Drop broken entries
        df = df.drop(broken.index).reset_index(drop=True)

        # Convert 'Time' to datetime and localize to US/Eastern
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        date_nat_count = df['date'].isna().sum()
        print(f"Number of NA values in Date after conversion: {date_nat_count}")
        df = df.dropna(subset=['date'])
        df['date'] = df['date'].apply(lambda ts: ts.tz_localize('US/Eastern'))

        # Sort df by 'Date' in ascending order
        df = df.sort_values(by='date')

        # export to CSV
        df.to_csv(PROCESSED_FILE, index=False)

        self.df = df

    def to_standard_format(self) -> pd.DataFrame:
        if self.df.empty:
            raise ValueError("DataFrame is empty. Call load() before to_standard_format().")

        return self.df