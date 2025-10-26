from datasets import load_dataset, concatenate_datasets
import pandas as pd
import Sentiment.Datasets.NIFTY.nifty_adapter
from Sentiment.Datasets.NIFTY.nifty_adapter import NiftyAdapter

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

nifty_adapter = NiftyAdapter()
if not nifty_adapter.try_load_preprocessed():
    nifty_adapter.load()
df = nifty_adapter.to_standard_format()

# 4. Show result
print(df.head(100))

# Basic info
print("🧾 DataFrame Info:")
print(df.info(), "\n")

# Check for missing values
print("🧩 Missing Values per Column:")
print(df.isna().sum(), "\n")