import Utils.dataset_plots as utils
import pandas as pd

df = pd.read_csv("Sentiment/Datasets/Headlines_2017_12_to_2020_7_USEastern/processed_headlines.csv")

utils.visualize_headline_count_daily(df, "2017-12-31", "2020-07-19")
utils.visualize_headline_count_hourly(df, "2018-02-01", "2018-02-06")

# df = pd.read_csv("./Headlines_2017_12_to_2020_7_USEastern/processed_headlines.csv")
# headlines_of_day = df[df['Date'].str.startswith('2018-02-01')]
# print(headlines_of_day)