import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

cnbc = pd.read_csv("cnbc_headlines.csv")
cnbc['Time'] = pd.to_datetime(cnbc['Time'].str.replace('ET','', regex=False).str.strip())
print(cnbc.head())

reuters = pd.read_csv("reuters_headlines.csv")
print(reuters.head())

guardian = pd.read_csv("guardian_headlines.csv")
print(guardian.head())

cnbc.dropna()
guardian.dropna()
reuters.dropna()

# Combine DataFrames
df = pd.concat([cnbc, guardian, reuters], ignore_index=True)
print(df.head())
print(df.describe())

from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon') # later use better sentiment analysis model

sia = SentimentIntensityAnalyzer()

def classify_sentiment(text):
    sentiment_score = sia.polarity_scores(text)['compound']
    if sentiment_score >= 0.05:
        return 'positive'
    elif sentiment_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def classify_sentiment_polarity(text):
    if pd.isna(text):   # check if text is NaN
        return None     # or return 0 if you prefer a neutral sentiment
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score

#df = df.dropna()
df = df.dropna(subset=['Headlines'])

# Apply sentiment classification to both 'Headlines' and 'Description'
df['Headline_Sentiment'] = df['Headlines'].apply(classify_sentiment_polarity)
df['Description_Sentiment'] = df['Description'].apply(classify_sentiment_polarity)

na_count = df['Description_Sentiment'].isna().sum()
print(f"Number of NA values in Description_Sentiment: {na_count}")

def use_sentiment(row):
    if pd.notna(row['Description_Sentiment']):
        return row['Description_Sentiment']
    else:
        return row['Headline_Sentiment']

df['sentiment'] = df.apply(use_sentiment, axis=1)
no_sentiment_count = df['sentiment'].isna().sum()
print(f"Number of rows without sentiment: {no_sentiment_count}")


def convert_to_us_eastern(ts):
    ts = ts.tz_localize('US/Eastern')
    return ts

# Convert to datetime first
df['Date'] = pd.to_datetime(df['Time'], errors='coerce')
date_nat_count = df['Date'].isna().sum()
print(f"Number of NA values in Date after conversion: {date_nat_count}")
df = df.dropna(subset=['Date'])
df['Date'] = df['Date'].apply(convert_to_us_eastern)

# Sort df by 'Date' in ascending order
df = df.sort_values(by='Date')

#prepare for export
df_export = df[['Date', 'sentiment']]
df_export.to_csv('processed_headlines.csv', index=False)

