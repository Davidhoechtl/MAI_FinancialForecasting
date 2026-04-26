import os
import pandas as pd
from matplotlib import pyplot as plt

from FeatureMatrixPipeline import get_feature_matrix
from Impact.ImpactScoreAnalyzerEnums import ImpactModel
from SP500_Prices.PriceAnalyzer import TechnicalIndicators
from Sentiment.SentimentAnalyzer import DatasetSources, SentimentModel, GranularityLevel

os.environ["KMP_DUPLICATE_LIB_OK"] = "1"  # prevent OpenMP conflict early

# Change working directory to project root
os.chdir("D:/Studium/Master/Masterarbeit/MAI_FinancialForecasting/Program")

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

start_date = "2010/02/03"
end_date = "2020/07/18"
# start_date = "2017/12/17"
# end_date = "2020/06/04"
df_combined_impact = get_feature_matrix(
    start_date=start_date,
    end_date=end_date,
    impact_model=ImpactModel.LLAMA_3_1_Instruct,
    tech_indicators=[TechnicalIndicators.VOLATILITY],
    sentiment_sources=[DatasetSources.AENLLE, DatasetSources.LUCASPHAM],
    sentiment_model=SentimentModel.FINBERT,
    granularity_level=GranularityLevel.DAILY
)
df_combined = get_feature_matrix(
    start_date=start_date,
    end_date=end_date,
    impact_model=ImpactModel.NONE,
    tech_indicators=[TechnicalIndicators.VOLATILITY],
    sentiment_sources=[DatasetSources.AENLLE, DatasetSources.LUCASPHAM],
    sentiment_model=SentimentModel.FINBERT,
    granularity_level=GranularityLevel.DAILY
)

df_combined['weighted_sentiment'] = df_combined_impact['weighted_sentiment']

fig, ax1 = plt.subplots(figsize=(12, 6))
df_combined['Close_Next_Day'] = df_combined['Close'].shift(-1)

# Drop NaNs for both sentiment models to ensure clean plotting
copy = df_combined.dropna(subset=['Close_Next_Day', 'weighted_sentiment', 'sentiment'])

# 1. Plot price on the primary y-axis (left)
ax1.plot(copy['Date'], copy['Close_Next_Day'], color='blue', label='Price (Next Day)')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(copy['Date'], copy['sentiment'], color='green', alpha=0.7, label='sentiment')
ax2.plot(copy['Date'], copy['weighted_sentiment'], color='red', alpha=0.7, label='weighted_sentiment')
ax2.set_ylabel('Sentiment Score', color='black')
ax2.tick_params(axis='y', labelcolor='black')

fig.suptitle('Price vs Sentiment vs Weighted Sentiment Over Time', fontsize=14)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()