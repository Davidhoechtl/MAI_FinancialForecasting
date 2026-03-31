import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Custom module imports
from FeatureMatrixPipeline import get_feature_matrix
from Impact.ImpactScoreAnalyzerEnums import ImpactModel
from SP500_Prices.PriceAnalyzer import TechnicalIndicators
from Sentiment.SentimentAnalyzer import DatasetSources, SentimentModel, GranularityLevel

# Setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "1"
os.chdir("D:/Studium/Master/Masterarbeit/MAI_FinancialForecasting/Program")

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# --- 1. Data Fetching & Preparation ---
start_date = "03/02/2010"
end_date = "18/07/2020"
impact_model = ImpactModel.NONE

# Fetch feature matrix
df_combined = get_feature_matrix(
    start_date=start_date,
    end_date=end_date,
    impact_model=impact_model,
    tech_indicators=[TechnicalIndicators.VOLATILITY],
    sentiment_sources=[DatasetSources.LUCASPHAM],
    sentiment_model=SentimentModel.FINBERT,
    granularity_level=GranularityLevel.DAILY
)

print("mean (1d): " + str(df_combined['Pct_Change_next'].mean()))
print("std (1d): " + str(df_combined['Pct_Change_next'].std()))

# --- 2. Visualization: Target Distribution ---
plt.figure(figsize=(10, 6))

df_combined['Target_3d_Return'] = df_combined['Close'].pct_change(periods=3).shift(-3)
df_combined['Target_5d_Return'] = df_combined['Close'].pct_change(periods=5).shift(-5)
df_combined['Target_20d_Return'] = df_combined['Close'].pct_change(periods=20).shift(-20)

# Plot Kernel Density Estimates (KDE) for the distributions
sns.kdeplot(df_combined['Pct_Change_next'], label='1d Return', fill=True, alpha=0.3)
# sns.kdeplot(df_combined['Target_3d_Return'], label='3d Return', fill=True, alpha=0.3)
# sns.kdeplot(df_combined['Target_5d_Return'], label='5d Return', fill=True, alpha=0.3)
# sns.kdeplot(df_combined['Target_20d_Return'], label='20d Return', fill=True, alpha=0.3)

# Formatting the plot
# plt.title('Distribution of Future Returns (1,3,5,20 days)')
plt.title('Distribution of Future Returns (1 day)')
plt.xlabel('Return (Percentage Change)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Display the plot
plt.show()