import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

start_date = "17/12/2017"
end_date = "18/07/2020"
impact_model = ImpactModel.LLAMA_3_1_Instruct
df_combined = get_feature_matrix(
    start_date=start_date,
    end_date=end_date,
    impact_model=impact_model,
    tech_indicators=[TechnicalIndicators.VOLATILITY],
    sentiment_sources=[DatasetSources.NIFTY],
    sentiment_model=SentimentModel.FINBERT,
    granularity_level=GranularityLevel.DAILY
)

plt.figure(figsize=(10, 6))

# Plot histogram with a Kernel Density Estimate (KDE)
sns.histplot(df_combined['sentiment'], bins=30, kde=True, color='skyblue', edgecolor='black')

# Add mean line for reference
mean_sentiment = df_combined['sentiment'].mean()
plt.axvline(mean_sentiment, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_sentiment:.4f}')

plt.title('Distribution of Sentiment Values')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.show()

# 1. Calculate Correlation Matrix
cols_to_analyze = ['Volatility', 'sentiment', 'weighted_sentiment']
corr_matrix = df_combined[cols_to_analyze].corr()

print("Correlation Coefficients:")
print(corr_matrix)

# 2. Create Visualization
plt.figure(figsize=(18, 5))

# --- Plot A: Heatmap ---
plt.subplot(1, 3, 1)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Matrix')

# --- Plot B: Scatter Plot ---
plt.subplot(1, 3, 2)
sns.regplot(data=df_combined, x='sentiment', y='Volatility',
            scatter_kws={'alpha':0.3, 's':15}, line_kws={'color':'red'})
plt.title('Sentiment vs. Volatility')
plt.grid(True, alpha=0.3)

# --- Plot C: Rolling Correlation (60-day window) ---
# Adjust window size (e.g., 30 or 90) based on your preference
rolling_corr = df_combined['sentiment'].rolling(window=60).corr(df_combined['Volatility'])

plt.subplot(1, 3, 3)
plt.plot(df_combined['Date'], rolling_corr, color='#2c3e50', linewidth=1.5)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('60-Day Rolling Correlation\n(Sentiment vs. Volatility)')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
#
# # 1. Calculate Correlation Matrix
# df_combined["Volatility_next"] = df_combined["Volatility"].shift(-1)
# df_combined.dropna(subset=["Volatility_next"], inplace=True)
# cols_to_analyze = ['Volatility_next', 'sentiment', 'weighted_sentiment']
# corr_matrix = df_combined[cols_to_analyze].corr()
#
# print("Correlation Coefficients:")
# print(corr_matrix)
#
# # 2. Create Visualization
# plt.figure(figsize=(18, 5))
#
# # --- Plot A: Heatmap ---
# plt.subplot(1, 3, 1)
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
# plt.title('Correlation Matrix')
#
# # --- Plot B: Scatter Plot ---
# plt.subplot(1, 3, 2)
# sns.regplot(data=df_combined, x='sentiment', y='Volatility_next',
#             scatter_kws={'alpha':0.3, 's':15}, line_kws={'color':'red'})
# plt.title('Sentiment vs. Volatility_next')
# plt.grid(True, alpha=0.3)
#
# # --- Plot C: Rolling Correlation (60-day window) ---
# # Adjust window size (e.g., 30 or 90) based on your preference
# rolling_corr = df_combined['sentiment'].rolling(window=60).corr(df_combined['Volatility_next'])
#
# plt.subplot(1, 3, 3)
# plt.plot(df_combined['Date'], rolling_corr, color='#2c3e50', linewidth=1.5)
# plt.axhline(0, color='red', linestyle='--', linewidth=1)
# plt.title('60-Day Rolling Correlation\n(Sentiment vs. Volatility_next)')
# plt.ylabel('Correlation Coefficient')
# plt.xticks(rotation=45)
# plt.grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.show()
#
# print(df_combined.head(20))