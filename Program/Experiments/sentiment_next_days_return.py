import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from FeatureMatrixPipeline import get_feature_matrix
from Forecasting.ARMA import ARMAForecastingModel
from Forecasting.XGBoost import XGBoostForecastingModel
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
    sentiment_sources=[DatasetSources.LUCASPHAM],
    sentiment_model=SentimentModel.FINBERT,
    granularity_level=GranularityLevel.DAILY
)
print(df_combined.head(10))

df_combined['Target_3d_Return'] = df_combined['Close'].pct_change(periods=3).shift(-3)
df_combined['Target_5d_Return'] = df_combined['Close'].pct_change(periods=5).shift(-5)
df_combined['Target_20d_Return'] = df_combined['Close'].pct_change(periods=20).shift(-20)

# --- Create target (next-day direction) ---
df_combined["Target_5d_Return_binary"] = (df_combined["Target_5d_Return"] > 0).astype(int) # 1 if next day's pct change > 0 else 0

df_combined.dropna(subset=['Target_3d_Return', 'Target_5d_Return', 'Target_20d_Return', 'Target_5d_Return_binary'], inplace=True)

print('Mean 3d Return:', df_combined['Target_3d_Return'].mean())
print('Mean 5d Return:', df_combined['Target_5d_Return'].mean())
print('Mean 20d Return:', df_combined['Target_20d_Return'].mean())
print('Standard Deviation 3d Return:', df_combined['Target_3d_Return'].std())
print('Standard Deviation 5d Return:', df_combined['Target_5d_Return'].std())
print('Standard Deviation 20d Return:', df_combined['Target_20d_Return'].std())

# --- 2. Visualization ---
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Top Row 1: Correlation Matrix
corr_cols = ['Target_3d_Return', 'Target_5d_Return', 'Target_20d_Return',
             'sentiment', 'weighted_sentiment', 'Volatility']
corr_matrix = df_combined[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f', ax=axes[0, 0])
axes[0, 0].set_title('Correlation Matrix')

# Top Row 2: Distribution of Returns
sns.kdeplot(df_combined['Target_3d_Return'], label='3d Return', fill=True, alpha=0.2, ax=axes[0, 1])
sns.kdeplot(df_combined['Target_5d_Return'], label='5d Return', fill=True, alpha=0.2, ax=axes[0, 1])
sns.kdeplot(df_combined['Target_20d_Return'], label='20d Return', fill=True, alpha=0.2, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Future Returns')
axes[0, 1].set_xlabel('Return')
axes[0, 1].legend()

# Top Row 3: Mean Return by Sentiment Decile (The "New" Plot)
# We bin sentiment into 5 quantiles (quintiles) for clarity
df_combined['Sentiment_Quintile'] = pd.qcut(df_combined['sentiment'], q=5, labels=['Very Low', 'Low', 'Neutral', 'High', 'Very High'])
quintile_returns = df_combined.groupby('Sentiment_Quintile')['Target_5d_Return'].mean().reset_index()

# Bar plot
sns.barplot(data=quintile_returns, x='Sentiment_Quintile', y='Target_5d_Return', palette='viridis', ax=axes[0, 2])
axes[0, 2].set_title('Avg 5-Day Return by Sentiment Quintile')
axes[0, 2].set_xlabel('Sentiment Level')
axes[0, 2].set_ylabel('Mean 5-Day Return')
axes[0, 2].axhline(0, color='black', linewidth=0.8)

# Bottom Row: Sentiment vs Each Target
targets = ['Target_3d_Return', 'Target_5d_Return', 'Target_20d_Return']
titles = ['Sentiment vs. 3-Day Return', 'Sentiment vs. 5-Day Return', 'Sentiment vs. 20-Day Return']

for i, target in enumerate(targets):
    sns.regplot(data=df_combined, x='sentiment', y=target,
                scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'}, ax=axes[1, i])
    axes[1, i].set_title(titles[i])
    axes[1, i].set_xlabel('Sentiment Score')
    axes[1, i].set_ylabel(target)
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

eval_model = ARMAForecastingModel()
eval_model.experiment(df_combined, target_col="Target_3d_Return", predictor_cols=['sentiment'])
eval_model.plot_results()

eval_model = ARMAForecastingModel()
eval_model.experiment(df_combined, target_col="Target_5d_Return", predictor_cols=['sentiment'])
eval_model.plot_results()

eval_model = XGBoostForecastingModel()
eval_model.experiment(df_combined, target_col="Target_5d_Return_binary", predictor_cols=['sentiment'])
eval_model.plot_results()

eval_model = ARMAForecastingModel()
eval_model.experiment(df_combined, target_col="Target_20d_Return", predictor_cols=['sentiment'])
eval_model.plot_results()
