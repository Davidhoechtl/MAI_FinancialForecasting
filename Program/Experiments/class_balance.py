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

start_date = "03/02/2010"
end_date = "18/07/2020"
# end_date = "18/07/2020"
impact_model = ImpactModel.NONE
df_combined = get_feature_matrix(
    start_date=start_date,
    end_date=end_date,
    impact_model=impact_model,
    tech_indicators=[TechnicalIndicators.VOLATILITY],
    sentiment_sources=[DatasetSources.AENLLE, DatasetSources.LUCASPHAM],
    sentiment_model=SentimentModel.FINBERT,
    granularity_level=GranularityLevel.DAILY
)

print(df_combined.tail(10))


def plot_horizon_distribution(df, horizon, ax):
    """Calculates distribution for a single horizon and plots it onto a specific axis."""
    target = (df['Close'].pct_change(periods=horizon).shift(-horizon).dropna() > 0).astype(int)
    counts = target.value_counts().sort_index()

    down_count = counts.get(0, 0)
    up_count = counts.get(1, 0)
    labels = [f"Down (0)\n({down_count} days)", f"Up (1)\n({up_count} days)"]

    ax.pie([counts.get(0, 0), counts.get(1, 0)], labels=labels,
           colors=['#d9534f', '#5cb85c'], autopct='%1.1f%%', startangle=90, explode=(0.02, 0.02))
    ax.set_title(f"{horizon}-Day Horizon")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

horizons = [1, 3, 5, 20]

for h, ax in zip(horizons, axes.flatten()):
    plot_horizon_distribution(df_combined, horizon=h, ax=ax)

plt.suptitle("S&P 500 Market Direction Class Balance by Target Horizon",
             fontsize=16, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
# 1. Create a single figure with 4 subplots side-by-side
#fig, axes = plt.subplots(2, 2, figsize=(20, 5))

# target_horizon = 20
# df_combined['Target'] = df_combined['Close'].pct_change(periods=target_horizon).shift(-target_horizon)
# df_combined.dropna(subset=['Target'], inplace=True)
#
# # --- Create target (next-day direction) ---
# df_combined["Target"] = (df_combined["Target"] > 0).astype(int) # 1 if next day's pct change > 0 else 0
# labels = ["Down (0)", "Up (1)"]
# class_counts = df_combined["Target"].value_counts().sort_index()
#
# print("Class distribution:")
# print("up: " + str(class_counts[1]))
# print("down: " + str(class_counts[0]))
#
# # plot class balance
# plt.figure(figsize=(8, 6))
# bars = plt.bar(labels, class_counts.values, color=['#d9534f', '#5cb85c'])
# plt.title("Class Balance: Next-Day Market Direction of S&P 500")
# plt.xlabel("Market Direction")
# plt.ylabel("Number of Days")
# plt.show()