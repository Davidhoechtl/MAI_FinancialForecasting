import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr, pearsonr

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
print(df_combined.describe())
print(df_combined.tail(10))

print("sentiment correlation:")
spearman_corr, spearman_p_value = spearmanr(df_combined['Pct_Change_next'], df_combined['sentiment'])
pearson_corr, pearson_p_value = pearsonr(df_combined['Pct_Change_next'], df_combined['sentiment'])
print(f"Spearman correlation: {spearman_corr}, p_value: {spearman_p_value}" )
print(f"Pearson correlation: {pearson_corr}, p_value: {pearson_p_value}" )

# same day sentiment
print("Same day sentiment correlation:")
spearman_corr, spearman_p_value = spearmanr(df_combined['Pct_Change'], df_combined['sentiment'])
pearson_corr, pearson_p_value = pearsonr(df_combined['Pct_Change'], df_combined['sentiment'])
print(f"Spearman correlation: {spearman_corr}, p_value: {spearman_p_value}" )
print(f"Pearson correlation: {pearson_corr}, p_value: {pearson_p_value}" )

from Utils import result_plots as rp
rp.plot_price_change_sentiment_scatter(df_combined)
rp.sentiment_price_plot(df_combined)