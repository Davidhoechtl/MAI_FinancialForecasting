import os
import pandas as pd
import EvaluationPipeline
from FeatureMatrixPipeline import get_feature_matrix
from Forecasting.AlwaysUp import AlwaysUpModel
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
end_date = "06/04/2020"
# end_date = "18/07/2020"
impact_model = ImpactModel.NONE
df_combined = get_feature_matrix(
    start_date=start_date,
    end_date=end_date,
    impact_model=impact_model,
    tech_indicators=[TechnicalIndicators.VOLATILITY],
    sentiment_sources=[DatasetSources.AENLLE],
    sentiment_model=SentimentModel.FINBERT,
    granularity_level=GranularityLevel.DAILY
)

print(df_combined.tail(10))

# --- Create target (next-day direction) ---
feature_cols = ['Pct_Change']
df_combined["Target"] = (df_combined["Pct_Change_next"] > 0).astype(int) # 1 if next day's pct change > 0 else 0
always_up_baseline = AlwaysUpModel()
always_up_results = EvaluationPipeline.evaluate_model_on_classification(
    model=always_up_baseline,
    feature_matrix=df_combined,
    predictor_cols=feature_cols,
    target_col='Target',
    target_horizon_in_days=1
)

print("Always Up Baseline Results:")
print("Always Up Results:", always_up_results)