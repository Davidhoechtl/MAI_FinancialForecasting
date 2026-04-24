import hashlib
import os
import re

import numpy as np
import pandas as pd
from llama_cpp import Llama
from tqdm import tqdm
from enum import Enum

import Utils.pandas_helper as pandas_helper
from Impact.ImpactMapUtils import load_impact_map, save_impact_map
from Impact.ImpactScoreAnalyzerEnums import EvaluationMode, ImpactModel
from Impact.Models.GPT_OSS_20B.GptOss20B import GptOss20B
from Impact.Models.Llama3_1_Instruct.Llama3_1_Instruct import LlamaInstruct
from Sentiment.Datasets.Headlines_2017_12_to_2020_7_USEastern.dataset_adapter import Adapter1
from Sentiment.Datasets.Miguel_Aenlle.AenlleAdapter import AenlleAdapter

factories = [
    LlamaInstruct(),
    GptOss20B(),
]

def format_prompt_llama(system_prompt: str, user_prompt: str) -> str:
    return f"""<|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

def _predict_impact_single_classification(llm:Llama, headline: str, max_retries: int = 2):
    """Call the LLM for a single headline, retry if no number is found."""
    system_prompt = """    
    Act as a Senior US Economic Strategist. Classify the news headline's relevance to the US Economy using these four categories:
    
    A (Critical US Macro): Direct US economic indicators only (Fed news, US Inflation, US GDP, US Jobs, US Treasury Yields).
    B (Major US Corporate): Events moving US large companies or sectors (big Deals, Antitrust, Major M&A, product launches).
    C (Global Spillover): International events with secondary US effects (Oil/OPEC, Global Supply Chains, Major Geopolitics, foreign investments).
    D (Noise/Irrelevant): Foreign domestic news (UK/EU/China specific politics or economy), small-caps, sports, local crime.
    
    Output format: ONLY valid JSON: {"score": <class>}
    """
    user_prompt = f"Headline: {headline}"

    if llm.metadata['general.architecture'] == 'llama':
        prompt = format_prompt_llama(system_prompt, user_prompt)
    else:
        prompt = f"{system_prompt}\n{user_prompt}"
    #print("Prompt: " + prompt)
    for attempt in range(1, max_retries + 1):
        output = llm(
            prompt,
            max_tokens=128,  # Hard limit: Generate only the number (e.g., "0.7")
            stop=["<|eot_id|>", "\n"],  # Stop immediately after the number
            temperature=0.005,  # Deterministic (0.0 is faster/safer for classification)
            echo=False,
        )

        import json

        try:
            # 1. Get text and find JSON structure
            text_response = output['choices'][0]['text'].strip()
            match = re.search(r'\{.*\}', text_response, re.DOTALL)

            if match:
                # Use json.loads for better safety than eval()
                data = json.loads(match.group())

                # 2. Extract the letter (Checking 'class' first, then 'score')
                # The prompt asks for "score", but your request mentioned "class"
                letter = data.get("score").upper()

                # 3. Map to numerical weight
                mapping = {
                    "A": 1.0,
                    "B": 0.66,
                    "C": 0.33,
                    "D": 0.0
                }

                # Check if we got a valid letter, otherwise retry
                if letter in mapping:
                    return mapping[letter]
        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed parsing: {e} | Output: {text_response[:50]}...")

    return None

def _predict_impact_single_regression(llm:Llama, headline: str, max_retries: int = 3):
    """Call the LLM for a single headline, retry if no number is found."""
    system_prompt = """You are an expert financial analyst specializing in the US Economy. 
    Analyze the news headline and determine its relevance to the 'US Economy'.

    Scoring Rules (0.0 to 1.0):
    - 0.9-1.0: Direct US Macro impact (Fed rates, US Inflation, US GDP, Major US regulations).
    - 0.7-0.8: Significant US Corporate News (Mergers of US firms, Antitrust involving US firms like Nvidia/Google, US Labor strikes).
    - 0.3-0.5: Global news with indirect US impact (Oil prices, Global supply chains).
    - 0.0-0.2: Irrelevant (Foreign domestic news, UK/EU specific with no US spillover, Sports, Quizzes, localized crimes).

    Crucial Relationships to Spot:
    - 'Fed' / 'Federal Reserve' -> High Relevance.
    - '£' or 'Euro' symbols often imply UK/EU context -> Low Relevance (unless global).
    - 'Nvidia', 'Apple', 'Tesla' are US companies -> Moderate/High Relevance.
    - 'Protests in Peru', 'UK Inflation' -> Low Relevance.

    Output ONLY the float number. Do not explain.
    """
    user_prompt = f"Headline: {headline}"

    if llm.metadata['general.architecture'] == 'llama':
        prompt = format_prompt_llama(system_prompt, user_prompt)
    else:
        prompt = f"{system_prompt}\n{user_prompt}"

    for attempt in range(1, max_retries + 1):
        output = llm(
            prompt,
            max_tokens=6,  # Hard limit: Generate only the number (e.g., "0.7")
            stop=["<|eot_id|>", "\n"],  # Stop immediately after the number
            temperature=0.0,  # Deterministic (0.0 is faster/safer for classification)
            echo=False,
        )

        text = output["choices"][0]["text"].strip()
        # Extract a 0–1 float
        match = re.search(r"\b(0\.\d+|1\.0|1)\b", text)
        if match:
            return float(match.group(1))


    # After all retries → return None or a fallback
    print("❌ No valid number extracted after retries. Returning None.")
    return None

# === Step 5. Prediction logic ===
def predict_impact(
        llm: Llama,
        headlines: pd.Series,
        evaluation_mode: EvaluationMode = EvaluationMode.REGRESSION,
        impact_map=None,
        verbose: bool = False):
    """
    Predict LLM-based market impact for a series of headlines.

    Args:
        llm (Llama): The LLM instance to use for predictions.
        headlines (pd.Series): Series of news headlines.
        evaluation_mode (EvaluationMode): Mode of evaluation (classification or regression).
        impact_map (dict): Optional existing impact cache.
        verbose (bool): If True, prints detailed progress messages.

    Returns:
        (pd.Series, dict): Series of impact scores and updated impact_map.
    """
    if impact_map is None:
        impact_map = {}

    results = []
    new_items = 0

    # tqdm progress bar
    for h in tqdm(headlines, desc="🔮 Evaluating headline impact", unit="headline"):
        if not isinstance(h, str):
            h = str(h)
        h_hash = hashlib.sha256(h.encode("utf-8")).hexdigest()

        entry = impact_map.get(h_hash)
        if entry is not None:
            val = entry.get("impact")
            if verbose:
                print(f"🟢 Cached impact for: {h[:80]}... → {val}")
        else:
            try:
                if evaluation_mode == EvaluationMode.CLASSIFICATION:
                    val = _predict_impact_single_classification(llm, h)
                else:
                    val = _predict_impact_single_regression(llm, h)

                if verbose:
                    print(f"🧠 Evaluated new headline: {h[:100]} → {val}")

                new_items += 1
            except Exception as e:
                if verbose:
                    print(f"⚠️  Error for '{h[:80]}': {e}")
                val = None
            impact_map[h_hash] = {"headline": h, "impact": val}

        results.append(np.nan if val is None else float(val))

    print(f"\n✅ Completed evaluation: {len(headlines)} total, {new_items} newly processed.")
    return pd.Series(results, index=headlines.index), impact_map

def get_impact_model_factory(model_type: ImpactModel):
    if model_type == ImpactModel.NONE:
        raise ValueError("ImpactModel.NONE does not have a factory.")

    for factory in factories:
        if factory.get_model_type() == model_type:
            return factory

    raise ValueError(f"No factory found for ImpactModel: {model_type}")

def load_impact_score(headline: pd.Series, impact_model: ImpactModel, evaluation_mode: EvaluationMode) -> pd.Series:
    factory = get_impact_model_factory(impact_model)
    llm = factory.create()
    impact_file_path = factory.get_impact_file_path(evaluation_mode)

    # Load previous cache
    impact_map = load_impact_map(impact_file_path)

    # Predict and update cache
    impact_series, impact_map = predict_impact(
        llm=llm,
        headlines=headline,
        evaluation_mode=evaluation_mode,
        impact_map=impact_map,
        verbose=False
    )

    # Save updated cache
    save_impact_map(impact_map,impact_file_path)

    return impact_series


# DEBUG / TESTING
#------------------------------
if __name__ == "__main__":
    # Change working directory to project root
    os.chdir("D:/Studium/Master/Masterarbeit/MAI_FinancialForecasting/Program")

    start_date = "17/12/2017"
    # start_date = "10/12/2019"
    end_date = "18/07/2020"
    adapter1 = AenlleAdapter()
    if not adapter1.try_load_preprocessed():
        adapter1.load()
    df1 = adapter1.to_standard_format()
    df1 = pandas_helper.filter_dataset_by_dates(df1, start_date, end_date)

    adapter1 = Adapter1()
    if not adapter1.try_load_preprocessed():
        adapter1.load()
    df2 = adapter1.to_standard_format()
    df2 = pandas_helper.filter_dataset_by_dates(df2, start_date, end_date)

    # concat df1 and df2
    df = pd.concat([df1, df2], ignore_index=True)

    df["impact"] = load_impact_score(df["headline"], ImpactModel.LLAMA_3_1_Instruct, EvaluationMode.CLASSIFICATION)

    import matplotlib.pyplot as plt
    # Assuming df["impact"] contains the numeric impact scores (0–1)
    plt.figure(figsize=(8, 5))
    plt.hist(df["impact"], bins=20, edgecolor="black")
    plt.title("Distribution of Predicted Headline Impact Scores")
    plt.xlabel("Impact Score (0 = no impact, 1 = strong impact)")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.3)
    plt.show()