import re
import pandas as pd
from llama_cpp import Llama
import huggingface_hub as hf_hub
import hashlib
import numpy as np
import os
from Sentiment.Datasets.Headlines_2017_12_to_2020_7_USEastern.dataset_adapter import Adapter1
from tqdm import tqdm
import Utils.pandas_helper as pandas_helper

IMPACT_MAP_FILE = "impact_map.csv"

# === Step 1. Download manually with Hugging Face Hub ===
# model_repo = "second-state/FinGPT-MT-Llama-3-8B-LoRA-GGUF"
# model_file = "FinGPT-MT-Llama-3-8B-LoRA-Q5_K_M.gguf"

model_repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
model_file="Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"

print("🔹 Downloading model from Hugging Face ...")
local_path = hf_hub.hf_hub_download(
    repo_id=model_repo,
    filename=model_file,
    repo_type="model",
)
print(f"✅ Model downloaded to: {local_path}")

# === Step 2. Load model from the local GGUF path ===
llm = Llama(
    model_path=local_path,
    n_ctx=8192,
    n_gpu_layers=-1,  # full GPU offload if possible
    verbose=False
)
print(llm("Say Hello!", max_tokens=10))

# 2️⃣ Prompt template
def make_prompt(headline: str) -> str:
    return f""" You are a financial analyst. Evaluate how strongly the following headline could impact the S&P 500 in the next trading day. Give a number between 0 (no measurable impact) and 1 (very strong market-moving impact). Answer with the number only. Headline: "{headline}". """

# 3️⃣ Inference helper
def _predict_impact_single(headline: str):
    """Call the LLM for a single headline and parse a float between 0 and 1, or return None."""
    prompt = make_prompt(headline)
    output = llm(
        prompt,
        max_tokens=3
    )
    text = output["choices"][0]["text"].strip()
    match = re.search(r"([0]\.[0-9]+|1\.0|1)", text)
    return float(match.group(1)) if match else None

# === Step 4. Persistence helpers ===

def load_impact_map():
    """Load saved impact map from CSV (if it exists)."""
    if os.path.exists(IMPACT_MAP_FILE):
        df = pd.read_csv(IMPACT_MAP_FILE)
        return {
            row["hash"]: {"headline": row["headline"], "impact": row["impact"]}
            for _, row in df.iterrows()
        }
    return {}

def save_impact_map(impact_map):
    """Save impact map to CSV."""
    df = pd.DataFrame([
        {"hash": h, "headline": v["headline"], "impact": v["impact"]}
        for h, v in impact_map.items()
    ])
    df.to_csv(IMPACT_MAP_FILE, index=False)
    print(f"✅ Saved {len(df)} impact entries to {IMPACT_MAP_FILE}")

# === Step 5. Prediction logic ===
def predict_impact(headlines: pd.Series, impact_map=None, verbose: bool = False):
    """
    Predict LLM-based market impact for a series of headlines.

    Args:
        headlines (pd.Series): Series of news headlines.
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
                val = _predict_impact_single(h)
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

def load_impact_score(headline: pd.Series) -> pd.Series:
    # Load previous cache
    impact_map = load_impact_map()

    # Predict and update cache
    df["impact"], impact_map = predict_impact(headline, impact_map, verbose=True)

    # Save updated cache
    save_impact_map(impact_map)

    return df["impact"]

if __name__ == "__main__":
    # Change working directory to project root
    os.chdir("D:/Studium/Master/Masterarbeit/MAI_FinancialForecasting/Program")

    start_date = "01/01/2020"
    end_date = "18/07/2020"
    adapter1 = Adapter1()
    if not adapter1.try_load_preprocessed():
        adapter1.load()
    df = adapter1.to_standard_format()
    df = pandas_helper.filter_dataset_by_dates(df, start_date, end_date)

    df["impact"] = load_impact_score(df["Headlines"])

    import matplotlib.pyplot as plt
    # Assuming df["impact"] contains the numeric impact scores (0–1)
    plt.figure(figsize=(8, 5))
    plt.hist(df["impact"], bins=20, edgecolor="black")
    plt.title("Distribution of Predicted Headline Impact Scores")
    plt.xlabel("Impact Score (0 = no impact, 1 = strong impact)")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.3)
    plt.show()