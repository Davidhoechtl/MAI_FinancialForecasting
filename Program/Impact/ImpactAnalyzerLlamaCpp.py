# pip install llama-cpp-python

import re
from llama_cpp import Llama
import huggingface_hub as hf_hub
import os

# === Step 1. Download manually with Hugging Face Hub (shows progress) ===
model_repo = "second-state/FinGPT-MT-Llama-3-8B-LoRA-GGUF"
model_file = "FinGPT-MT-Llama-3-8B-LoRA-Q5_K_M.gguf"

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
    n_ctx=1024,
    n_gpu_layers=-1,  # full GPU offload if possible
    verbose=True
)

# 2️⃣ Prompt template
def make_prompt(headline: str) -> str:
    return f"""
You are a financial analyst.
Evaluate how strongly the following headline could impact the S&P 500 in the next trading day.
Give a number between 0 (no measurable impact) and 1 (very strong market-moving impact).
Headline: "{headline}"
Answer with only the number.
"""

# 3️⃣ Inference helper
def predict_impact(headline: str):
    prompt = make_prompt(headline)
    output = llm(
        prompt,
        max_tokens=20,
        temperature=0.1,
        stop=["\n"]
    )
    text = output["choices"][0]["text"].strip()
    match = re.search(r"([0]\.\d+|1\.0|1)", text)
    return float(match.group(1)) if match else None

# 4️⃣ Test headlines
headlines = [
    "Amazon Prime could face investigation over delivery complaints",
    "Federal Reserve signals higher interest rates for longer",
    "Local bakery opens new shop downtown",
]

for h in headlines:
    print(f"{h} -> {predict_impact(h)}")
