from transformers import LlamaTokenizerFast, LlamaForCausalLM
from peft import PeftModel
import torch, re

# Load base + LoRA model
base_model = "meta-llama/Meta-Llama-3-8B"
peft_model = "FinGPT/fingpt-mt_llama3-8b_lora"

tokenizer = LlamaTokenizerFast.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="cuda:0")
model = PeftModel.from_pretrained(model, peft_model)
model.eval()

# Prompt template → ask for numeric market impact
def make_prompt(headline: str) -> str:
    return f"""
    You are a financial analyst.
    Evaluate how strongly the following headline could impact the S&P 500 in the next trading day.
    Give a number between 0 (no measurable impact) and 1 (very strong market-moving impact).
    Headline: "{headline}"
    Answer with only the number.
    """

# Inference
def predict_impact(headline: str):
    prompt = make_prompt(headline)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Simple numeric parse
    match = re.search(r"([0]\.\d+|1\.0|1)", text)
    return float(match.group(1)) if match else None

# Example headlines
headlines = [
    "Amazon Prime could face investigation over delivery complaints",
    "Federal Reserve signals higher interest rates for longer",
    "Local bakery opens new shop downtown"
]

for h in headlines:
    print(f"{h} -> {predict_impact(h)}")
