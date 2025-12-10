import pandas as pd
from llama_cpp import Llama
import re

from Impact.Models.GPT_OSS_20B.GptOss20B import GptOss20B
from Impact.Models.Llama3_1_Instruct.Llama3_1_Instruct import LlamaInstruct

# 1. Initialize the Llama 3.1 Model
# Make sure to download the .gguf file first!
# Set n_ctx=4096 or higher to handle the system prompt + headline context
LlamaInstructFactory = LlamaInstruct()
# LlamaInstructFactory = GptOss20B()
llm = LlamaInstructFactory.create()


def evaluate_headline_relevance(headline):
    """
    Asks the LLM to rate the relevance of a headline to the US Economy.
    Returns a score (0.0 - 1.0) and a brief reason.
    """

    # The System Prompt defines the "Nuanced Relationships"
    system_instruction = """
    Act as a Senior US Economic Strategist with. Classify the news headline's relevance to the US Economy using these four categories:
    
    A (Critical US Macro): Direct US economic indicators only (Fed news, US Inflation, US GDP, US Jobs, US Treasury Yields).
    B (Major US Corporate): Events moving US large companies or sectors (big Deals, Antitrust, Major M&A, product launches).
    C (Global Spillover): International events with secondary US effects (Oil/OPEC, Global Supply Chains, Major Geopolitics, foreign investments).
    D (Noise/Irrelevant): Foreign domestic news (UK/EU/China specific politics or economy), small-caps, sports, local crime.
    
    Output format: ONLY valid JSON: {"score": <class>}
    """

    prompt = f"""<|start_header_id|>system<|end_header_id|>
{system_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>
Headline: "{headline}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
#     prompt = system_instruction + f"\nHeadline: \"{headline}\""

    # Run Inference
    output = llm(
        prompt,
        max_tokens=128,
        stop=["<|eot_id|>"],
        temperature=0.0,  # Low temperature for consistent classification
        echo=False
    )

    try:
        # Parse the JSON output (simple regex to grab the JSON part if model chats too much)
        text_response = output['choices'][0]['text'].strip()
        # Use regex to find the JSON object in case of extra whitespace
        match = re.search(r'\{.*\}', text_response, re.DOTALL)
        if match:
            return eval(match.group())  # Safe for local strictly formatted output
        return {"score": 0.0, "reason": "Error parsing output"}
    except Exception as e:
        return {"score": 0.0, "reason": f"Error: {str(e)}"}


# --- Your Data ---
# Headline;Class
df = pd.read_csv("HeadlineImpactClassification_ManualGroundTruth.csv", sep=";")
print(df.head(10))
# A simple numeric mapping to measure distance
class_map = {"A": 3, "B": 2, "C": 1, "D": 0}

print(df.head(10))

print(f"{'GUESS':<6} | {'TRUE':<6} | {'TYPE':<12} | {'HEADLINE (Truncated)':<60} | {'REASONING'}")
print("-" * 160)

correct = 0
big_wrongs = 0
small_wrongs = 0

for idx, row in df.iterrows():
    headline = row["Headline"]
    true_class = row["Class"]

    result = evaluate_headline_relevance(headline)

    # Extract guess safely
    guess = result.get("score", "?")
    reasoning = result.get("reason", "")

    # Determine correctness category
    if guess == true_class:
        error_type = "CORRECT"
        correct += 1
    else:
        # Compute numeric class difference
        if guess in class_map and true_class in class_map:
            diff = abs(class_map[guess] - class_map[true_class])
        else:
            diff = None

        # Categorize errors
        if diff is None:
            error_type = "UNKNOWN"
        elif diff >= 2:
            error_type = "BIG WRONG"
            big_wrongs += 1
        else:
            error_type = "SMALL WRONG"
            small_wrongs += 1

    # Print nicely formatted row
    print(f"{guess:<6} | {true_class:<6} | {error_type:<12} | {headline} | {reasoning}")

# --- Summary ---
total = len(df)
accuracy = correct / total * 100

print("\n" + "-" * 160)
print(f"Correct:      {correct}/{total}   ({accuracy:.2f}%)")
print(f"Small wrongs: {small_wrongs}")
print(f"Big wrongs:   {big_wrongs}")
print("-" * 160)