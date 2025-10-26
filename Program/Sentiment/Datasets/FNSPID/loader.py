from datasets import load_dataset, Features, Value, config
# config.USE_PARQUET = False
# config.USE_ARROW_EXTENSION_ARRAYS = False

# Load first few lines of the CSV (streaming skips Arrow)
# ds = load_dataset("Zihan1004/FNSPID", split="train", streaming=True)
# first = next(iter(ds))
# print(first.keys())


features = Features({
    "Date": Value("string"),
    "Article_title": Value("string"),
    "Stock_symbol": Value("string"),
    "Url": Value("string"),
    "Publisher": Value("string"),
    "Author": Value("string"),
    "Article": Value("string"),
    "Lsa_summary": Value("string"),
    "Luhn_summary": Value("string"),
    "Textrank_summary": Value("string"),
    "Lexrank_summary": Value("string"),
})

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("Zihan1004/FNSPID", split="train", features=features)

