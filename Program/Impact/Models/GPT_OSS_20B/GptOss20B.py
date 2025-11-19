from Impact.ImpactScoreAnalyzerEnums import EvaluationMode, ImpactModel
from Impact.Models.ImpactModelBase import ImpactModelFactoryBase
from pathlib import Path
from llama_cpp import Llama
import huggingface_hub as hf_hub

N_CTX = 1024

BASE_PATH = Path(__file__).resolve().parent
IMPACT_MAP_FILE_REGRESSION = BASE_PATH / "impact_map_regression.csv"
IMPACT_MAP_FILE_CLASSIFICATION = BASE_PATH / "impact_map_classification.csv"

MODEL_REPO="unsloth/gpt-oss-20b-GGUF"
MODEL_FILE="gpt-oss-20b-Q8_0.gguf"

class GptOss20B(ImpactModelFactoryBase):
    def __init__(self):
        super().__init__()

    def create(self):
        print("🔹 Downloading model from Hugging Face ...")
        local_path = hf_hub.hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            repo_type="model",
        )
        print(f"✅ Model downloaded to: {local_path}")

        # === Step 2. Load model from the local GGUF path ===
        llm = Llama(
            model_path=local_path,
            n_ctx=N_CTX,
            n_gpu_layers=-1,  # full GPU offload if possible
            verbose=False,
        )
        print(llm("Say Hello!", max_tokens=10))
        return llm

    def get_impact_file_path(self, eval_mode: EvaluationMode) -> Path:
        return IMPACT_MAP_FILE_REGRESSION if eval_mode == EvaluationMode.REGRESSION else IMPACT_MAP_FILE_CLASSIFICATION

    def get_model_type(self) -> ImpactModel:
        return ImpactModel.GPT_OSS_20B