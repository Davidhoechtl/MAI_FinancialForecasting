from enum import Enum

class EvaluationMode(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2

class ImpactModel(Enum):
    NONE = 1,
    LLAMA_3_1_Instruct = 2,
    GPT_OSS_20B = 3