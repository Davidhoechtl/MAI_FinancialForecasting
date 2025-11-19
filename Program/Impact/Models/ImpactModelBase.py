from abc import ABC, abstractmethod
from llama_cpp import Llama
from pathlib import Path

from Impact.ImpactScoreAnalyzerEnums import EvaluationMode, ImpactModel

class ImpactModelFactoryBase(ABC):
    @abstractmethod
    def create(self) -> Llama:
        """
        Create and return an instance of the impact model.
        """
        pass

    @abstractmethod
    def get_impact_file_path(self, eval_mode: EvaluationMode) -> Path:
        """
        Get the file path for storing/loading the impact map.
        """
        pass

    @abstractmethod
    def get_model_type(self) -> ImpactModel:
        """
        Get the type/name of the impact model.
        """
        pass

    def format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        Format the prompt for the specific model, if needed.
        Default implementation returns the user prompt as is.
        """
        # Cleaner formatting that avoids sticking extra indentation into the string
        return (
            f"{system_prompt}\n"
            f"{user_prompt}"
        )
