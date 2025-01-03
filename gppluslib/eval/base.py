from abc import ABC, abstractmethod
import torch

from gppluslib.models.gpr import GPR

class BaseEvaluator(torch.nn.Module, ABC):
    """
    Evaluator base class.
    """

    def __init__(
        self,
        model: GPR
    ):
        super().__init__()
        
        self.model = model

    @abstractmethod
    def evaluate(self, x: torch.Tensor):
        raise NotImplementedError