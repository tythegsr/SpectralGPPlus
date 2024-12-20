import torch

from gppluslib.models.gpr import GPR
from gppluslib.eval.base import BaseEvaluator

class Evaluator(BaseEvaluator):
    """
    Evaluator class.
    """

    def __init__(
        self,
        model: GPR
    ):
        super().__init__(model)

    def evaluate(self, x: torch.Tensor):
        return self.model.predict(x, return_std=True, include_noise=True)