import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from gpplus.models.gpr import GPR
from gpplus.loss.base import Loss

class ExactMarginalLogLikelihoodLoss(Loss):
    """
    Exact Marginal Log Likelihood Loss class.
    """

    def __init__(
        self,
        model: GPR,
    ):

        super().__init__(model)

        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def forward(self) -> torch.Tensor:
        """
        Exact Marginal Log Likelihood forward pas.
        """
        # output from model
        output = self.model(*self.model.train_inputs)
        # calculate loss and backprop gradients
        loss = -self.mll(output, self.model.train_targets)
        return loss