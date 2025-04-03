from abc import ABC, abstractmethod

import torch

from gpplus.models.gpr import GPR


class Loss(torch.nn.Module, ABC):
    """
    Loss base class.
    """

    def __init__(
        self,
        model: GPR,
    ):
        super().__init__()

        self.model = model

    @abstractmethod
    def forward(self) -> torch.Tensor:
        raise NotImplementedError
