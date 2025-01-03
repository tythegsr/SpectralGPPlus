from abc import ABC, abstractmethod
import torch

class BaseTrainer(torch.nn.Module, ABC):
    """
    Trainer base class.
    """

    def __init__(self):
        
        super().__init__()
    
    @abstractmethod
    def fit(self):
        raise NotImplementedError