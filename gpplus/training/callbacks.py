from abc import ABC, abstractmethod
from typing import Any, TypedDict


class CallbackEpochContext(TypedDict):
    epoch: int
    loss: float
    model: Any
    trainer: Any
    device: str


class Callback(ABC):
    @abstractmethod
    def on_epoch_end(self, context: CallbackEpochContext):
        """
        Called at the end of each epoch during training.

        Args:
            context (dict): A dictionary containing training state info.
        """
        raise NotImplementedError


class PrintLossCallback(Callback):
    def on_epoch_end(self, context: dict):
        print(f"Epoch {context['epoch']} - Loss: {context['loss']:.4f}")
