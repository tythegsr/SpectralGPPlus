from abc import ABC
from typing import Any, TypedDict


class CallbackOnEpochStartContext(TypedDict):
    epoch: int
    model: Any
    trainer: Any
    device: str


class CallbackOnEpochEndContext(TypedDict):
    epoch: int
    model: Any
    trainer: Any
    loss: float
    device: str


class CallbackOnTrainStartContext(TypedDict):
    model: Any
    trainer: Any
    device: str


class CallbackOnTrainEndContext(TypedDict):
    epoch: int
    model: Any
    trainer: Any
    best_loss: float
    best_state_dict: Any
    device: str


class Callback(ABC):
    def on_epoch_start(self, context: CallbackOnEpochStartContext):
        """
        Called at the start of each epoch during training.

        Args:
            context (dict): A dictionary containing training state info.
        """
        pass

    def on_epoch_end(self, context: CallbackOnEpochEndContext):
        """
        Called at the end of each epoch during training.

        Args:
            context (dict): A dictionary containing training state info.
        """
        pass

    def on_train_start(self, context: CallbackOnTrainStartContext):
        """
        Called at the start of each training.

        Args:
            context (dict): A dictionary containing training state info.
        """
        pass

    def on_train_end(self, context: CallbackOnTrainEndContext):
        """
        Called at the end of each training.

        Args:
            context (dict): A dictionary containing training state info.
        """
        pass


class PrintLossCallback(Callback):
    def on_epoch_end(self, context: dict):
        print(f"Epoch {context['epoch']} - Loss: {context['loss']:.4f}")
        