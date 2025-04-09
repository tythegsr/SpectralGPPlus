import copy
from typing import List, Optional

import gpytorch
import torch

from ..config import logger
from .callbacks import (
    Callback,
    CallbackOnEpochEndContext,
    CallbackOnEpochStartContext,
    CallbackOnTrainEndContext,
    CallbackOnTrainStartContext,
)


class GPTrainerSingleProcess:
    def __init__(
        self,
        model,
        optimizer_class,
        optimizer_kwargs,
        mll_class,
        num_epochs,
        convergence_patience,
        cholesky_jitter: float,
        callbacks: Optional[List[Callback]] = None,
        device: str = None,
    ):
        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.mll_class = mll_class
        self.num_epochs = num_epochs
        self.convergence_patience = convergence_patience
        self.cholesky_jitter = cholesky_jitter
        self.callbacks = callbacks or []
        self.device = device
        # Move only the training data to device
        # (assuming the model has train_inputs[0], train_targets)
        # If you have multiple train_inputs, adapt accordingly.
        self.train_x = self.model.train_inputs[0].to(self.device)
        self.train_y = self.model.train_targets.to(self.device)

    def train(self):
        """
        Train the GP model.
        """
        # Create an optimizer instance
        optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)
        # Create mll instance
        mll = self.mll_class(self.model.likelihood, self.model)

        if isinstance(optimizer, torch.optim.LBFGS):
            train_epoch = self._train_lbfgs_epoch
        else:
            train_epoch = self._train_standard_epoch

        # Local variables for early stopping
        best_loss = float("inf")
        best_state_dict = None
        no_improvement_epochs = 0

        # ---------------------------
        # on_train_start
        # ---------------------------
        ctx: CallbackOnTrainStartContext = {
            "model": self.model,
            "trainer": self,
            "device": self.device,
        }
        for cb in self.callbacks:
            cb.on_train_start(ctx)

        with gpytorch.settings.cholesky_jitter(self.cholesky_jitter):
            # Set the model to training mode
            self.model.train()

            logger.info(f"Starting training for {self.num_epochs} epochs.")

            for epoch in range(self.num_epochs):
                # ---------------------------
                # on_epoch_start
                # ---------------------------
                ctx: CallbackOnEpochStartContext = {
                    "epoch": epoch,
                    "model": self.model,
                    "trainer": self,
                    "device": self.device,
                }
                for cb in self.callbacks:
                    cb.on_epoch_start(self, ctx)

                # Train for a single epoch
                loss = train_epoch(optimizer, mll)

                # ---------------------------
                # on_epoch_end
                # ---------------------------
                ctx: CallbackOnEpochEndContext = {
                    "epoch": epoch,
                    "model": self.model,
                    "trainer": self,
                    "loss": loss,
                    "device": self.device,
                }
                for cb in self.callbacks:
                    cb.on_epoch_end(ctx)

                # Update best-loss and best-state tracking
                if loss < best_loss:
                    best_loss = loss
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1

                # Check for early stopping
                if self.convergence_patience is not None and no_improvement_epochs >= self.convergence_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}. Best loss: {best_loss}")
                    break  # Stop training

        # ---------------------------
        # on_train_end
        # ---------------------------
        ctx: CallbackOnTrainEndContext = {
            "epoch": epoch,
            "model": self.model,
            "trainer": self,
            "best_loss": best_loss,
            "best_state_dict": best_state_dict,
            "device": self.device,
        }
        for cb in self.callbacks:
            cb.on_train_end(self, self.model, best_loss, best_state_dict)

        return {"loss": best_loss, "state_dict": best_state_dict}

    def _train_standard_epoch(self, optimizer, mll):
        """
        Train the model for a single epoch with standard optimizers.

        Parameters:
            model: The Gaussian Process model being trained.
            optimizer: The LBFGS optimizer.
            mll: Marginal Log Likelihood loss.

        Returns:
            loss (float): The loss value after training for one epoch.
        """
        optimizer.zero_grad()
        output = self.model(self.train_x)
        loss = -mll(output, self.train_y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _train_lbfgs_epoch(self, optimizer, mll):
        """
        Train the model for a single epoch using LBFGS optimizer.

        Parameters:
            model: The Gaussian Process model being trained.
            optimizer: The LBFGS optimizer.
            mll: Marginal Log Likelihood loss.

        Returns:
            float: The loss value after training for one epoch.
        """
        # Get the closure function
        closure = self._lbfgs_closure(self.model, optimizer, mll)
        # Perform the optimizer step using the closure
        loss = optimizer.step(closure)
        return loss.item()

    def _lbfgs_closure(self, model, optimizer, mll):
        """
        Defines the closure for LBFGS optimizer.
        This method is reused across LBFGS training epochs.

        Parameters:
            model: The Gaussian Process model being trained.
            optimizer: The LBFGS optimizer.
            mll: Marginal Log Likelihood loss.

        Returns:
            Callable: The closure function.
        """

        def closure():
            optimizer.zero_grad()
            output = model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            return loss

        return closure
