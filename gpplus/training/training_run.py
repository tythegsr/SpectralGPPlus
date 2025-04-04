import copy
from typing import List, Optional

import gpytorch
import torch

from ..config import logger
from .callbacks import Callback, CallbackEpochContext
from .parameter_initializer import ParameterInitializer


class GPTrainingRun:
    def __init__(
        self,
        model,
        sobol_sample,
        optimizer_class,
        optimizer_kwargs,
        mll_class,
        num_epochs,
        convergence_patience,
        callbacks: Optional[List[Callback]] = None,
        device: str = None,
    ):
        self.model = model
        self.train_x = self.model.train_inputs[0]
        self.train_y = self.model.train_targets
        self.sobol_sample = sobol_sample
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.mll_class = mll_class
        self.num_epochs = num_epochs
        self.convergence_patience = convergence_patience
        self.callbacks = callbacks or []
        self.device = device

    def train(self):
        """
        Train the GP model.
        """
        ParameterInitializer.initialize_parameters(self.model, self.sobol_sample)

        # Create an optimizer instance
        optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)
        # Create mll instance
        mll = self.mll_class(self.model.likelihood, self.model).to(self.device)

        if isinstance(optimizer, torch.optim.LBFGS):
            train_epoch = self._train_lbfgs_epoch
        else:
            train_epoch = self._train_standard_epoch

        # Local variables for early stopping
        best_loss = float("inf")
        best_state_dict = None
        no_improvement_epochs = 0

        with gpytorch.settings.cholesky_jitter(1e-6):
            # Set the model to training mode
            self.model.train()
            self.model.likelihood.train()

            for epoch in range(self.num_epochs):
                # Train for a single epoch
                loss = train_epoch(optimizer, mll)

                # Epoch callbacks
                ctx: CallbackEpochContext = {
                    "epoch": epoch,
                    "loss": loss,
                    "model": self.model,
                    "trainer": self,
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
