import copy
from typing import List, Optional

import gpytorch
import linear_operator
import torch

from ..config import logger
from .callbacks import (
    Callback,
    CallbackOnEpochEndContext,
    CallbackOnEpochStartContext,
    CallbackOnTrainEndContext,
    CallbackOnTrainStartContext,
)
from .optimizers import LBFGSScipy


class GPTrainerSingleProcess:
    def __init__(
        self,
        model,
        optimizer_class,
        optimizer_kwargs,
        mll_class,
        num_epochs,
        convergence_patience,
        cholesky_jitter: float = 1e-6,
        callbacks: Optional[List[Callback]] = None,
        device: str = None,
        min_loss_change: float = 1e-7,
        scheduler_class: torch.optim.lr_scheduler.LRScheduler = None,
        scheduler_kwargs: dict = None,
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
        self.min_loss_change = min_loss_change
        # Get dtype from the model (which should be set from input data)
        if hasattr(model, "dtype") and model.dtype is not None:
            self.dtype = model.dtype
        else:
            self.dtype = torch.float64
            logger.warning(f"Model has no dtype attribute. Using {self.dtype} as fallback.")
        # Move the model to device and convert to specified dtype
        self.model = self.model.to(self.device, dtype=self.dtype)

        # Update the model's internal training data to be on the same device and dtype
        # This is crucial for GPyTorch models to work correctly
        self.model.set_train_data(
            self.model.train_inputs[0].to(self.device, dtype=self.dtype),
            self.model.train_targets.to(self.device, dtype=self.dtype),
            strict=False,
        )

        # Store training data for easy access
        self.train_x = self.model.train_inputs[0]
        self.train_y = self.model.train_targets
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs

    def train(self):
        """
        Train the GP model with optional gradual jitter decrease.
        """
        # Create an optimizer instance
        optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

        if self.scheduler_class is not None:
            self.scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
        else:
            self.scheduler = None

        # Create mll instance
        mll = self.mll_class(self.model.likelihood, self.model)

        # Determine which training function to use based on optimizer type
        if isinstance(optimizer, torch.optim.LBFGS):
            train_epoch = self._train_lbfgs_epoch
        elif isinstance(optimizer, LBFGSScipy):
            train_epoch = self._train_scipy_lbfgs_epoch
        else:
            train_epoch = self._train_standard_epoch

        # Local variables for early stopping
        best_loss = float("inf")
        best_state_dict = None
        no_improvement_epochs = 0
        previous_loss = None

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

        # Set jitter in both gpytorch and linear_operator settings
        # Set the same jitter for both float32 and float64 to ensure consistent behavior
        with (
            gpytorch.settings.cholesky_jitter(self.cholesky_jitter),
            linear_operator.settings.cholesky_jitter(
                float_value=self.cholesky_jitter, double_value=self.cholesky_jitter
            ),
        ):
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
                    cb.on_epoch_start(ctx)

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

                # Check for early stopping conditions
                early_stop_triggered = False
                early_stop_reason = ""

                # Condition 1: No improvement for convergence_patience epochs
                if self.convergence_patience is not None and no_improvement_epochs >= self.convergence_patience:
                    early_stop_triggered = True
                    early_stop_reason = f"No improvement for {self.convergence_patience} epochs"

                # Condition 2: Absolute loss change is below threshold (OR condition)
                if previous_loss is not None:
                    loss_change = abs(previous_loss - loss)
                    if loss_change < self.min_loss_change:
                        early_stop_triggered = True
                        if early_stop_reason:
                            early_stop_reason += f" OR absolute loss change below {self.min_loss_change:.1e}"
                        else:
                            early_stop_reason = f"absolute loss change below {self.min_loss_change:.1e}"

                if early_stop_triggered:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"Reason: {early_stop_reason}. Best loss: {best_loss:.6f}"
                    )
                    break  # Stop training

                # Update previous_loss for next iteration
                previous_loss = loss

        # Log training completion
        logger.info(f"Training completed. Best loss: {best_loss:.6f}")
        logger.info(f"Total epochs trained: {epoch + 1}")

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
            cb.on_train_end(ctx)

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
        # Ensure training data is in correct dtype
        train_x = self.train_x.to(dtype=self.dtype)
        train_y = self.train_y.to(dtype=self.dtype)
        output = self.model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    def _train_lbfgs_epoch(self, optimizer, mll):
        """
        Train the model for a single epoch using LBFGS optimizer.

        Parameters:
            optimizer: The LBFGS optimizer.
            mll: Marginal Log Likelihood loss.
            epoch: Current epoch number (unused but kept for consistency).

        Returns:
            float: The loss value after training for one epoch.
        """
        # Get the closure function
        closure = self._lbfgs_closure(optimizer, mll)
        # Perform the optimizer step using the closure
        loss = optimizer.step(closure)
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    def _train_scipy_lbfgs_epoch(self, optimizer, mll):
        """
        Train the model for a single epoch using Scipy LBFGS optimizer.
        Note: cholesky_jitter context is set in the main training loop.

        Parameters:
            optimizer: The Scipy LBFGS optimizer.
            mll: Marginal Log Likelihood loss.

        Returns:
            float: The loss value after training for one epoch.
        """
        # Get the closure function for scipy LBFGS
        closure = self._lbfgs_closure(optimizer, mll)
        # Perform the optimizer step using the closure
        optimizer.step(closure)
        # The loss is stored in optimizer._last_loss after step()
        loss = optimizer._last_loss
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    def _lbfgs_closure(self, optimizer, mll):
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

            # Ensure training data is on the same device as the model
            model_device = next(self.model.parameters()).device

            # Ensure training data is in correct dtype
            train_x = self.train_x.to(dtype=self.dtype, device=model_device)
            train_y = self.train_y.to(dtype=self.dtype, device=model_device)

            output = self.model(train_x)

            loss = -mll(output, train_y)

            loss.backward()

            return loss.detach()

        return closure
