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
        map_prior: bool = False,
        callbacks: Optional[List[Callback]] = None,
        device: str = None,
        track_loocv: bool = True,
        loocv_log_freq: int = 50,
        use_loocv_objective: bool = False,
        min_loss_change: float = 1e-7,
        dtype: torch.dtype = torch.float32,
        scheduler_class: torch.optim.lr_scheduler._LRScheduler = None,
        scheduler_kwargs: dict = None,
        use_gradual_jitter: bool = True,
        jitter_start: float = 1e-3,
        jitter_end: float = 1e-6,
        jitter_schedule: str = "linear",  # "linear", "exponential", "cosine"
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
        self.dtype = dtype
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
        self.map_prior = map_prior
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs

        # Gradual jitter parameters
        self.use_gradual_jitter = use_gradual_jitter
        self.jitter_start = jitter_start
        self.jitter_end = jitter_end
        self.jitter_schedule = jitter_schedule

        # Ensure multifidelity likelihood knows true sample order
        try:
            if hasattr(self.model, "likelihood") and hasattr(self.model.likelihood, "set_training_data"):
                self.model.likelihood.set_training_data(self.train_x)
        except Exception:
            logger.warning("Likelihood does not have set_training_data method")

        # Track LOOCV for comparison
        self.track_loocv = track_loocv
        self.loocv_log_freq = loocv_log_freq
        self.use_loocv_objective = use_loocv_objective

    def _compute_jitter(self, epoch: int):
        """
        Compute the jitter value for the current epoch.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            Scalar value in the model's dtype: Jitter value for this epoch
        """
        if not self.use_gradual_jitter:
            return self.cholesky_jitter

        if self.num_epochs <= 1:
            return self.jitter_start

        # Progress from 0 to 1 over all epochs
        progress = torch.tensor(epoch / (self.num_epochs - 1), dtype=self.dtype)
        progress = torch.clamp(progress, 0.0, 1.0)  # Clamp to [0, 1]

        if self.jitter_schedule == "linear":
            # Linear interpolation
            jitter = self.jitter_start + progress * (self.jitter_end - self.jitter_start)

        elif self.jitter_schedule == "exponential":
            # Exponential decay (log-linear in jitter space)
            log_start = torch.log(torch.tensor(self.jitter_start, dtype=self.dtype))
            log_end = torch.log(torch.tensor(self.jitter_end, dtype=self.dtype))
            log_jitter = log_start + progress * (log_end - log_start)
            jitter = torch.exp(log_jitter)

        elif self.jitter_schedule == "cosine":
            # Cosine annealing (smooth start, accelerates in middle, smooth end)
            pi = torch.tensor(torch.pi, dtype=self.dtype)
            cosine_progress = 0.5 * (1 + torch.cos(pi * progress))
            jitter = self.jitter_end + (self.jitter_start - self.jitter_end) * cosine_progress

        else:
            raise ValueError(f"Unknown jitter schedule: {self.jitter_schedule}")

        return jitter.item()

    def calculate_both_likelihoods(self, mll):
        """
        Calculate both MLL and LOOCV likelihoods simultaneously for comparison.

        Args:
            mll: Marginal log likelihood object

        Returns:
            dict: Contains both MLL and LOOCV likelihoods
        """
        try:
            # Calculate MLL loss
            mll_loss = mll(self.model(self.train_x), self.train_y)

            # Calculate LOOCV loss using GPyTorch's built-in class
            loocv_mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(mll.likelihood, self.model)
            loocv_loss = -loocv_mll(self.model(self.train_x), self.train_y)

            return {
                "mll_loss": mll_loss.item(),
                "mll_likelihood": -mll_loss.item(),
                "loocv_loss": loocv_loss.item(),
                "loocv_likelihood": -loocv_loss.item(),
                "difference": mll_loss.item() - loocv_loss.item(),  # MLL - LOOCV
            }

        except Exception as e:
            logger.warning(f"LOOCV calculation failed: {e}")
            # Return MLL only if LOOCV fails, but still continue tracking
            try:
                mll_loss = mll(self.model(self.train_x), self.train_y)
                return {
                    "mll_loss": mll_loss.item(),
                    "mll_likelihood": -mll_loss.item(),
                    "loocv_loss": float("nan"),  # Mark as NaN when LOOCV fails
                    "loocv_likelihood": float("nan"),
                    "difference": float("nan"),
                }
            except Exception as e2:
                logger.warning(f"MLL calculation also failed: {e2}")
                return None

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

        # Create mll instance
        if self.use_loocv_objective:
            # Use LOOCV as the training objective
            mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(self.model.likelihood, self.model)
            logger.info("Using LOOCV as training objective")
        else:
            # Use standard MLL
            mll = self.mll_class(self.model.likelihood, self.model)
            logger.info("Using standard MLL as training objective")

        # Local variables for early stopping
        best_loss = float("inf")
        best_state_dict = None
        no_improvement_epochs = 0
        previous_loss = None

        # Traces for logging/return
        loss_trace: List[float] = []
        loocv_trace: List[dict] = []

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

        # Set the model to training mode
        self.model.train()

        if self.use_gradual_jitter:
            logger.info(f"Starting training for {self.num_epochs} epochs with gradual jitter decrease.")
            logger.info(
                f"Jitter schedule: {self.jitter_schedule}, from {self.jitter_start:.2e} to {self.jitter_end:.2e}"
            )
        else:
            logger.info(f"Starting training for {self.num_epochs} epochs with fixed jitter {self.cholesky_jitter:.2e}.")

        for epoch in range(self.num_epochs):
            # Compute jitter for this epoch
            current_jitter = self._compute_jitter(epoch)

            # Log jitter changes periodically
            if self.use_gradual_jitter and (
                epoch == 0 or epoch % max(1, self.num_epochs // 100) == 0 or epoch == self.num_epochs - 1
            ):
                logger.info(f"Epoch {epoch}: jitter = {current_jitter:.2e}")
            # --- DIAGNOSTIC: Print optimizer parameter names and check for raw_lengthscales ---
            # print("[DIAGNOSTIC][OPTIMIZER] Parameter names and requires_grad status:")
            # param_id_to_name = {}
            # for name, param in self.model.named_parameters():
            #     print(f"  {name}: requires_grad={param.requires_grad}, id={id(param)}")
            #     param_id_to_name[id(param)] = name
            # print("[DIAGNOSTIC][OPTIMIZER] Parameters in optimizer:")
            # for i, group in enumerate(optimizer.param_groups):
            #     print(f"  Param group {i}:")
            #     for param in group['params']:
            #         pname = param_id_to_name.get(id(param), "<unnamed>")
            #         print(f"    {pname}: requires_grad={param.requires_grad}, id={id(param)}")

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

            # Train for a single epoch with current jitter
            with gpytorch.settings.cholesky_jitter(current_jitter):
                loss = train_epoch(optimizer, mll)
            loss_trace.append(float(loss))

            # Calculate both MLL and LOOCV every N epochs
            likelihood_data = None
            if self.track_loocv and epoch % self.loocv_log_freq == 0:
                likelihood_data = self.calculate_both_likelihoods(mll)
                if likelihood_data is not None:
                    loocv_trace.append(
                        {
                            "epoch": int(epoch),
                            "mll_loss": float(likelihood_data.get("mll_loss", float("nan"))),
                            "loocv_loss": float(likelihood_data.get("loocv_loss", float("nan"))),
                        }
                    )
                else:
                    # Even if calculation fails, add an entry with NaN to maintain trace continuity
                    loocv_trace.append(
                        {
                            "epoch": int(epoch),
                            "mll_loss": float("nan"),
                            "loocv_loss": float("nan"),
                        }
                    )

            if epoch % 500 == 0:
                # Log epoch and loss using the existing logger
                log_msg = f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {loss:.6f}"
                if likelihood_data is not None:
                    log_msg += f"\n  MLL: {likelihood_data['mll_likelihood']:.6f}, "
                    log_msg += f"LOOCV: {likelihood_data['loocv_likelihood']:.6f}, "
                    log_msg += f"Diff: {likelihood_data['difference']:.6f}"

                    # Add overfitting indicator
                    if likelihood_data["difference"] > 0.5:
                        log_msg += " ⚠️ (Overfitting risk)"
                    elif likelihood_data["difference"] < -0.1:
                        log_msg += " ✓ (Good generalization)"
                    else:
                        log_msg += " ✓ (Well-aligned)"
                logger.info(log_msg)

            # ---------------------------
            # on_epoch_end
            # ---------------------------
            ctx: CallbackOnEpochEndContext = {
                "epoch": epoch,
                "model": self.model,
                "trainer": self,
                "loss": loss,
                "likelihood_data": likelihood_data,
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

        # Final training LOOCV loss (last computed)
        last_loocv = None
        if loocv_trace:
            last_loocv = float(loocv_trace[-1].get("loocv_loss", float("nan")))

        # Extract validation trace from callbacks if available (for plotting/saving outside trainer)
        validation_trace = None
        for cb in self.callbacks:
            if hasattr(cb, "get_validation_data"):
                try:
                    validation_trace = cb.get_validation_data()
                    break
                except Exception:
                    validation_trace = None

        return {
            "loss": best_loss,
            "state_dict": best_state_dict,
            "loss_trace": loss_trace,
            "loocv_trace": loocv_trace,
            "train_loocv_loss": last_loocv,
            "validation_trace": validation_trace,
        }

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

            return loss

        return closure
