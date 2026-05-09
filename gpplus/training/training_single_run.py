import copy
from typing import List, Optional

import gpytorch
import linear_operator
import torch

try:
    from linear_operator.utils.errors import NotPSDError
except ImportError:
    NotPSDError = Exception

from ..config import logger
from .callbacks import (
    Callback,
    CallbackOnEpochEndContext,
    CallbackOnEpochStartContext,
    CallbackOnTrainEndContext,
    CallbackOnTrainStartContext,
)
from .optimizers import LBFGSScipy
from .stop_conditions import (
    ConvergencePatienceStopCondition,
    MinLossChangeStopCondition,
    StopCondition,
    StopConditionContext,
)


class GPTrainerSingleProcess:
    def __init__(
        self,
        model,
        optimizer_class,
        optimizer_kwargs,
        num_epochs: int,
        mll_class: gpytorch.mlls.MarginalLogLikelihood = None,
        cholesky_jitter: float = 1e-6,
        callbacks: Optional[List[Callback]] = None,
        device: str | torch.device | None = None,
        scheduler_class: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        scheduler_kwargs: Optional[dict] = None,
        stop_conditions: Optional[List[StopCondition]] = None,
        min_epochs: int = 0,
    ):
        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.mll_class = mll_class
        self.num_epochs = num_epochs
        self.cholesky_jitter = cholesky_jitter
        self.min_epochs = min_epochs
        self.callbacks = callbacks or []
        self.device = device

        if stop_conditions is None:
            self.stop_conditions = [
                ConvergencePatienceStopCondition(patience=20),
                MinLossChangeStopCondition(min_loss_change=1e-7),
            ]
        else:
            self.stop_conditions = stop_conditions

        if hasattr(model, "dtype") and model.dtype is not None:
            self.dtype = model.dtype
        else:
            self.dtype = torch.float64
            logger.warning("Model has no dtype attribute. Using %s as fallback.", self.dtype)

        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.set_train_data(
            self.model.train_inputs[0].to(self.device, dtype=self.dtype),
            self.model.train_targets.to(self.device, dtype=self.dtype),
            strict=False,
        )

        self.train_x = self.model.train_inputs[0]
        self.train_y = self.model.train_targets
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or {}

    def train(self):
        """
        Train the GP model with optional gradual jitter increase on NotPSDError.
        """
        optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

        if self.scheduler_class is not None:
            self.scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
        else:
            self.scheduler = None

        mll = self.mll_class(self.model.likelihood, self.model)

        if isinstance(optimizer, torch.optim.LBFGS):
            train_epoch = self._train_lbfgs_epoch
        elif isinstance(optimizer, LBFGSScipy):
            train_epoch = self._train_scipy_lbfgs_epoch
        else:
            train_epoch = self._train_standard_epoch

        best_loss = float("inf")
        best_state_dict = None
        no_improvement_epochs = 0
        previous_loss = None

        # ---------------------------
        # on_train_start
        # ---------------------------
        ctx_start: CallbackOnTrainStartContext = {
            "model": self.model,
            "trainer": self,
            "device": str(self.device),
        }
        for cb in self.callbacks:
            cb.on_train_start(ctx_start)

        if isinstance(optimizer, LBFGSScipy):
            for cb in self.callbacks:
                if hasattr(cb, "register_with_optimizer"):
                    cb.register_with_optimizer(optimizer, model=self.model, trainer=self)

        run_jitter = float(self.cholesky_jitter)
        max_jitter = 1e-3
        self.current_jitter = run_jitter
        self.model.train()
        logger.info("Starting training for %d epochs.", self.num_epochs)

        for epoch in range(self.num_epochs):
            ctx_epoch_start: CallbackOnEpochStartContext = {
                "epoch": epoch,
                "model": self.model,
                "trainer": self,
                "device": str(self.device),
            }
            for cb in self.callbacks:
                cb.on_epoch_start(ctx_epoch_start)

            self._current_epoch = epoch
            epoch_successful = False
            while not epoch_successful:
                self._current_run_jitter = run_jitter
                self.current_jitter = run_jitter
                with (
                    gpytorch.settings.cholesky_jitter(run_jitter),
                    linear_operator.settings.cholesky_jitter(float_value=run_jitter, double_value=run_jitter),
                ):
                    try:
                        loss = train_epoch(optimizer, mll)
                        epoch_successful = True
                    except NotPSDError:
                        if run_jitter < max_jitter:
                            run_jitter = min(run_jitter * 10.0, max_jitter)
                            self._current_run_jitter = run_jitter
                            self.current_jitter = run_jitter
                            logger.warning("NotPSDError detected. Increasing jitter to %.1e.", run_jitter)
                        else:
                            logger.warning("NotPSDError persists with jitter=%.1e. Re-raising.", run_jitter)
                            raise
                    except (RuntimeError, ValueError) as exc:
                        err_str = str(exc).lower()
                        if "notpsd" in err_str or "not p.d." in err_str or "not positive definite" in err_str:
                            if run_jitter < max_jitter:
                                run_jitter = min(run_jitter * 10.0, max_jitter)
                                self._current_run_jitter = run_jitter
                                self.current_jitter = run_jitter
                                logger.warning("NotPSD error detected. Increasing jitter to %.1e.", run_jitter)
                            else:
                                raise
                        else:
                            raise

            ctx_epoch_end: CallbackOnEpochEndContext = {
                "epoch": epoch,
                "model": self.model,
                "trainer": self,
                "loss": loss,
                "device": str(self.device),
                "jitter": run_jitter,
            }
            for cb in self.callbacks:
                cb.on_epoch_end(ctx_epoch_end)

            if loss < best_loss:
                best_loss = loss
                best_state_dict = copy.deepcopy(self.model.state_dict())
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            stop_context: StopConditionContext = {
                "epoch": epoch,
                "model": self.model,
                "trainer": self,
                "loss": loss,
                "previous_loss": previous_loss,
                "best_loss": best_loss,
                "no_improvement_epochs": no_improvement_epochs,
                "device": str(self.device),
            }

            early_stop_triggered = False
            early_stop_reasons = []
            if (epoch + 1) >= self.min_epochs:
                for stop_condition in self.stop_conditions:
                    should_stop, reason = stop_condition.should_stop(stop_context)
                    if should_stop:
                        early_stop_triggered = True
                        if reason:
                            early_stop_reasons.append(reason)

            if early_stop_triggered:
                early_stop_reason = " OR ".join(early_stop_reasons) if early_stop_reasons else "Stop condition met"
                logger.info(
                    "Early stopping triggered at epoch %d. Reason: %s. Best loss: %.6f",
                    epoch + 1,
                    early_stop_reason,
                    best_loss,
                )
                break
            previous_loss = loss

        logger.info("Training completed. Best loss: %.6f", best_loss)
        logger.info("Total epochs trained: %d", epoch + 1)
        self.model.cholesky_jitter = run_jitter

        ctx_end: CallbackOnTrainEndContext = {
            "epoch": epoch,
            "model": self.model,
            "trainer": self,
            "best_loss": best_loss,
            "best_state_dict": best_state_dict,
            "device": str(self.device),
        }
        lbfgs_stop_reason = getattr(self, "_lbfgs_stop_reason", None)
        if lbfgs_stop_reason is not None:
            ctx_end["lbfgs_stop_reason"] = lbfgs_stop_reason
        jitter_max = getattr(self, "jitter_max", None)
        if jitter_max is not None:
            ctx_end["jitter_max"] = jitter_max
        for cb in self.callbacks:
            cb.on_train_end(ctx_end)

        callback_data = {}
        for cb in self.callbacks:
            if hasattr(cb, "get_stored_parameters"):
                cb_name = cb.__class__.__name__
                stored_params = cb.get_stored_parameters()
                if stored_params:
                    callback_data[cb_name] = stored_params

        out = {
            "loss": best_loss,
            "state_dict": best_state_dict,
            "callback_data": callback_data,
            "cholesky_jitter": run_jitter,
        }
        return out

    def _train_standard_epoch(self, optimizer, mll):
        """
        Train the model for a single epoch with standard optimizers.
        """
        optimizer.zero_grad()
        train_x = self.train_x.to(dtype=self.dtype)
        train_y = self.train_y.to(dtype=self.dtype)
        output = self.model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return float(loss.item())

    def _train_lbfgs_epoch(self, optimizer, mll):
        """
        Train the model for a single epoch using torch.optim.LBFGS.
        """
        closure = self._lbfgs_closure(optimizer, mll)
        loss = optimizer.step(closure)
        if self.scheduler is not None:
            self.scheduler.step()

        return float(loss.item())

    def _train_scipy_lbfgs_epoch(self, optimizer, mll):
        """
        Train the model for a single epoch using LBFGSScipy.
        """
        closure = self._lbfgs_closure(optimizer, mll)
        optimizer.step(closure)
        loss = optimizer._last_loss
        self._lbfgs_stop_reason = getattr(optimizer, "_lbfgs_stop_reason", None)
        if self.scheduler is not None:
            self.scheduler.step()
        return float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)

    def _lbfgs_closure(self, optimizer, mll):
        """
        Defines the closure for LBFGS-style optimizers.
        """

        def closure():
            optimizer.zero_grad()
            model_device = next(self.model.parameters()).device
            train_x = self.train_x.to(dtype=self.dtype, device=model_device)
            train_y = self.train_y.to(dtype=self.dtype, device=model_device)
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            return loss

        return closure
