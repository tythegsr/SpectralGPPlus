import copy
from collections.abc import Callable
from functools import partial
from typing import List, Optional

import gpytorch
import linear_operator
import torch
from torch import nn

from ..config import logger
from .callbacks import Callback
from .optimizers import LBFGSScipy
from .stop_conditions import (
    ConvergencePatienceStopCondition,
    MinLossChangeStopCondition,
    StopCondition,
)
from .lrnn_mll import LRNNWoodburyMarginalLogLikelihood
from .rff_mll import RFFWoodburyMarginalLogLikelihood
from .rff_mt_mll import RFFMTWoodburyMarginalLogLikelihood

_WOODBURY_MLL_TYPES = (
    RFFWoodburyMarginalLogLikelihood,
    RFFMTWoodburyMarginalLogLikelihood,
    LRNNWoodburyMarginalLogLikelihood,
)
from .trainer_utils import (
    SingleRunResult,
    check_early_stop,
    configure_woodbury_matmul_precision,
    select_epoch_train_fn,
)


def _optimizer_lr(optimizer) -> float | None:
    groups = getattr(optimizer, "param_groups", None)
    if not groups:
        return None
    lr = groups[0].get("lr")
    return float(lr) if lr is not None else None


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
        param_groups_fn: Callable[[nn.Module], list[dict]] | None = None,
        stop_conditions: Optional[List[StopCondition]] = None,
        dtype: torch.dtype = torch.float64,
        min_epochs: int = 0,
        run_index: Optional[int] = None,
        num_inits: Optional[int] = None,
    ):
        """
        Initialize a single training run for one model initialization.

        Args:
            model: GP model instance to optimize for this run.
            optimizer_class: Optimizer class used for this run.
            optimizer_kwargs: Optimizer kwargs (without `params`).
            num_epochs: Number of epochs to run for this initialization.
            mll_class: Marginal log likelihood class used as objective.
            cholesky_jitter: Cholesky jitter used during the run.
            callbacks: Optional callback instances invoked during training.
            device: Torch device (or device string) for this run.
            scheduler_class: Optional learning-rate scheduler class.
            scheduler_kwargs: Optional scheduler kwargs.
            param_groups_fn: Optional callable ``model -> Adam param_groups``.
                When set, ignores ``optimizer_kwargs`` for param construction.
            stop_conditions: Optional early-stop conditions for this run.
            dtype: Tensor dtype used in forward/loss computations.
            min_epochs: Minimum epochs before stop conditions can terminate training.
        """
        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.mll_class = mll_class
        self.num_epochs = num_epochs
        self.cholesky_jitter = cholesky_jitter
        self.min_epochs = min_epochs
        self.callbacks = callbacks or []
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.param_groups_fn = param_groups_fn
        self.scheduler = None
        self.dtype = dtype
        self.run_index = run_index
        self.num_inits = num_inits

        if stop_conditions is None:
            self.stop_conditions = [
                ConvergencePatienceStopCondition(patience=20),
                MinLossChangeStopCondition(min_loss_change=1e-7),
            ]
        else:
            self.stop_conditions = stop_conditions

        if not isinstance(self.dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype, got {type(self.dtype).__name__}.")
        if not hasattr(self.model, "train_inputs") or not hasattr(self.model, "train_targets"):
            raise AttributeError("model must expose train_inputs and train_targets before training.")
        self.train_x = self.model.train_inputs[0]
        self.train_y = self.model.train_targets

    def _emit_callbacks(self, hook_name: str, ctx: dict) -> None:
        ctx = {**ctx, "run_index": self.run_index, "num_inits": self.num_inits}
        for cb in self.callbacks:
            getattr(cb, hook_name)(ctx)

    def _negative_mll_loss(self, mll, train_x: torch.Tensor, train_y: torch.Tensor) -> torch.Tensor:
        """MLL loss; skips ExactGP forward when using Woodbury MLL."""
        if isinstance(mll, _WOODBURY_MLL_TYPES):
            return -mll(None, train_y)
        output = self.model(train_x)
        return -mll(output, train_y)

    def _register_optimizer_callbacks(self, optimizer) -> None:
        """Attach optimizer-level callbacks that implement register_with_optimizer."""
        for cb in self.callbacks:
            register = getattr(cb, "register_with_optimizer", None)
            if callable(register):
                register(optimizer, model=self.model, trainer=self)

    def _step_scheduler(self, loss: float) -> None:
        """Step LR scheduler if one was configured."""
        if self.scheduler is None:
            return
        self.scheduler.step()

    def train(self) -> SingleRunResult:
        # Disable TF32 for float32 CUDA Woodbury runs: TF32 Phi^T Phi can
        # corrupt MLL grads and collapse validation while train loss improves.
        if isinstance(self.mll_class, type) and issubclass(self.mll_class, _WOODBURY_MLL_TYPES):
            configure_woodbury_matmul_precision(device=self.device, dtype=self.dtype)
        if self.param_groups_fn is not None:
            optimizer = self.optimizer_class(self.param_groups_fn(self.model))
        else:
            optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)
        if isinstance(optimizer, LBFGSScipy) and self.num_epochs > 1:
            logger.warning(
                "LBFGSScipy performs internal iterations per optimizer step; "
                "prefer num_epochs=1 and tune max_iter/max_eval in optimizer_kwargs."
            )
        if self.scheduler_class is not None:
            self.scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
        else:
            self.scheduler = None

        if isinstance(self.mll_class, type) and issubclass(self.mll_class, _WOODBURY_MLL_TYPES):
            mll = self.mll_class(
                self.model.likelihood, self.model, jitter=self.cholesky_jitter
            )
        else:
            mll = self.mll_class(self.model.likelihood, self.model)
        train_epoch = select_epoch_train_fn(
            optimizer=optimizer,
            standard_epoch_fn=self._train_standard_epoch,
            lbfgs_epoch_fn=self._train_lbfgs_epoch,
            scipy_lbfgs_epoch_fn=self._train_scipy_lbfgs_epoch,
        )

        best_loss = float("inf")
        best_state_dict = None
        no_improvement_epochs = 0
        previous_loss = None
        epochs_trained = 0

        self._emit_callbacks(
            "on_train_start",
            {"model": self.model, "trainer": self, "device": self.device},
        )
        self._register_optimizer_callbacks(optimizer)
        with (
            gpytorch.settings.cholesky_jitter(self.cholesky_jitter),
            linear_operator.settings.cholesky_jitter(
                float_value=self.cholesky_jitter, double_value=self.cholesky_jitter
            ),
        ):
            self.model.train()
            logger.info("Starting training for %s epochs.", self.num_epochs)
            for epoch in range(self.num_epochs):
                epochs_trained = epoch + 1
                self._emit_callbacks(
                    "on_epoch_start",
                    {"epoch": epoch, "model": self.model, "trainer": self, "device": self.device},
                )
                loss = train_epoch(optimizer, mll)
                self._emit_callbacks(
                    "on_epoch_end",
                    {
                        "epoch": epoch,
                        "model": self.model,
                        "trainer": self,
                        "loss": loss,
                        "device": self.device,
                    },
                )
                if loss < best_loss:
                    best_loss = loss
                    best_state_dict = copy.deepcopy(self.model.state_dict())
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1
                stop_context = {
                    "epoch": epoch,
                    "model": self.model,
                    "trainer": self,
                    "loss": loss,
                    "previous_loss": previous_loss,
                    "best_loss": best_loss,
                    "no_improvement_epochs": no_improvement_epochs,
                    "device": self.device,
                    "current_lr": _optimizer_lr(optimizer),
                }
                if check_early_stop(
                    self.stop_conditions,
                    stop_context,
                    epoch,
                    best_loss,
                    min_epochs=self.min_epochs,
                ):
                    break
                previous_loss = loss

        final_lr = _optimizer_lr(optimizer)
        logger.info("Training completed. Best loss: %.6f", best_loss)
        logger.info("Total epochs trained: %s", epochs_trained)
        if final_lr is not None:
            logger.info("Final learning rate: %g", final_lr)
        if best_state_dict is None:
            logger.warning("No model state was captured during training; verify epoch count and optimizer behavior.")

        final_epoch = max(0, epochs_trained - 1)
        self._emit_callbacks(
            "on_train_end",
            {
                "epoch": final_epoch,
                "model": self.model,
                "trainer": self,
                "best_loss": best_loss,
                "best_state_dict": best_state_dict,
                "device": self.device,
            },
        )
        callback_data: dict = {}
        for cb in self.callbacks:
            if hasattr(cb, "get_stored_parameters"):
                stored_params = cb.get_stored_parameters()
                if stored_params:
                    callback_data[cb.__class__.__name__] = stored_params

        result: SingleRunResult = {
            "loss": best_loss,
            "state_dict": best_state_dict,
            "callback_data": callback_data,
        }
        if final_lr is not None:
            result["final_lr"] = final_lr
        return result

    def _train_standard_epoch(self, optimizer, mll) -> float:
        optimizer.zero_grad()
        train_x = self.train_x.to(dtype=self.dtype)
        train_y = self.train_y.to(dtype=self.dtype)
        loss = self._negative_mll_loss(mll, train_x, train_y)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        self._step_scheduler(loss_value)
        return loss_value

    def _train_lbfgs_epoch(self, optimizer, mll) -> float:
        closure = partial(self._lbfgs_step, optimizer=optimizer, mll=mll)
        loss = optimizer.step(closure)
        loss_value = float(loss.item())
        self._step_scheduler(loss_value)
        return loss_value

    def _train_scipy_lbfgs_epoch(self, optimizer, mll) -> float:
        closure = partial(self._lbfgs_step, optimizer=optimizer, mll=mll)
        optimizer.step(closure)
        loss = optimizer._last_loss  # pylint: disable=protected-access
        loss_value = float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
        self._step_scheduler(loss_value)
        return loss_value

    def _lbfgs_step(self, optimizer, mll):
        optimizer.zero_grad()
        train_x = self.train_x.to(dtype=self.dtype)
        train_y = self.train_y.to(dtype=self.dtype)
        loss = self._negative_mll_loss(mll, train_x, train_y)
        loss.backward()
        return loss.detach()
