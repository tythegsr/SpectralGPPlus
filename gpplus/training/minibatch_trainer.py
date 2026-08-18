"""Stochastic gradient training over minibatches for variational GPs.

Mirrors :mod:`gpplus.training.training_single_run` -- same callback hooks, stop
conditions, best-checkpoint tracking and salvage-on-exception, and the same
multi-init orchestration from :class:`~gpplus.training.trainer.GPTrainer` -- but
each epoch is a shuffled pass over minibatches instead of one full-batch step.

Supports the two model families whose objective decomposes over data points:
:class:`~gpplus.models.svgp_gpr.SVGPR` (inducing-point ELBO) and
:class:`~gpplus.models.vi_rff_gpr.VIRFFGPR` (variational random features). The
Woodbury marginal likelihoods do not decompose, so they are rejected here.
"""

from __future__ import annotations

import copy
from typing import List, Optional

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
from .trainer import GPTrainer
from .trainer_utils import (
    RunResult,
    SingleRunResult,
    check_early_stop,
    configure_woodbury_matmul_precision,
)
from .svgp_elbo import SVGPELBO
from .vi_rff_elbo import VIRFFELBO

VARIATIONAL_PARAM_PREFIX = "raw_variational_"
# SVGP keeps q(u) and the inducing locations under the variational strategy.
SVGP_VARIATIONAL_PARAM_PREFIX = "variational_strategy"


def _is_variational_param(name: str) -> bool:
    return VARIATIONAL_PARAM_PREFIX in name or SVGP_VARIATIONAL_PARAM_PREFIX in name


def resolve_elbo_class(model: nn.Module):
    """Pick the ELBO matching the model family."""
    from ..models.svgp_gpr import SVGPR
    from ..models.vi_rff_gpr import VIRFFGPR

    if isinstance(model, SVGPR):
        return SVGPELBO
    if isinstance(model, VIRFFGPR):
        return VIRFFELBO
    raise TypeError(
        "Minibatch training requires an SVGPR or VIRFFGPR model, got "
        f"{type(model).__name__}. The Woodbury marginal likelihoods do not "
        "decompose over data points and cannot be minibatched."
    )


def _optimizer_lr(optimizer) -> float | None:
    groups = getattr(optimizer, "param_groups", None)
    if not groups:
        return None
    lr = groups[0].get("lr")
    return float(lr) if lr is not None else None


def _is_lbfgs_like(optimizer_class) -> bool:
    if optimizer_class is None:
        return False
    if optimizer_class in (LBFGSScipy, torch.optim.LBFGS):
        return True
    return isinstance(optimizer_class, type) and issubclass(
        optimizer_class, (LBFGSScipy, torch.optim.LBFGS)
    )


def variational_param_groups(
    model: nn.Module,
    base_kwargs: dict,
    variational_lr: float | None,
) -> list[dict]:
    """
    Split parameters into hyperparameter and variational groups.

    The variational mean and covariance (and, for SVGP, the inducing locations)
    typically tolerate (and benefit from) a larger step size than the kernel
    hyperparameters, since the ELBO is convex in them for fixed
    hyperparameters.
    """
    variational, hyper = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (variational if _is_variational_param(name) else hyper).append(param)

    groups = []
    if hyper:
        groups.append({"params": hyper, **base_kwargs})
    if variational:
        group = {"params": variational, **base_kwargs}
        if variational_lr is not None:
            group["lr"] = float(variational_lr)
        groups.append(group)
    if not groups:
        raise ValueError("Model has no trainable parameters.")
    return groups


class MinibatchGPTrainerSingleProcess:
    """One random start: shuffled minibatch SGD on the variational ELBO."""

    def __init__(
        self,
        model,
        optimizer_class,
        optimizer_kwargs: dict,
        num_epochs: int,
        batch_size: int,
        elbo_class=None,
        kl_beta: float = 1.0,
        variational_lr: float | None = None,
        warm_start_points: int = 0,
        callbacks: Optional[List[Callback]] = None,
        device: str | torch.device | None = None,
        scheduler_class: type[torch.optim.lr_scheduler.LRScheduler] | None = None,
        scheduler_kwargs: Optional[dict] = None,
        stop_conditions: Optional[List[StopCondition]] = None,
        dtype: torch.dtype = torch.float64,
        min_epochs: int = 0,
        run_index: Optional[int] = None,
        num_inits: Optional[int] = None,
        seed: Optional[int] = None,
        cholesky_jitter: float = 1e-6,
    ):
        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.num_epochs = int(num_epochs)
        self.elbo_class = elbo_class or resolve_elbo_class(model)
        self.kl_beta = float(kl_beta)
        self.variational_lr = variational_lr
        self.warm_start_points = int(warm_start_points)
        self.callbacks = callbacks or []
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.scheduler = None
        self.dtype = dtype
        self.min_epochs = int(min_epochs)
        self.run_index = run_index
        self.num_inits = num_inits
        self.seed = seed
        self.cholesky_jitter = float(cholesky_jitter)

        if stop_conditions is None:
            self.stop_conditions = [
                ConvergencePatienceStopCondition(patience=20),
                MinLossChangeStopCondition(min_loss_change=1e-7),
            ]
        else:
            self.stop_conditions = stop_conditions

        if _is_lbfgs_like(self.optimizer_class):
            raise ValueError(
                "Minibatch training requires a stochastic optimizer (Adam/SGD); "
                "LBFGS assumes a deterministic objective and cannot be used."
            )
        if not hasattr(self.model, "train_inputs") or not hasattr(self.model, "train_targets"):
            raise AttributeError("model must expose train_inputs and train_targets before training.")

        self.train_x = self.model.train_inputs[0]
        self.train_y = self.model.train_targets
        if self.train_x.dim() == 3 and self.train_x.shape[0] == 1:
            self.train_x = self.train_x.squeeze(0)
        if self.train_y.dim() == 2 and self.train_y.shape[0] == 1:
            self.train_y = self.train_y.squeeze(0)

        self.num_data = int(self.train_x.shape[0])
        resolved = self.num_data if int(batch_size) <= 0 else int(batch_size)
        self.batch_size = min(resolved, self.num_data)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    def _emit_callbacks(self, hook_name: str, ctx: dict) -> None:
        ctx = {**ctx, "run_index": self.run_index, "num_inits": self.num_inits}
        for cb in self.callbacks:
            getattr(cb, hook_name)(ctx)

    def _step_scheduler(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    def _make_generator(self) -> torch.Generator:
        seed = 0 if self.seed is None else int(self.seed)
        seed += 1000 * (0 if self.run_index is None else int(self.run_index))
        return torch.Generator(device="cpu").manual_seed(seed)

    def _warn_on_expensive_cpu_covariance(self) -> None:
        if self.device.type != "cpu":
            return
        m = int(getattr(self.model, "num_inducing", 0))
        if m:
            logger.warning(
                "SVGP on CPU costs O(B M^2) per step with M=%s inducing points; "
                "a GPU is strongly preferred.",
                m,
            )
            return
        if getattr(self.model, "variational_cov", None) != "chol":
            return
        logger.warning(
            "variational_cov='chol' on CPU costs O(B m^2) per step with m=%s; "
            "consider variational_cov='diag' while iterating.",
            int(getattr(self.model, "num_features", 0)),
        )

    def _train_epoch(self, optimizer, elbo, generator) -> float:
        """One shuffled pass; returns the mean per-observation loss over batches."""
        perm = torch.randperm(self.num_data, generator=generator).to(self.train_x.device)
        total = 0.0
        num_batches = 0
        for start in range(0, self.num_data, self.batch_size):
            idx = perm[start : start + self.batch_size]
            x_b = self.train_x[idx].to(dtype=self.dtype)
            y_b = self.train_y[idx].to(dtype=self.dtype)
            optimizer.zero_grad(set_to_none=True)
            loss = -elbo(x_b, y_b, self.num_data)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            num_batches += 1
        self._step_scheduler()
        return total / max(1, num_batches)

    def train(self) -> SingleRunResult:
        configure_woodbury_matmul_precision(device=self.device, dtype=self.dtype)
        self._warn_on_expensive_cpu_covariance()

        generator = self._make_generator()
        if self.warm_start_points > 0 and hasattr(self.model, "warm_start_from_train"):
            self.model.warm_start_from_train(
                max_points=self.warm_start_points,
                generator=generator,
                jitter=self.cholesky_jitter,
            )

        optimizer = self.optimizer_class(
            variational_param_groups(self.model, self.optimizer_kwargs, self.variational_lr)
        )
        if self.scheduler_class is not None:
            self.scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
        elbo = self.elbo_class(self.model.likelihood, self.model, kl_beta=self.kl_beta)

        best_loss = float("inf")
        best_state_dict = None
        no_improvement_epochs = 0
        previous_loss = None
        epochs_trained = 0
        abort_error: str | None = None
        abort_epoch: int | None = None

        self._emit_callbacks(
            "on_train_start",
            {"model": self.model, "trainer": self, "device": self.device},
        )
        for cb in self.callbacks:
            register = getattr(cb, "register_with_optimizer", None)
            if callable(register):
                register(optimizer, model=self.model, trainer=self)

        self.model.train()
        steps_per_epoch = -(-self.num_data // self.batch_size)
        logger.info(
            "Starting minibatch training: %s epochs x %s steps (N=%s, batch_size=%s).",
            self.num_epochs,
            steps_per_epoch,
            self.num_data,
            self.batch_size,
        )
        for epoch in range(self.num_epochs):
            epochs_trained = epoch + 1
            self._emit_callbacks(
                "on_epoch_start",
                {"epoch": epoch, "model": self.model, "trainer": self, "device": self.device},
            )
            try:
                loss = self._train_epoch(optimizer, elbo, generator)
            except Exception as exc:
                if best_state_dict is not None:
                    abort_error = str(exc)
                    abort_epoch = epoch
                    logger.exception(
                        "Minibatch training aborted at epoch %s/%s; recovering best "
                        "checkpoint (best_loss=%.6f). Error: %s",
                        epoch + 1,
                        self.num_epochs,
                        best_loss,
                        exc,
                    )
                    break
                raise
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
            elif epoch + 1 > self.min_epochs:
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
                self.stop_conditions, stop_context, epoch, best_loss, min_epochs=self.min_epochs
            ):
                break
            previous_loss = loss

        final_lr = _optimizer_lr(optimizer)
        if abort_error is not None:
            logger.warning(
                "Minibatch training ended early via salvage. Best loss: %.6f (abort at epoch %s).",
                best_loss,
                abort_epoch + 1 if abort_epoch is not None else "?",
            )
        else:
            logger.info("Minibatch training completed. Best loss: %.6f", best_loss)
        logger.info("Total epochs trained: %s", epochs_trained)
        if best_state_dict is None:
            logger.warning("No model state was captured during training; verify epoch count.")
        else:
            self.model.load_state_dict(best_state_dict)

        self._emit_callbacks(
            "on_train_end",
            {
                "epoch": max(0, epochs_trained - 1),
                "model": self.model,
                "trainer": self,
                "best_loss": best_loss,
                "best_state_dict": best_state_dict,
                "device": self.device,
                "aborted": abort_error is not None,
                "abort_error": abort_error,
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
        if abort_error is not None:
            result["error"] = abort_error
            result["aborted"] = True
            if abort_epoch is not None:
                result["aborted_epoch"] = abort_epoch
        return result


class MinibatchGPTrainer(GPTrainer):
    """
    Multi-init orchestration for minibatch ELBO training.

    Reuses :class:`~gpplus.training.trainer.GPTrainer` for device/dtype
    preparation, Sobol parameter initialization, parallel dispatch over random
    starts and best-run selection; only the per-run loop differs. Batched
    multi-init (``train_mode="batched"``) is not supported, since minibatch
    gradient noise makes per-init early stopping within a shared wave ambiguous.

    Parameters
    ----------
    batch_size :
        Minibatch size. ``0`` or ``None`` means full batch, which turns SGD into
        deterministic gradient descent on the exact ELBO.
    variational_lr :
        Optional separate learning rate for the variational parameters.
    kl_beta :
        KL weight passed to the ELBO.
    warm_start_points :
        When positive and the model supports it (``VIRFFGPR``), initialize ``q``
        to the analytically optimal posterior on a random subsample of this many
        training points before the first step. Ignored for ``SVGPR``.
    elbo_class :
        Defaults to the ELBO matching the model family: ``SVGPELBO`` for
        ``SVGPR``, ``VIRFFELBO`` for ``VIRFFGPR``.
    """

    def __init__(
        self,
        model,
        *,
        batch_size: int | None = 1024,
        variational_lr: float | None = None,
        kl_beta: float = 1.0,
        warm_start_points: int = 0,
        elbo_class=None,
        optimizer_class: type[torch.optim.Optimizer] | None = None,
        **gp_trainer_kwargs,
    ):
        elbo_class = elbo_class or resolve_elbo_class(model)
        if _is_lbfgs_like(optimizer_class):
            raise ValueError(
                "Minibatch training requires a stochastic optimizer (Adam/SGD); "
                f"got {getattr(optimizer_class, '__name__', optimizer_class)}."
            )
        if gp_trainer_kwargs.get("train_mode", "independent") != "independent":
            raise ValueError(
                "MinibatchGPTrainer supports train_mode='independent' only."
            )
        if gp_trainer_kwargs.get("param_groups_fn") is not None:
            raise ValueError(
                "MinibatchGPTrainer builds its own param groups; pass "
                "variational_lr instead of param_groups_fn."
            )
        gp_trainer_kwargs.pop("mll_class", None)

        super().__init__(
            model,
            optimizer_class=optimizer_class or torch.optim.Adam,
            mll_class=elbo_class,
            **gp_trainer_kwargs,
        )
        self.elbo_class = elbo_class
        self.batch_size = 0 if batch_size is None else int(batch_size)
        self.variational_lr = variational_lr
        self.kl_beta = float(kl_beta)
        self.warm_start_points = int(warm_start_points)
        logger.info(
            "MinibatchGPTrainer[%s]: batch_size=%s, kl_beta=%g, variational_lr=%s, "
            "warm_start_points=%s.",
            type(model).__name__,
            self.batch_size or "full",
            self.kl_beta,
            self.variational_lr,
            self.warm_start_points,
        )

    def train_single_process(
        self, run_index: int, run_device: Optional[torch.device] = None
    ) -> RunResult:
        target_device = run_device or self.device
        base_model = copy.deepcopy(self.model)
        self.initializer.initialize(base_model, run_index)
        base_model = base_model.to(target_device, dtype=self.dtype)

        if self.num_inits == 1:
            callbacks_copy = self.callbacks
            stop_conditions_copy = self.stop_conditions
            for cb in callbacks_copy:
                if hasattr(cb, "set_run_index"):
                    cb.set_run_index(run_index)
        else:
            callbacks_copy = []
            for cb in self.callbacks:
                cb_copy = copy.deepcopy(cb)
                if hasattr(cb_copy, "set_run_index"):
                    cb_copy.set_run_index(run_index)
                callbacks_copy.append(cb_copy)
            stop_conditions_copy = (
                [copy.deepcopy(sc) for sc in self.stop_conditions] if self.stop_conditions else None
            )

        run = MinibatchGPTrainerSingleProcess(
            model=base_model,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            elbo_class=self.elbo_class,
            kl_beta=self.kl_beta,
            variational_lr=self.variational_lr,
            warm_start_points=self.warm_start_points,
            callbacks=callbacks_copy,
            device=target_device,
            scheduler_class=self.scheduler_class,
            scheduler_kwargs=self.scheduler_kwargs,
            stop_conditions=stop_conditions_copy,
            dtype=self.dtype,
            min_epochs=self.min_epochs,
            run_index=run_index,
            num_inits=self.num_inits,
            seed=self.seed,
            cholesky_jitter=self.cholesky_jitter,
        )
        train_result = run.train()
        return {"run_index": run_index, **train_result}


__all__ = [
    "MinibatchGPTrainer",
    "MinibatchGPTrainerSingleProcess",
    "resolve_elbo_class",
    "variational_param_groups",
]
