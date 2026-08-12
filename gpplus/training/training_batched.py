"""Batched multi-init training (single model, leading init batch dimension)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import List, Optional

import gpytorch
import linear_operator
import torch
from torch import nn

from ..config import logger
from .batch_utils import (
    assert_batched_trainable_parameters,
    clear_optimizer_state_for_inits,
    log_batched_training_banner,
    model_init_batch_size,
    restore_init_from_state_dict,
    snapshot_init_into_state_dict,
    zero_grads_for_inactive_inits,
)
from .callbacks import Callback
from .optimizers import LBFGSScipy
from .stop_conditions import (
    ConvergencePatienceStopCondition,
    MinLossChangeStopCondition,
    StopCondition,
)
from .trainer_utils import (
    RunResult,
    configure_woodbury_matmul_precision,
)
from .training_single_run import _WOODBURY_MLL_TYPES


class BatchedGPTrainer:
    """
    Train ``num_inits`` initializations as one batched model on a single device.

    Requires the model to expose ``batch_shape=torch.Size([num_inits])`` (or
    equivalent batched modules). Uses Adam-style PyTorch optimizers only.
    Per-init early stopping freezes inactive slices (restore best + clear Adam
    moments) so ``init_batch_size`` does not change optimization dynamics beyond
    VRAM / throughput.
    """

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
        num_inits: int = 1,
        patience: int | None = 20,
        min_loss_change: float | None = 1e-7,
        flat_patience: int | None = None,
    ):
        if optimizer_class is LBFGSScipy or (
            isinstance(optimizer_class, type) and issubclass(optimizer_class, LBFGSScipy)
        ):
            raise ValueError(
                "train_mode='batched' does not support LBFGSScipy; use Adam "
                "or train_mode='independent'."
            )
        if optimizer_class is torch.optim.LBFGS or (
            isinstance(optimizer_class, type) and issubclass(optimizer_class, torch.optim.LBFGS)
        ):
            raise ValueError(
                "train_mode='batched' does not support torch.optim.LBFGS; use Adam "
                "or train_mode='independent'."
            )

        self.model = model
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.mll_class = mll_class or gpytorch.mlls.ExactMarginalLogLikelihood
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
        self.num_inits = num_inits
        self.patience = patience
        self.min_loss_change = min_loss_change
        # Consecutive flat epochs required before flat-stop (not one-shot).
        if flat_patience is not None:
            self.flat_patience = int(flat_patience)
        elif patience is not None:
            self.flat_patience = max(1, int(patience))
        else:
            self.flat_patience = 5

        if stop_conditions is None:
            self.stop_conditions = [
                ConvergencePatienceStopCondition(patience=patience),
                MinLossChangeStopCondition(min_loss_change=min_loss_change or 1e-7),
            ]
        else:
            self.stop_conditions = stop_conditions

        batch_size = model_init_batch_size(model)
        if num_inits > 1 and batch_size != num_inits:
            raise ValueError(
                f"Batched training requires model batch size {num_inits}, got {batch_size}. "
                "Construct the model with batch_shape=torch.Size([num_inits])."
            )
        if num_inits > 1:
            assert_batched_trainable_parameters(model, batch_size)

        self.train_x = self.model.train_inputs[0]
        self.train_y = self.model.train_targets

    def _emit_callbacks(self, hook_name: str, ctx: dict) -> dict:
        ctx = {**ctx, "num_inits": self.num_inits, "train_mode": "batched"}
        for cb in self.callbacks:
            getattr(cb, hook_name)(ctx)
        return ctx

    def _negative_mll_losses(self, mll, train_x: torch.Tensor, train_y: torch.Tensor) -> torch.Tensor:
        """Return per-init negative MLL losses shaped ``(B,)``."""
        if isinstance(mll, _WOODBURY_MLL_TYPES):
            nll = -mll(None, train_y)
        else:
            output = self.model(train_x)
            nll = -mll(output, train_y)
        if nll.dim() == 0:
            return nll.unsqueeze(0)
        return nll.reshape(-1)

    def _deactivate_inits(
        self,
        to_deactivate: torch.Tensor,
        *,
        active: torch.Tensor,
        best_state: dict,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        reason: str,
    ) -> None:
        """Freeze newly inactive inits: restore best/initial weights and clear Adam moments."""
        newly = to_deactivate & active
        idxs = torch.where(newly)[0].tolist()
        if not idxs:
            active[to_deactivate] = False
            return
        for i in idxs:
            # Always restore: best_state holds either the last finite improvement
            # or the initial deepcopy. Skipping left NaN/Inf weights in the batch.
            restore_init_from_state_dict(self.model, best_state, i, batch_size)
            logger.info("Batched init #%s deactivated (%s).", i, reason)
        clear_optimizer_state_for_inits(
            optimizer, idxs, batch_size, model=self.model
        )
        active[to_deactivate] = False

    def train(self) -> list[RunResult]:
        if isinstance(self.mll_class, type) and issubclass(self.mll_class, _WOODBURY_MLL_TYPES):
            configure_woodbury_matmul_precision(device=self.device, dtype=self.dtype)

        log_batched_training_banner(self.num_inits, self.device)
        B = self.num_inits
        if B > 1:
            assert_batched_trainable_parameters(self.model, B)

        if self.param_groups_fn is not None:
            optimizer = self.optimizer_class(self.param_groups_fn(self.model))
        else:
            optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

        if self.scheduler_class is not None:
            self.scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
        else:
            self.scheduler = None

        if isinstance(self.mll_class, type) and issubclass(self.mll_class, _WOODBURY_MLL_TYPES):
            mll = self.mll_class(self.model.likelihood, self.model, jitter=self.cholesky_jitter)
        else:
            mll = self.mll_class(self.model.likelihood, self.model)

        best_losses = torch.full((B,), float("inf"), dtype=torch.float64)
        best_state = copy.deepcopy(self.model.state_dict())
        no_improvement = torch.zeros(B, dtype=torch.long)
        flat_epochs = torch.zeros(B, dtype=torch.long)
        active = torch.ones(B, dtype=torch.bool)
        previous_losses: torch.Tensor | None = None
        epochs_trained = 0

        self._emit_callbacks(
            "on_train_start",
            {"model": self.model, "trainer": self, "device": self.device},
        )

        with (
            gpytorch.settings.cholesky_jitter(self.cholesky_jitter),
            linear_operator.settings.cholesky_jitter(
                float_value=self.cholesky_jitter, double_value=self.cholesky_jitter
            ),
        ):
            self.model.train()
            logger.info("Starting batched training for %s epochs over %s inits.", self.num_epochs, B)
            for epoch in range(self.num_epochs):
                epochs_trained = epoch + 1
                self._emit_callbacks(
                    "on_epoch_start",
                    {
                        "epoch": epoch,
                        "model": self.model,
                        "trainer": self,
                        "device": self.device,
                        "active": active.clone(),
                    },
                )

                optimizer.zero_grad()
                train_x = self.train_x.to(dtype=self.dtype)
                train_y = self.train_y.to(dtype=self.dtype)
                losses = self._negative_mll_losses(mll, train_x, train_y)
                if losses.numel() != B:
                    raise RuntimeError(
                        f"Batched MLL returned {losses.numel()} losses, expected {B}."
                    )

                # Control-flow in float64 on CPU; keep a device bool mask for the step.
                losses_cpu = losses.detach().to(dtype=torch.float64).cpu()
                finite_cpu = torch.isfinite(losses_cpu)
                finite = finite_cpu.to(device=losses.device)

                # Quarantine non-finite inits before the step.
                nonfinite = active & (~finite_cpu)
                if bool(nonfinite.any()):
                    self._deactivate_inits(
                        nonfinite,
                        active=active,
                        best_state=best_state,
                        optimizer=optimizer,
                        batch_size=B,
                        reason="non-finite train NLL",
                    )

                step_mask_cpu = active & finite_cpu
                if not bool(step_mask_cpu.any()):
                    logger.info(
                        "Batched early stopping at epoch %s: no finite active inits. "
                        "Best losses=%s",
                        epoch + 1,
                        best_losses.tolist(),
                    )
                    break

                # Snapshot BEST params at the θ that produced the loss (before Adam step).
                for i in range(B):
                    if not bool(step_mask_cpu[i]):
                        continue
                    li = float(losses_cpu[i].item())
                    if li < float(best_losses[i].item()):
                        best_losses[i] = li
                        no_improvement[i] = 0
                        snapshot_init_into_state_dict(best_state, self.model, i, B)
                    elif epoch + 1 > self.min_epochs:
                        # Do not accumulate patience before min_epochs (e.g. NIGP freeze).
                        no_improvement[i] += 1

                # Sum (not mean): per-init grads match independent training scale.
                step_f = step_mask_cpu.to(device=losses.device, dtype=losses.dtype)
                loss = (
                    torch.nan_to_num(losses, nan=0.0, posinf=0.0, neginf=0.0) * step_f
                ).sum()
                loss.backward()
                inactive_now = ~active
                if B > 1 and bool(inactive_now.any()):
                    zero_grads_for_inactive_inits(self.model, inactive_now, B)
                optimizer.step()
                # Re-apply freeze after Adam (zero grad ≠ no update with weight decay /
                # stale moments). Always restore from best_state (initial or best).
                for i in torch.where(~active)[0].tolist():
                    restore_init_from_state_dict(self.model, best_state, i, B)
                if self.scheduler is not None:
                    self.scheduler.step()

                n_step = float(step_mask_cpu.sum().clamp_min(1).item())
                loss_mean = float(
                    (
                        torch.nan_to_num(losses.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                        * step_f.detach()
                    )
                    .sum()
                    .item()
                    / n_step
                )

                self._emit_callbacks(
                    "on_epoch_end",
                    {
                        "epoch": epoch,
                        "model": self.model,
                        "trainer": self,
                        "loss": loss_mean,
                        "loss_sum": float(loss.detach().item()),
                        "losses": losses_cpu.clone(),
                        "active": active.clone(),
                        "device": self.device,
                    },
                )

                if epoch + 1 >= self.min_epochs:
                    to_stop = torch.zeros(B, dtype=torch.bool)
                    if self.patience is not None:
                        to_stop = to_stop | (no_improvement >= int(self.patience))
                    if self.min_loss_change is not None and previous_losses is not None:
                        delta = (losses_cpu - previous_losses).abs()
                        # Non-finite deltas do not count as flat.
                        flat = torch.isfinite(delta) & (delta < float(self.min_loss_change))
                        # Only accumulate flat while not improving vs best.
                        growing_flat = flat & (no_improvement > 0) & active
                        flat_epochs = torch.where(
                            growing_flat, flat_epochs + 1, torch.zeros_like(flat_epochs)
                        )
                        to_stop = to_stop | (flat_epochs >= int(self.flat_patience))
                    else:
                        flat_epochs.zero_()

                    if bool((to_stop & active).any()):
                        self._deactivate_inits(
                            to_stop & active,
                            active=active,
                            best_state=best_state,
                            optimizer=optimizer,
                            batch_size=B,
                            reason="early stop",
                        )
                    if not bool(active.any()):
                        logger.info(
                            "Batched early stopping at epoch %s: all inits inactive. "
                            "Best losses=%s",
                            epoch + 1,
                            best_losses.tolist(),
                        )
                        break

                previous_losses = losses_cpu.clone()

        best_index = int(torch.argmin(best_losses).item())
        best_loss = float(best_losses[best_index].item())
        from .batch_utils import (
            _remap_lrnn_feature_net_keys,
            build_unbatched_model_shell,
            slice_state_dict,
        )

        ref_sd = build_unbatched_model_shell(self.model).state_dict()
        unbatched = _remap_lrnn_feature_net_keys(
            slice_state_dict(
                best_state,
                best_index,
                batch_size=B,
                reference_state_dict=ref_sd,
            ),
            best_index,
        )
        logger.info(
            "Batched training completed. Best init=#%s loss=%.6f (epochs=%s).",
            best_index,
            best_loss,
            epochs_trained,
        )

        results: list[RunResult] = []
        for i in range(B):
            results.append(
                {
                    "run_index": i,
                    "loss": float(best_losses[i].item()),
                    "state_dict": _remap_lrnn_feature_net_keys(
                        slice_state_dict(
                            best_state,
                            i,
                            batch_size=B,
                            reference_state_dict=ref_sd,
                        ),
                        i,
                    ),
                }
            )

        self._emit_callbacks(
            "on_train_end",
            {
                "epoch": max(0, epochs_trained - 1),
                "model": self.model,
                "trainer": self,
                "best_loss": best_loss,
                "best_run_index": best_index,
                "best_state_dict": unbatched,
                "losses": best_losses.clone(),
                "device": self.device,
            },
        )
        self._best_unbatched_state = unbatched
        self._best_loss = best_loss
        self._best_run_index = best_index
        return results
