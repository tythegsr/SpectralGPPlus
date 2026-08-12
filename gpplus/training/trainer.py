import copy
from collections.abc import Callable
from typing import List, Optional

import gpytorch
import torch
from torch import nn

from ..config import logger
from .callbacks import Callback
from .optimizers import LBFGSScipy
from .parameter_initializer import DefaultParameterInitializer, ParameterInitializer
from .stop_conditions import StopCondition
from .trainer_utils import (
    RunResult,
    build_run_error,
    get_effective_optimizer_kwargs,
    run_parallel_initializations,
    select_best_run,
)
from .batch_utils import materialize_unbatched_model
from .training_batched import BatchedGPTrainer
from .training_single_run import GPTrainerSingleProcess


class GPTrainer:
    def __init__(
        self,
        model,
        optimizer_class: torch.optim.Optimizer = None,
        optimizer_kwargs: dict = None,
        scheduler_class: torch.optim.lr_scheduler.LRScheduler = None,
        scheduler_kwargs: dict = None,
        param_groups_fn: Callable[[nn.Module], list[dict]] | None = None,
        num_epochs: int = 1000,
        seed: int = None,
        num_inits: int = 64,
        mll_class: gpytorch.mlls.MarginalLogLikelihood = None,
        cholesky_jitter: float = 1e-6,
        callbacks: Optional[List[Callback]] = None,
        initializer_class: ParameterInitializer = None,
        initializer_kwargs: dict = None,
        device: str = "cpu",
        stop_conditions: Optional[List[StopCondition]] = None,
        min_epochs: int = 0,
        n_jobs: Optional[int] = None,
        inner_max_num_threads: Optional[int] = 1,
        dtype: torch.dtype = torch.float64,
        parallel_verbose: int = 0,
        train_mode: str = "independent",
        init_batch_size: int | None = None,
    ):
        #! TODO: Update so LBFGS and adam use different trainers to minimize 'if' lines
        """
        Initialize the multi-run GP trainer.

        Args:
            model: GP model instance with `train_inputs` and `train_targets`.
            optimizer_class: Optimizer class used for each run. Defaults to `LBFGSScipy`.
            optimizer_kwargs: Optimizer kwargs (without `params`).
            scheduler_class: Optional learning-rate scheduler class.
            scheduler_kwargs: Optional scheduler kwargs.
            param_groups_fn: Optional callable ``model -> Adam param_groups``.
                When set, each run builds the optimizer from groups instead of
                ``model.parameters()`` + ``optimizer_kwargs``.
            num_epochs: Number of epochs per run.
            seed: Random seed for parameter initialization.
            num_inits: Total number of random starts to evaluate.
            mll_class: Marginal log likelihood class. Defaults to exact MLL.
            cholesky_jitter: Cholesky jitter used during training.
            callbacks: Optional callback instances applied during training.
            initializer_class: Parameter initializer class for per-run starts.
            initializer_kwargs: Optional kwargs for `initializer_class`.
            device: Target device string (falls back to CPU if CUDA is unavailable).
            stop_conditions: Optional early-stop conditions. Defaults are applied when omitted.
            min_epochs: Minimum epochs before stop conditions can terminate a run.
            n_jobs: Optional parallel job cap used by run dispatch (independent mode).
                On CUDA, ``None`` means one worker per GPU; a positive value is the
                total concurrent init cap (may place multiple jobs on one GPU).
            inner_max_num_threads: Optional torch thread cap per run worker.
            dtype: Tensor dtype used for model and training data.
            parallel_verbose: joblib Parallel verbosity (0=quiet, 10=progress).
            train_mode: ``"independent"`` (default) runs deepcopy+joblib multi-init;
                ``"batched"`` trains one model with ``batch_shape=[init_batch_size]`` on a
                single device (Adam-style optimizers only), in sequential waves until
                ``num_inits`` starts are done.
            init_batch_size: Concurrent inits per wave in batched mode (model
                ``batch_shape``). Defaults to ``num_inits`` (single wave) or
                ``model.init_batch_size`` when set. ``num_inits`` must be divisible by
                this value. Ignored in independent mode. After batched freeze /
                early-stop fixes this is a **VRAM / throughput** knob only — it
                should not change per-init Adam dynamics vs a smaller wave.
        """
        if train_mode not in {"independent", "batched"}:
            raise ValueError(f"train_mode must be 'independent' or 'batched', got {train_mode!r}.")
        self.train_mode = train_mode

        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)
        logger.info("Using device: %s", self.device)

        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype, got {type(dtype).__name__}.")
        self.dtype = dtype
        self._prepare_model_and_data(model)

        self.num_epochs = num_epochs
        if int(num_inits) < 1:
            raise ValueError(f"num_inits must be >= 1, got {num_inits}.")
        self.num_inits = int(num_inits)
        self.total_inits = self.num_inits

        resolved_ibs = init_batch_size
        if resolved_ibs is None:
            resolved_ibs = getattr(model, "init_batch_size", None)
        if resolved_ibs is None:
            resolved_ibs = self.num_inits
        resolved_ibs = int(resolved_ibs)
        if resolved_ibs < 1:
            raise ValueError(f"init_batch_size must be >= 1, got {resolved_ibs}.")
        self.init_batch_size = min(resolved_ibs, self.num_inits)

        if self.train_mode == "batched":
            if self.num_inits % self.init_batch_size != 0:
                raise ValueError(
                    f"num_inits ({self.num_inits}) must be divisible by "
                    f"init_batch_size ({self.init_batch_size})."
                )
            self.num_batches = self.num_inits // self.init_batch_size
        else:
            if init_batch_size is not None and int(init_batch_size) != self.num_inits:
                logger.warning(
                    "init_batch_size=%s is ignored when train_mode=%r.",
                    init_batch_size,
                    self.train_mode,
                )
            self.init_batch_size = self.num_inits
            self.num_batches = 1

        self.seed = seed
        self.callbacks = callbacks or []
        self.cholesky_jitter = cholesky_jitter
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.param_groups_fn = param_groups_fn
        self.min_epochs = min_epochs
        self.n_jobs = n_jobs
        self.inner_max_num_threads = inner_max_num_threads
        self.parallel_verbose = parallel_verbose

        if stop_conditions is None:
            from .stop_conditions import ConvergencePatienceStopCondition, MinLossChangeStopCondition

            self.stop_conditions = [
                ConvergencePatienceStopCondition(patience=20),
                MinLossChangeStopCondition(min_loss_change=1e-7),
            ]
        else:
            self.stop_conditions = stop_conditions

        if initializer_class is None:
            self.initializer = DefaultParameterInitializer(
                num_inits=self.num_inits, seed=self.seed
            )
        else:
            self.initializer = initializer_class(
                num_inits=self.num_inits,
                seed=self.seed,
                **(initializer_kwargs or {}),
            )
        self.initializer.setup(self.model)

        if optimizer_class is None:
            self.optimizer_class = LBFGSScipy
            logger.warning(
                "No optimizer class passed (input=%s). Defaulting to optimizer class=%s.",
                optimizer_class,
                self.optimizer_class.__name__,
            )
        else:
            self.optimizer_class = optimizer_class

        self.optimizer_kwargs = optimizer_kwargs or {}
        if optimizer_kwargs is None:
            logger.warning("No optimizer kwargs passed (input=%s). Using optimizer class defaults.", optimizer_kwargs)
        else:
            logger.info("Optimizer class: %s, kwargs: %s", self.optimizer_class.__name__, optimizer_kwargs)

        if mll_class is None:
            self.mll_class = gpytorch.mlls.ExactMarginalLogLikelihood
            logger.warning("No MLL class passed. Defaulting to ExactMarginalLogLikelihood.")
        else:
            self.mll_class = mll_class

        is_lbfgs_like = (
            self.optimizer_class is LBFGSScipy
            or (isinstance(self.optimizer_class, type) and issubclass(self.optimizer_class, LBFGSScipy))
            or self.optimizer_class is torch.optim.LBFGS
            or (isinstance(self.optimizer_class, type) and issubclass(self.optimizer_class, torch.optim.LBFGS))
        )
        if is_lbfgs_like and self.num_epochs != 1:
            logger.info("Overriding num_epochs=%s to 1 for LBFGS-style optimizer.", self.num_epochs)
            self.num_epochs = 1

        if self.train_mode == "batched" and is_lbfgs_like:
            raise ValueError(
                "train_mode='batched' requires an Adam-style PyTorch optimizer; "
                "LBFGS is only supported with train_mode='independent'."
            )

        optimizer_name = getattr(self.optimizer_class, "__name__", str(self.optimizer_class))
        effective_optimizer_kwargs = get_effective_optimizer_kwargs(self.optimizer_class, self.optimizer_kwargs)
        logger.info(
            "Trainer optimizer configured: class=%s, effective_kwargs=%s, train_mode=%s",
            optimizer_name,
            effective_optimizer_kwargs,
            self.train_mode,
        )

    def _prepare_model_and_data(self, model) -> None:
        if not hasattr(model, "train_inputs") or not hasattr(model, "train_targets"):
            raise AttributeError("model must expose train_inputs and train_targets before training.")

        train_x = model.train_inputs[0]
        train_y = model.train_targets
        if not isinstance(train_x, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            raise TypeError("train_inputs and train_targets must be torch.Tensor instances.")
        if train_x.dtype != train_y.dtype:
            raise TypeError(f"Training data dtype mismatch: train_x is {train_x.dtype}, train_y is {train_y.dtype}.")

        if train_x.dtype != self.dtype:
            logger.info("Converting model training data from %s to %s on %s.", train_x.dtype, self.dtype, self.device)

        self.model = model.to(self.device, dtype=self.dtype)
        self.model.set_train_data(
            train_x.to(self.device, dtype=self.dtype),
            train_y.to(self.device, dtype=self.dtype),
            strict=False,
        )
        self.model.dtype = self.dtype
        self.train_x = self.model.train_inputs[0]
        self.train_y = self.model.train_targets

    def train_single_process(self, run_index: int, run_device: Optional[torch.device] = None) -> RunResult:
        target_device = run_device or self.device
        base_model = copy.deepcopy(self.model)
        self.initializer.initialize(base_model, run_index)
        base_model = base_model.to(target_device, dtype=self.dtype)

        if self.num_inits == 1:
            # Preserve callback state for single-run workflows (e.g., plotting callbacks in examples).
            callbacks_copy = self.callbacks
            stop_conditions_copy = self.stop_conditions
        else:
            callbacks_copy = []
            for cb in self.callbacks:
                cb_copy = copy.deepcopy(cb)
                if hasattr(cb_copy, "set_run_index"):
                    cb_copy.set_run_index(run_index)
                callbacks_copy.append(cb_copy)
            stop_conditions_copy = [copy.deepcopy(sc) for sc in self.stop_conditions] if self.stop_conditions else None

        if self.num_inits == 1:
            for cb in callbacks_copy:
                if hasattr(cb, "set_run_index"):
                    cb.set_run_index(run_index)

        run = GPTrainerSingleProcess(
            model=base_model,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            mll_class=self.mll_class,
            num_epochs=self.num_epochs,
            cholesky_jitter=self.cholesky_jitter,
            callbacks=callbacks_copy,
            device=target_device,
            scheduler_class=self.scheduler_class,
            scheduler_kwargs=self.scheduler_kwargs,
            param_groups_fn=self.param_groups_fn,
            stop_conditions=stop_conditions_copy,
            min_epochs=self.min_epochs,
            dtype=self.dtype,
            run_index=run_index,
            num_inits=self.num_inits,
        )
        train_result = run.train()
        return {"run_index": run_index, **train_result}

    def _train_single_process_safe(self, run_index: int, run_device: torch.device) -> RunResult:
        previous_num_threads = None
        try:
            if self.inner_max_num_threads is not None:
                previous_num_threads = torch.get_num_threads()
                torch.set_num_threads(max(1, self.inner_max_num_threads))
            return self.train_single_process(run_index, run_device=run_device)
        except Exception as exc:
            return build_run_error(run_index, exc)
        finally:
            if previous_num_threads is not None:
                torch.set_num_threads(previous_num_threads)

    def train_multiple_process_parallel(self) -> list[RunResult]:
        results = run_parallel_initializations(
            num_inits=self.num_inits,
            trainer_device=self.device,
            run_callable=self._train_single_process_safe,
            n_jobs=self.n_jobs,
            parallel_verbose=self.parallel_verbose,
        )
        logger.info("Training completed.")
        return results

    def train_batched(self) -> list[RunResult]:
        """Train ``num_inits`` starts in waves of ``init_batch_size``; load the best unbatched state."""
        if not hasattr(self.initializer, "initialize_batched"):
            raise TypeError(
                f"Initializer {type(self.initializer).__name__} does not support initialize_batched()."
            )

        from .batch_utils import model_init_batch_size

        model_batch = model_init_batch_size(self.model)
        if self.init_batch_size > 1 and model_batch != self.init_batch_size:
            raise ValueError(
                f"Batched model batch size is {model_batch}, but init_batch_size="
                f"{self.init_batch_size}. Construct the model with "
                f"batch_shape=torch.Size([{self.init_batch_size}])."
            )

        patience = None
        min_loss_change = None
        for sc in self.stop_conditions:
            if hasattr(sc, "patience"):
                patience = sc.patience
            if hasattr(sc, "min_loss_change"):
                min_loss_change = sc.min_loss_change

        all_results: list[RunResult] = []
        best_unbatched = None
        best_index = None
        best_loss = None
        wave_size = self.init_batch_size

        for wave in range(self.num_batches):
            offset = wave * wave_size
            logger.info(
                "Batched init wave %s/%s (inits %s..%s of %s total, concurrent=%s).",
                wave + 1,
                self.num_batches,
                offset,
                offset + wave_size - 1,
                self.num_inits,
                wave_size,
            )
            self.initializer.initialize_batched(self.model, start_index=offset)

            runner = BatchedGPTrainer(
                model=self.model,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                mll_class=self.mll_class,
                num_epochs=self.num_epochs,
                cholesky_jitter=self.cholesky_jitter,
                callbacks=self.callbacks,
                device=self.device,
                scheduler_class=self.scheduler_class,
                scheduler_kwargs=self.scheduler_kwargs,
                param_groups_fn=self.param_groups_fn,
                stop_conditions=self.stop_conditions,
                min_epochs=self.min_epochs,
                dtype=self.dtype,
                num_inits=wave_size,
                patience=patience,
                min_loss_change=min_loss_change,
            )
            wave_results = runner.train()
            for result in wave_results:
                local_idx = result.get("run_index")
                if local_idx is not None:
                    result = {**result, "run_index": offset + int(local_idx), "batch_index": wave}
                else:
                    result = {**result, "batch_index": wave}
                all_results.append(result)

            wave_state = getattr(runner, "_best_unbatched_state", None)
            wave_best_index = getattr(runner, "_best_run_index", None)
            wave_best_loss = getattr(runner, "_best_loss", None)
            if wave_state is not None and wave_best_loss is not None:
                if best_loss is None or float(wave_best_loss) < float(best_loss):
                    best_unbatched = wave_state
                    best_loss = float(wave_best_loss)
                    best_index = (
                        offset + int(wave_best_index) if wave_best_index is not None else offset
                    )

        if best_unbatched is not None:
            self.model = materialize_unbatched_model(self.model, best_unbatched)
            self.train_x = self.model.train_inputs[0]
            self.train_y = self.model.train_targets
            logger.info(
                "Best batched init #%s materialized into unbatched model (loss=%.4f, "
                "waves=%s, init_batch_size=%s, total_inits=%s).",
                best_index,
                best_loss if best_loss is not None else float("nan"),
                self.num_batches,
                self.init_batch_size,
                self.num_inits,
            )
        else:
            logger.warning("No batched winner state found; model left in batched form.")
        logger.info("Batched training completed.")
        return all_results

    def train(self) -> list[RunResult]:
        if self.train_mode == "batched":
            return self.train_batched()

        results = self.train_multiple_process_parallel()
        failed_runs = [result for result in results if result.get("error")]
        if failed_runs:
            logger.warning(
                "%s/%s runs failed. Check run-level error payloads for details.",
                len(failed_runs),
                len(results),
            )

        best_run = select_best_run(results)
        if best_run is not None:
            best_loss = best_run["loss"]
            self.model.load_state_dict(best_run["state_dict"])
            if best_run.get("aborted"):
                logger.warning(
                    "Best run found: #%s with salvaged loss=%.4f (mid-train abort: %s).",
                    best_run["run_index"],
                    best_loss,
                    best_run.get("error", "unknown"),
                )
            else:
                logger.info(
                    "Best run found: #%s with loss=%.4f.", best_run["run_index"], best_loss
                )
        else:
            logger.warning("No valid best run found. Model was not updated.")
        return results
