import copy
import importlib
import os
from contextlib import nullcontext
from typing import List, Optional

import gpytorch
import torch
from joblib import Parallel, delayed, parallel_config

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:
    threadpool_info = None
    threadpool_limits = None

from ..config import logger
from .callbacks import Callback
from .optimizers import LBFGSScipy
from .parameter_initializer import DefaultParameterInitializer, ParameterInitializer
from .stop_conditions import StopCondition
from .training_single_run import GPTrainerSingleProcess


class _SettingsNoop:
    def apply(self):
        return None


def get_settings():
    settings_module_spec = importlib.util.find_spec(f"{__package__.rsplit('.', 1)[0]}.config.settings")
    if settings_module_spec is None:
        return _SettingsNoop()
    settings_module = importlib.import_module(f"{__package__.rsplit('.', 1)[0]}.config.settings")
    getter = getattr(settings_module, "get_settings", None)
    if callable(getter):
        return getter()
    return _SettingsNoop()


class GPTrainer:
    """
    GPTrainer handles the training process of a Gaussian Process model.

    Parameters:
        model (GPModel): The Gaussian Process model to train.
        optimizer_class (torch.optim.Optimizer, optional): The optimizer class to use for training.
        optimizer_kwargs (dict, optional): The arguments for the optimizer, excluding 'params'.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 50.
        seed (int, optional): Random seed for parameter initialization. Defaults to None.
        num_inits (int, optional): Number of initializations. Defaults to 64.
        mll_class (gpytorch.mlls.MarginalLogLikelihood, optional): The Marginal Log Likelihood class to use.
        cholesky_jitter (float, optional): Jitter term for numerical stability in Cholesky. Defaults to 1e-6.
        callbacks (list[Callback]): Optional list of callback objects.
        stop_conditions (list[StopCondition], optional): List of stop conditions to check after each epoch.
            If None, defaults to ConvergencePatienceStopCondition(patience=20) and
            MinLossChangeStopCondition(min_loss_change=1e-7).
        device (str, optional): Device to run on. Defaults to "cpu", but set to "cuda" or "cuda:0"
                                if you have a GPU and want GPU training.
    """

    def __init__(
        self,
        model,
        optimizer_class: torch.optim.Optimizer = None,
        optimizer_kwargs: dict = None,
        scheduler_class: torch.optim.lr_scheduler.LRScheduler = None,
        scheduler_kwargs: dict = None,
        num_epochs: int = 50,
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
    ):
        self.n_jobs = n_jobs
        self.inner_max_num_threads = inner_max_num_threads
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        self.model = model
        logger.info("Model stays on CPU in the constructor.")

        self.num_epochs = num_epochs
        self.num_inits = num_inits
        self.seed = seed
        self.callbacks = callbacks or []
        self.cholesky_jitter = cholesky_jitter
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs
        self.min_epochs = min_epochs

        if stop_conditions is None:
            from .stop_conditions import ConvergencePatienceStopCondition, MinLossChangeStopCondition

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
            logger.warning(f"Model has no dtype attribute. Using {self.dtype} as fallback.")

        if initializer_class is None:
            self.initializer = DefaultParameterInitializer(self.num_inits, seed=self.seed)
        else:
            if initializer_kwargs is None:
                initializer_kwargs = {}
            self.initializer = initializer_class(self.num_inits, seed=self.seed, **initializer_kwargs)

        self.initializer.setup(model)

        if optimizer_class is None:
            self.optimizer_class = LBFGSScipy
            logger.warning("No optimizer class passed. Defaulting to LBFGS Scipy optimizer.")
        else:
            self.optimizer_class = optimizer_class

        if optimizer_kwargs is not None:
            self.optimizer_kwargs = optimizer_kwargs
        else:
            opt_cls = self.optimizer_class
            if opt_cls is LBFGSScipy or (hasattr(opt_cls, "__name__") and opt_cls.__name__ == "LBFGSScipy"):
                self.optimizer_kwargs = {
                    "max_iter": 2000,
                    "max_eval": 5000,
                    "tolerance_grad": 1e-5,
                    "tolerance_change": 1e-9,
                    "history_size": 10,
                }
            elif opt_cls is torch.optim.Adam or (isinstance(opt_cls, type) and issubclass(opt_cls, torch.optim.Adam)):
                self.optimizer_kwargs = {"lr": 0.01}
            else:
                self.optimizer_kwargs = {"max_iter": 20}
                logger.warning(
                    "No optimizer arguments passed and no built-in defaults for "
                    f"{getattr(opt_cls, '__name__', opt_cls)}. Using max_iter=20."
                )

        if mll_class is None:
            self.mll_class = gpytorch.mlls.ExactMarginalLogLikelihood
            logger.warning("No MLL class passed. Defaulting to ExactMarginalLogLikelihood.")
        else:
            self.mll_class = mll_class

    def train_single_process(self, init_index):
        base_model = copy.deepcopy(self.model)
        self.initializer.initialize(base_model, init_index)
        base_model = base_model.to(self.device)

        use_original_callbacks = self.num_inits == 1
        callbacks_copy = self.callbacks if use_original_callbacks else [copy.deepcopy(cb) for cb in self.callbacks]
        stop_conditions_copy = [copy.deepcopy(sc) for sc in self.stop_conditions] if self.stop_conditions else None

        opt_cls = self.optimizer_class
        is_lbfgs_like = (
            opt_cls is LBFGSScipy
            or (isinstance(opt_cls, type) and issubclass(opt_cls, LBFGSScipy))
            or opt_cls is torch.optim.LBFGS
            or (isinstance(opt_cls, type) and issubclass(opt_cls, torch.optim.LBFGS))
        )
        num_epochs_for_run = 1 if is_lbfgs_like else self.num_epochs
        min_epochs_for_run = 1 if is_lbfgs_like else self.min_epochs

        run = GPTrainerSingleProcess(
            model=base_model,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            mll_class=self.mll_class,
            num_epochs=num_epochs_for_run,
            cholesky_jitter=self.cholesky_jitter,
            callbacks=callbacks_copy,
            device=self.device,
            scheduler_class=self.scheduler_class,
            scheduler_kwargs=self.scheduler_kwargs,
            stop_conditions=stop_conditions_copy,
            min_epochs=min_epochs_for_run,
        )
        train_result = run.train()

        with torch.no_grad():
            for (_, param), (_, trained_param) in zip(self.model.named_parameters(), base_model.named_parameters()):
                if param.requires_grad:
                    param.data.copy_(trained_param.data.to(dtype=param.dtype))

        callback_data = {}
        for cb in callbacks_copy:
            if hasattr(cb, "get_stored_parameters"):
                cb_name = cb.__class__.__name__
                stored_params = cb.get_stored_parameters()
                if stored_params:
                    callback_data[cb_name] = stored_params

        if "callback_data" in train_result:
            for key, value in train_result["callback_data"].items():
                if key not in callback_data:
                    callback_data[key] = value
        train_result["callback_data"] = callback_data

        return {"init_index": init_index, **train_result}

    def train_multiple_process_parallel(self, init_indices=None):
        if init_indices is None:
            init_indices = list(range(self.num_inits))
        num_inits_to_train = len(init_indices)
        threadpool_logged = [False]

        def _log_threadpool_once(tag):
            if threadpool_logged[0] or threadpool_info is None:
                return
            try:
                info = threadpool_info()
                summary = ", ".join(
                    f"{e.get('user_api', e.get('prefix', '?'))}={e.get('num_threads', '?')}" for e in info
                )
                logger.info("threadpool_info[%s]: %s", tag, summary)
            except Exception as exc:
                logger.warning("threadpool_info() failed: %s", exc)
            threadpool_logged[0] = True

        def safe_single_process(init_index, device_override=None, tag="worker"):
            try:
                original_device = self.device
                if device_override is not None:
                    self.device = device_override
                _worker_init()
                _log_threadpool_once(tag)
                result = self.train_single_process(init_index)
                self.device = original_device
                return result
            except Exception as e:
                logger.exception(f"Error in training init #{init_index}: {e}")
                return {
                    "init_index": init_index,
                    "state_dict": None,
                    "loss": None,
                    "error": str(e),
                }

        def _worker_init():
            get_settings().apply()
            try:
                seed_val = self.seed if self.seed is not None else 0
                torch.manual_seed(int(seed_val))
            except Exception as exc:
                logger.warning("torch.manual_seed failed in worker: %s", exc)
            if self.device.type == "cpu":
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception as exc:
                    logger.warning("torch.use_deterministic_algorithms failed: %s", exc)
            else:
                try:
                    torch.use_deterministic_algorithms(False)
                except Exception as exc:
                    logger.warning("torch.use_deterministic_algorithms disable failed: %s", exc)

        try:
            seed_val = self.seed if self.seed is not None else 0
            torch.manual_seed(int(seed_val))
        except Exception as exc:
            logger.warning("torch.manual_seed failed in main process: %s", exc)
        if self.device.type == "cpu":
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as exc:
                logger.warning("torch.use_deterministic_algorithms failed in main: %s", exc)
        else:
            try:
                torch.use_deterministic_algorithms(False)
            except Exception as exc:
                logger.warning("torch.use_deterministic_algorithms disable failed in main: %s", exc)

        if self.device.type == "cpu":
            requested_jobs = self.n_jobs if self.n_jobs is not None else max(1, (os.cpu_count() or 1) - 2)
            max_jobs = min(num_inits_to_train, max(1, requested_jobs))
            inner = self.inner_max_num_threads
            blas_ctx = (
                threadpool_limits(limits=inner)
                if (max_jobs == 1 and threadpool_limits is not None and inner is not None)
                else nullcontext()
            )

            if max_jobs == 1:
                logger.info(
                    "Running %d inits in series (n_jobs=1) with BLAS threads pinned to %s.",
                    num_inits_to_train,
                    inner if inner is not None else "default",
                )
                with blas_ctx:
                    results = [safe_single_process(init_index, tag="series_main") for init_index in init_indices]
            else:
                logger.info(
                    "Running %d inits using %d parallel processes on CPU (joblib 'loky', inner_max_num_threads=%s).",
                    num_inits_to_train,
                    max_jobs,
                    inner if inner is not None else "default",
                )
                pc_kwargs = {}
                if inner is not None:
                    pc_kwargs["inner_max_num_threads"] = inner
                with parallel_config(backend="loky", **pc_kwargs):
                    results = Parallel(n_jobs=max_jobs, verbose=0)(
                        delayed(safe_single_process)(init_index, tag="loky_worker") for init_index in init_indices
                    )
        elif str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
            num_gpus = torch.cuda.device_count()
            requested_jobs = self.n_jobs if self.n_jobs is not None else (num_gpus if num_gpus > 0 else 1)
            max_jobs = min(num_inits_to_train, max(1, requested_jobs))
            logger.info(
                f"Running {num_inits_to_train} inits distributed across {num_gpus} GPUs "
                f"(using joblib 'threading' backend with verbose=0; n_jobs={max_jobs})."
            )
            results = Parallel(n_jobs=max_jobs, backend="threading", verbose=0)(
                delayed(safe_single_process)(
                    init_index,
                    device_override=torch.device(f"cuda:{i % max(1, num_gpus)}"),
                    tag="cuda_worker",
                )
                for i, init_index in enumerate(init_indices)
            )
        else:
            results = [safe_single_process(init_index, tag="local") for init_index in init_indices]

        return results

    def train(self):
        results = self.train_multiple_process_parallel(init_indices=None)

        logger.info("Training completed.")

        best_run = None
        best_loss = float("inf")

        for run_result in results:
            if (
                run_result.get("loss") is not None
                and run_result["loss"] < best_loss
                and run_result.get("state_dict") is not None
            ):
                best_loss = run_result["loss"]
                best_run = run_result

        if best_run is not None and best_run.get("state_dict") is not None:
            state = best_run["state_dict"]
            target_device = self.device
            state_on_device = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in state.items()}
            self.model.load_state_dict(state_on_device)
            self.model = self.model.to(target_device)
            jitter = best_run.get("cholesky_jitter")
            if jitter is not None:
                self.model.cholesky_jitter = jitter

            logger.info(
                f"Best init found: #{best_run.get('init_index', 'N/A')} with loss={best_loss:.4f}. "
                "Original model state_dict updated with best weights."
            )
        else:
            logger.warning("No valid best run found. Model was not updated.")

        return results
