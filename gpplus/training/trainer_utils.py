import inspect
import os
from typing import Any, Callable, NotRequired, Optional, TypedDict

import torch
from joblib import Parallel, delayed

from ..config import logger
from .optimizers import LBFGSScipy

_WOODBURY_MATMUL_PRECISION_LOGGED = False


def configure_woodbury_matmul_precision(
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    """
    Force full float32 matmul precision for Woodbury / RFF training.

    TF32 on ``Phi^T Phi`` can corrupt eigenvalues / MLL grads enough to let
    train loss decrease while validation collapses. Keep TF32 opt-in only via
    :func:`enable_fast_float32_matmul` for speed experiments — not the default.

    Always re-applies the CUDA flags (safe if TF32 was toggled elsewhere).
    """
    global _WOODBURY_MATMUL_PRECISION_LOGGED
    if dtype is not None and dtype != torch.float32:
        return
    dev = torch.device(device) if device is not None else None
    if dev is not None and dev.type != "cuda":
        return
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    if not _WOODBURY_MATMUL_PRECISION_LOGGED:
        _WOODBURY_MATMUL_PRECISION_LOGGED = True
        logger.info(
            "Woodbury training: TF32 disabled (float32_matmul_precision=highest) "
            "for Gram / MLL numerical stability."
        )


def enable_fast_float32_matmul(
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> bool:
    """
    Opt-in TF32 / high float32 matmul throughput (speed experiments only).

    Not used by default for Woodbury training — see
    :func:`configure_woodbury_matmul_precision`.
    """
    if dtype is not None and dtype != torch.float32:
        return False
    dev = torch.device(device) if device is not None else None
    if dev is not None and dev.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    logger.warning(
        "Enabled fast float32 matmul (TF32). Can degrade Woodbury MLL / "
        "validation; use only for throughput experiments."
    )
    return True


class SingleRunResult(TypedDict):
    loss: float
    state_dict: dict[str, Any]
    callback_data: NotRequired[dict[str, Any]]
    final_lr: NotRequired[float]
    # Set when training aborted mid-run but the best earlier epoch was salvaged.
    error: NotRequired[str]
    aborted: NotRequired[bool]
    aborted_epoch: NotRequired[int]


class RunResult(TypedDict, total=False):
    run_index: int
    loss: Optional[float]
    state_dict: Optional[dict[str, Any]]
    error: str
    aborted: bool
    aborted_epoch: int


def build_run_error(run_index: int, exc: Exception) -> RunResult:
    """Return a consistent error payload for failed runs."""
    logger.exception(f"Error in training run #{run_index}: {exc}")
    return {
        "run_index": run_index,
        "state_dict": None,
        "loss": None,
        "error": str(exc),
    }


def get_optimizer_default_kwargs(optimizer_class) -> dict:
    """Best-effort extraction of constructor defaults excluding params/self."""
    try:
        signature = inspect.signature(optimizer_class.__init__)
    except (TypeError, ValueError):
        return {}

    defaults = {}
    for name, parameter in signature.parameters.items():
        if name in {"self", "params"}:
            continue
        if parameter.default is not inspect.Signature.empty:
            defaults[name] = parameter.default
    return defaults


def get_effective_optimizer_kwargs(optimizer_class, user_kwargs: Optional[dict]) -> dict:
    """Merge optimizer defaults with user overrides for logging."""
    effective = get_optimizer_default_kwargs(optimizer_class)
    if user_kwargs:
        effective.update(user_kwargs)
    return effective


def select_best_run(results: list[RunResult]) -> Optional[RunResult]:
    """Return the best run by minimum loss.

    Includes runs that aborted mid-training but still returned a salvaged
    ``state_dict`` / ``loss`` from an earlier best epoch.
    """
    valid_runs = [
        run_result
        for run_result in results
        if run_result.get("loss") is not None and run_result.get("state_dict") is not None
    ]
    if not valid_runs:
        return None
    return min(valid_runs, key=lambda run_result: run_result["loss"])


def _cpu_parallel_jobs(num_inits: int, n_jobs: Optional[int] = None) -> int:
    if n_jobs is not None:
        return min(num_inits, max(1, n_jobs))
    cpu_count = os.cpu_count() or 1
    # TODO: expose reserved cores as config for cluster-specific tuning.
    reserved_cores = 2
    return min(num_inits, max(1, cpu_count - reserved_cores))


def _collect_parallel_results(
    parallel: Parallel,
    tasks,
    *,
    num_inits: int,
) -> list[RunResult]:
    """Run tasks and log each completion as results stream back from joblib."""
    results: list[RunResult] = []
    for done, result in enumerate(parallel(tasks), start=1):
        run_index = result.get("run_index")
        error = result.get("error")
        loss = result.get("loss")
        if error and result.get("state_dict") is not None and loss is not None:
            logger.warning(
                "Run %s/%s finished (#%s) with mid-train abort; salvaged best "
                "loss=%.6f (epoch crash: %s)",
                done,
                num_inits,
                run_index,
                float(loss),
                error,
            )
        elif error:
            logger.warning(
                "Run %s/%s finished (#%s) with error: %s",
                done,
                num_inits,
                run_index,
                error,
            )
        else:
            loss_text = f"{loss:.6f}" if loss is not None else "n/a"
            logger.info(
                "Run %s/%s finished (#%s) loss=%s",
                done,
                num_inits,
                run_index,
                loss_text,
            )
        results.append(result)
    return results


def run_parallel_initializations(
    num_inits: int,
    trainer_device: torch.device,
    run_callable: Callable[[int, torch.device], RunResult],
    n_jobs: Optional[int] = None,
    parallel_verbose: int = 0,
) -> list[RunResult]:
    """Execute initialization runs across CPU cores or available GPUs."""
    if trainer_device.type == "cpu":
        max_jobs = _cpu_parallel_jobs(num_inits, n_jobs=n_jobs)
        logger.info(f"Running {num_inits} runs using {max_jobs} parallel jobs on {os.cpu_count()} available CPU cores.")
        tasks = (delayed(run_callable)(run_index, trainer_device) for run_index in range(num_inits))
        parallel = Parallel(
            n_jobs=max_jobs,
            backend="loky",
            verbose=parallel_verbose,
            return_as="generator",
        )
        return _collect_parallel_results(parallel, tasks, num_inits=num_inits)

    if trainer_device.type == "cuda":
        torch.cuda.empty_cache()
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            logger.warning("CUDA device selected but no GPUs were detected. Falling back to CPU.")
            cpu_device = torch.device("cpu")
            max_jobs = _cpu_parallel_jobs(num_inits, n_jobs=n_jobs)
            tasks = (delayed(run_callable)(run_index, cpu_device) for run_index in range(num_inits))
            parallel = Parallel(
                n_jobs=max_jobs,
                backend="loky",
                verbose=parallel_verbose,
                return_as="generator",
            )
            return _collect_parallel_results(parallel, tasks, num_inits=num_inits)

        max_jobs = min(num_inits, num_gpus)
        if n_jobs is not None:
            max_jobs = min(max_jobs, max(1, n_jobs))
        logger.info(f"Running {num_inits} runs distributed across {num_gpus} GPUs.")
        tasks = (
            delayed(run_callable)(run_index, torch.device(f"cuda:{run_index % num_gpus}"))
            for run_index in range(num_inits)
        )
        parallel = Parallel(
            n_jobs=max_jobs,
            backend="threading",
            verbose=parallel_verbose,
            return_as="generator",
        )
        return _collect_parallel_results(parallel, tasks, num_inits=num_inits)

    raise ValueError(f"Unsupported training device: {trainer_device}")


def select_epoch_train_fn(
    optimizer,
    standard_epoch_fn: Callable,
    lbfgs_epoch_fn: Callable,
    scipy_lbfgs_epoch_fn: Callable,
) -> Callable:
    """Choose the per-epoch training function based on optimizer class."""
    if isinstance(optimizer, torch.optim.LBFGS):
        return lbfgs_epoch_fn
    if isinstance(optimizer, LBFGSScipy):
        return scipy_lbfgs_epoch_fn
    return standard_epoch_fn


def check_early_stop(
    stop_conditions: list,
    stop_context: dict,
    epoch: int,
    best_loss: float,
    min_epochs: int = 0,
) -> bool:
    """Evaluate stop conditions and log the first stop signal batch."""
    if epoch + 1 < min_epochs:
        return False
    reasons = []
    for stop_condition in stop_conditions:
        stop_now, reason = stop_condition.should_stop(stop_context)
        if stop_now and reason:
            reasons.append(reason)
        if stop_now and not reason:
            reasons.append("Stop condition met")
    if reasons:
        logger.info(
            "Early stopping triggered at epoch %s. Reason: %s. Best loss: %.6f",
            epoch + 1,
            " OR ".join(reasons),
            best_loss,
        )
        return True
    return False
