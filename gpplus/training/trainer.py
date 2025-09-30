import copy
import os
from typing import List, Optional

import gpytorch
import scipy
import torch
from joblib import Parallel, delayed

from ..config import logger
from .callbacks import Callback
from .parameter_initializer import DefaultParameterInitializer, ParameterInitializer
from .training_single_run import GPTrainerSingleProcess


class GPTrainer:
    """
    GPTrainer handles the training process of a Gaussian Process model.

    Parameters:
        model (GPModel): The Gaussian Process model to train.
        optimizer_class (torch.optim.Optimizer, optional): The optimizer class to use for training.
        optimizer_kwargs (dict, optional): The arguments for the optimizer, excluding 'params'.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 50.
        convergence_patience (int, optional): Early stopping patience. Defaults to 20.
        seed (int, optional): Random seed for parameter initialization. Defaults to None.
        num_runs (int, optional): Number of runs (initializations). Defaults to 64.
        mll_class (gpytorch.mlls.MarginalLogLikelihood, optional): The Marginal Log Likelihood class to use.
        cholesky_jitter (float, optional): Jitter term for numerical stability in Cholesky. Defaults to 1e-6.
        callbacks (list[Callback]): Optional list of callback objects.
        device (str, optional): Device to run on. Defaults to "cpu", but set to "cuda" or "cuda:0"
                                if you have a GPU and want GPU training.
    """

    def __init__(
        self,
        model,
        optimizer_class: torch.optim.Optimizer = None,
        optimizer_kwargs: dict = None,
        num_epochs: int = 50,
        convergence_patience=20,  # Stop if no improvement for 20 epochs
        seed: int = None,
        num_runs: int = 64,
        mll_class: gpytorch.mlls.MarginalLogLikelihood = None,
        cholesky_jitter: float = 1e-6,
        callbacks: Optional[List[Callback]] = None,
        initializer_class: ParameterInitializer = None,
        initializer_kwargs: dict = None,
        device: str = "cpu",
        map_prior: bool = False,
        track_loocv: bool = True,
        loocv_log_freq: int = 50,
        use_loocv_objective: bool = False,
        min_loss_change: float = 1e-7,
        dtype: torch.dtype = torch.float64,
    ):
        # -------------------------------------------------------
        # Set up the device (CPU or CUDA)
        # -------------------------------------------------------
        # If the user sets device="cuda" but CUDA is not available, fall back to CPU.
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        # Keep the original model on CPU
        self.model = model  # no .to(self.device) here
        logger.info("Model stays on CPU in the constructor.")

        # --------------------------------------------------
        #  CORE CONFIG
        # --------------------------------------------------
        self.num_epochs = num_epochs
        self.convergence_patience = convergence_patience
        self.num_runs = num_runs
        self.seed = seed
        self.callbacks = callbacks or []
        self.cholesky_jitter = cholesky_jitter
        self.map_prior = map_prior
        self.track_loocv = track_loocv
        self.loocv_log_freq = loocv_log_freq
        self.use_loocv_objective = use_loocv_objective
        self.min_loss_change = min_loss_change
        self.dtype = dtype

        """
        # Initialize model parameters if requested
        if initialize_params:
            self.initialize_parameters(seed)
        """
        # Set up the initializer; use a default one if none is provided.
        if initializer_class is None:
            self.initializer = DefaultParameterInitializer(num_runs=self.num_runs, seed=self.seed)
        else:
            # Pass initializer_kwargs if provided, otherwise use empty dictionary
            if initializer_kwargs is None:
                initializer_kwargs = {}
            self.initializer = initializer_class(num_runs=self.num_runs, seed=self.seed, **initializer_kwargs)

        # Precompute number of parameters and Sobol samples.
        self.initializer.setup(model)

        # --------------------------------------------------
        #  OPTIMIZER
        # --------------------------------------------------
        # Handle optimizer class, use LBFGS as default
        if optimizer_class is None:
            self.optimizer_class = scipy.optimize.LBFGS  # torch.optim.Adam # scipy.optimize.LBFGS # torch.optim.LBFGS
            logger.warning("No optimizer class passed. Defaulting to LBFGS optimizer.")
        else:
            self.optimizer_class = optimizer_class

        # Handle optimizer arguments
        if optimizer_kwargs is None:
            self.optimizer_kwargs = {"lr": 0.1, "line_search_fn": "strong_wolfe"}
            logger.warning("No optimizer arguments passed. Defaulting to learning rate of 0.1")
        else:
            self.optimizer_kwargs = optimizer_kwargs

        # Handle MLL class
        if mll_class is None:
            # Use the GPytorch MLL (marginal log likelihood) as the loss function
            self.mll_class = gpytorch.mlls.ExactMarginalLogLikelihood
            logger.warning("No MLL class passed. Defaulting to ExactMarginalLogLikelihood.")
        else:
            self.mll_class = mll_class

    def train_single_process(self, run_index):
        """
        Runs training for a single initialization (run_index).
        - Copy the master CPU-based model
        - Initialize on CPU
        - Move the copy to GPU (if device is CUDA)
        - Train the copy
        - Return best loss + best state
        """
        # Copy the model (which is on CPU)
        base_model = copy.deepcopy(self.model)

        # Initialize parameters for the model copy on CPU using the initializer
        self.initializer.initialize(base_model, run_index)
        # Snapshot initialized state dict before training
        initial_state_dict = copy.deepcopy(base_model.state_dict())

        # Move model_copy to device
        base_model = base_model.to(self.device)

        # Train the model
        # Create isolated callback instances per run to avoid cross-run state mixing
        callbacks_copy = [copy.deepcopy(cb) for cb in self.callbacks] if self.callbacks else []

        run = GPTrainerSingleProcess(
            model=base_model,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            mll_class=self.mll_class,
            num_epochs=self.num_epochs,
            convergence_patience=self.convergence_patience,
            cholesky_jitter=self.cholesky_jitter,
            callbacks=callbacks_copy,
            device=self.device,
            map_prior=self.map_prior,
            track_loocv=self.track_loocv,
            loocv_log_freq=self.loocv_log_freq,
            use_loocv_objective=self.use_loocv_objective,
            min_loss_change=self.min_loss_change,
            dtype=self.dtype,
        )
        train_result = run.train()

        # Copy the trained parameters back to the original model
        # This ensures constraint enforcement is preserved
        with torch.no_grad():
            for (name, param), (_, trained_param) in zip(self.model.named_parameters(), base_model.named_parameters()):
                if param.requires_grad:
                    param.data.copy_(trained_param.data.to(dtype=param.dtype))

        return {"run_index": run_index, "initial_state_dict": initial_state_dict, **train_result}

    def train_multiple_process_parallel(self):
        """
        Train the model in parallel using different initialization runs.

        Returns:
            list[dict]: A list of dictionaries containing training results
                        for each run (including error info if something fails).
        """

        # defining a small wrapper to handle errors gracefully
        def safe_single_process(run_index, device_override=None):
            try:
                # Run the actual training job
                original_device = self.device
                if device_override is not None:
                    # Temporarily override the device for this run.
                    self.device = device_override
                result = self.train_single_process(run_index)
                # Restore the original device.
                self.device = original_device
                return result
            except Exception as e:
                # Log and return an error record for that run
                logger.exception(f"Error in training run #{run_index}: {e}")
                return {
                    "run_index": run_index,
                    "state_dict": None,
                    "loss": None,
                    "error": str(e),
                }

        # Cap the number of parallel jobs
        if self.device.type == "cpu":
            max_jobs = min(self.num_runs, max(1, (os.cpu_count() or 1) - 2))
            logger.info(
                f"Running {self.num_runs} runs using {max_jobs} parallel jobs on {os.cpu_count()} available CPU cores."
            )
            results = Parallel(n_jobs=max_jobs, backend="threading", verbose=11)(
                delayed(safe_single_process)(run_index) for run_index in range(self.num_runs)
            )

        elif str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
            num_gpus = torch.cuda.device_count()
            # Allow as many parallel jobs as there are GPUs.
            max_jobs = min(self.num_runs, num_gpus)
            logger.info(f"Running {self.num_runs} runs distributed across {num_gpus} GPUs.")

            results = Parallel(n_jobs=max_jobs, backend="threading", verbose=11)(
                # For each run, choose a GPU device based on the run index.
                delayed(safe_single_process)(run_index, device_override=torch.device(f"cuda:{run_index % num_gpus}"))
                for run_index in range(self.num_runs)
            )

        logger.info("Training completed.")
        return results

    def train(self):
        # Call the multiple_process() method that trains using different initializations
        results = self.train_multiple_process_parallel()

        # ------------------------------------------------------
        #  Select the best run by comparing the 'loss' values
        # ------------------------------------------------------
        best_run = None
        best_loss = float("inf")

        for run_result in results:
            if (
                run_result["loss"] is not None
                and run_result["loss"] < best_loss
                and run_result["state_dict"] is not None
            ):
                best_loss = run_result["loss"]
                best_run = run_result

        # ------------------------------------------------------
        #  If a valid best run was found, load it into self.model
        # ------------------------------------------------------
        if best_run is not None and best_run["state_dict"] is not None:
            self.model.load_state_dict(best_run["state_dict"])

            logger.info(
                f"Best run found: #{best_run['run_index']} with loss={best_loss:.4f}. "
                "Original model state_dict updated with best weights."
            )
        else:
            logger.warning("No valid best run found. Model was not updated.")

        # Attach best initial state dict to results for external consumers
        for r in results:
            if r.get("run_index") == (best_run.get("run_index") if best_run else None):
                r["best_initial_state_dict"] = best_run.get("initial_state_dict") if best_run else None
                break

        return results
