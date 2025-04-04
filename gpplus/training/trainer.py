import copy
import os
from typing import List, Optional

import gpytorch
import torch
from joblib import Parallel, delayed
from torch.quasirandom import SobolEngine

from ..config import logger
from .callbacks import Callback
from .training_run import GPTrainingRun


class GPTrainer:
    def __init__(
        self,
        model,
        optimizer_class=None,
        optimizer_kwargs=None,
        mll_class: gpytorch.mlls.MarginalLogLikelihood = None,
        callbacks: Optional[List[Callback]] = None,
        num_epochs=50,
        convergence_patience=20,
        seed=None,
        num_runs=64,
        device: str = "cpu",
    ):
        # Determine the device (CPU or GPU)
        self.device = device

        self.model = model
        self.callbacks = callbacks or []
        self.num_epochs = num_epochs
        self.convergence_patience = convergence_patience
        self.seed = seed
        self.num_runs = num_runs

        self.optimizer_class = optimizer_class or torch.optim.LBFGS
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 0.01, "line_search_fn": "strong_wolfe"}
        self.mll_class = mll_class or gpytorch.mlls.ExactMarginalLogLikelihood

        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        sobol_engine = SobolEngine(dimension=self.num_params, scramble=True, seed=self.seed)
        self.sobol_samples = sobol_engine.draw(self.num_runs)

    def train_single_process(self, run_index):
        sobol_sample = self.sobol_samples[run_index]

        # Copy the model and move the internal model inputs and targets to the same device as the model
        base_model = copy.deepcopy(self.model).to(self.device)
        base_model.likelihood.to(self.device)
        base_model.train_inputs = [x.to(self.device) for x in self.model.train_inputs]  # Move each input tensor
        base_model.train_targets = self.model.train_targets.to(self.device)  # Move targets to the same device

        # Train the model
        run = GPTrainingRun(
            model=base_model,
            sobol_sample=sobol_sample,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            mll_class=self.mll_class,
            num_epochs=self.num_epochs,
            convergence_patience=self.convergence_patience,
            callbacks=self.callbacks,
            device=self.device,
        )
        return {"num_run": run_index, **run.train()}

    def train_multiple_runs_parallel(self):
        # We define a small wrapper to handle errors gracefully
        def safe_single_process(index):
            try:
                # Run the actual training job
                return self.train_single_process(index)
            except Exception as e:
                # Log and return an error record for that run
                logger.exception(f"Run {index} failed.")
                return {"num_run": index, "loss": None, "state_dict": None, "error": str(e)}

        if self.device == "cpu":
            max_jobs = min(self.num_runs, max(1, (os.cpu_count() or 1) - 2))
            logger.info(f"Running {self.num_runs} runs using {max_jobs} parallel jobs.")
            return Parallel(n_jobs=max_jobs)(delayed(safe_single_process)(i) for i in range(self.num_runs))
        elif self.device == "cuda":
            logger.warning(f"Using device {self.device}, only 1 run is allowed.")
            self.num_runs = 1
            num_gpus = torch.cuda.device_count()
            max_jobs = min(self.num_runs, num_gpus)
            logger.info(f"Running {self.num_runs} runs using {num_gpus} GPUs.")
            return Parallel(n_jobs=max_jobs)(delayed(safe_single_process)(i) for i in range(self.num_runs))
        else:
            raise TypeError(f"Device {self.device} not supported")

    def train(self):
        # Call the multiple_process() method that trains using different initializations
        results = self.train_multiple_runs_parallel()

        # ------------------------------------------------------
        #  Select the best run by comparing the 'loss' values
        # ------------------------------------------------------
        best_run = None
        best_loss = float("inf")

        for r in results:
            if r["state_dict"] and r["loss"] < best_loss:
                best_loss = r["loss"]
                best_run = r

        # ------------------------------------------------------
        #  If a valid best run was found, load it into self.model
        # ------------------------------------------------------
        if best_run:
            self.model.load_state_dict(best_run["state_dict"])
            logger.info(f"Best run: #{best_run['num_run']} with loss={best_loss:.4f}")
        else:
            logger.warning("No valid runs found.")

        return results
