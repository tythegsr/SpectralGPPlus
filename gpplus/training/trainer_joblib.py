import copy
import logging
import os  # Required to get the number of CPU cores

import gpytorch
import torch
from joblib import Parallel, delayed
from torch.quasirandom import SobolEngine

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


###############################################
###############################################
class GPTrainer:
    """
    GPTrainer handles the training process of a Gaussian Process model.

    Parameters:
        model (GPModel): The Gaussian Process model to train.
        optimizer_class (torch.optim.Optimizer, optional): The optimizer class to use for training.
        optimizer_kwargs (dict, optional): The arguments for the optimizer, excluding 'params'.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 50.
        initialize_params (bool, optional): Whether to initialize model parameters. Defaults to False.
        seed (int, optional): Random seed for parameter initialization. Defaults to None.
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
    ):
        self.model = model
        self.train_x = self.model.train_inputs[0]
        self.train_y = self.model.train_targets
        self.num_epochs = num_epochs
        self.convergence_patience = convergence_patience
        self.num_runs = num_runs
        self.seed = seed
        """
        # Initialize model parameters if requested
        if initialize_params:
            self.initialize_parameters(seed)
        """
        # Get the number of learnable parameters
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Initialize Sobol Engine
        sobol_engine = SobolEngine(dimension=self.num_params, scramble=True, seed=self.seed)

        # Generate all initialization points at once
        self.sobol_samples = sobol_engine.draw(self.num_runs)
        print(f"self.sobol_samples: {self.sobol_samples}")
        print(f"self.sobol_samples.shape: {self.sobol_samples.shape}")

        # Handle optimizer class
        if optimizer_class is None:
            self.optimizer_class = torch.optim.LBFGS  # torch.optim.Adam
            logger.warning("No optimizer class passed. Defaulting to LBFGS optimizer.")
        else:
            self.optimizer_class = optimizer_class

        # Handle optimizer arguments
        if optimizer_kwargs is None:
            # self.optimizer_kwargs = {'lr': 0.01} # , 'line_search_fn': 'strong_wolfe'}
            self.optimizer_kwargs = {"lr": 0.01, "line_search_fn": "strong_wolfe"}
            logger.warning("No optimizer arguments passed. Defaulting to learning rate of 0.01")
        else:
            self.optimizer_kwargs = optimizer_kwargs

        # Select epoch training method based on optimizer
        if self.optimizer_class == torch.optim.LBFGS:
            self._train_single_instance_epoch = self._train_lbfgs_epoch
        else:
            self._train_single_instance_epoch = self._train_standard_epoch

        # Use the GPytorch MLL (marginal log likelihood) as the loss function
        # self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood

    def initialize_parameters(self, model, num_run):
        """
        Initialize the learnable parameters of the model.

        Parameters:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        # Set the random seed for reproducibility, if provided
        # if seed is not None:
        # torch.manual_seed(self.seed)

        idx = 0

        # Loop over each parameter in the model and initialize based on name
        with torch.no_grad():
            for name, param in model.named_parameters():
                param_length = param.numel()
                initial_values = self.sobol_samples[num_run, idx : idx + param_length]
                initial_values = initial_values.reshape(param.shape)

                if param.requires_grad:
                    if ".lengthscale" in name:
                        # Initialize lengthscale and outputscale to 1.0
                        # torch.nn.init.normal_(param, mean=1.0, std=2.0)
                        scale = 3
                        initial_values = initial_values * 2 * scale - scale
                        logger.info(f"lengthscale before for #{num_run}: {param}")
                        param.data = initial_values
                        logger.info(f"lengthscale after for #{num_run}: {param}")

                    elif ".outputscale" in name:
                        # Initialize outputscale to 1.0
                        # torch.nn.init.normal_(param, mean=1.0, std=2.0)
                        lower = 0.1
                        upper = 10
                        initial_values = lower + (upper - lower) * initial_values
                        param.data = initial_values

                    # elif '.noise' in name:
                    #     # Initialize noise parameter to a small positive constant
                    #     # torch.nn.init.constant_(param, 0.0)
                    #     lower = 1e-5
                    #     upper = 1e-2
                    #     initial_values = lower + (upper - lower) * initial_values
                    #     param.data = initial_values

                    elif "weight" in name:
                        # Xavier uniform initialization for weight parameters
                        # torch.nn.init.xavier_uniform_(param)
                        # print(f'param.shape: {name}: {param.shape[0]}, {param.shape[1]}, {param.shape}')

                        logger.info(f"weights before for #{num_run}: {param}")
                        # Xavier/Glorot scaling
                        limit = torch.sqrt(
                            torch.tensor(0.2 / (param.size(1) + param.size(0)))
                        )  # torch.tensor(6.0 / (fan_in + fan_out))
                        print(f"limit: {limit}")

                        initial_values = (initial_values * 2 - 1) * limit

                        param.data = initial_values

                        logger.info(f"weights after for #{num_run}: {param}")

                    elif "bias" in name:
                        # Zero initialization for bias parameters
                        torch.nn.init.zeros_(param)

                    elif "power" in name:
                        # Initialize power to 1.0
                        # torch.nn.init.normal_(param, mean=1.0, std=2.0)
                        lower = -5
                        upper = 10
                        initial_values = lower + (upper - lower) * initial_values
                        param.data = initial_values

                    elif ".raw_noise" in name:
                        # Initialize noise parameter to a small positive constant
                        # torch.nn.init.constant_(param, 0.0)
                        lower = -6 + 0  # Nima  -6 +
                        upper = -6 + 1e-2  # Nima   -6 +
                        initial_values = lower + (upper - lower) * initial_values
                        param.data = initial_values

                    else:  # variants of means and all raws
                        logger.info("*Else*")
                        param.data = 10 * initial_values - 5

                    idx += param_length

                logger.info("Num Param #: {}".format(param_length))

        logger.info("Model parameters initialized with run #: {}".format(num_run))

    def _lbfgs_closure(self, model, optimizer, mll):
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
            output = model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            return loss

        return closure

    def _train_standard_epoch(self, model, optimizer, mll):
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
        output = model(self.train_x)
        loss = -mll(output, self.train_y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _train_lbfgs_epoch(self, model, optimizer, mll):
        """
        Train the model for a single epoch using LBFGS optimizer.

        Parameters:
            model: The Gaussian Process model being trained.
            optimizer: The LBFGS optimizer.
            mll: Marginal Log Likelihood loss.

        Returns:
            float: The loss value after training for one epoch.
        """
        # Get the closure function
        closure = self._lbfgs_closure(model, optimizer, mll)

        # Perform the optimizer step using the closure
        loss = optimizer.step(closure)
        return loss.item()

    # def _train_single_instance(self, model, optimizer, mll):
    #     """
    #     Train the GP model using the specified number of epochs.
    #     """
    #     # Set the model to training mode
    #     model.train()
    #     #self.likelihood.train()

    #     for epoch in range(self.num_epochs):
    #         # Train for a single epoch
    #         loss = self._train_single_instance_epoch(model, optimizer, mll)

    #     return loss

    def _train_single_instance(self, model, optimizer, mll):
        """
        Train the GP model using the specified number of epochs, with optional early stopping.

        Parameters:
            model: The Gaussian Process model being trained.
            optimizer: The optimizer instance for the model.
            mll: Marginal Log Likelihood loss.

        Returns:
            float: The best loss value achieved during training.
        """
        with gpytorch.settings.cholesky_jitter(1e-6):  # Nima
            # Set the model to training mode
            model.train()

            # Local variables for early stopping & Track best values
            best_loss = float("inf")
            best_state_dict = None
            no_improvement_epochs = 0

            for epoch in range(self.num_epochs):
                # Train for a single epoch
                loss = self._train_single_instance_epoch(model, optimizer, mll)

                # Update best-loss and best-state tracking
                if loss < best_loss:
                    best_loss = loss
                    best_state_dict = copy.deepcopy(model.state_dict())
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1

                # Check for early stopping
                if self.convergence_patience is not None and no_improvement_epochs >= self.convergence_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}. Best loss: {best_loss}")
                    break  # Stop training

        # Return the best loss AND the best set of parameters
        return best_loss, best_state_dict

    def single_process(self, num_run):
        # Copy the model
        model_copy = copy.deepcopy(self.model)

        # Initialize parameters for the model copy
        self.initialize_parameters(model_copy, num_run)

        # Create a new optimizer instance for the model copy
        optimizer_copy = self.optimizer_class(model_copy.parameters(), **self.optimizer_kwargs)

        # Create a new MLL for the model copy
        mll_copy = self.mll(model_copy.likelihood, model_copy)

        logger.info(f"Starting training for {self.num_epochs} epochs.")
        # Train the model copy
        best_loss, best_state_dict = self._train_single_instance(model_copy, optimizer_copy, mll_copy)

        print(f"num_run: {num_run}, loss: {best_loss}")
        return {
            "num_run": num_run,
            "state_dict": best_state_dict,  # model_copy.state_dict(),
            "loss": best_loss,
        }  # {seed: model_copy.state_dict(),} # 'loss': loss}

    def multiple_process(self):
        """
        Train the model in parallel using different initialization runs.

        Returns:
            list[dict]: A list of dictionaries containing training results
                        for each run (including error info if something fails).
        """

        # We define a small wrapper to handle errors gracefully
        def safe_single_process(num_run):
            try:
                # Run the actual training job
                return self.single_process(num_run)
            except Exception as e:
                # Log and return an error record for that run
                logger.exception(f"Error in training run #{num_run}: {e}")
                return {
                    "num_run": num_run,
                    "state_dict": None,
                    "loss": None,
                    "error": str(e),
                }

        # Cap the number of parallel jobs to the lesser of available cores or number of runs
        max_jobs = min(
            self.num_runs, max(1, (os.cpu_count() or 1) - 2)
        )  # min(self.num_runs, os.cpu_count() or 1)    # Use 1 if os.cpu_count() returns None

        # Log the number of jobs being used
        logger.info(f"Using {max_jobs} parallel jobs out of {os.cpu_count()} available CPU cores.")

        # Run all the jobs in parallel. If any single job fails in `safe_single_process`,
        # it will just return an error record instead of crashing the entire pool.
        results = Parallel(n_jobs=max_jobs)(delayed(safe_single_process)(num_run) for num_run in range(self.num_runs))

        logger.info("Training completed.")
        return results

    # def multiple_process(self):
    #     # Use joblib to run the worker function in parallel
    #     """
    #     Train the model in parallel using different seeds.

    #     Parameters:
    #         seeds (list[int]): A list of seeds for initializing the model parameters.

    #     Returns:
    #         list[dict]: A list of dictionaries containing training results for each seed.
    #     """
    #     # Cap the number of parallel jobs to the lesser of available cores or number of seeds
    #     #max_jobs = min(len(seeds), os.cpu_count())
    #     max_jobs = min(self.num_runs, os.cpu_count() or 1)  # Use 1 if os.cpu_count() returns None

    #     # Log the number of jobs being used
    #     logger.info(f"Using {max_jobs} parallel jobs out of {os.cpu_count()} available CPU cores.")

    #      # Run the training in parallel
    #     results = Parallel(n_jobs=max_jobs)(delayed(self.single_process)(num_run) for num_run in range(self.num_runs))

    #     logger.info("Training completed.")
    #     return results

    def train(self):
        # Call the multiple_process() method that trains using different initializations
        results = self.multiple_process()

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
            print(f"self_model_before:{self.model.state_dict()}")
            print(f"best_model:{best_run['state_dict']}")

            self.model.load_state_dict(best_run["state_dict"])
            print(f"self_model_after:{self.model.state_dict()}")

            logger.info(
                f"Best run found: #{best_run['num_run']} with loss={best_loss:.4f}. "
                "Original model state_dict updated with best weights."
            )
        else:
            logger.warning("No valid best run found. Model was not updated.")

        return results


###############################################
###############################################
