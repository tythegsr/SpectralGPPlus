import torch
import gpytorch
import gpplus
import logging
from joblib import Parallel, delayed
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPTrainer_v3:
    """
    GPTrainer handles the training process of a Gaussian Process model.

    Parameters:
        model (GPModel): The Gaussian Process model to train.
        train_x (torch.Tensor): Training data features.
        train_y (torch.Tensor): Training data targets.
        optimizer (torch.optim.Optimizer, optional): The optimizer to use for training.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 50.
        initialize_params (bool, optional): Whether to initialize model parameters. Defaults to False.
        seed (int, optional): Random seed for parameter initialization. Defaults to None.
    """

    def __init__(self, 
                 model,
                 train_x: torch.Tensor, 
                 train_y: torch.Tensor, 
                 optimizer: torch.optim.Optimizer = None,
                 num_epochs: int = 50,
                 initialize_params: bool = False,
                 seed: int = None):
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.num_epochs = num_epochs
        '''
        # Initialize model parameters if requested
        if initialize_params:
            self.initialize_parameters(seed)
        '''
        # If no optimizer is passed, use Adam as the default optimizer with learning rate 0.1
        if optimizer is None:
            self.learning_rate = 0.1  # Internal default learning rate
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            logger.warning("No optimizer passed. Defaulting to Adam optimizer with learning rate of {}".format(self.learning_rate))
        else:
            self.optimizer = optimizer
            self.learning_rate = self.optimizer.param_groups[0]["lr"]

        # Use the GPytorch MLL (marginal log likelihood) as the loss function
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

    def initialize_parameters(self, model, seed=None):
        """
        Initialize the learnable parameters of the model.

        Parameters:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        # Set the random seed for reproducibility, if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Loop over each parameter in the model and initialize based on name
        for name, param in model.named_parameters():
            if 'lengthscale' in name or 'outputscale' in name:
                # Initialize lengthscale and outputscale to 1.0
                torch.nn.init.normal_(param, mean=1.0, std=2.0)
            elif 'noise' in name:
                # Initialize noise parameter to a small positive constant
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                # Xavier uniform initialization for weight parameters
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Zero initialization for bias parameters
                torch.nn.init.zeros_(param)

        logger.info("Model parameters initialized with seed: {}".format(seed))


    # def _train_epoch(self):
    #     """
    #     Train the model for a single epoch.

    #     Returns:
    #         loss (float): The loss value after training for one epoch.
    #     """
    #     # Zero gradients from the previous step
    #     self.optimizer.zero_grad()
        
    #     # Output from model
    #     output = self.model(self.train_x)

    #     # Calculate loss and backpropagate gradients
    #     loss = -self.mll(output, self.train_y)
    #     loss.backward()

    #     # Take a step in the direction of the gradient
    #     self.optimizer.step()

    #     return loss.item()
    

    # def train(self):
    #     """
    #     Train the GP model using the specified number of epochs.
    #     """
    #     # Set the model to training mode
    #     self.model.train()
    #     #self.likelihood.train()

    #     logger.info(f"Starting training for {self.num_epochs} epochs.")

    #     for epoch in range(self.num_epochs):
    #         # Train for a single epoch
    #         loss = self._train_epoch()

    #         # Logging and tracking
    #         if (epoch + 1) % 10 == 0:
    #             logger.info(f"Iteration {epoch + 1}/{self.num_epochs} - Loss: {loss:.4f}")

    #     logger.info("Training completed.")


    def _train_single_instance_epoch(self, model, optimizer, mll):
        """
        Train the model for a single epoch.

        Returns:
            loss (float): The loss value after training for one epoch.
        """
        # Zero gradients from the previous step
        optimizer.zero_grad()
        
        # Output from model
        output = model(self.train_x)

        # Calculate loss and backpropagate gradients
        loss = -mll(output, self.train_y)
        loss.backward()

        # Take a step in the direction of the gradient
        optimizer.step()

        return loss.item()


    def _train_single_instance(self, model, optimizer, mll):
        """
        Train the GP model using the specified number of epochs.
        """
        # Set the model to training mode
        model.train()
        #self.likelihood.train()

        logger.info(f"Starting training for {self.num_epochs} epochs.")

        for epoch in range(self.num_epochs):
            # Train for a single epoch
            loss = self._train_single_instance_epoch(model, optimizer, mll)

        return loss


    def evaluate(self, test_x: torch.Tensor):
        """
        Evaluate the model on test data.

        Parameters:
            test_x (torch.Tensor): Test data features.

        Returns:
            mean (torch.Tensor): Predictive mean for each test point.
            lower (torch.Tensor): Lower confidence bound for each test point.
            upper (torch.Tensor): Upper confidence bound for each test point.
        """
        # Set the model to evaluation mode
        self.model.eval()
        #self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Make predictions
            observed_pred = self.model.likelihood(self.model(test_x))

            # Get the mean, lower and upper confidence bounds
            mean = observed_pred.mean
            lower, upper = observed_pred.confidence_region()

            logger.info("Evaluation completed.")
            return mean, lower, upper
        

    
    def single_process(self, seed):
        # Copy the model
        model_copy = copy.deepcopy(self.model)
        #optimizer_copy = copy.deepcopy(self.optimizer)
        #mll_copy = copy.deepcopy(self.mll)
        # Create a new optimizer for the model copy
        optimizer_copy = torch.optim.Adam(model_copy.parameters(), lr=self.learning_rate)

        # Create a new MLL for the model copy
        mll_copy = gpytorch.mlls.ExactMarginalLogLikelihood(model_copy.likelihood, model_copy)

        # Initialize parameters for the model copy
        self.initialize_parameters(model_copy, seed)
        loss = self._train_single_instance(model_copy, optimizer_copy, mll_copy)

        # Collecting the state_dict and named_parameters in a readable format
        #state_dict_output = "State Dict:\n" + "\n".join([f"{key}: {value}" for key, value in model_copy.state_dict().items()])
        #named_parameters_output = "\nNamed Parameters:\n" + "\n".join([f"{name}: {param}" for name, param in model_copy.named_parameters()])

        #return f"Results for seed {seed}:\n{state_dict_output}\n{named_parameters_output}\n"

        print(f'seed: {seed}, loss: {loss}')
        return {'seed': seed, 'state_dict': model_copy.state_dict(), 'loss': loss}    #{seed: model_copy.state_dict(),} # 'loss': loss}
    
    def multiple_process(self, seeds):
        # Use joblib to run the worker function in parallel
        results = Parallel(n_jobs=len(seeds))(delayed(self.single_process)(seed) for seed in seeds)
        return results
    









###############################################
###############################################
class GPTrainer_v4:
    """
    GPTrainer handles the training process of a Gaussian Process model.

    Parameters:
        model (GPModel): The Gaussian Process model to train.
        train_x (torch.Tensor): Training data features.
        train_y (torch.Tensor): Training data targets.
        optimizer_class (torch.optim.Optimizer, optional): The optimizer class to use for training.
        optimizer_kwargs (dict, optional): The arguments for the optimizer, excluding 'params'.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 50.
        initialize_params (bool, optional): Whether to initialize model parameters. Defaults to False.
        seed (int, optional): Random seed for parameter initialization. Defaults to None.
    """

    def __init__(self, 
                 model,
                 train_x: torch.Tensor, 
                 train_y: torch.Tensor, 
                 optimizer_class: torch.optim.Optimizer = None,
                 optimizer_kwargs: dict = None,
                 num_epochs: int = 50,
                 initialize_params: bool = False,
                 seed: int = None):
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.num_epochs = num_epochs
        '''
        # Initialize model parameters if requested
        if initialize_params:
            self.initialize_parameters(seed)
        '''

        # Handle optimizer class
        if optimizer_class is None:
            self.optimizer_class = torch.optim.Adam
            logger.warning("No optimizer class passed. Defaulting to Adam optimizer.")
        else:
            self.optimizer_class = optimizer_class

        # Handle optimizer arguments
        if optimizer_kwargs is None:
            self.optimizer_kwargs = {'lr': 0.1}
            logger.warning("No optimizer arguments passed. Defaulting to learning rate of 0.1")
        else:
            self.optimizer_kwargs = optimizer_kwargs

        # Use the GPytorch MLL (marginal log likelihood) as the loss function
        #self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood

    def initialize_parameters(self, model, seed=None):
        """
        Initialize the learnable parameters of the model.

        Parameters:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        # Set the random seed for reproducibility, if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Loop over each parameter in the model and initialize based on name
        for name, param in model.named_parameters():
            if 'lengthscale' in name or 'outputscale' in name:
                # Initialize lengthscale and outputscale to 1.0
                torch.nn.init.normal_(param, mean=1.0, std=2.0)
            elif 'noise' in name:
                # Initialize noise parameter to a small positive constant
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                # Xavier uniform initialization for weight parameters
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Zero initialization for bias parameters
                torch.nn.init.zeros_(param)

        logger.info("Model parameters initialized with seed: {}".format(seed))



    def _train_single_instance_epoch(self, model, optimizer, mll):
        """
        Train the model for a single epoch.

        Returns:
            loss (float): The loss value after training for one epoch.
        """
        # Zero gradients from the previous step
        optimizer.zero_grad()
        
        # Output from model
        output = model(self.train_x)

        # Calculate loss and backpropagate gradients
        loss = -mll(output, self.train_y)
        loss.backward()

        # Take a step in the direction of the gradient
        optimizer.step()

        return loss.item()


    def _train_single_instance(self, model, optimizer, mll):
        """
        Train the GP model using the specified number of epochs.
        """
        # Set the model to training mode
        model.train()
        #self.likelihood.train()

        for epoch in range(self.num_epochs):
            # Train for a single epoch
            loss = self._train_single_instance_epoch(model, optimizer, mll)

        return loss

            
    def single_process(self, seed):
        # Copy the model
        model_copy = copy.deepcopy(self.model)

        # Initialize parameters for the model copy
        self.initialize_parameters(model_copy, seed)

        # Create a new optimizer instance for the model copy
        optimizer_copy = self.optimizer_class(model_copy.parameters(), **self.optimizer_kwargs)

        # Create a new MLL for the model copy
        mll_copy = self.mll(model_copy.likelihood, model_copy)

        logger.info(f"Starting training for {self.num_epochs} epochs.")
        # Train the model copy
        loss = self._train_single_instance(model_copy, optimizer_copy, mll_copy)


        print(f'seed: {seed}, loss: {loss}')
        return {'seed': seed, 'state_dict': model_copy.state_dict(), 'loss': loss}    #{seed: model_copy.state_dict(),} # 'loss': loss}
    
    def multiple_process(self, seeds):
        # Use joblib to run the worker function in parallel
        results = Parallel(n_jobs=len(seeds))(delayed(self.single_process)(seed) for seed in seeds)
        logger.info("Training completed.")
        return results
    