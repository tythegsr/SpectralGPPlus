import torch
import gpytorch
import gpplus
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPTrainer_v2:
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

        # Initialize model parameters if requested
        if initialize_params:
            self.initialize_parameters(seed)

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

    def initialize_parameters(self, seed=None):
        """
        Initialize the learnable parameters of the model.

        Parameters:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        # Set the random seed for reproducibility, if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Loop over each parameter in the model and initialize based on name
        for name, param in self.model.named_parameters():
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


    def _train_epoch(self):
        """
        Train the model for a single epoch.

        Returns:
            loss (float): The loss value after training for one epoch.
        """
        # Zero gradients from the previous step
        self.optimizer.zero_grad()
        
        # Output from model
        output = self.model(self.train_x)

        # Calculate loss and backpropagate gradients
        loss = -self.mll(output, self.train_y)
        loss.backward()

        # Take a step in the direction of the gradient
        self.optimizer.step()

        return loss.item()
    

    def train(self):
        """
        Train the GP model using the specified number of epochs.
        """
        # Set the model to training mode
        self.model.train()
        #self.likelihood.train()

        logger.info(f"Starting training for {self.num_epochs} epochs.")

        for epoch in range(self.num_epochs):
            # Train for a single epoch
            loss = self._train_epoch()

            # Logging and tracking
            if (epoch + 1) % 10 == 0:
                logger.info(f"Iteration {epoch + 1}/{self.num_epochs} - Loss: {loss:.4f}")


        logger.info("Training completed.")

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