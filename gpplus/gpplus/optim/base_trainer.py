import torch
import gpytorch
import gpplus
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPTrainer:
    """
    GPTrainer handles the training process of a Gaussian Process model.

    Parameters:
        model (GPModel): The Gaussian Process model to train.
        train_x (torch.Tensor): Training data features.
        train_y (torch.Tensor): Training data targets.
        optimizer (torch.optim.Optimizer, optional): The optimizer to use for training.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 50.
    """

    def __init__(self, 
                 model,
                 train_x: torch.Tensor, 
                 train_y: torch.Tensor, 
                 optimizer: torch.optim.Optimizer = None,
                 num_epochs: int = 50):
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.num_epochs = num_epochs

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

    def train(self):
        """
        Train the GP model using the specified number of epochs.
        """
        # Set the model to training mode
        self.model.train()
        #self.likelihood.train()

        logger.info(f"Starting training for {self.num_epochs} epochs.")

        for epoch in range(self.num_epochs):
            # Zero gradients from previous step
            self.optimizer.zero_grad()
            
            # Output from model
            output = self.model(self.train_x)

            # Calculate loss and backpropagate gradients
            loss = -self.mll(output, self.train_y)
            loss.backward()

            logger.info(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item()}')

            # Take a step in the direction of the gradient
            self.optimizer.step()

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