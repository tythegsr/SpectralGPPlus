import logging
import torch
import gpytorch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseKernel(torch.nn.Module):

    def __init__():
        pass

    def forward():
        return NotImplemented
    
class GYPytorch_RBFKErnel(BaseKernel):

    def __init__(self):
        self.kernel = gpytorch.kernels.GaussianKernel()

    def forward(self, x):
        self.kernel(x)

class RBFKErnelKronecker(BaseKernel):

    def __init__(self):
        self.kernel_1 = gpytorch.kernels.GaussianKernel()
        self.kernel_2 = gpytorch.kernels.GaussianKernel()
        self.kernel_3 = gpytorch.kernels.GaussianKernel()

    def forward(self, x):
        return self.kernel1(x)*self.kernel2(x)*self.kernel3(x)*


class GPModel():
    """
    GPModel is a Gaussian Process model for regression using GPPlus.

    Parameters:
        train_x (torch.Tensor): Training data features.
        train_y (torch.Tensor): Training data targets.
        likelihood (gpytorch.likelihoods.Likelihood): The likelihood function.
        mean_module (gpytorch.means.Mean, optional): Mean function of the GP.
            Defaults to ConstantMean if None.
        kernel_module (gpytorch.kernels.Kernel, optional): Kernel (covariance) function.
            Defaults to RBFKernel if None.
    
    Raises:
        ValueError: If train_x or train_y are None.
        TypeError: If train_x, train_y, or x in forward is not a torch.Tensor.
    """

    def __init__(self, 
                 train_x: torch.Tensor, 
                 train_y: torch.Tensor, 
                 likelihood: gpytorch.likelihoods.Likelihood,
                 mean_module: gpytorch.means.Mean = None, 
                 kernel_module: BaseKernel = None):
        if mean_module is None:
            mean_module = gpytorch.means.ConstantMean()
            logger.warning("No mean_module provided. Using ConstantMean as default.")
        if kernel_module is None:
            kernel_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            logger.warning("No kernel_module provided. Using RBF Kernel as default.")

        if not isinstance(train_x, torch.Tensor) or not isinstance(train_y, torch.Tensor):
            logger.error("train_x and train_y must be torch.Tensor instances.")
            raise TypeError("train_x and train_y must be torch.Tensor instances.")
        if not isinstance(likelihood, gpytorch.likelihoods.Likelihood):
            logger.error("likelihood must be an instance of gpytorch.likelihoods.Likelihood.")
            raise TypeError("likelihood must be an instance of gpytorch.likelihoods.Likelihood.")

        # Input dimensionality check
        '''
        if train_x.ndimension() != 2:
            logger.error(f"Expected train_x to be 2D, but got {train_x.ndimension()}D.")
            raise ValueError(f"train_x must be 2D, got {train_x.ndimension()}D.")
        if train_y.ndimension() != 1:
            logger.error(f"Expected train_y to be 1D, but got {train_y.ndimension()}D.")
            raise ValueError(f"train_y must be 1D, got {train_y.ndimension()}D.")
        '''

        logger.debug(f'train_x shape: {train_x.shape}, train_y shape: {train_y.shape}')

        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = kernel_module

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Gaussian Process model.

        Parameters:
            x (torch.Tensor): Test data features to make predictions.

        Returns:
            gpytorch.distributions.MultivariateNormal: Predicted mean and covariance for the input data.
        """
        if not isinstance(x, torch.Tensor):
            logger.error("Input x must be a torch.Tensor instance.")
            raise TypeError("Input x must be a torch.Tensor.")

        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return mean, covar
