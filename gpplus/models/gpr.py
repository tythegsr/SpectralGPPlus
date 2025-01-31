import os
from typing import Tuple, Union

from gpytorch.models import ExactGP
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.distributions import MultivariateNormal
from gpytorch import settings as gpysettings

import torch

class GPR(ExactGP):
    """
    Gaussian Process Regression base model.
    """
    def __init__(
            self,
            train_x: torch.Tensor,
            train_y: torch.Tensor,
            likelihood: Likelihood,
            mean_module: Mean,
            kernel_module: Kernel,
            **kwargs
        ):
        
        self.kwargs = kwargs

        # Normalize
        train_y = train_y.reshape(-1)
        y_mean = torch.zeros(train_y.shape)
        y_std = torch.ones(train_y.shape)
        train_y_sc = (train_y-y_mean)/y_std

        # Registering mean and std of the raw response
        super(GPR, self).__init__(train_x, train_y_sc, likelihood)

        self.mean_module = mean_module
        self.covar_module = kernel_module

        # Registering mean and std of the raw response
        self.register_buffer('y_mean',y_mean)
        self.register_buffer('y_std',y_std)
        self.register_buffer('y_scaled',train_y_sc)


    def forward(self, x: torch.Tensor, **kwargs) -> MultivariateNormal:
        """
        Forward pass of the Gaussian Process model.

        Parameters:
            x (torch.Tensor): Test data features to make predictions.

        Returns:
            gpytorch.distributions.MultivariateNormal: Predicted mean and covariance for the input data.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor.")
        
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)

    def predict(
        self,
        x: torch.Tensor,
        return_std:bool = False,
        include_noise:bool = False
    )-> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """Returns the predictive mean, and optionally the standard deviation at the given points

        :param x: The input variables at which the predictions are sought. 
        :type x: torch.Tensor
        :param return_std: Standard deviation is returned along the predictions  if `True`. 
            Defaults to `False`.
        :type return_std: bool, optional
        :param include_noise: Noise variance is included in the standard deviation if `True`. 
            Defaults to `False`.
        :type include_noise: bool
        """
        self.eval()
        with gpysettings.fast_computations(log_prob=False):
            output = self(x)
            
            if return_std and include_noise:
                output = self.likelihood(output)

            out_mean = self.y_mean + self.y_std*output.mean

            if return_std:
                out_std = output.variance.sqrt()*self.y_std
                return out_mean, out_std

            return out_mean
        
    def save(self) -> None:
        """
        Save model.
        """
        torch.save(self.state_dict(), os.path.join('./', 'model_weights.pth'))

    def load(self) -> None:
        """
        Load model.
        """
        model_path = os.path.join('./' 'model_weights.pth')
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            self.eval()  # Set the model to evaluation mode after loading
        else:
            raise FileNotFoundError(f"No model weights found at {model_path}")