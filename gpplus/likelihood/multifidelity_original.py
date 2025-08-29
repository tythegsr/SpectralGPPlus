# Copyright © 2025, Dr. Bostanabad's research group at the University of California, Irvine.
# 
# GP+ Intellectual Property Notice:

from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import _HomoskedasticNoiseBase
import torch
from typing import Any, Optional
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor, ConstantDiagLazyTensor
from torch import Tensor

class MultifidelityLikelihood(_GaussianLikelihoodBase):
    
    """
    Multifidelity Gaussian likelihood that adds different noise levels
    depending on a fidelity index for each data point.

    Args:
        fidel_indices (torch.Tensor):
            Tensor of shape (N,) indicating the fidelity level for each point.
        noise_indices (List[int], optional):
            Which fidelity levels should have learnable noise. Defaults to [1].
        noise_prior (Prior, optional):
            Prior over the noise parameters.
        noise_constraint (Constraint, optional):
            Constraint on the noise parameters.
        learn_additional_noise (bool, optional):
            If True, allows an extra global noise term to be learned.
        batch_shape (torch.Size, optional):
            Batch shape for batched inputs.
        **kwargs:
            Other keyword arguments forwarded to _GaussianLikelihoodBase.
    """

    def __init__(self, fidel_indices: Tensor,noise_indices: list = [1], noise_prior = None, noise_constraint = None, 
        learn_additional_noise = False, batch_shape = torch.Size(), **kwargs) -> None:
        # num_noises = len(noise_indices)
        num_noises = len(noise_indices)
        noise_covar = MultifidelityNoise(noise_prior=noise_prior, 
        noise_constraint=noise_constraint, batch_shape=batch_shape, num_noises = num_noises)
        
        """
        Initialize the multifidelity likelihood.

        Args:
            fidel_indices (torch.Tensor):
                Fidelity indices per training point.
            noise_indices (List[int], optional):
                Fidelity levels treated as noisy.
            noise_prior (Prior, optional):
                Prior distribution on each noise level.
            noise_constraint (Constraint, optional):
                Constraint applied to each noise parameter.
            learn_additional_noise (bool, optional):
                Whether to learn an extra homoskedastic noise term.
            batch_shape (torch.Size, optional):
                Batch dimensions for batched likelihoods.
        """
                
        super().__init__(noise_covar = noise_covar)

        self.fidel_indices = fidel_indices
        self.noise_indices = noise_indices

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)


    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        # This runs the forward method in noise class
        # Shape is not used any more.
        return self.noise_covar(*params, fidel_indices = self.fidel_indices, noise_indices = self.noise_indices)
    

    def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)





class MultifidelityNoise(_HomoskedasticNoiseBase):
    def __init__(self, noise_prior=None, noise_constraint=None, batch_shape= torch.Size(), num_noises=1):
        super().__init__(noise_prior, noise_constraint, batch_shape, num_tasks= num_noises)


    def forward(self, *params: Any, shape: Optional[torch.Size] = None, 
        fidel_indices: Tensor, noise_indices: list, **kwargs: Any) -> DiagLazyTensor:
        
        """
        Compute a diagonal noise covariance where each point’s variance
        is selected by its fidelity index.

        Args:
            *params:
                Parameters from the base noise model.
            shape (torch.Size, optional):
                Unused—shape is inferred from fidel_indices.
            fidel_indices (torch.Tensor):
                Tensor of shape (N,) mapping each point to a fidelity level.
            noise_indices (List[int]):
                Which fidelity levels correspond to each learned noise parameter.
            **kwargs:
                Unused.

        Returns:
            DiagLazyTensor:
                A (N×N) diagonal covariance tensor where entry i,i
                = noise[fidelity_indices[i]].
        """

        if len(fidel_indices) ==0 or fidel_indices is None:
            raise ValueError('You need to specify a list of indices for noise such as [1,3]')
        # This contains a list of diagonal matrices with defined noise. Crates [batch * 1 * noise_size * n * n]
        covar = super().forward(*params, shape= fidel_indices.shape, **kwargs)

        if covar.dim() > 2:
            if covar.shape[1] is not len(noise_indices):
                raise ValueError('Something is wrong, number of noise and indices are not the same')

        if covar.dim() == 4: # no batch
            covar = covar.squeeze(0)
        elif covar.dim() == 5: # for batch
            covar = covar.squeeze(1) 


        # This part is for categorical_indices
        temp = ConstantDiagLazyTensor(torch.tensor([0.0]), len(fidel_indices))
        temp = temp.to(dtype=covar.dtype, device=covar.device)
        for i in range(len(noise_indices)):
            if i ==0:
                diag = DiagLazyTensor( (fidel_indices == noise_indices[i]) )#.type(torch.int32)
                if covar.dim() == 4: # batch
                    temp += diag * covar[:,i,...]
                
                elif covar.dim() == 3: # 
                    temp += diag * covar[i,...]

                elif covar.dim() == 2:
                    temp += diag * covar
                else:
                    raise ValueError('Covar is 1D? why?')
            else :
                diag = DiagLazyTensor( (fidel_indices == noise_indices[i]).type(torch.int32) )
                if covar.dim() == 4: # batch
                    temp += diag * covar[:,i,...]
                
                elif covar.dim() == 3: # 
                    temp += diag * covar[i,...]

                elif covar.dim() == 2:
                    temp += diag * covar
                else:
                    raise ValueError('Covar is 1D? why?')
        
        return temp

