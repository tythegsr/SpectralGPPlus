import torch

from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import NormalPrior
from gpytorch.kernels import ScaleKernel
from gpytorch.constraints import Positive

from gpplus.models.mogp import KroneckerMOGP
from gpplus.priors.horseshoe import LogHalfHorseshoePrior
from gpplus.kernels.factory import KernelFactory, KernelType
from gpplus.utils.transforms import softplus, inv_softplus

class FileldModel(KroneckerMOGP):
    """ Multi output Kronecker Gaussian Process
    """
    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        **kwargs
    ) -> None:   

        # Likelihood
        lb_noise = 1e-12
        noise_constraint=GreaterThan(lb_noise, transform = torch.exp, inv_transform = torch.log)
        likelihood = GaussianLikelihood(num_tasks = 1, noise_constraint = noise_constraint)

        # Split data to exploit structure
        self.q = 2
        phi_indices = list(range(self.q, x_train.shape[1]))
        y_indices = list(range(self.q))
        column_indices = [phi_indices, y_indices]

        # Kernel
        kernel_phi = KernelFactory.create_kernel(
            KernelType.RBFKernel,
            ard_num_dims=len(phi_indices),
        )

        kernel_phi_scaled = ScaleKernel(
            base_kernel = kernel_phi,
            outputscale_constraint = Positive(transform = softplus, inv_transform = inv_softplus),
        )

        kernel_y = KernelFactory.create_kernel(
            KernelType.RBFKernel,
            ard_num_dims=len(y_indices),
        )

        kernels = [kernel_phi_scaled, kernel_y]

        # Mean
        mean_module = ConstantMean(prior = NormalPrior(0.,1.))

        # Parent class initialization
        KroneckerMOGP.__init__(self, x_train, y_train, likelihood, mean_module, kernels, column_indices, **kwargs)

        # Initialize likelihood
        self.likelihood.initialize(noise=0.05**2)
        self.likelihood.register_prior('noise_prior', LogHalfHorseshoePrior(0.2,lb_noise),'raw_noise')

        # Initialize mean
        self.mean_module.constant.data = torch.tensor([0.0])
        self.mean_module.constant.requires_grad = False