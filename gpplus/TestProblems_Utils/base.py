import numpy as np
import torch


class RangeException(Exception):
    pass

class DimensionException(Exception):
    pass

class BenchmarkProblem():

    """
    Base class for Bayesian Optimization benchmark problems.
    """

    def __init__(self, dim = 1, num_obj = 1, num_cons = 0, bounds = None, 
                 optimizers = [[]], optimum = [[]], ref_point = None, 
                 to_verify = True, out_type = torch, tags = [], is_mixed = False, 
                 to_print_Xscaled = False, is_constrained = False, flag = ''):
        self.dim = dim
        self.num_obj = num_obj
        self.num_cons = num_cons
        self.bounds = bounds
        self.optimizers = optimizers
        self.optimum = optimum
        self.ref_point = ref_point
        self.to_verify = to_verify
        self.out_type = out_type
        self.tags = tags
        self.is_mixed = is_mixed
        self.to_print_Xscaled = to_print_Xscaled
        self.is_constrained = is_constrained
        self.flag = flag


    def scale(self, X, to_verify):
        """
        (Optionally) verifies that X is in the correct range [0, 1] and has the correct dimensions.
        Converts X to a torch.Tensor if necessary and scales X to the problem's bounds.

        Parameters:
            X (array, np.array, or torch.Tensor): data in range of [0, 1]

        Returns:
            X (Torch.tensor): data scaled to bounds

        """

        if not torch.is_tensor(X):
            X = torch.tensor(X)

        if self.to_verify:
            if X.size(1) != self.dim:
                raise DimensionException("Incorrect X dimensions.")
            if torch.max(X) > 1 or torch.min(X) < 0:
                raise RangeException("Incorrect X range: must be [0, 1].")

        if not torch.is_tensor(self.bounds):
            self.bounds = torch.tensor(self.bounds)

        self.bounds = self.bounds.to(X.device)
        
        X_scaled = torch.add(torch.mul(X, (self.bounds[:, 1] - self.bounds[:, 0])), self.bounds[:, 0])

            
        return X_scaled


    def cont_to_disc(self, x, disc_values):
        # Convert continuous value to discrete value
        # Input:
        #   x: continuous value in [0, 1]
        #   disc_values: discrete values
        # Output: discrete value
        idx = torch.floor(x * len(disc_values)).long()
        return disc_values[torch.clamp(idx, 0, len(disc_values)-1)]
    