import torch
from torch.autograd import Function
from gpytorch.constraints import Interval


class SoftClampFunction(Function):
    """
    Custom autograd function for SoftClamp transform with proper gradient handling.
    This avoids the torch.where gradient bug.
    """
    @staticmethod
    def forward(ctx, raw_tensor, a, b, eps):
        xL, xR = a + eps, b - eps
        
        # Compute all branches
        left = a + eps * torch.exp((raw_tensor - xL) / eps)
        mid = raw_tensor
        right = b - eps * torch.exp(-(raw_tensor - xR) / eps)
        
        # Create masks
        mask_left = raw_tensor < xL
        mask_right = raw_tensor > xR
        
        # Select output
        result = torch.where(mask_left, left, torch.where(mask_right, right, mid))
        
        # Save for backward
        ctx.save_for_backward(raw_tensor, a, b, eps, mask_left, mask_right)
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        raw_tensor, a, b, eps, mask_left, mask_right = ctx.saved_tensors
        xL, xR = a + eps, b - eps
        
        # Compute gradients for each branch
        # Left branch: d/dx[a + ε * exp((x - xL)/ε)] = exp((x - xL)/ε)
        grad_left = torch.exp((raw_tensor - xL) / eps)
        
        # Middle branch: d/dx[x] = 1
        grad_mid = torch.ones_like(raw_tensor)
        
        # Right branch: d/dx[b - ε * exp(-(x - xR)/ε)] = exp(-(x - xR)/ε)
        grad_right = torch.exp(-(raw_tensor - xR) / eps)
        
        # Select gradient based on which branch was taken
        grad_raw = torch.where(mask_left, grad_left, torch.where(mask_right, grad_right, grad_mid))
        
        # Multiply by upstream gradient
        grad_raw = grad_raw * grad_output
        
        # Return gradients (matching number of forward inputs: raw_tensor, a, b, eps)
        return grad_raw, None, None, None


class SoftClamp(Interval):
    """
    Identity core + exponential tails.
    lower_bound=a, upper_bound=b, margin=ε with 0 < ε < (b-a)/2
    """

    def __init__(self, lower_bound, upper_bound, margin: float = 1e-2, initial_value=None):
        # Let Interval validate/store bounds & initial_value
        super().__init__(lower_bound, upper_bound, transform=None, inv_transform=None, initial_value=None)

        # Prepare margin as a buffer on the right dtype/device
        eps = torch.as_tensor(margin, dtype=self.lower_bound.dtype, device=self.lower_bound.device)

        # ---- VALIDATION: ε > 0 and ε < (b-a)/2 ----
        width = self.upper_bound - self.lower_bound
        tiny_eps = torch.finfo(self.lower_bound.dtype).eps
        if not torch.all(eps > 0):
            raise ValueError(f"margin must be > 0 (got {margin}).")
        if not torch.all(eps < (width / 2) - tiny_eps):
            raise ValueError(
                f"margin must be < (upper_bound - lower_bound)/2. "
                f"Got margin={margin}, width={width.item() if width.numel() == 1 else 'tensor'}."
            )

        self.register_buffer("margin", eps)

        if initial_value is not None:
            self._initial_value = self.inverse_transform(
                torch.as_tensor(initial_value, dtype=self.lower_bound.dtype, device=self.lower_bound.device)
            )

    def transform(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        # a = lower bound, b = upper bound, ε = margin
        # x_raw is the unconstrained value; transform maps x_raw -> x_act in (a, b)
        # Use custom autograd function for proper gradient handling
        return SoftClampFunction.apply(raw_tensor, self.lower_bound, self.upper_bound, self.margin)

    def inverse_transform(self, transformed_tensor: torch.Tensor) -> torch.Tensor:
        # Inverse of the piecewise map (x_act -> x_raw), with safe guards for log(0).
        # Define the join points once and reuse them:
        #   xL = a + ε,  xR = b - ε
        # Branches:
        #   Left  (x_act <= a+ε):   x_raw = xL + ε * log((x_act - a)/ε)          <-- uses log(·)
        #   Middle (a+ε < x_act < b-ε): x_raw = x_act                            <-- identity
        #   Right (x_act >= b-ε):   x_raw = xR - ε * log((b - x_act)/ε)          <-- uses log(·)
        a, b, eps = self.lower_bound, self.upper_bound, self.margin
        xL, xR = a + eps, b - eps

        # avoid log(0): clamp the normalized arguments strictly positive
        # tiny = smallest positive *normal* for the dtype, ensures (·) >= tiny
        tiny = torch.finfo(transformed_tensor.dtype).eps

        left = xL + eps * torch.log(torch.clamp(((transformed_tensor - a) / eps), min=tiny))
        mid = transformed_tensor
        right = xR - eps * torch.log(torch.clamp(((b - transformed_tensor) / eps), min=tiny))

        return torch.where(transformed_tensor < xL, left, torch.where(transformed_tensor > xR, right, mid))

    def __repr__(self):
        return f"SoftClamp({self.lower_bound.item()}, {self.upper_bound.item()}, margin={self.margin.item()})"