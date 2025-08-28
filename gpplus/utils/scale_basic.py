import os
import math
import torch
import torch.nn.functional as F

def scale(X, l_bounds, u_bounds):
    """Scale samples from [0, 1] to [l_bounds, u_bounds]."""
    return X * (u_bounds.to(X.device) - l_bounds.to(X.device)) + l_bounds.to(X.device)