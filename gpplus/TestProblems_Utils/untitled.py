import numpy as np
import torch

from .base import BenchmarkProblem

# from .cont_to_disc import cont_to_disc


#
#
#   Car: 11D objective, 10 constraints
#
#   Reference:
#     Gandomi AH, Yang XS, Alavi AH (2011) Mixed
#     variable structural optimization using firefly
#     algorithm. Computers & Structures 89(23-
#     24):2325–2336
#
#

class bbob(BenchmarkProblem):
    def __init__(self):
        super().__init__(dim = 11,
                         num_obj = 1, 
                         num_cons = 0,
                         bounds = [[1.5, 0.5], [1.35, 0.45], [1.5, 0.5], 
                                    [1.5, 0.5], [1.5, 0.5], [1.5, 0.5], 
                                    [1.5, 0.5], [0.345, 0.192], [0.345, 0.192],
                                    [0.0, -20], [0.0, -20] ])