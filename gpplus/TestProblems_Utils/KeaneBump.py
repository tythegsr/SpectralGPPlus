import torch

from .base import BenchmarkProblem


class KeaneBump(BenchmarkProblem):

    r'''
    Keane A (1994) Experiences with optimizers instructural design. In: Proceedings of the conference
    on adaptive computing in engineering design and control, pp 14–27
    '''

    # N-D objective, 2 constraints, X = n-by-dim

    tags = {"single_objective", "constrained", "continuous", "ND"}

    def __init__(self, dim=18, is_constrained=False):
        super().__init__(dim, 
                         num_obj = 1, num_cons = 2, 
                         bounds = [[0, 10] * dim], 
                         is_constrained = is_constrained,
                        )

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)

        fx = torch.zeros(X.shape[0], 1).to(torch.float64)
        gx1 = torch.zeros(X.shape[0], 1).to(torch.float64)
        gx2 = torch.zeros(X.shape[0], 1).to(torch.float64)
        
        
        
        for i in range(X.shape[0]):
            x = X[i,:]
            
            cos4 = 0
            cos2 = 1
            sq_denom = 0
            
            pi_sum = 1
            sigma_sum = 0
            
            for j in range(X.shape[1]):
                cos4 += torch.cos(x[j]) ** 4
                cos2 *= torch.cos(x[j]) ** 2
                sq_denom += (j+1) * (x[j])**2
                
                pi_sum *= x[j]
                sigma_sum += x[j]
            
            
            # Objective
            test_function = torch.abs(  (cos4 - 2*cos2) / torch.sqrt(sq_denom)  )
            fx[i] = test_function
    
            # Constraints
            gx1[i] = 0.75 - pi_sum
            gx2[i] = sigma_sum - 7.5* (X.shape[1])
            
        gx = torch.cat((gx1, gx2), 1)
        
        if self.is_constrained:
            return gx, fx
        else:
            return None, fx
