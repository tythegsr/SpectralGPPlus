import math

import torch

from .base import BenchmarkProblem


class Test_PFN(BenchmarkProblem):


    def __init__(self, dim=2, flag = None):
        super().__init__(dim, 
                         num_obj = 1, 
                         num_cons = 2, 
                         optimizers = [[0] * dim], 
                         optimum = [[0]], 
                         bounds = [[-5, 10]],
                         flag = flag,
                        )

    def evaluate(self, X, to_verify = True):
        
        from botorch.test_functions import Ackley as Ackley_imported
        from botorch.test_functions.synthetic import Rosenbrock as Rosenbrock_imported

        device = torch.device("cpu")
        dtype = torch.double

        if self.flag == 'single_obj, Unconstrained':
            X = super().scale(X, to_verify)
            n = X.size(0)
            gx = torch.zeros((n, self.num_cons))
    
            fun = Ackley_imported(dim=self.dim, negate=True).to(dtype=dtype, device=device)
            fun.bounds[0, :].fill_(-5)
            fun.bounds[1, :].fill_(10)
    
            fx = fun(X)
            fx = fx.reshape((n, 1))
    
            gx[:, 0] = torch.sum(X,1)
            gx[:, 1] = (torch.norm(X, p=2, dim=1)-5)
            print('The testing function is using Ackley\n')
    
            return None, fx
            
        elif self.flag == 'single_obj, Constrained':
            X = super().scale(X, to_verify)
            n = X.size(0)
            gx = torch.zeros((n, self.num_cons))
    
            fun = Ackley_imported(dim=self.dim, negate=True).to(dtype=dtype, device=device)
            fun.bounds[0, :].fill_(-5)
            fun.bounds[1, :].fill_(10)
    
            fx = fun(X)
            fx = fx.reshape((n, 1))
    
            gx[:, 0] = torch.sum(X,1)
            gx[:, 1] = (torch.norm(X, p=2, dim=1)-5)

            print('The testing function is using Ackley\n')
    
            return gx, fx
            
        elif self.flag == 'multi_obj, Unconstrained':
            X = super().scale(X, to_verify)
            n = X.size(0)
            gx = torch.zeros((n, self.num_cons))

            # Ackley
            fun = Ackley_imported(dim=self.dim, negate=True).to(dtype=dtype, device=device)
            fun.bounds[0, :].fill_(-5)
            fun.bounds[1, :].fill_(10)
    
            fx = fun(X)
            fx = fx.reshape((n, 1))
    
            gx[:, 0] = torch.sum(X,1)
            gx[:, 1] = (torch.norm(X, p=2, dim=1)-5)

            # Rosenbrock
            from botorch.test_functions.synthetic import Rosenbrock as Rosenbrock_imported

            fun = Rosenbrock_imported(dim=self.dim, negate=True)
            fun.bounds[0, :].fill_(-5)
            fun.bounds[1, :].fill_(10)
            print(fun(X))
    
            fx = torch.cat((fx, fun(X).reshape((n, 1))), 1)

            print('The first obj is using Ackley\n and the second is Rosenbrock')
    
            return None, fx

            
        elif self.flag == 'multi_obj, Constrained':
            X = super().scale(X, to_verify)
            n = X.size(0)
            gx = torch.zeros((n, self.num_cons))

            # Ackley
            fun = Ackley_imported(dim=self.dim, negate=True).to(dtype=dtype, device=device)
            fun.bounds[0, :].fill_(-5)
            fun.bounds[1, :].fill_(10)
    
            fx = fun(X)
            fx = fx.reshape((n, 1))
    
            gx[:, 0] = torch.sum(X,1)
            gx[:, 1] = (torch.norm(X, p=2, dim=1)-5)

            # Michalewicz
            from botorch.test_functions.synthetic import Michalewicz as Michalewicz_imported

            fun = Michalewicz_imported(dim=self.dim, negate=True)
            fun.bounds[0, :].fill_(0)
            fun.bounds[1, :].fill_(math.pi)
            # print(fun(X))
    
            n = X.size(0)
    
            fx = torch.cat((fx, fun(X).reshape((n, 1))), 1)

            print('The first obj is using Ackley\n and the second is Michalewicz')
    
            return gx, fx
            

        