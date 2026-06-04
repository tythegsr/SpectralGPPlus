import torch

from .base import BenchmarkProblem


class Goldstein(BenchmarkProblem):

    r'''
    LVGP paper: https://www.nature.com/articles/s41598-020-60652-9
    '''

    # 2D objective, 0 constraints, X = n-by-2

    tags = {"single_objective", "unconstrained", "continuous", "mixed", "2D"}

    def __init__(self, is_mixed = True, to_print_Xscaled = False):
        super().__init__(dim = 4, 
                         num_obj = 1, 
                         num_cons = 0, 
                         optimizers = [[0, -1]], 
                         optimum = [-3], 
                         bounds = [[-2, 2], [0, 1]],
                         is_mixed = is_mixed,
                         to_print_Xscaled = to_print_Xscaled,
                        )

    def evaluate(self, X, to_verify = True):


        # x0: [-2, 2]
        # x1: {-2, -1, 0, 1, 2}
        X = super().scale(X, to_verify)

        if self.is_mixed:
            X[:,1] = super().cont_to_disc(X[:,1], torch.tensor([-2, -1, 0, 1, 2]))

        if self.to_print_Xscaled:
            print(f'X: {X}')

        fx = ((1 + (X[:,0] + X[:,1] +1)**2
            * (19 - 14*X[:,0] + 3*X[:,0]**2 -14*X[:,1]
                +6*X[:,0]*X[:,1] + 3*X[:,1]**2
                )
            ) *
            (
                30 + (2*X[:,0] - 3*X[:,1])**2
                * (18- 32*X[:,0] + 12*X[:,0]**2 + 48*X[:,1]
                    -36*X[:,0]*X[:,1] + 27*X[:,1]**2
                )
            ))

        return None, -fx.reshape(-1, 1)
