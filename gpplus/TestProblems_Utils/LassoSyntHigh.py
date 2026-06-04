import torch

from .base import BenchmarkProblem


class LassoSyntHigh(BenchmarkProblem):

    r'''
    ...
    '''

    # ND objective, 0 constraints, X = n-by-dim

    tags = {"single_objective", "unconstrained", "continuous", "ND", "extra_imports"}

    def __init__(self):
        super().__init__(dim=300, num_obj = 1, num_cons = 0, bounds = [[-1, 1]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)
        DEVICE = X.device

        import LassoBench
        fx = torch.zeros(X.shape[0],1)
        synt_bench = LassoBench.SyntheticBenchmark(pick_bench='synt_high')
        for i in range(X.shape[0]):
            fx[i,0] = -synt_bench.evaluate(X[i,:].to(torch.double).detach().cpu().numpy())


        return None, fx.to(DEVICE)