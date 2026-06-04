import torch

from .base import BenchmarkProblem


class LassoDNA(BenchmarkProblem):

    r'''
    ...
    '''

    # ND objective, 0 constraints, X = n-by-dim

    tags = {"single_objective", "unconstrained", "continuous", "ND", "extra_imports"}

    def __init__(self):
        super().__init__(dim=180, num_obj = 1, num_cons = 0, bounds = [[-1, 1]])

    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)
        DEVICE = X.device

        import LassoBench
        fx = torch.zeros(X.shape[0],1)
        real_bench = LassoBench.RealBenchmark(pick_data='DNA')
        for i in range(X.shape[0]):
            # loss = real_bench.evaluate(X[i,:].numpy())
            fx[i,0] = -real_bench.evaluate(X[i,:].to(torch.double).detach().cpu().numpy())


        return None, fx.to(DEVICE)