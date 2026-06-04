import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from .base import BenchmarkProblem


class SVM(BenchmarkProblem):

    r'''
    https://www.sfu.ca/~ssurjano/stybtang.html
    '''

    # 10D objective, 0 constraints, X = n-by-10

    tags = {"single_objective", "unconstrained", "continuous", "10D", "extra_imports"}

    def __init__(self, dim=388):
        
        super().__init__(dim = dim, 
                         num_obj = 1, 
                         num_cons = 0, 
                         optimizers = [[0] * dim], 
                         optimum = [[0] * dim], 
                         bounds = [[0, 1]])
        
    def evaluate(self, X, to_verify = True):
        X = super().scale(X, to_verify)
        n = X.size(0)

        
        DEVICE = X.device

        fx = torch.zeros(X.shape[0],1)
        func = SVMBenchmark()

        for i in range(X.shape[0]):
            fx[i,0] = func(X[i,:].to(torch.double).detach().cpu().numpy())


        return None, fx.to(DEVICE)












class SVMBenchmark:
    def __init__(
            self,
    ):
        self.dims = 388
        self.lb = np.zeros(388,)
        self.ub = np.ones(388,)
        self.X, self.y = self._load_data()

        idxs = np.random.choice(np.arange(len(self.X)), min(10000, len(self.X)), replace=False)
        half = len(idxs) // 2
        self._X_train = self.X[idxs[:half]]
        self._X_test = self.X[idxs[half:]]
        self._y_train = self.y[idxs[:half]]
        self._y_test = self.y[idxs[half:]]

    def _load_data(self):
      #   dir_path = os.path.dirname(os.path.realpath(__file__))
      #   data_path = os.path.join(dir_path, "data", "slice_localization_data.csv")
        data_path = "/home/turbo/rosenyu/TestProblems_Utils/slice_localization_data.csv"
        data = pd.read_csv(data_path).to_numpy()
        X = data[:, :385]
        y = data[:, -1]
        X = MinMaxScaler().fit_transform(X)
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).squeeze()
        return X, y

    def __call__(self, x: np.array):
        x = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        assert x.ndim == 1
        x = x ** 2

        C = 0.01 * (500 ** x[387])
        gamma = 0.1 * (30 ** x[386])
        epsilon = 0.01 * (100 ** x[385])
        length_scales = np.exp(4 * x[:385] - 2)

        svr = SVR(gamma=gamma, epsilon=epsilon, C=C, cache_size=1500, tol=0.001)
        svr.fit(self._X_train / length_scales, self._y_train)
        pred = svr.predict(self._X_test / length_scales)
        error = np.sqrt(np.mean(np.square(pred - self._y_test)))

        return -error
