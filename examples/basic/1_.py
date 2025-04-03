import gpytorch
import torch

import gpplus
import gpplus.models

# 1. Define a toy dataset
train_x = torch.linspace(0, 1, 10)
train_y = torch.sin(train_x * (2 * torch.pi)) + 0.1 * torch.randn(train_x.size())

# # 2. Define the GP model and likelihood
# class ExactGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel()
#         )

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = gpplus.models.GPR(train_x, train_y, likelihood)

# 3. Train the model
training_iter = 50
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    if (i + 1) % 10 == 0:
        print(f"Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f}")

# 4. Check the training data from inside the model
#    Note: train_inputs is a tuple. If you had multiple training inputs, they’d be in here as well.
internal_train_x = model.train_inputs[0]
internal_train_y = model.train_targets

print("Accessing training data from the model:")
print("internal_train_x:", internal_train_x)
print("internal_train_y:", internal_train_y)

# 5. Evaluate (testing phase)
#    Normally you'd pass in new test_x here, but let's just reuse the same data for demonstration
model.eval()
likelihood.eval()

test_x = torch.linspace(0, 1, 51)  # More dense for "testing"
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = likelihood(model(test_x))
    mean = pred.mean
    lower, upper = pred.confidence_region()

# 6. Print out the predictions for demonstration
print("\nPredictions on test_x:")
for i, x in enumerate(test_x):
    print(
        f"x = {x.item():.3f} | mean = {mean[i].item():.3f} | "
        f"lower = {lower[i].item():.3f} | "
        f"upper = {upper[i].item():.3f}"
    )
