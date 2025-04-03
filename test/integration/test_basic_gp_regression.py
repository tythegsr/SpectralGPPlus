import gpytorch
import torch

import gpplus.models


def test_gpr_training():
    # Define toy dataset
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.sin(train_x * (2 * torch.pi)) + 0.1 * torch.randn(train_x.size())

    # Define the model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = gpplus.models.GPR(train_x, train_y, likelihood)

    # Set training mode
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    initial_loss = None
    final_loss = None
    training_iter = 50

    # Training loop
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if i == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    # Assertions to verify training
    assert final_loss < initial_loss, f"Loss did not decrease! Initial: {initial_loss}, Final: {final_loss}"
    assert model.train_inputs[0].equal(train_x), "Model's stored train_x does not match input!"
    assert model.train_targets.equal(train_y), "Model's stored train_y does not match input!"

    # Switch to eval mode
    model.eval()
    likelihood.eval()

    # Generate test points and make predictions
    test_x = torch.linspace(0, 1, 51)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x))
        mean = pred.mean

    # Assertions to verify inference
    assert mean.shape == test_x.shape, "Prediction mean has incorrect shape!"
    assert torch.isfinite(mean).all(), "Predictions contain NaN or Inf!"
