import gpytorch
import torch

from gpplus.models import GPR
from gpplus.training import GPTrainer, evaluate_gp_model


def test_gpr_training_single_job():
    # Define toy dataset
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.sin(train_x * (2 * torch.pi)) + 0.1 * torch.randn(train_x.size())

    # Define the model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPR(train_x, train_y, likelihood)

    # Training loop
    trainer = GPTrainer(
        model=model,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.1},
        num_runs=1,
    )
    trainer.train()

    assert model.train_inputs[0].equal(train_x.unsqueeze(1)), "Model's stored train_x does not match input!"
    assert model.train_targets.equal(train_y), "Model's stored train_y does not match input!"

    # Evaluate
    test_x = torch.linspace(0, 1, 51)
    mean, _, _, _ = evaluate_gp_model(model, test_x)

    # Assertions to verify inference
    assert mean.shape == test_x.shape, "Prediction mean has incorrect shape!"
    assert torch.isfinite(mean).all(), "Predictions contain NaN or Inf!"
