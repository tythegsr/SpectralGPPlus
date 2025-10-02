# Import needed libraries
import logging

import gpytorch
import matplotlib.pyplot as plt
import torch

import gpplus
from gpplus.config import configure_logger
from gpplus.training.eval import evaluate_gp_model
from gpplus.training.trainer import GPTrainer
from gpplus.utils.set_seed import set_seed

configure_logger(logging.WARNING)

set_seed(42)

train_x = torch.linspace(0, 1, 10)
train_y = torch.sin(train_x * (2 * torch.pi)) + 0.1 * torch.randn(train_x.size())
test_x = torch.linspace(0, 1, 1000)
test_y = torch.sin(test_x * (2 * torch.pi)) + 0.1 * torch.randn(test_x.size())
# Plot the training data
plt.figure(figsize=(6, 4))
plt.scatter(train_x.numpy(), train_y.numpy(), color="red", label="Train Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training Data")
plt.legend()
plt.show()

# DEfine the GP model and likelihood
likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
model1 = gpplus.models.GPR(train_x, train_y, likelihood1)

print("\n\nNormal likelihood model... \n")
print(model1)

# Train
training_iter = 50

optimizer = torch.optim.Adam(model1.parameters(), lr=0.1)


class PrintLossCallback(gpplus.training.callbacks.Callback):
    def on_epoch_end(self, context: dict):
        print(f"Epoch {context['epoch']} - Loss: {context['loss']:.4f}")


class LossCallback(gpplus.training.callbacks.Callback):
    def __init__(self):
        self.loss = []

    def on_epoch_end(self, context: dict):
        self.loss.append(context["loss"])


printCallback1 = PrintLossCallback()
lossCallback1 = LossCallback()
cllbcks1 = [printCallback1, lossCallback1]

trainer = GPTrainer(
    model=model1, num_runs=1, num_epochs=training_iter, callbacks=cllbcks1, device="cuda", convergence_patience=150
)
trainer.train()

# Evaluate

y_pred, lower, upper, y_std_pred = evaluate_gp_model(model1, test_x)

print("MSE: ", torch.mean((y_pred - test_y) ** 2))

# Plot training loss for model 1
plt.figure(figsize=(6, 4))
plt.plot(range(len(lossCallback1.loss)), lossCallback1.loss, label="Normal Likelihood Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss over Iterations - Normal Likelihood")
plt.legend()
plt.show()

## Learnable priors likelihood model
likelihood2 = gpplus.likelihoods.LearnablePriorsLikelihood()  # change which likelihood is used to test difference
model2 = gpplus.models.GPR(train_x, train_y, likelihood2)

print("\n\nLearnable priors likelihood model... \n")
print(model2)

# Train
training_iter = 50

optimizer = torch.optim.Adam(model2.parameters(), lr=0.1)

# Create new callbacks for the second model
printCallback2 = PrintLossCallback()
lossCallback2 = LossCallback()
cllbcks2 = [printCallback2, lossCallback2]

trainer = GPTrainer(
    model=model2, num_runs=1, num_epochs=training_iter, callbacks=cllbcks2, device="cuda", convergence_patience=150
)
trainer.train()

# Evaluate
y_pred, lower, upper, y_std_pred = evaluate_gp_model(model2, test_x)

print("MSE: ", torch.mean((y_pred - test_y) ** 2))

# Plot training loss for model 2
plt.figure(figsize=(6, 4))
plt.plot(range(len(lossCallback2.loss)), lossCallback2.loss, label="Learnable Priors Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss over Iterations - Learnable Priors Likelihood")
plt.legend()
plt.show()
