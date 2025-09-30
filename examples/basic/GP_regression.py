# Import needed libraries
import logging

import gpytorch
import matplotlib.pyplot as plt
import torch

import gpplus
from gpplus.config import configure_logger
from gpplus.training.trainer import GPTrainer

configure_logger(logging.WARNING)

train_x = torch.linspace(0, 1, 10)
train_y = torch.sin(train_x * (2 * torch.pi)) + 0.1 * torch.randn(train_x.size())

# Plot the training data
plt.figure(figsize=(6, 4))
plt.scatter(train_x.numpy(), train_y.numpy(), color="red", label="Train Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training Data")
plt.legend()
plt.show()

# DEfine the GP model and likelihood

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = gpplus.models.GPR(train_x, train_y, likelihood)

# Train

training_iter = 50
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


class PrintLossCallback(gpplus.training.callbacks.Callback):
    def on_epoch_end(self, context: dict):
        print(f"Epoch {context['epoch']} - Loss: {context['loss']:.4f}")


class LossCallback(gpplus.training.callbacks.Callback):
    def __init__(self):
        self.loss = []

    def on_epoch_end(self, context: dict):
        self.loss.append(context["loss"])


printCallback = PrintLossCallback()
lossCallback = LossCallback()
cllbcks = [printCallback, lossCallback]

trainer = GPTrainer(model=model, num_runs=1, callbacks=cllbcks, device="cuda", convergence_patience=150)
trainer.train()

# Plot training loss
plt.figure(figsize=(6, 4))
plt.plot(range(len(lossCallback.loss)), lossCallback.loss, label="Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss over Iterations")
plt.legend()
plt.show()
