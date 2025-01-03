from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

from gppluslib.train.trainer import TorchTrainer
from gppluslib.loss.mll import ExactMarginalLogLikelihoodLoss

from model import FileldModel
from data import generate_data

def main():

    # Generate data
    train_x, train_y = generate_data(nx = 30, ny = 30, nsamples = 2)

    # Model
    model = FileldModel(
        x_train = train_x,
        y_train = train_y
    )

    # Optim
    num_iter = 1000
    lr = 0.01
    optimizer = Adam(model.parameters(), lr = lr)
    scheduler = MultiStepLR(optimizer, milestones= np.linspace(0, num_iter, 4).tolist(), gamma=0.75)

    # Loss
    loss = ExactMarginalLogLikelihoodLoss(model = model)

    # Train
    trainer = TorchTrainer(
        model = model,
        loss_func = loss,
        optimizer = optimizer,
        scheduler = scheduler,
        num_iter = num_iter
    )

    trainer.fit()

    model.save()

if __name__ == "__main__":
    main()