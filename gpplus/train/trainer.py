from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from gpplus.train.base import BaseTrainer
from gpplus.models.gpr import GPR
from gpplus.loss.base import Loss

class TorchTrainer(BaseTrainer):
    """
    Trainer.
    """

    def __init__(
        self,
        model: GPR,
        loss_func: Loss,
        optimizer: Optimizer,
        scheduler: LRScheduler = None,
        num_iter: int = 1000
    ):
        super().__init__()

        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_iter = num_iter
    
    def fit(self) -> float:
        """
        Optimize the parameters of the GP model
        """
        self.model.train()
        
        epochs_iter = tqdm(range(self.num_iter), desc='Epoch', position=0, leave=True)
        for j in epochs_iter:        
            self.optimizer.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            desc = f'Epoch {j} - loss {loss.item():.3e}%'
            epochs_iter.set_description(desc)
            epochs_iter.update(1)