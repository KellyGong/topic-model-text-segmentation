import numpy as np
import logging


class EarlyStop:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, lr, val_metric_max=None, patience=5, delta=0, factor=0.1):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.counter = 0
        self.val_metric_max = -np.Inf if val_metric_max is None else val_metric_max
        self.delta = delta
        self.lr = lr
        self.init_lr = lr
        self.factor = factor

    def update(self, validation_metric):
        if validation_metric > self.val_metric_max + self.delta:
            self.val_metric_max = validation_metric
            self.counter = 0
            self.lr = self.init_lr

        else:
            self.counter += 1

        if self.counter == self.patience:
            # reload best checkpoint and update lr
            self.counter = 0
            self.val_metric_max = -np.Inf
            self.lr = self.lr * self.factor
            logging.info('reload best model, induce learning rate!')
            return False

        return True



