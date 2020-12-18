import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn


class Solver:
    def __init__(self, args, train_set=None, valid_set=None, test_set=None):
        self.device = args.device
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.batch_size = args.batch_size
        self.lr_scheduler = args.lr_scheduler
        self.enforced_teach = args.enforced_teach
        self.factor = args.factor
        self.min_lr = args.min_lr
        self.save_path = args.save_path
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.seed = args.seed
        self.epoch = args.epoch
        self.model = None

    def init_optimizer(self):
        if self.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            return NotImplementedError

    def init_optimizer_scheduler(self):
        if self.lr_scheduler == 'None':
            return None

        elif self.lr_scheduler == 'Plateau':
            assert 0 < self.factor < 1
            assert self.min_lr < self.lr
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=self.factor, min_lr=self.min_lr)

        else:
            raise NotImplementedError

    def init_segmentation_loss(self):
        return nn.BCELoss(reduction='sum').to(self.device)

    def init_topic_classification_loss(self):
        return nn.CrossEntropyLoss(reduction='sum').to(self.device)

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError



