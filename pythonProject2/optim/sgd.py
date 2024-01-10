from .optimier import Optimizer
import numpy as np


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr,
        momentum=0.,
        weight_decay=0.,
    ):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = [np.zeros(param.shape) for param in self.params]

    def step(self):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.weight_decay * self.params[i].data
            self.v[i] = self.v[i] * self.momentum + grad
            self.params[i].data -= self.lr * self.v[i]
